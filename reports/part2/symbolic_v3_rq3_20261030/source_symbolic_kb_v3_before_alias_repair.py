"""Dynamic run-local Wikidata fact store and conservative rule verifier.

Facts come from Wikidata responses, never the input claim. Unsupported wording,
ambiguity and uncertain statements abstain. HTTP failures remain explicit.
"""
import calendar
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

import requests

API='https://www.wikidata.org/w/api.php'
MONTHS={name.casefold():i for i,name in enumerate(calendar.month_name) if name}
for i,name in enumerate(calendar.month_abbr):
    if name:MONTHS[name.casefold()]=i
TIMEWORDS=set(MONTHS)|{x.casefold() for x in calendar.day_name}|{x.casefold() for x in calendar.day_abbr}|{'yesterday','today','tomorrow','century','spring','summer','autumn','fall','winter'}
OCCUPATIONS={'actor','actress','singer','writer','politician','director','footballer','musician','scientist','artist','dancer'}


def now():return datetime.now(timezone.utc).isoformat()
def norm(text):return ' '.join(re.sub(r'[^\w]+',' ',unicodedata.normalize('NFKC',text).casefold()).split())


def parse_date(text):
    text=text.strip().strip('.').strip()
    if re.fullmatch(r'(?:the )?\d{3}0s',text,re.I):
        year=int(re.search(r'\d{4}',text)[0]);return dict(year=year,decade=True)
    if re.fullmatch(r'\d{4}',text):return dict(year=int(text))
    m=re.fullmatch(r'([A-Za-z]+)(?:\s+(\d{1,2})(?:st|nd|rd|th)?)?(?:,?\s+(\d{4}))?',text,re.I)
    if m and m[1].casefold() in MONTHS:
        result=dict(month=MONTHS[m[1].casefold()])
        if m[2]:result['day']=int(m[2])
        if m[3]:result['year']=int(m[3])
        if 'day' in result and not 1<=result['day']<=calendar.monthrange(result.get('year',2000),result['month'])[1]:return None
        return result
    m=re.fullmatch(r'(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?',text,re.I)
    if m and m[2].casefold() in MONTHS:
        return parse_date(m[2]+' '+m[1]+(', '+m[3] if m[3] else ''))
    m=re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})',text)
    if m:
        try:datetime(int(m[1]),int(m[2]),int(m[3]))
        except ValueError:return None
        return dict(year=int(m[1]),month=int(m[2]),day=int(m[3]))
    return None


def plan(claim):
    text=claim.strip().rstrip('.').strip()
    m=re.fullmatch(r'(.+?) ((?:was|is) born|died) (?:in|on) (.+)',text,re.I)
    if m:
        subject,event,value=m.groups();date_prop='P570' if event.casefold()=='died' else 'P569'
        location_prop='P20' if date_prop=='P570' else 'P19'
        date=parse_date(value)
        if date:return dict(subject=subject,human=True,parts=[dict(property=date_prop,kind='date',expected=date)])
        compound=re.fullmatch(r'(.+?) on (.+)',value,re.I)
        if compound and parse_date(compound[2]):
            return dict(subject=subject,human=True,parts=[dict(property=location_prop,kind='entity',expected=compound[1]),
                dict(property=date_prop,kind='date',expected=parse_date(compound[2]))])
        # Never send recognizable or malformed dates to a location property.
        if re.search(r'\d|\b(?:'+ '|'.join(TIMEWORDS)+r')\b',value,re.I):return None
        if re.search(r'\b(?:and|or|not|only|before|after|during)\b',value,re.I):return None
        return dict(subject=subject,human=True,parts=[dict(property=location_prop,kind='entity',expected=value)])
    m=re.fullmatch(r'(.+?) (?:is|was) an? (.+)',text,re.I)
    if m and m[2].casefold() in OCCUPATIONS:
        return dict(subject=m[1],human=True,parts=[dict(property='P106',kind='entity',expected=m[2])])
    m=re.fullmatch(r'(.+?) is a member of (.+)',text,re.I)
    if m and not re.search(r'\b(?:and|or|not|only|since|until)\b',m[2],re.I):
        return dict(subject=m[1],human=False,parts=[dict(property='P463',kind='entity',expected=m[2])])
    return None


def labels(entity):
    return {norm(x) for x in [entity.get('labels',{}).get('en',{}).get('value','')]+
            [a['value'] for a in entity.get('aliases',{}).get('en',[])] if x}


def statements(entity,prop):
    return [s for s in entity.get('claims',{}).get(prop,[]) if s.get('rank')!='deprecated']


def compare_date(active,expected):
    """Only directly contradict an adequately precise, unique Gregorian date."""
    values=[]
    for s in active:
        if s.get('qualifiers'):return None,'qualified_date_requires_richer_reasoning'
        v=s.get('mainsnak',{}).get('datavalue',{}).get('value')
        if not isinstance(v,dict):return None,'unknown_date_value'
        if not v.get('calendarmodel','').endswith('/Q1985727') or v.get('before',0) or v.get('after',0):
            return None,'uncertain_or_unsupported_calendar'
        m=re.fullmatch(r'\+(\d+)-(\d{2})-(\d{2})T00:00:00Z',v.get('time',''))
        required=11 if 'day' in expected else 10 if 'month' in expected else 9
        if not m or v.get('precision',0)<required:return None,'insufficient_date_precision'
        actual=tuple(int(x) for x in m.groups())
        if not 1<=actual[0]<=9999:return None,'unsupported_year'
        if v.get('precision',0)>=10 and not 1<=actual[1]<=12:return None,'invalid_month_value'
        if v.get('precision',0)>=11:
            try:datetime(*actual)
            except ValueError:return None,'invalid_day_value'
        values.append(actual)
    if not values:return None,'missing_date'
    if len(set(values))!=1:return None,'conflicting_date_values'
    actual=dict(zip(['year','month','day'],values[0]))
    matches=all((value<=actual['year']<=value+9) if key=='year' and expected.get('decade') else actual[key]==value
                for key,value in expected.items() if key!='decade')
    return ('SUPPORTS' if matches else 'REFUTES'),'direct_date_comparison'


class FactStore:
    def __init__(self,folder,offline=False):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.snapshots=self.folder/'snapshots';self.snapshots.mkdir(exist_ok=True)
        self.db=sqlite3.connect(self.folder/'knowledge.sqlite')
        self.db.executescript('CREATE TABLE IF NOT EXISTS entities(qid TEXT PRIMARY KEY,payload TEXT,source TEXT);'
                             'CREATE TABLE IF NOT EXISTS facts(qid TEXT,pid TEXT,statement_id TEXT,payload TEXT,source TEXT,PRIMARY KEY(qid,pid,statement_id));')
        self.session=requests.Session();self.session.trust_env=False
        self.session.headers['User-Agent']='COSC874SymbolicVerifier/0.3 (academic fact-checking experiment)'
        self.offline=offline;self.calls=0;self.hits=0;self.last_request=0.;self.events=[]
    def query(self,**params):
        key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest();path=self.snapshots/(key+'.json')
        if path.exists():
            saved=json.loads(path.read_text());self.hits+=1
            if saved['params']!=params:raise ValueError('Snapshot key mismatch')
        else:
            if self.offline:raise RuntimeError('Missing offline snapshot')
            for attempt in range(3):
                time.sleep(max(0,.5-(time.monotonic()-self.last_request)))
                self.last_request=time.monotonic();self.calls+=1
                try:
                    response=self.session.get(API,params={**params,'format':'json','maxlag':5},timeout=(5,10))
                    response.raise_for_status();data=response.json()
                    if 'error' in data:raise RuntimeError(str(data['error']))
                    break
                except (requests.RequestException,ValueError,RuntimeError):
                    if attempt==2:raise RuntimeError('Wikidata unavailable; explicit operational fallback')
                    time.sleep(2**attempt)
            saved=dict(params=params,response=data,fetched_utc=now(),response_sha256=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest())
            with path.open('x') as stream:json.dump(saved,stream,ensure_ascii=False)
        self.events.append(dict(params=params,snapshot_file=path.name,fetched_utc=saved['fetched_utc']))
        return saved['response']
    def entity(self,qid):
        cached=self.db.execute('SELECT payload FROM entities WHERE qid=?',(qid,)).fetchone()
        if cached:
            self.events.append(dict(entity_cache_hit=qid));return json.loads(cached[0])
        data=self.query(action='wbgetentities',ids=qid,props='info|claims|labels|aliases',languages='en')
        entity=data.get('entities',{}).get(qid,{})
        if not entity or 'missing' in entity:raise RuntimeError('Entity unavailable')
        source=json.dumps(self.events[-1])
        self.db.execute('INSERT INTO entities VALUES(?,?,?)',(qid,json.dumps(entity),source))
        for pid,ss in entity.get('claims',{}).items():
            for s in ss:
                ident=s.get('id') or hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
                self.db.execute('INSERT INTO facts VALUES(?,?,?,?,?)',(qid,pid,ident,json.dumps(s),source))
        self.db.commit();return entity
    def resolve(self,name,human=False):
        result=self.query(action='wbsearchentities',search=name,language='en',limit=50)
        if 'search-continue' in result:return None,'truncated_entity_search'
        candidates=[r['id'] for r in result.get('search',[]) if norm(name) in
                    {norm(r.get('label',''))}|{norm(a) for a in r.get('aliases',[])}]
        entities=[]
        for qid in dict.fromkeys(candidates):
            entity=self.entity(qid)
            if norm(name) not in labels(entity):continue
            if human and not any(s.get('mainsnak',{}).get('datavalue',{}).get('value',{}).get('id')=='Q5'
                                 for s in statements(entity,'P31')):continue
            entities.append(entity)
        if len(entities)!=1:return None,'unresolved_or_ambiguous_entity'
        return entities[0],'unique_label_or_alias'
    def close(self):self.session.close();self.db.close()


def verify(claim,store):
    started=time.perf_counter();before=store.calls;store.events=[]
    result=dict(claim=claim,label=None,reason='unsupported_claim_structure',parts=[])
    try:
        parsed=plan(claim);result['parsed']=parsed
        if parsed is None:return result
        entity,reason=store.resolve(parsed['subject'],parsed['human'])
        result['entity_resolution']=reason
        if entity is None:
            result['reason']=reason;return result
        qid=entity['id'];result['subject_id']=qid
        result['source_url']='https://www.wikidata.org/wiki/'+qid
        result['source_revision']=entity.get('lastrevid')
        for part in parsed['parts']:
            active=statements(entity,part['property']);decision=None
            if part['kind']=='date':decision,reason=compare_date(active,part['expected'])
            else:
                reason='missing_or_unmatched_property'
                place_ids={s.get('mainsnak',{}).get('datavalue',{}).get('value',{}).get('id')
                           for s in active if isinstance(s.get('mainsnak',{}).get('datavalue',{}).get('value'),dict)}
                ambiguous_place=part['property'] in {'P19','P20'} and len(place_ids-{None})>1
                if ambiguous_place:reason='multiple_place_values'
                for s in ([] if ambiguous_place else active):
                    if s.get('qualifiers'):continue
                    value=s.get('mainsnak',{}).get('datavalue',{}).get('value')
                    if isinstance(value,dict) and value.get('id'):
                        target=store.entity(value['id'])
                        if norm(part['expected']) in labels(target):decision,reason='SUPPORTS','exact_value_label_or_alias';break
                # A different occupation/place/membership does not refute the requested fact.
            result['parts'].append(dict(**part,verdict=decision,reason=reason,statements=active))
        verdicts=[p['verdict'] for p in result['parts']]
        result['label']='REFUTES' if 'REFUTES' in verdicts else 'SUPPORTS' if verdicts and all(v=='SUPPORTS' for v in verdicts) else None
        result['reason']='complete_symbolic_comparison' if result['label'] else 'incomplete_symbolic_evidence'
    except (requests.RequestException,RuntimeError) as exc:
        result['reason']='service_failure_fallback';result['error']=str(exc)
    finally:
        result.update(queries=list(store.events),http_calls=store.calls-before,wall_seconds=time.perf_counter()-started)
    return result
