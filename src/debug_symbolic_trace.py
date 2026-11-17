"""Replay the historical symbolic verifier with Python line and query traces.

Reads saved Wikidata snapshots only. Does not change rules, labels or scores.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

from experiment_symbolic_hybrid import WikidataVerifier
from experiment_symbolic_v2 import SnapshotVerifier

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/part2/step2_final_20261023/heldout100'
FILES=[ROOT/'src/experiment_symbolic_v2.py',ROOT/'src/experiment_symbolic_hybrid.py']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def short(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,re.Match):return {'groups':list(value.groups()),'matched':value.group(0)}
    if isinstance(value,(list,tuple,set)):
        values=sorted(value) if isinstance(value,set) else list(value)
        return [short(v) for v in values[:15]]+(['[additional values omitted from line summary; raw response retained]'] if len(values)>15 else [])
    if isinstance(value,dict):
        return dict(keys=list(value)[:15],entity_id=value.get('id'),
                    claim_properties=list(value.get('claims',{})))
    return type(value).__name__


class AuditVerifier(SnapshotVerifier):
    def __init__(self,snapshot):
        super().__init__(snapshot,offline=True)
        self.queries=[]
        def forbid(*args,**kwargs):raise AssertionError('Network forbidden in historical replay')
        self.session.get=forbid
    def query(self,**params):
        result=super().query(**params)
        key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
        self.queries.append(dict(params=params,snapshot_file=key+'.json',response=result))
        return result


def replay_claim(verifier,claim):
    steps=[]
    source_lines={str(p):p.read_text().splitlines() for p in FILES}
    tracked={'claim','subject','object_text','property_id','subject_id','name','human',
             'matches','match','ids','names','candidates','year','decade','prop','entity_id'}
    def trace(frame,event,arg):
        path=frame.f_code.co_filename
        if path not in source_lines:return None
        if event in ('line','return'):
            line=frame.f_lineno
            entry=dict(event=event,function=frame.f_code.co_name,file=str(Path(path).relative_to(ROOT)),line=line,
                       source=source_lines[path][line-1],locals={k:short(v) for k,v in frame.f_locals.items() if k in tracked})
            if event=='return':entry['returned']=short(arg)
            steps.append(entry)
        return trace
    verifier.queries=[]
    previous=sys.gettrace()
    try:
        sys.settrace(trace)
        prediction,reason=verifier.verify(claim)
    finally:sys.settrace(previous)
    return dict(symbolic_prediction=prediction,reason=reason,steps=steps,queries=list(verifier.queries))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    original=json.loads((SOURCE/'summary.json').read_text())
    for name,expected in original['snapshot_sha256'].items():
        if digest(SOURCE/'wikidata_snapshot'/name)!=expected:raise ValueError('Historical snapshot changed')
    before={str(p):digest(p) for p in FILES}
    claims={r['id']:r['claim'] for r in map(json.loads,(SOURCE/'claims_without_labels.jsonl').read_text().splitlines())}
    saved=[r for r in map(json.loads,(SOURCE/'pipeline_events.jsonl').read_text().splitlines())
           if r['pass']=='initial_online' and r['condition']=='hybrid_v2']
    verifier=AuditVerifier(SOURCE/'wikidata_snapshot');cases=[]
    try:
        for row in saved:
            result=replay_claim(verifier,claims[row['id']])
            if result['symbolic_prediction']!=row['symbolic_prediction'] or result['reason']!=row['reason']:
                raise ValueError('Historical symbolic decision did not reproduce: '+str(row['id']))
            cases.append(dict(id=row['id'],claim=claims[row['id']],saved_fallback_label=row['prediction'],
                              saved_reference_label=row['gold_label_id'],**result))
        if verifier.calls!=0:raise AssertionError('Historical replay made HTTP requests')
    finally:verifier.session.close()
    if any(digest(p)!=value for p,value in before.items()):raise ValueError('Source changed during replay')
    code=[]
    for function in [SnapshotVerifier.verify,WikidataVerifier.search_exact,WikidataVerifier.verify]:
        lines,start=inspect.getsourcelines(function)
        code.append(dict(name=function.__qualname__,file=str(Path(inspect.getsourcefile(function)).relative_to(ROOT)),
                         start=start,text=''.join(f'{i:4d} {s}' for i,s in enumerate(lines,start))))
    result=dict(n=len(cases),replayed_matches=len(cases),http_requests=0,
                reason_counts=dict(Counter(c['reason'] for c in cases)),source_sha256=before,
                snapshot='Historical October 23 heldout100 Wikidata snapshot; no current facts fetched.',
                controls='Rules unchanged. Python execution traced; predictions/reasons checked against every saved event. '
                         'Debug timings are not a new latency benchmark.',
                findings=[
                    '96 claims returned unsupported_claim_pattern before any Wikidata query: narrow rule coverage.',
                    'Samuel Witwer search returned Sam Witwer with Samuel Witwer as an alias; inherited search_exact checks labels only.',
                    'October, May and November expressions pass the temporal guard and reach P19 birthplace rules: a parsing defect.',
                    'David Beckham search contained two exact labels; abstention reflects unresolved ambiguity, not an HTTP outage.',
                    'The Charcot response already contained birth date P569 = 1825-11-29, but the parser checked P19 places instead.',
                    'Halle Berry object_text retained Texas on August 14, 1966 and was compared with Cleveland label/aliases as one string.',
                    'V2 inherits place matching from v1. These results audit this narrow implementation, not symbolic verification in general.'
                ],code=code,cases=cases)
    out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    template=(ROOT/'web/evidence_explorer/symbolic-debug.html').read_text()
    embedded=json.dumps(result,ensure_ascii=False).replace('<','\\u003c')
    (out/'index.html').write_text(template.replace('/* EMBEDDED_AUDIT */',embedded))
    (out/'README.md').write_text('# Historical symbolic verifier audit\n\nAll 100 symbolic decisions and reasons reproduced offline from the original snapshots. No rules, models, labels or recorded results were changed.\n\n'+ '\n'.join('- '+f for f in result['findings'])+'\n')
    print(json.dumps({k:result[k] for k in ['n','replayed_matches','http_requests','reason_counts','findings']},indent=2))


if __name__=='__main__':main()
