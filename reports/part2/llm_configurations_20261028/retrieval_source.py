"""Experimental exact-prefix entity anchoring plus local sentence BM25.

No gold labels or annotations are read. Retains original retrieval separately.
"""
from rank_bm25 import BM25Okapi

from retrieve_title_aware import STOPWORDS, find_titles, tokenize


def exact_subject_page(connection, claim):
    words=claim.strip().split()
    versions=[words]
    if words and words[0].lower() in ('the','a','an'):versions.append(words[1:])
    choices={}
    for version in versions:
        for count in range(1,min(10,len(version))+1):
            phrase=' '.join(version[:count]).strip(' "“”')
            if not phrase or phrase.casefold() in STOPWORDS:continue
            for title in {phrase,phrase[:1].upper()+phrase[1:]}:
                key=title.replace(' ','_').replace('(','-LRB-').replace(')','-RRB-').replace(':','-COLON-')
                choices[key]=max(choices.get(key,0),count)
    if not choices:return None
    keys=sorted(choices)
    rows=connection.execute('SELECT page_id FROM wiki_titles WHERE page_id IN ('+','.join('?' for _ in keys)+')',keys).fetchall()
    if not rows:return None
    return min((r[0] for r in rows),key=lambda k:(-choices[k],k))


def retrieve_entity_anchored(connection, claim, top_k=5):
    anchor=exact_subject_page(connection,claim)
    titles=list(dict.fromkeys(([anchor] if anchor else [])+find_titles(connection,claim,10)))[:11]
    if not titles:return []
    rows=connection.execute('SELECT page_id,sentence_id,text FROM page_sentences WHERE page_id IN ('+','.join('?' for _ in titles)+')',titles).fetchall()
    if not rows:return []
    stop=STOPWORDS-{'not'}
    def terms(text):return [w for w in tokenize(text) if w not in stop and len(w)>1]
    corpus=[terms(r[2]) for r in rows]
    scores=BM25Okapi(corpus).get_scores(terms(claim)) if any(corpus) else [0.]*len(rows)
    positions={page:i for i,page in enumerate(titles)}
    ranked=[dict(page_id=page,sentence_id=int(sid),text=text,
        score=float(score)+(2. if page==anchor else 0.)+.05/(1+positions[page]),
        entity_anchor=page==anchor) for (page,sid,text),score in zip(rows,scores)]
    ranked.sort(key=lambda r:(-r['score'],positions[r['page_id']],r['sentence_id']))
    return ranked[:top_k]
