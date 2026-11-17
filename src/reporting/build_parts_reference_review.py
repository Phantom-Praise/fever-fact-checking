"""Show FEVER reference evidence for the original parts prompt's three errors.

Read-only inspection: preserve reference labels, predictions and scores.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/part2/parts_prompt_reference_review_20261029'
RUN=ROOT/'reports/part2/parts_prompt_replay_20261029'
DATA=ROOT/'data/official_fever/shared_task_dev.jsonl'

NOTES={130325:('Possible annotation inconsistency','FEVER marks this REFUTES, but its own reference says dark matter is distinct from neutrinos. The cited sentence appears to support the claim. Flag this for review; keep the original label and score unchanged.'),
40442:('Evidence was present; the model missed it','The saved model input already contained E8: “While the film won some minor international awards, it was widely viewed as a critical failure.” The NEI answer overlooked explicit evidence of awards.'),
4807:('Inspect evidence selection and interpretation','FEVER cites an explicit distinction between dark matter and dark energy. Our model received different excerpts from newer Wikipedia renders and answered NEI. The reference sentence and actual supplied text are shown separately; this page does not automatically attribute the failure to retrieval alone.')}


def main():
    rows=[json.loads(l) for l in (RUN/'results.jsonl').read_text().splitlines()]
    gold={r['id']:r for r in map(json.loads,DATA.read_text().splitlines())}
    errors=[r for r in rows if r['derived_label']!=gold[r['id']]['label']]
    assert {r['id'] for r in errors}==set(NOTES)
    conn=sqlite3.connect('file:'+str(ROOT/'data/processed/wiki_fts.db')+'?mode=ro',uri=True);cases=[]
    try:
        for r in errors:
            g=gold[r['id']];trace=json.loads((RUN/'traces'/('%s.json'%r['id'])).read_text());groups=[]
            for group in g['evidence']:
                refs=[]
                for ann in group:
                    if ann[2] is None:continue
                    text=conn.execute('SELECT text FROM page_sentences WHERE page_id=? AND sentence_id=?',(ann[2],ann[3])).fetchone()
                    if text is None:raise ValueError('Reference sentence missing from local FEVER snapshot')
                    title=ann[2].replace('-LRB-','(').replace('-RRB-',')')
                    refs.append(dict(page_id=ann[2],sentence_id=ann[3],text=text[0],text_sha256=hashlib.sha256(text[0].encode()).hexdigest(),current_url='https://en.wikipedia.org/wiki/'+quote(title,safe='_()')))
                groups.append(refs)
            excerpts=[]
            for ident,e in trace['supplied_evidence'].items():
                relevant=any(w in e['text'].lower() for w in (['award','won'] if r['id']==40442 else ['neutrino','dark energy']))
                if relevant:excerpts.append(dict(id=ident,**e))
            cases.append(dict(id=r['id'],claim=g['claim'],reference_label=g['label'],model_label=trace['model_label'],derived_label=trace['derived_label'],reference_groups=groups,reference_raw=g['evidence'],model_parts=trace['parts'],model_excerpts=excerpts,all_supplied_evidence=trace['supplied_evidence'],note_title=NOTES[r['id']][0],note=NOTES[r['id']][1]))
    finally:conn.close()
    OUT.mkdir(exist_ok=True)
    payload=dict(cases=cases,dataset=str(DATA.relative_to(ROOT)),dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),scoring_unchanged=True,reference_text_source='Local FEVER Wikipedia sentence snapshot; current Wikipedia may differ.')
    (OUT/'evidence.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FEVER · Reference evidence review</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f6f5;color:#183c42;font:15px/1.65 system-ui}main{max-width:1150px;margin:auto;padding:30px}h1{font-size:36px;letter-spacing:-1px;margin-bottom:8px}h2{font-size:23px;line-height:1.4}a{color:#126b68}nav{display:flex;gap:15px;flex-wrap:wrap;margin:22px 0}.card{background:white;border:1px solid #d6e2df;border-radius:15px;padding:24px;margin:22px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.tag{display:inline-block;background:#e7eeef;padding:7px 12px;margin:5px 7px 5px 0;border-radius:7px;font-size:12px}.note{background:#fff4df;border-left:4px solid #c7994e;padding:15px;margin:18px 0}blockquote{margin:12px 0;padding:15px;border-left:4px solid #338c78;background:#eff7f3}small,.muted{color:#617b7e;font-size:12px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:450px;overflow:auto;background:#f0f4f4;padding:15px;font:12px/1.6 monospace}details{margin:18px 0}button{font:inherit;border:1px solid #a6c8bb;background:white;border-radius:8px;padding:9px 14px;cursor:pointer}@media(max-width:750px){main{padding:18px}.grid{grid-template-columns:1fr}.card{padding:18px}h1{font-size:29px}}
</style><main><a href="/claim-parts">← Original 7/10 parts-prompt experiment</a><h1>What FEVER actually cites.</h1><p>The three disagreements with FEVER in the code-derived results. These are reference sentences, not explanations written by FEVER.</p><p class="muted">Quotes come from the local FEVER Wikipedia snapshot. Source links open current Wikipedia, which may have changed. All labels and recorded scores remain unchanged.</p><nav id="nav"></nav><div id="cases"></div><button id="download">Download reference evidence JSON ↓</button></main><script type="application/json" id="data">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);function el(t,text,c){const n=document.createElement(t);if(text!=null)n.textContent=text;if(c)n.className=c;return n;}function detail(title,value){const d=el('details');d.append(el('summary',title),el('pre',typeof value==='string'?value:JSON.stringify(value,null,2)));return d;}function readable(s){return s.replaceAll('-LRB-','(').replaceAll('-RRB-',')');}
for(const c of D.cases){const link=el('a','#'+c.id);link.href='#'+c.id;$('nav').append(link);const card=el('section',null,'card');card.id=c.id;card.append(el('small','#'+c.id),el('h2',c.claim));for(const [name,v]of [['FEVER reference',c.reference_label],['Llama overall',c.model_label],['Code-derived',c.derived_label]])card.append(el('span',name+': '+v,'tag'));const note=el('div',null,'note');note.append(el('strong',c.note_title),el('div',c.note));card.append(note);const grid=el('div',null,'grid'),gold=el('div'),model=el('div');gold.append(el('h3','FEVER’s reference evidence'));for(const [index,group]of c.reference_groups.entries()){gold.append(el('small','Sufficient evidence set '+(index+1)));for(const ref of group){gold.append(el('b',ref.page_id+' · sentence '+ref.sentence_id),el('blockquote',readable(ref.text)),detail('Exact FEVER sentence text',ref.text));const a=el('a','Open current Wikipedia article ↗');a.href=ref.current_url;gold.append(a);}}model.append(el('h3','What our model concluded'));for(const p of c.model_parts)model.append(el('strong',p.verdict),el('p',p.part),el('p',p.justification,'muted'));model.append(el('h4','Example from the actual model input'),el('small','Shown for inspection; not necessarily cited by the model.'));const keyId=({130325:'E9',40442:'E8',4807:'E22'})[c.id];for(const e of c.model_excerpts.filter(e=>e.id===keyId))model.append(el('small',e.id+' · '+e.title),el('blockquote',e.text));if(!c.model_excerpts.length)model.append(el('p','Expand all supplied excerpts below to inspect the complete model input.'));grid.append(gold,model);card.append(grid,detail('All actual model-input excerpts',c.all_supplied_evidence),detail('Original FEVER annotation references',c.reference_raw));const trace=el('a','Open the full original prompt and model response →');trace.href='/claim-parts#'+c.id;card.append(trace);$('cases').append(card);}$('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(D,null,2)],{type:'application/json'})),a=el('a');a.href=url;a.download='fever-reference-evidence-three-cases.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
</script></html>'''
    (OUT/'index.html').write_text(page.replace('PAYLOAD',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
    print('Built reference review for',len(cases),'claims. Scores unchanged.')

if __name__=='__main__':main()
