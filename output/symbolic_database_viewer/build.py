import json, sqlite3, html
from pathlib import Path
root=Path(__file__).resolve().parents[2]
run=root/'reports/part2/symbolic_v3_rq3_20261030_v2'
c=sqlite3.connect('file:'+str(run/'knowledge_base/knowledge.sqlite')+'?mode=ro',uri=True)
entities={q:json.loads(p) for q,p in c.execute('select qid,payload from entities')}
labels={'P569':'Date of birth','P570':'Date of death','P19':'Place of birth','P20':'Place of death','P31':'Instance of','P106':'Occupation','P27':'Citizenship','P21':'Sex or gender','P463':'Member of'}
rows=[]
for q,p,s,payload,source in c.execute('select qid,pid,statement_id,payload,source from facts'):
 d=json.loads(payload); v=d.get('mainsnak',{}).get('datavalue',{}).get('value',d.get('mainsnak',{}).get('snaktype'))
 if isinstance(v,dict): display=v.get('time',v.get('id',v.get('text',json.dumps(v,ensure_ascii=False))))
 else: display=str(v)
 rows.append(dict(qid=q,entity=entities[q].get('labels',{}).get('en',{}).get('value',q),property=p,property_label=labels.get(p,p),value=display,rank=d.get('rank'),raw=d,source=json.loads(source),revision=entities[q].get('lastrevid')))
traces=[]
for p in (run/'traces').glob('*.json'):
 t=json.loads(p.read_text()); s=t['conditions']['symbolic_first']['symbolic'];traces.append(dict(id=t['id'],claim=t['claim'],symbolic=s))
data=json.dumps(dict(rows=rows,traces=traces),ensure_ascii=False).replace('<','\\u003c')
page='''<!doctype html><html><meta charset="utf-8"><title>Saved symbolic knowledge base</title><style>
body{font:16px system-ui;background:#f4f6fa;color:#182537;margin:0}main{max-width:1200px;margin:35px auto;padding:20px}h1{margin-bottom:8px}p{line-height:1.6}.card,details{background:white;padding:20px;border:1px solid #dce2ec;border-radius:12px;margin:16px 0}.metrics{display:flex;gap:20px;flex-wrap:wrap}.metrics strong{font-size:26px}input,select{font:inherit;padding:10px;border:1px solid #bcc7d5;border-radius:7px;max-width:100%}input{width:420px}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #e0e6ef;vertical-align:top}a{color:#155abc}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:440px;overflow:auto}.scroll{overflow:auto}button{padding:9px;cursor:pointer}small{color:#53637a}summary{cursor:pointer}#facts details{margin:0;padding:5px}label{display:inline-block;margin:5px} .good{color:#156842}</style><main>
<a href="http://127.0.0.1:8766/">← Main experiment</a><h1>The saved symbolic database</h1><p>This is the actual SQLite data collected during the <b>new 100-claim RQ3 experiment</b>. This page reads saved results; it makes no new model calls or Wikidata requests.</p>
<div class="metrics"><div class="card"><strong>7</strong><br>Stored entity records</div><div class="card"><strong>293</strong><br>Imported statements</div><div class="card"><strong>1 / 100</strong><br>Claims resolved symbolically</div></div>
<div class="card"><h2>A fact the system actually used</h2><p><b>Claim:</b> Oh Yeon-seo was born in 1987.</p><p><b>Saved fact:</b> Oh Yeon-seo → date of birth (P569) → June 22, 1987.</p><p class="good"><b>Symbolic verdict: SUPPORTS.</b> The stored birth year matches 1987.</p><p>RoBERTa also answered this correctly, so this did not increase accuracy.</p><a href="https://www.wikidata.org/wiki/Q486536?oldid=2547976438" target="_blank" rel="noopener">Wikidata source revision</a></div>
<h2>Browse every stored statement</h2><p>Some records are search candidates with the same name. Stored statements include identifiers and other metadata; 293 statements does not mean 293 usable verification facts. They were imported from Wikidata, not independently checked by a person.</p>
<label>Entity <select id="entity"><option value="">All entities</option></select></label><label>Search <input id="search" placeholder="Try P569, birth, 1987, or a name"></label><p id="count"></p><div class="card scroll"><table><thead><tr><th>Entity / source</th><th>Property</th><th>Stored value</th><th>Rank / full record</th></tr></thead><tbody id="facts"></tbody></table></div>
<h2>What happened for each claim?</h2><p>Select a claim to inspect the parser, lookup outcome, and saved symbolic trace.</p><select id="claims"></select><pre class="card" id="trace"></pre><p><small>Original database: reports/part2/symbolic_v3_rq3_20261030_v2/knowledge_base/knowledge.sqlite. This viewer leaves it unchanged.</small></p></main><script>
const data=DATA; const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const entity=document.querySelector('#entity'),search=document.querySelector('#search');
for(const q of [...new Set(data.rows.map(r=>r.qid))]){let r=data.rows.find(r=>r.qid===q);entity.add(new Option(r.entity+' ('+q+')',q));}
function render(){const text=search.value.toLowerCase(); const rows=data.rows.filter(r=>(!entity.value||r.qid===entity.value)&&(!text||[r.entity,r.qid,r.property,r.property_label,r.value].join(' ').toLowerCase().includes(text)));document.querySelector('#count').textContent=rows.length+' of '+data.rows.length+' statements';document.querySelector('#facts').innerHTML=rows.map(r=>`<tr><td><a href="https://www.wikidata.org/wiki/${esc(r.qid)}?oldid=${esc(r.revision)}" target="_blank" rel="noopener">${esc(r.entity)}</a><br><small>${esc(r.qid)} · revision ${esc(r.revision)}</small></td><td><a href="https://www.wikidata.org/wiki/Property:${esc(r.property)}" target="_blank" rel="noopener">${esc(r.property_label)}</a><br><small>${esc(r.property)}</small></td><td>${esc(r.value)}</td><td>${esc(r.rank)}<details><summary>Full statement + provenance</summary><pre>${esc(JSON.stringify({statement:r.raw,source:r.source},null,2))}</pre></details></td></tr>`).join('');}entity.onchange=search.oninput=render;render();
const claims=document.querySelector('#claims');data.traces.sort((a,b)=>(b.id===218745)-(a.id===218745));for(const t of data.traces)claims.add(new Option(t.id+' · '+t.claim,t.id));claims.onchange=()=>document.querySelector('#trace').textContent=JSON.stringify(data.traces.find(t=>String(t.id)===claims.value),null,2);claims.onchange();
</script></html>'''.replace('DATA',data)
(root/'output/symbolic_database_viewer/index.html').write_text(page)
print('Built viewer:',len(rows),'statements;',len(traces),'claims')
