"""Read-only before/after viewer for the approved three-addition prompt test."""
import ast
import json
from pathlib import Path
from replay_claim_parts_v2 import OUT,PREVIOUS,ROOT


def main():
    summary=json.loads((OUT/'summary.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    if 'before_derived_correct' not in summary:raise ValueError('Complete the paired audit first')
    rows=json.loads((OUT/'comparison.json').read_text());cases=[]
    for row in rows:
        before=json.loads((PREVIOUS/'traces'/('%s.json'%row['id'])).read_text())
        after=json.loads((OUT/'traces'/('%s.json'%row['id'])).read_text())
        cases.append(dict(id=row['id'],claim=row['claim'],gold=row['gold'],before=before,after=after))
    # Reuse the existing trace-viewer layout without changing the earlier report.
    tree=ast.parse((ROOT/'src/reporting/build_claim_parts_report.py').read_text())
    page=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='page' for t in n.targets))
    page=page.replace('← Original temporary-RAG pilot','← Previous parts-prompt test').replace('href="/ephemeral-rag"','href="/claim-parts"')
    page=page.replace('FEVER · Essential-parts prompt replay','FEVER · Parts prompt v2').replace('Essential parts. Explicit verdicts.','Three additions. Same evidence.')
    page=page.replace('<h1>Three additions. Same evidence.</h1>','<h1>Three additions. Same evidence.</h1><a href="#instruction">See the new prompt — additions underlined ↓</a>')
    page=page.replace('The user-supplied prompt requests part verdicts and an overall label. The application computes a separate label for comparison; disagreement is shown rather than hidden.','Only the three approved general instruction additions changed. The evidence, model, schema and generation settings match the earlier parts-prompt test.')
    page=page.replace("['Previous RAG',r.before.label]","['Before: model',r.before.model_label],['Before: derived',r.before.derived_label]")
    page=page.replace('Model label fixed an error','Derived label fixed an error').replace('Model label introduced an error','Derived label introduced an error')
    page=page.replace("r.before.label!==r.gold&&r.after.model_label===r.gold","r.before.derived_label!==r.gold&&r.after.derived_label===r.gold").replace("r.before.label===r.gold&&r.after.model_label!==r.gold","r.before.derived_label===r.gold&&r.after.derived_label!==r.gold")
    page=page.replace("r.before.label+' → '+r.after.model_label","r.before.derived_label+' → '+r.after.derived_label")
    page=page.replace('r.before.label','r.before.derived_label')
    page=page.replace('Previous RAG response','Previous parts-prompt response')
    page=page.replace("[['Previous RAG',D.summary.previous_rag_correct],['Updated model label',D.summary.model_label_correct],['Code-derived label',D.summary.derived_label_correct]]", "[['Before: model',D.summary.before_model_correct],['Before: derived',D.summary.before_derived_correct],['After: model',D.summary.model_label_correct],['After: derived',D.summary.derived_label_correct]]")
    page=page.replace('grid-template-columns:repeat(3,1fr)','grid-template-columns:repeat(4,1fr)')
    page=page.replace('Exact user-approved instruction','Exact tested instruction · new additions underlined')
    page=page.replace("$('instruction').textContent=D.protocol.prompt;", "let rest=D.protocol.prompt;for(const addition of D.protocol.additions){const index=rest.indexOf(addition);if(index<0)throw Error('Prompt addition missing');$('instruction').append(document.createTextNode(rest.slice(0,index)),el('u',addition));rest=rest.slice(index+addition.length);}$('instruction').append(document.createTextNode(rest));")
    data=json.dumps(dict(summary=summary,protocol=protocol,cases=cases),ensure_ascii=False).replace('<','\\u003c')
    (OUT/'index.html').write_text(page.replace('PAYLOAD',data));print(OUT/'index.html')

if __name__=='__main__':main()
