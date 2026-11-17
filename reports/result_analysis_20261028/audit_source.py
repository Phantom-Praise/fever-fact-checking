import csv,hashlib,json,sys,zipfile
from pathlib import Path
from html.parser import HTMLParser
sys.path.insert(0,str(Path.cwd()/'src'))
from experiment_token_budget import per_class
root=Path.cwd();out=root/'reports/result_analysis_20261028'
manifest=json.loads((out/'evidence_manifest.json').read_text())
for source in manifest:assert hashlib.sha256((root/source['path']).read_bytes()).hexdigest()==source['sha256'],source['id']
main=list(csv.DictReader((out/'main_benchmark.csv').open()))
original=json.loads((root/'reports/part2/official_fever_dev_20261023/summary.json').read_text())
for row,key in zip(main[:3],['claim_only_H2','retrieved_roberta','calibrated_ensemble']):
 s=original['conditions'][key];assert row['Accuracy']==f"{100*s['label_accuracy']:.2f}%";assert row['Macro-F1']==f"{s['classification_macro_f1']:.4f}";assert row['Strict FEVER']==f"{100*s['fever_score']:.2f}%"
assert main[3]['Accuracy']=='74.84%' and main[3]['Strict FEVER']=='70.69%'
assert main[4]['Accuracy']=='78.29%' and main[4]['Strict FEVER']=='76.11%'
h=list(csv.DictReader((out/'hypothesis_evidence.csv').open()));assert len(h)==5
assert h[1]['Verdict']=='No convincing support';assert 'uncertain' in h[3]['Verdict'];assert h[4]['Verdict'].startswith('Not supported')
text=(out/'RESULT_ANALYSIS.md').read_text();scope=(out/'ADOPTED_SCOPE.md').read_text()
for phrase in ['retrospective','not preregistered','unanswered','−2.50 to +18.33','7/20 to 13/20','41.67%']:assert phrase in text,phrase
assert 'Adopted on October 28, 2026' in scope
assert 'Renaming the question does not complete that work' in scope
ref=json.loads((out/'derived_assessment_neural_reference.json').read_text());assert ref['n']==120 and ref['fever_score']==50/120
cases=json.loads((out/'evidence_cases.json').read_text());assert len(cases)==6
assert cases[-1]['claim_override'] is False
assert 'Correct label, poor justification' in cases[-1]['interpretation']
HTMLParser().feed((out/'index.html').read_text())
with zipfile.ZipFile(out/'Result_Analysis_Evidence.zip') as z:
 assert z.testzip() is None
 assert 'RESULT_ANALYSIS.md' in z.namelist() and 'ADOPTED_SCOPE.md' in z.namelist()
 assert 'hypothesis_evidence.csv' in z.namelist()
 assert not any(n.startswith('/') or '..' in Path(n).parts for n in z.namelist())
audit=dict(status='passed',checks=['Every evidence-manifest SHA256 matches the preserved source artifact','Main local metrics exactly reproduce the official benchmark summaries','Published development rows match the verified KGAT Table 2 values','Five explicit hypothesis verdicts retain negative/inconclusive findings','Retrospective adoption and the unanswered original question are disclosed','Derived 120-claim neural strict reference checked','Six case examples preserve evidence and explanation limits','HTML parsed and submission ZIP integrity/path checks passed'])
(out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
