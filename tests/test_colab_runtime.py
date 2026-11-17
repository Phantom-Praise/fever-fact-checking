"""Offline checks; no Hub access, GPU execution, training or model downloads."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('colab_runtime',ROOT/'colab/fever_colab_runtime.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
class RuntimeChecks(unittest.TestCase):
 def answer(self):return {'parts':[{'part':'A claim.','verdict':'SUPPORTS','evidence_ids':['E1'],'justification':'The passage establishes it.'}],'label':'SUPPORTS'}
 def test_aggregation_retains_model_disagreement(self):
  a=self.answer();a['parts'].append({'part':'Another part.','verdict':'NOT ENOUGH INFO','evidence_ids':[],'justification':'Not stated.'})
  r=runtime.parse_parts(json.dumps(a),{'supplied_evidence':{'E1':{}}},True)
  self.assertEqual(r['label'],'NOT ENOUGH INFO');self.assertEqual(r['model_label'],'SUPPORTS');self.assertFalse(r['consistent'])
 def test_truncated_response_is_invalid_even_with_partial_verdict(self):
  with self.assertRaisesRegex(ValueError,'token limit'):runtime.parse_parts(json.dumps(self.answer()),{'supplied_evidence':{'E1':{}}},False)
 def test_complete_markdown_wrapper_is_accepted(self):
  raw='```json\n'+json.dumps(self.answer())+'\n```'
  result=runtime.parse_parts(raw,{'supplied_evidence':{'E1':{}}},True)
  self.assertEqual(result['label'],'SUPPORTS');self.assertEqual(result['response_format'],'markdown_json')
 def test_observed_qwen_response_derives_missing_overall_label(self):
  part={'part':'Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.',
        'verdict':'NOT ENOUGH INFO','evidence_ids':[],
        'justification':'There is no direct evidence linking Nikolaj Coster-Waldau to the Fox Broadcasting Company.'}
  raw='```json\n'+json.dumps({'parts':[part]})+'\n```'
  result=runtime.parse_parts(raw,{'supplied_evidence':{'E34':{}}},True)
  self.assertEqual(result['label'],'NOT ENOUGH INFO');self.assertIsNone(result['model_label'])
  self.assertTrue(result['model_label_missing']);self.assertIsNone(result['consistent'])
 def test_wrapper_does_not_allow_prose_truncation_or_schema_repairs(self):
  answer=json.dumps(self.answer());context={'supplied_evidence':{'E1':{}}}
  for raw in ['Here is the answer:\n'+answer,'```json\n'+answer,answer+' trailing',
              '```json\n'+answer+'\n```\nextra',json.dumps({'parts':self.answer()['parts'],'label':'UNKNOWN'})]:
   with self.subTest(raw=raw),self.assertRaises(ValueError):runtime.parse_parts(raw,context,True)
  with self.assertRaisesRegex(ValueError,'token limit'):runtime.parse_parts('```json\n'+answer+'\n```',context,False)
 def test_invented_or_missing_citations_are_rejected(self):
  for ids in [['E99'],[]]:
   a=self.answer();a['parts'][0]['evidence_ids']=ids
   with self.assertRaises(ValueError):runtime.parse_parts(json.dumps(a),{'supplied_evidence':{'E1':{}}},True)
 def test_prediction_artifacts_are_immutable(self):
  with tempfile.TemporaryDirectory() as work:
   p=Path(work)/'trace.json';runtime.write_once(p,{'label':'SUPPORTS'});old=p.read_bytes()
   with self.assertRaisesRegex(ValueError,'immutable'):runtime.write_once(p,{'label':'REFUTES'})
   self.assertEqual(old,p.read_bytes())
 def test_resume_rejects_changed_protocol(self):
  with tempfile.TemporaryDirectory() as work:
   runtime.freeze(work,{'model':'one','budget':512})
   with self.assertRaisesRegex(ValueError,'Protocol differs'):runtime.freeze(work,{'model':'one','budget':256})
 def test_last_n_blocks_and_head_only_are_trainable(self):
  from transformers import RobertaConfig,RobertaForSequenceClassification
  for n in [1,2,3]:
   config=RobertaConfig(hidden_size=24,num_hidden_layers=4,num_attention_heads=4,intermediate_size=32,vocab_size=50,num_labels=3)
   model=RobertaForSequenceClassification(config)
   info=runtime.unfreeze_last_layers(model,n)
   self.assertGreater(info['trainable_parameters'],0)
   self.assertTrue(all(not p.requires_grad for p in model.roberta.embeddings.parameters()))
   self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
   for i,block in enumerate(model.roberta.encoder.layer):self.assertTrue(all(p.requires_grad==(i>=4-n) for p in block.parameters()))
 def test_partial_evaluation_cannot_produce_final_accuracy(self):
  with tempfile.TemporaryDirectory() as work:
   root=Path(work)/'project';out=Path(work)/'run';(root/'data').mkdir(parents=True);pred=out/'predictions/model';pred.mkdir(parents=True)
   inputs=[{'id':i,'claim':'claim '+str(i)} for i in range(1,19999)]
   (root/'data/eval_inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in inputs))
   (root/'data/eval_labels.json').write_text(json.dumps({str(i):'SUPPORTS' if i==1 else 'REFUTES' for i in range(1,19999)}))
   (pred/'1.json').write_text(json.dumps({'id':1,'label':'SUPPORTS'}))
   self.assertEqual(runtime.score(root,out,['model'],require_full=True),{})
   (pred/'2.json').write_text(json.dumps({'id':2,'label':'INVALID'}))
   # Do not call two predictions a complete 19,998-claim run.
   result=runtime.score(root,out,['model'],require_full=False)['model']
   self.assertEqual(result['status'],'partial_not_final');self.assertEqual(result['accuracy'],.5);self.assertEqual(result['invalid'],1)

class MonitoringChecks(unittest.TestCase):
 def test_diagnostic_manifest_enforces_labels_features_and_checksums(self):
  with tempfile.TemporaryDirectory() as work:
   root=Path(work);(root/'data').mkdir()
   rows=[{'id':i,'claim':'diagnostic claim '+str(i),'ranked_articles':[]} for i in range(90)]
   labels={str(i):runtime.LABELS[i//30] for i in range(90)}
   (root/'data/eval_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
   (root/'data/eval_labels.json').write_text(json.dumps(labels))
   manifest={'files':{name:runtime.sha_file(root/'data'/name) for name in ['eval_inputs.jsonl','eval_labels.json']}}
   (root/'diagnostic.json').write_text(json.dumps(manifest))
   self.assertEqual(runtime.verify_diagnostic(root),rows)
   rows[0]['label']='SUPPORTS'
   (root/'data/eval_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
   with self.assertRaisesRegex(ValueError,'checksum'):runtime.verify_diagnostic(root)
   manifest['files']['eval_inputs.jsonl']=runtime.sha_file(root/'data/eval_inputs.jsonl')
   (root/'diagnostic.json').write_text(json.dumps(manifest))
   with self.assertRaisesRegex(ValueError,'unexpected fields'):runtime.verify_diagnostic(root)
 def test_heartbeat_and_terminal_state_are_persisted(self):
  import threading
  with tempfile.TemporaryDirectory() as work:
   monitor=runtime.RunMonitor(work,'fake',total=19998,show_progress=False,heartbeat_seconds=.05)
   monitor.event('generating_response',claim_id=42,input_tokens=123)
   threading.Event().wait(.18)
   monitor.close('paused','test budget')
   events=[json.loads(s) for s in monitor.log_path.read_text().splitlines()]
   self.assertTrue(any(e['event']=='heartbeat' and e['stage']=='generating_response' for e in events))
   status=json.loads((Path(work)/'status.json').read_text())
   self.assertEqual(status['status'],'paused');self.assertEqual(status['claim_id'],42)
   self.assertFalse(monitor.thread.is_alive())
 def fixture(self,work):
  root=Path(work)/'project';out=Path(work)/'run';(root/'data').mkdir(parents=True);(root/'src').mkdir()
  rows=[{'id':i,'claim':'claim '+str(i)} for i in range(1,19999)]
  (root/'data/eval_inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
  (root/'src/fever_colab_runtime.py').write_text('frozen test adapter')
  (root/'parts_prompt.txt').write_text('fixed test prompt')
  return root,out,rows
 def execute(self,root,out,rows,predict,max_claims=2,invalid_limit=10,scope='official_dev'):
  from contextlib import ExitStack
  from unittest.mock import patch
  class Encoder:
   def __init__(self,*a,**kw):pass
  class Verifier:
   def __init__(self,*a,**kw):pass
   def predict(self,row,context,**kw):return predict(row,context,**kw)
   def close(self):pass
  def context(row,*args,progress=None,**kwargs):
   if progress:
    progress('fetching_article',article=1,total_articles=1,url='https://en.wikipedia.org/wiki/Example')
    progress('embedding_started',chunks=1)
   return {'status':'READY','payload':{'claim':row['claim']},'supplied_evidence':{}}
  with ExitStack() as stack:
   for name,value in [('ResidentEmbedder',Encoder),('CausalVerifier',Verifier),('context_for',context),('release',lambda:None),
                      ('environment',lambda:{'test':'offline'}),('context_protocol',lambda *a:{'test':'fixed'})]:
    stack.enter_context(patch.object(runtime,name,value))
   runtime.run_condition(rows,root,out,'fake','qwen',{'repo':'fake'},{'repo':'embed'},minutes=1,
                         show_progress=False,heartbeat_seconds=.1,max_new_claims=max_claims,max_consecutive_invalid=invalid_limit,scope=scope)
 def test_development_run_resumes_and_reports_errors_without_guessing_causes(self):
  with tempfile.TemporaryDirectory() as work:
   root,out,rows=self.fixture(work);rows=rows[:90]
   refs={str(r['id']):{'label':runtime.LABELS[i//30],'annotated_evidence':[]} for i,r in enumerate(rows)}
   (root/'data/eval_inputs.jsonl').write_text(''.join(json.dumps(dict(r,ranked_articles=[]))+'\n' for r in rows))
   rows=runtime.read_rows(root/'data/eval_inputs.jsonl')
   (root/'data/eval_labels.json').write_text(json.dumps({k:v['label'] for k,v in refs.items()}))
   (root/'data/references.json').write_text(json.dumps(refs))
   (root/'diagnostic.json').write_text(json.dumps({'files':{name:runtime.sha_file(root/'data'/name) for name in ['eval_inputs.jsonl','eval_labels.json','references.json']}}))
   def predict(row,context,**kwargs):
    self.assertNotIn('label',row);self.assertNotIn('annotated_evidence',row)
    return {'label':'INVALID' if row['id']==2 else 'SUPPORTS','response':'raw preserved','error':'bad json' if row['id']==2 else ''}
   self.execute(root,out,rows,predict,scope='development90')
   saved=(out/'predictions/fake/1.json').read_bytes()
   with self.assertRaisesRegex(ValueError,'19,998'):self.execute(root,out,rows,predict)
   self.execute(root,out,rows,predict,max_claims=90,scope='development90')
   self.assertEqual(saved,(out/'predictions/fake/1.json').read_bytes())
   report=runtime.diagnostic_report(root,out,['fake'])['fake']
   self.assertEqual(report['status'],'development_diagnostic_complete');self.assertEqual(report['n'],90)
   self.assertEqual(report['invalid'],1);self.assertEqual(len(report['confusion_matrix'][0]),4)
   import csv
   with next(out.glob('review_*.csv')).open() as stream:sheet=list(csv.DictReader(stream))
   self.assertEqual(sheet[1]['review_category'],'invalid_model_output')
   self.assertEqual(sheet[-1]['review_category'],'evidence_or_reasoning_review_required')
 def test_bounded_run_logs_claims_then_resumes_without_replacing_them(self):
  def predict(row,context,progress=None,**kwargs):
   if progress:progress('generating_response',input_tokens=10)
   return {'label':'SUPPORTS','response':'exact raw output','input_tokens':10,'output_tokens':5}
  with tempfile.TemporaryDirectory() as work:
   root,out,rows=self.fixture(work);self.execute(root,out,rows,predict)
   p=out/'predictions/fake/1.json';before=p.read_bytes()
   status=json.loads((out/'monitor/fake/status.json').read_text())
   self.assertEqual(status['completed'],2);self.assertEqual(status['status'],'paused')
   self.execute(root,out,rows,predict)
   self.assertEqual(p.read_bytes(),before)
   status=json.loads((out/'monitor/fake/status.json').read_text());self.assertEqual(status['completed'],4)
   records=[]
   for log in (out/'monitor/fake').glob('*.jsonl'):records.extend(json.loads(s) for s in log.read_text().splitlines())
   self.assertTrue(any(e['event']=='fetching_article' for e in records))
   saved=[e for e in records if e['event']=='claim_saved']
   self.assertEqual(len(saved),4);self.assertTrue(all(e['prediction']=='SUPPORTS' for e in saved))
 def test_repeated_invalid_outputs_stop_and_remain_counted(self):
  def predict(*args,**kwargs):return {'label':'INVALID','response':'unfinished json','error':'output limit'}
  with tempfile.TemporaryDirectory() as work:
   root,out,rows=self.fixture(work)
   with self.assertRaisesRegex(RuntimeError,'2 consecutive INVALID'):
    self.execute(root,out,rows,predict,max_claims=5,invalid_limit=2)
   saved=[p for p in (out/'predictions/fake').glob('*.json') if p.name!='protocol.json']
   self.assertEqual(len(saved),2)
   self.assertTrue(all(json.loads(p.read_text())['label']=='INVALID' for p in saved))
   status=json.loads((out/'monitor/fake/status.json').read_text());self.assertEqual(status['status'],'failed');self.assertEqual(status['invalid_total'],2)
 def test_keyboard_interrupt_logs_pending_claim_without_inventing_a_prediction(self):
  def predict(*args,**kwargs):raise KeyboardInterrupt
  with tempfile.TemporaryDirectory() as work:
   root,out,rows=self.fixture(work)
   with self.assertRaises(KeyboardInterrupt):self.execute(root,out,rows,predict)
   self.assertFalse((out/'predictions/fake/1.json').exists())
   status=json.loads((out/'monitor/fake/status.json').read_text())
   self.assertEqual(status['status'],'interrupted');self.assertEqual(status['completed'],0)

 def test_oversized_prompt_preserves_debug_input_without_calling_model(self):
  from types import SimpleNamespace
  class Encoded(dict):
   def to(self,device):return self
  class Tokenizer:
   def apply_chat_template(self,*args,**kwargs):return 'rendered full prompt'
   def __call__(self,*args,**kwargs):return Encoded(input_ids=SimpleNamespace(shape=(1,3900)))
  verifier=object.__new__(runtime.CausalVerifier);verifier.prompt='fixed instruction';verifier.tokenizer=Tokenizer()
  events=[]
  result=verifier.predict({'claim':'example'},{'status':'READY','payload':{'claim':'example'},'supplied_evidence':{}},
                          max_new_tokens=512,progress=lambda event,**fields:events.append(event))
  self.assertEqual(result['label'],'INVALID');self.assertFalse(result['model_called'])
  self.assertIn('example',result['prompt']);self.assertEqual(result['rendered_prompt'],'rendered full prompt')
  self.assertIn('prompt_rejected',events)
 def test_saved_inspector_retains_raw_output_and_only_opt_in_reference(self):
  from contextlib import redirect_stdout
  from io import StringIO
  with tempfile.TemporaryDirectory() as work:
   root=Path(work)/'project';out=Path(work)/'run';(root/'data').mkdir(parents=True);folder=out/'predictions/fake';folder.mkdir(parents=True)
   (root/'data/eval_labels.json').write_text(json.dumps({'42':'REFUTES'}))
   p=folder/'42.json';p.write_text(json.dumps({'id':42,'claim':'Example','label':'INVALID','prompt':'EXACT INPUT','response':'{ unfinished raw','error':'length'}))
   before=p.read_bytes();capture=StringIO()
   with redirect_stdout(capture):summary=runtime.inspect_saved(out,'fake',42,show_full=True)
   self.assertNotIn('reference_label',summary);self.assertIn('{ unfinished raw',capture.getvalue());self.assertIn('EXACT INPUT',capture.getvalue())
   with redirect_stdout(StringIO()):summary=runtime.inspect_saved(out,'fake',42,show_reference=True,root=root)
   self.assertEqual(summary['reference_label'],'REFUTES');self.assertFalse(summary['matches_reference']);self.assertEqual(before,p.read_bytes())

if __name__=='__main__':unittest.main()
