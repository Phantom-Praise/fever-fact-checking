"""Offline checks for validation selection and paired inference/resume integrity."""
from contextlib import ExitStack
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'colab'))
import fever_colab_runtime as core
import paired_roberta_evaluation as paired

class SelectionChecks(unittest.TestCase):
 def fixture(self,path):
  for n,metric in [(1,.66),(2,.68),(3,.69)]:
   folder=path/('roberta_full_last'+str(n));checkpoint=folder/'checkpoints/checkpoint-20';checkpoint.mkdir(parents=True)
   protocol={'scope':'full','train_n':116356,'validation_n':14534,'seed':42,'unfreezing':{'last_blocks':n}}
   (folder/'protocol.json').write_text(json.dumps(protocol))
   (folder/'complete.json').write_text(json.dumps({'steps':20,'best_validation_checkpoint':'/old/session/checkpoint-10'}))
   (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':20,'best_metric':metric,
     'best_model_checkpoint':'/old/session/checkpoint-10','log_history':[
      {'step':10,'eval_macro_f1':metric,'eval_loss':.6},{'step':20,'eval_macro_f1':metric-.01,'eval_loss':.7}]}))
 def test_selection_uses_best_validation_and_not_last_step(self):
  with tempfile.TemporaryDirectory() as d:
   self.fixture(Path(d));selection=paired.select_validation(d)
   self.assertEqual(selection['winner']['last_blocks'],3)
   self.assertEqual(selection['winner']['selected_step'],10)
   self.assertFalse(selection['official_evaluation_used_for_selection'])
 def test_inconsistent_training_or_history_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d);self.fixture(path);file=path/'roberta_full_last2/protocol.json'
   p=json.loads(file.read_text());p['seed']=43;file.write_text(json.dumps(p))
   with self.assertRaisesRegex(ValueError,'settings differ'):paired.select_validation(path)
 def test_final_weights_must_match_selected_checkpoint(self):
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'final';b=Path(d)/'best';a.mkdir();b.mkdir()
   (a/'model.safetensors').write_bytes(b'final');(b/'model.safetensors').write_bytes(b'different')
   with self.assertRaisesRegex(ValueError,'weights differ'):
    paired.verify_selected_weights({'winner':{'final_path':str(a),'best_checkpoint_path':str(b)}})

class PairChecks(unittest.TestCase):
 def fixture(self,path):
  root=path/'project';out=path/'out';(root/'data').mkdir(parents=True)
  rows=[{'id':i,'claim':'claim '+str(i),'ranked_articles':[]} for i in range(19998)]
  (root/'data/eval_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
  return root,out,rows
 def run_fake(self,root,out,rows,mismatch=False,corpus=None):
  class Encoder:
   def __init__(self,*args):pass
  class Model:
   def __init__(self,path,*args):self.original=path!='selected'
   def predict(self,row,context,**kwargs):
    if 'label' in row:raise AssertionError('Reference label leaked into input')
    return {'label':'SUPPORTS' if self.original else 'REFUTES','retained_input':
      ('different' if mismatch and not self.original else row['claim']),'input_tokens':10}
   def close(self):pass
  with ExitStack() as stack:
   for obj,name,value in [(paired,'GPUEmbedder',Encoder),(core,'RobertaVerifier',Model),
    (paired,'verify_selected_weights',lambda *a:{'weights':'selected'}),
    (core,'checkpoint_identity',lambda *a:{'weights':'original'}),
    (core,'context_protocol',lambda *a:{'fixed':'evidence'}),(core,'environment',lambda:{}),(core,'release',lambda:None),
    (core,'context_for',lambda row,*a,**kw:{'id':row['id'],'status':'READY','payload':{},'supplied_evidence':{}})]:
    stack.enter_context(patch.object(obj,name,value))
   if corpus is not None:stack.enter_context(patch.object(core,'context_for',side_effect=AssertionError('Live evidence route forbidden')))
   return paired.run_pairs(rows,root,out,{'winner':{'last_blocks':3,'final_path':'selected'}},
    {'repo':'fixed'},minutes=1,max_new_claims=2,show_progress=False,heartbeat_seconds=.05,corpus=corpus)
 def test_snapshot_uses_offline_route_and_cannot_mix_with_live_predictions(self):
  class Corpus:
   def __init__(self):self.seen=[]
   def protocol(self,*args):return {'source':'FEVER snapshot','snapshot_sha256':'fixed'}
   def context_for(self,row,*args,**kwargs):
    self.seen.append(row['id'])
    return {'id':row['id'],'status':'READY','payload':{},'supplied_evidence':{}}
  with tempfile.TemporaryDirectory() as d:
   root,out,rows=self.fixture(Path(d));corpus=Corpus();self.run_fake(root,out,rows,corpus=corpus)
   self.assertEqual(corpus.seen,[0,1])
   protocol=json.loads((out/'pairs/protocol.json').read_text());self.assertEqual(protocol['evidence_source'],'FEVER Wikipedia snapshot')
   with self.assertRaisesRegex(ValueError,'Protocol differs'):self.run_fake(root,out,rows)
 def test_bounded_pairing_and_resume_preserve_both_predictions(self):
  with tempfile.TemporaryDirectory() as d:
   root,out,rows=self.fixture(Path(d));names=self.run_fake(root,out,rows)
   a=out/'predictions'/names[0]/'0.json';b=out/'predictions'/names[1]/'0.json'
   before=[a.read_bytes(),b.read_bytes()]
   self.assertEqual(json.loads(before[0])['evidence_identity'],json.loads(before[1])['evidence_identity'])
   self.run_fake(root,out,rows)
   self.assertEqual(before,[a.read_bytes(),b.read_bytes()])
   status=json.loads((out/'monitor/paired_roberta/status.json').read_text());self.assertEqual(status['completed'],4)
 def test_mismatched_retained_inputs_publish_no_new_predictions(self):
  with tempfile.TemporaryDirectory() as d:
   root,out,rows=self.fixture(Path(d))
   with self.assertRaisesRegex(ValueError,'identical retained'):self.run_fake(root,out,rows,mismatch=True)
   self.assertEqual(len(list((out/'predictions').glob('*/[0-9]*.json'))),0)
 def test_subset_cannot_be_called_full_evaluation(self):
  with tempfile.TemporaryDirectory() as d:
   root,out,rows=self.fixture(Path(d))
   with self.assertRaisesRegex(ValueError,'19,998'):self.run_fake(root,out,rows[:2])

if __name__=='__main__':unittest.main()
