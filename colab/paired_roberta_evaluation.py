"""Paired original/selected RoBERTa evaluation with one shared live-RAG context.

No training. No model or data downloads on import. The official label file is
read only by reporting; checkpoint selection uses training-pool validation.
"""
import json,math,time
from pathlib import Path
import fever_colab_runtime as core

def select_validation(training_root):
 training_root=Path(training_root);candidates=[];protocols=[]
 for blocks in [1,2,3]:
  folder=training_root/('roberta_full_last'+str(blocks))
  protocol=json.loads((folder/'protocol.json').read_text())
  complete=json.loads((folder/'complete.json').read_text())
  state_path=folder/'checkpoints'/('checkpoint-'+str(complete['steps']))/'trainer_state.json'
  state=json.loads(state_path.read_text())
  if protocol['scope']!='full' or protocol['train_n']!=116356 or protocol['validation_n']!=14534:
   raise ValueError('Unexpected full-training partition')
  if protocol['unfreezing']['last_blocks']!=blocks or state['global_step']!=complete['steps']:
   raise ValueError('Training condition/completion mismatch')
  evaluations=[r for r in state['log_history'] if 'eval_macro_f1' in r]
  best=max(evaluations,key=lambda r:r['eval_macro_f1'])
  if not math.isclose(best['eval_macro_f1'],state['best_metric'],abs_tol=1e-12):
   raise ValueError('Best validation metric does not match history')
  if Path(state['best_model_checkpoint']).name!='checkpoint-'+str(best['step']):
   raise ValueError('Best checkpoint does not match validation history')
  if Path(complete['best_validation_checkpoint']).name!=Path(state['best_model_checkpoint']).name:
   raise ValueError('Completion metadata selects a different checkpoint')
  final=folder/'final';best_folder=folder/'checkpoints'/Path(state['best_model_checkpoint']).name
  candidates.append({'last_blocks':blocks,'validation_macro_f1':best['eval_macro_f1'],
   'validation_loss':best['eval_loss'],'selected_step':best['step'],'final_path':str(final),
   'best_checkpoint_path':str(best_folder),'state_sha256':core.sha_file(state_path),
   'training_protocol_sha256':core.sha_file(folder/'protocol.json')})
  comparable=dict(protocol);comparable.pop('unfreezing');protocols.append(comparable)
 if not all(p==protocols[0] for p in protocols):raise ValueError('Training settings differ beyond unfrozen layers')
 winner=max(candidates,key=lambda r:(r['validation_macro_f1'],-r['last_blocks']))
 return {'selection_metric':'training-pool validation macro-F1; ties favor fewer unfrozen blocks',
         'candidates':candidates,'winner':winner,'official_evaluation_used_for_selection':False}

class GPUEmbedder(core.ResidentEmbedder):
 """Same MiniLM tokenizer/mean pooling; FP32 CUDA rather than CPU execution.

Device and source identity are recorded as a new evaluation protocol. Small
floating point differences from previous CPU embeddings are possible.
"""
 def __init__(self,metadata):
  super().__init__(metadata);self.model=self.model.to('cuda')
 def encode(self,texts,progress=None):
  import numpy as np
  result=[]
  for start in range(0,len(texts),32):
   batch=texts[start:start+32]
   if any(len(self.tokenizer.encode(x,add_special_tokens=True))>256 for x in batch):
    raise ValueError('Embedding text exceeds frozen 256-token limit')
   encoded=self.tokenizer(batch,padding=True,truncation=False,return_tensors='pt').to('cuda')
   with self.torch.inference_mode():
    hidden=self.model(**encoded).last_hidden_state;mask=encoded['attention_mask'].unsqueeze(-1)
    vector=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
    result.append(self.torch.nn.functional.normalize(vector,p=2,dim=1).cpu().numpy().astype('float32'))
   if start==0 or start+32>=len(texts) or (start//32+1)%10==0:
    core.notify(progress,'embedding_batch',embedded=min(start+32,len(texts)),total_texts=len(texts))
  return np.concatenate(result)

def verify_selected_weights(selection):
 winner=selection['winner'];final=Path(winner['final_path']);best=Path(winner['best_checkpoint_path'])
 final_identity=core.checkpoint_identity(final);best_identity=core.checkpoint_identity(best)
 weight_names=[name for name in final_identity if name.endswith(('.safetensors','.bin'))]
 if not weight_names or any(final_identity[name]!=best_identity.get(name) for name in weight_names):
  raise ValueError('Final weights differ from the validation-selected checkpoint')
 return final_identity

def run_pairs(rows,root,out,selection,embedding,minutes=180,show_progress=True,heartbeat_seconds=20,max_new_claims=None,corpus=None):
 import torch
 root=Path(root);out=Path(out)
 canonical=core.read_rows(root/'data/eval_inputs.jsonl')
 if len(rows)!=19998 or rows!=canonical or len({r['id'] for r in rows})!=19998:
  raise ValueError('Paired evaluation requires the exact full 19,998 official inputs')
 if any(set(r)!={'id','claim','ranked_articles'} for r in rows):raise ValueError('Unexpected evaluation feature fields')
 if minutes<=0:raise ValueError('Session budget must be positive')
 if max_new_claims is not None and (not isinstance(max_new_claims,int) or max_new_claims<1):raise ValueError('Use a positive session claim limit or None')
 torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 selected_identity=verify_selected_weights(selection)
 original_path=root/'checkpoint/roberta_evidence'
 names=['roberta_original_rag512','roberta_last'+str(selection['winner']['last_blocks'])+'_rag512']
 paths=[str(original_path),selection['winner']['final_path']]
 embedding={**embedding,'inference_device':'cuda','compute_dtype':'float32',
            'paired_adapter_sha256':core.sha_file(__file__)}
 evidence_protocol=core.context_protocol(root,embedding) if corpus is None else corpus.protocol(root,embedding)
 source='live Wikipedia' if corpus is None else 'FEVER Wikipedia snapshot'
 protocol={'scope':'paired original and validation-selected RoBERTa on all 19998 official development claims',
  'input_sha256':core.sha_file(root/'data/eval_inputs.jsonl'),'selection':selection,
  'models':dict(zip(names,[core.checkpoint_identity(original_path),selected_identity])),
  'max_length':512,'seed':42,'evidence_protocol':evidence_protocol,'evidence_source':source,
  'adapter_sha256':core.sha_file(__file__),'environment':core.environment(),
  'note':'New GPU-embedding RAG evaluation; not five-sentence evaluation, not an untouched test.'}
 pair_identity=core.freeze(out/'pairs',protocol)
 fingerprints={name:core.freeze(out/'predictions'/name,{'paired_protocol_identity':pair_identity,'condition':name}) for name in names}
 core.write_once(out/'validation_selection.json',selection)
 pending=[];complete_count=invalid_count=0
 for row in rows:
  present=[]
  for name in names:
   path=out/'predictions'/name/(str(row['id'])+'.json')
   if path.exists():
    trace=json.loads(path.read_text())
    if trace.get('id')!=row['id'] or trace.get('claim')!=row['claim'] or trace.get('protocol_identity')!=fingerprints[name]:
     raise ValueError('Saved prediction does not match the paired protocol')
    present.append(trace)
  if len(present)==2:
   if present[0]['evidence_identity']!=present[1]['evidence_identity']:
    raise ValueError('Previously saved paired contexts differ')
   if all(r['evidence_status']=='READY' for r in present) and present[0]['retained_input']!=present[1]['retained_input']:
    raise ValueError('Previously saved paired retained inputs differ')
   complete_count+=1;invalid_count+=any(r['label']=='INVALID' for r in present)
  else:pending.append(row)
 if not pending:print('Both conditions already cover all 19,998 claims.');return names
 monitor=core.RunMonitor(out/'monitor/paired_roberta','paired_roberta',len(rows),completed=complete_count,
  invalid=invalid_count,show_progress=show_progress,heartbeat_seconds=heartbeat_seconds)
 encoder=None;models=[];started=time.perf_counter();status='paused';reason='session budget reached';invalid_streak=0;new=0
 try:
  monitor.event('loading_embedding_model',model=embedding['repo'],device='cuda',dtype='float32')
  encoder=GPUEmbedder(embedding)
  for name,path in zip(names,paths):
   monitor.event('loading_classifier',condition=name);models.append(core.RobertaVerifier(path,512))
  for row in pending:
   if time.perf_counter()-started>=minutes*60:break
   if max_new_claims is not None and new>=max_new_claims:reason='per-session claim limit reached';break
   phase=time.perf_counter();monitor.event('claim_started',claim_id=row['id'],claim=row['claim'])
   context=core.context_for(row,root,out,encoder,embedding,progress=monitor.event) if corpus is None else corpus.context_for(row,root,out,encoder,embedding,progress=monitor.event)
   evidence_identity=core.signature(context);pair=[]
   for name,model in zip(names,models):
    path=out/'predictions'/name/(str(row['id'])+'.json')
    if path.exists():
     trace=json.loads(path.read_text())
     if trace['evidence_identity']!=evidence_identity:raise ValueError('Cached prediction context differs')
    else:
     trace={'id':row['id'],'claim':row['claim'],'label':'INVALID','created_utc':core.now(),
       'protocol_identity':fingerprints[name],'evidence_identity':evidence_identity,'evidence_status':context['status']}
     if context['status']!='READY':trace['error']=context.get('error','Evidence preparation failed')
     else:trace.update(model.predict(row,context,progress=monitor.event))
    pair.append(trace)
   if context['status']=='READY' and (pair[0]['retained_input']!=pair[1]['retained_input'] or pair[0]['input_tokens']!=pair[1]['input_tokens']):
    raise ValueError('Models did not receive identical retained claim/evidence text')
   # Validate both model inputs before publishing either new prediction.
   for name,trace in zip(names,pair):
    core.write_once(out/'predictions'/name/(str(row['id'])+'.json'),trace)
   invalid=any(r['label']=='INVALID' for r in pair)
   monitor.event('pair_predictions',claim_id=row['id'],original=pair[0]['label'],selected=pair[1]['label'],
                 identical_context=True,identical_retained_input=context['status']=='READY')
   monitor.saved({'id':row['id'],'claim':row['claim'],'label':'INVALID' if invalid else 'PAIR_COMPLETE',
      'wall_seconds':time.perf_counter()-phase})
   complete_count+=1;new+=1;invalid_streak=invalid_streak+1 if invalid else 0
   if invalid_streak>=3:raise RuntimeError('Three consecutive evidence failures; inspect saved traces before resuming')
  if complete_count==19998:status='complete';reason='both models evaluated on all 19,998 claims'
 except KeyboardInterrupt:
  status='interrupted';reason='user interrupted; incomplete pair resumes from saved model predictions';raise
 except Exception as exc:
  status='failed';reason=type(exc).__name__+': '+str(exc);monitor.event('run_failed',exception=reason);raise
 finally:
  for model in models:model.close()
  encoder=None;core.release();monitor.close(status,reason)
 print('Paired session:',status,';',complete_count,'/ 19998 claims complete.')
 return names

def report(root,out,names):
 """Partial progress is visible, but final metrics require full paired coverage."""
 out=Path(out)
 metrics=core.score(root,out,names,require_full=True)
 if len(metrics)!=2:return metrics
 source=json.loads((out/'pairs/protocol.json').read_text())['evidence_source']
 for metric in metrics.values():
  metric['evidence_source']=source
  metric['metric_note']='Label metrics only; supplied passage citations are not scored with the official strict FEVER evidence scorer.'
 labels=json.loads((Path(root)/'data/eval_labels.json').read_text())
 counts={'both_correct':0,'original_only_correct':0,'selected_only_correct':0,'both_wrong':0}
 for row in core.read_rows(Path(root)/'data/eval_inputs.jsonl'):
  a,b=[json.loads((out/'predictions'/name/(str(row['id'])+'.json')).read_text()) for name in names]
  if a['evidence_identity']!=b['evidence_identity']:raise ValueError('Paired evidence differs during scoring')
  correct_a=a['label']==labels[str(row['id'])];correct_b=b['label']==labels[str(row['id'])]
  key='both_correct' if correct_a and correct_b else 'original_only_correct' if correct_a else 'selected_only_correct' if correct_b else 'both_wrong'
  counts[key]+=1
 result={'n':19998,'scope':'full official development, paired retained evidence and token budget',
         'conditions':names,'paired_counts':counts,'evidence_source':source,'model_metrics':metrics,
         'accuracy_difference_percentage_points':100*(metrics[names[1]]['accuracy']-metrics[names[0]]['accuracy'])}
 core.write_once(out/('paired_comparison_'+str(time.time_ns())+'.json'),result)
 print(json.dumps(result,indent=2));return metrics
