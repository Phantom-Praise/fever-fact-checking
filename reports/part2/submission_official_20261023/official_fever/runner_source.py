"""Evaluate frozen local models on all official FEVER development claims.

No model selection, training, or gold-evidence input. Uses the pinned, unmodified
official scorer and exports standard FEVER submissions. Outputs are resumable
only when protocol inputs still match. Run from the repository root.
"""
import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from analyze_controlled import metrics, paired
from experiment_symbolic_v2 import digest
from verify_calibrated import CalibratedVerifier, DEFAULT_CONFIG, LABELS
from vendor.fever_scorer import fever_score, is_strictly_correct

CONDITIONS = ['claim_only_H2', 'retrieved_roberta', 'calibrated_ensemble']


def normalize(text):
    return ' '.join(text.casefold().split())


def official_metrics(gold, predictions):
    if len(gold) != len(predictions) or [r['id'] for r in gold] != [r['id'] for r in predictions]:
        raise ValueError('Official predictions must cover identical IDs in identical order')
    if not gold:
        raise ValueError('Empty evaluation')
    # Pass copies because the upstream scorer adds gold labels/evidence in place.
    strict, accuracy, precision, recall, f1 = fever_score([dict(r) for r in predictions], gold, max_evidence=5)
    labels = {label: i for i, label in enumerate(LABELS)}
    classification = metrics([labels[r['label']] for r in gold], [labels[r['predicted_label']] for r in predictions])
    return {'n':len(gold), 'fever_score':strict, 'label_accuracy':accuracy,
            'evidence_macro_precision':precision, 'evidence_macro_recall':recall, 'evidence_f1':f1,
            'classification_macro_f1':classification['macro_f1'], 'confusion_matrix':classification['confusion_matrix']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/official_fever/shared_task_dev.jsonl')
    parser.add_argument('--config-dir', default=DEFAULT_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    gold = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if len(gold) != 19998 or Counter(r['label'] for r in gold) != Counter({label:6666 for label in LABELS}):
        raise ValueError('Expected complete official balanced FEVER development set, not a custom sample')
    if len({r['id'] for r in gold}) != len(gold):
        raise ValueError('Duplicate official IDs')
    source_paths = [args.data, __file__, 'src/verify_calibrated.py', 'src/evidence_pipeline_v2.py',
                    'src/retrieve_title_aware.py', 'src/vendor/fever_scorer.py',
                    str(Path(args.config_dir)/'selected_config.json'), str(Path(args.config_dir)/'protocol.json')]
    frozen = json.loads((Path(args.config_dir)/'protocol.json').read_text())
    for checkpoint in frozen['checkpoint_paths']:
        source_paths.extend(str(path) for path in sorted(Path(checkpoint).iterdir()) if path.is_file())
    hashes = {path:digest(path) for path in source_paths}
    out = Path(args.output)
    if args.resume:
        protocol=json.loads((out/'protocol.json').read_text())
        if protocol['input_sha256'] != hashes:
            raise ValueError('Inputs changed; resume rejected')
    else:
        out.mkdir(parents=True,exist_ok=False)
        prior=[json.loads(line) for line in Path('data/raw/train.jsonl').read_text().splitlines()]
        prior_ids={r['id'] for r in prior}; prior_text={normalize(r['claim']) for r in prior}
        overlap={'id_overlap_with_original_training_pool':sum(r['id'] in prior_ids for r in gold),
                 'normalized_text_overlap_with_original_training_pool':sum(normalize(r['claim']) in prior_text for r in gold)}
        protocol={'population':'Complete official FEVER shared_task_dev.jsonl; original order; all 19998 claims.',
                  'n':len(gold),'conditions':CONDITIONS,'labels':LABELS,'max_evidence':5,
                  'gold_evidence_used_as_model_input':False,'tuning_on_official_dev':False,
                  'frozen_calibration_config':json.loads((Path(args.config_dir)/'selected_config.json').read_text()),
                  'batch_size':frozen['batch_size'],'max_length':frozen['max_length'],'seed':42,
                  'overlap_audit':overlap,'input_sha256':hashes,
                  'comparison':{'source':'https://aclanthology.org/2020.acl-main.655.pdf','table':2,
                    'split':'Dev, not Test','published_models_not_rerun':True,
                    'rows':{'GEAR':{'label_accuracy':.7484,'fever_score':.7069},
                            'KGAT_BERT_Base_ESIM':{'label_accuracy':.7551,'fever_score':.7161},
                            'KGAT_BERT_Base':{'label_accuracy':.7802,'fever_score':.7588},
                            'KGAT_RoBERTa_Large':{'label_accuracy':.7829,'fever_score':.7611}}},
                  'limitations':['Same official population and scoring, but not matched model size, training budget, or retrieval architecture.',
                    'Published development results are literature references, not independent local reproductions.',
                    'Full official sample is retained even if a normalized-text overlap exists; see audit.',
                    'The ensemble is frozen from earlier internal validation; do not retune on these results.',
                    'No symbolic/LLM paper is claimed reproduced. Training uses a subset of official training data.']}
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        (out/'runner_source.py').write_text(Path(__file__).read_text())
    raw_path=out/'outputs.jsonl'
    saved=[]
    if args.resume and raw_path.exists():
        # A partial final write is never silently truncated.
        saved=[json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        if [r['id'] for r in saved] != [r['id'] for r in gold[:len(saved)]]:
            raise ValueError('Saved prefix does not match official sample')
    if len(saved)>len(gold): raise ValueError('Too many saved rows')
    verifier=CalibratedVerifier(args.config_dir)
    started=time.perf_counter(); initial=len(saved)
    with raw_path.open('a' if args.resume else 'x') as stream:
        for offset in range(len(saved),len(gold),128):
            batch=gold[offset:offset+128]
            results=verifier.predict([r['claim'] for r in batch])
            for record,result in zip(batch,results):
                value={'id':record['id'],'claim':record['claim'],
                       'predicted_evidence':[[r['page_id'],int(r['sentence_id'])] for r in result['retrieved_evidence'][:5]],
                       'model_probabilities':result['model_probabilities'],
                       'predictions':{'claim_only_H2':int(np.argmax(result['model_probabilities']['claim'])),
                                      'retrieved_roberta':int(np.argmax(result['model_probabilities']['evidence'])),
                                      'calibrated_ensemble':result['label_id']}}
                stream.write(json.dumps(value)+'\n'); saved.append(value)
            stream.flush()
            elapsed=time.perf_counter()-started
            progress={'completed':len(saved),'total':len(gold),'session_seconds':elapsed,'session_claims':len(saved)-initial,
                      'device':str(verifier.device),'status':'inference_running'}
            (out/'progress.json').write_text(json.dumps(progress,indent=2))
            print(f"Completed {len(saved)}/{len(gold)}; session {elapsed:.1f}s",flush=True)
    results={}; predictions={}; targets=np.array([LABELS.index(r['label']) for r in gold])
    for condition in CONDITIONS:
        values=[{'id':r['id'],'predicted_label':LABELS[r['predictions'][condition]],
                 'predicted_evidence':[] if condition=='claim_only_H2' else r['predicted_evidence']} for r in saved]
        predictions[condition]=values
        with (out/(condition+'.jsonl')).open('x') as stream:
            stream.writelines(json.dumps(r)+'\n' for r in values)
        results[condition]=official_metrics(gold,values)
    comparisons={}
    for baseline in CONDITIONS[:2]:
        comparisons[baseline+'__vs__ensemble_label_accuracy']=paired(targets,
              np.array([r['predictions'][baseline] for r in saved]),
              np.array([r['predictions']['calibrated_ensemble'] for r in saved]),samples=2000)
        def strict_flags(name):
            return np.array([int(is_strictly_correct({**g,**p},5)) for g,p in zip(gold,predictions[name])])
        # The paired helper compares binary correctness flags with an all-one target.
        comparisons[baseline+'__vs__ensemble_fever_score']=paired(np.ones(len(gold)),strict_flags(baseline),strict_flags('calibrated_ensemble'),samples=2000)
    result={'protocol':protocol,'device':str(verifier.device),'conditions':results,'paired_comparisons':comparisons,
            'population_matched_to_published_dev':True,'scorer_matched':True,'authors_models_rerun':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    with (out/'comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','origin','n','label_accuracy','fever_score','classification_macro_f1'])
        writer.writeheader()
        for name,values in results.items():
            writer.writerow({'model':name,'origin':'local frozen evaluation',**{k:values[k] for k in writer.fieldnames if k in values}})
        for name,values in protocol['comparison']['rows'].items():
            writer.writerow({'model':name,'origin':'published Dev reference; not rerun','n':19998,**values})
    (out/'progress.json').write_text(json.dumps({'completed':len(saved),'total':len(gold),'status':'complete'},indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
