"""Isolated evidence-RoBERTa inference. Never imports FAISS or reads gold labels."""
import argparse
import time
from experiment_parts_test import (EvidenceBaseline, rows, completed, write_new, progress)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    inputs = rows(out / 'inputs.jsonl')
    start = time.perf_counter()
    baseline = EvidenceBaseline()
    write_new(out / ('baseline_setup_' + str(time.time_ns()) + '.json'), dict(seconds=time.perf_counter()-start))
    for row in inputs:
        if completed(out, 'evidence_roberta', row['id']) is not None:
            continue
        trace = baseline(row['claim'])
        write_new(out / 'traces/evidence_roberta' / (str(row['id']) + '.json'), dict(id=row['id'], **trace))
        progress(out, inputs, 'running_baseline', latest_id=row['id'])
        print('BASELINE_SAVED', row['id'], flush=True)


if __name__ == '__main__':
    main()
