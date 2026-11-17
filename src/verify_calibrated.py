"""Run the validation-calibrated neural fallback with existing local checkpoints.

This experimental classifier returns retrieved evidence for inspection; a
predicted label is not a guarantee of factual correctness. No API or training.
Run from the repository root using its existing virtual environment.
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evidence_pipeline_v2 import evidence_text, fuse_probabilities, retrieve_bm25
from retrieve_title_aware import retrieve

LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
DEFAULT_CONFIG = 'reports/part2/literature_improvements_20261023_corrected'


class CalibratedVerifier:
    def __init__(self, config_dir=DEFAULT_CONFIG, db='data/processed/wiki_fts.db', device='auto'):
        folder = Path(config_dir)
        self.config = json.loads((folder / 'selected_config.json').read_text())
        self.protocol = json.loads((folder / 'protocol.json').read_text())
        if self.config['variant'] not in self.protocol['variants']:
            raise ValueError('Unknown evidence variant')
        # Validate parameters even when the corresponding model has zero weight.
        fuse_probabilities([[1, 0, 0]], [[1, 0, 0]], self.config['claim_weight'], self.config['nei_bias'])
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.db = Path(db)
        self.models = {}
        for name, checkpoint, needed in zip(
                ('claim', 'evidence'), self.protocol['checkpoint_paths'],
                (self.config['claim_weight'] > 0, self.config['claim_weight'] < 1)):
            if needed:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
                self.models[name] = (tokenizer, model.to(self.device).eval())

    def predict(self, claims):
        if not claims or any(not isinstance(c, str) or not c.strip() for c in claims):
            raise ValueError('Provide at least one nonempty claim')
        variant = self.config['variant']
        rows = [[] for _ in claims]
        if 'evidence' in self.models:
            connection = sqlite3.connect(self.db.resolve().as_uri() + '?mode=ro', uri=True)
            try:
                for i, claim in enumerate(claims):
                    rows[i] = (retrieve(connection, claim, 5) if variant == 'original' else
                               retrieve_bm25(connection, claim, 5, link_subject=variant.startswith('subject_')))
            finally:
                connection.close()
        texts = [evidence_text(r, include_titles=variant.endswith('_titles')) for r in rows]
        probabilities = {}
        for name, (tokenizer, model) in self.models.items():
            values = []
            with torch.inference_mode():
                for index in range(0, len(claims), self.protocol['batch_size']):
                    batch = claims[index:index + self.protocol['batch_size']]
                    pairs = None if name == 'claim' else texts[index:index + len(batch)]
                    encoded = tokenizer(batch, text_pair=pairs, padding='max_length', truncation=True,
                                        max_length=self.protocol['max_length'], return_tensors='pt')
                    logits = model(**{k: v.to(self.device) for k, v in encoded.items()}).logits
                    values.extend(torch.softmax(logits, -1).cpu().tolist())
            probabilities[name] = np.asarray(values)
        neutral = np.full((len(claims), 3), 1 / 3)
        prediction = fuse_probabilities(probabilities.get('claim', neutral), probabilities.get('evidence', neutral),
                                       self.config['claim_weight'], self.config['nei_bias'])
        return [{'claim': claim, 'label': LABELS[int(label)], 'label_id': int(label),
                 'retrieved_evidence': rows[i], 'model_probabilities': {k: v[i].tolist() for k, v in probabilities.items()},
                 'decision': {k: self.config[k] for k in ('variant', 'claim_weight', 'nei_bias')},
                 'status': 'experimental_validation_calibration'}
                for i, (claim, label) in enumerate(zip(claims, prediction))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claim', action='append', required=True)
    parser.add_argument('--config-dir', default=DEFAULT_CONFIG)
    parser.add_argument('--db', default='data/processed/wiki_fts.db')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    verifier = CalibratedVerifier(args.config_dir, args.db, args.device)
    for result in verifier.predict(args.claim):
        print(json.dumps(result))


if __name__ == '__main__':
    main()
