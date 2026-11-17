"""Generate a disposable report to check metrics/figures without any model call."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment_parts_test import metrics
from report_parts_test import build


class ReportArtifacts(unittest.TestCase):
    def test_negative_results_and_invalids_remain_visible_in_report(self):
        labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
        baseline = dict(**metrics(labels, labels), mean_wall_seconds=.1)
        main = dict(**metrics(labels, ['INVALID']*3), mean_wall_seconds=20)
        summary = dict(n=3, conditions={'evidence_roberta': baseline, 'parts_rag': main,
                       'parts_model_label': metrics(labels, ['INVALID']*3)},
                       paired_accuracy_95_percentile_interval=[-1, -1], fixed=0, introduced=3, inconsistent=0)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'summary.json').write_text(json.dumps(summary))
            (folder/'protocol.json').write_text(json.dumps({'hypothesis': 'Improve both metrics.', 'limitations': ['Fixture only.']}))
            (folder/'scored_cases.json').write_text(json.dumps([dict(id=i, claim='Fixture '+str(i), gold=g,
                baseline=g, main='INVALID', model_label='INVALID', parts=[], cited_evidence=[], error='Incomplete output')
                for i, g in enumerate(labels)]))
            build(folder)
            text = (folder/'REPORT.md').read_text()
            self.assertIn('not supported on this sample', text)
            self.assertIn('Incomplete output', text)
            self.assertIn('0.00%', text)
            for name in ['comparison.png', 'confusion.png']:
                self.assertEqual((folder/name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(len((folder/'predictions.csv').read_text().splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
