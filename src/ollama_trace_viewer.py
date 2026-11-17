"""Read-only inspection of saved local Ollama calls; never invokes a model."""
import ast
import hashlib
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/part2/ollama_comparison_20261028'
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']


def source_function(path, name, display_path=None):
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return dict(path=display_path or str(path.relative_to(ROOT)), line=node.lineno,
                code=ast.get_source_segment(text, node))


class TraceStore:
    def __init__(self, folder=RUN):
        self.folder = Path(folder)
        self.protocol = json.loads((self.folder / 'protocol.json').read_text())
        samples = [json.loads(line) for line in (self.folder / 'sample.jsonl').read_text().splitlines() if line.strip()]
        responses = [json.loads(line) for line in (self.folder / 'responses.jsonl').read_text().splitlines() if line.strip()]
        if [r['id'] for r in samples] != [r['id'] for r in responses] or len(samples) != 300:
            raise ValueError('Expected the complete paired 300-claim sample')
        diagnostic = json.loads((self.folder / 'telemundo_input.json').read_text())
        diagnostic_response = json.loads((self.folder / 'telemundo_response.json').read_text())
        if diagnostic['id'] != diagnostic_response['id'] or diagnostic['id'] in {r['id'] for r in samples}:
            raise ValueError('Diagnostic must be separate from the sample')
        self.inputs = {str(r['id']): r for r in [diagnostic] + samples}
        self.responses = {str(r['id']): r for r in [diagnostic_response] + responses}
        self.catalog = []
        for row in [diagnostic] + samples:
            response = self.responses[str(row['id'])]
            self.catalog.append(dict(id=row['id'], claim=row['claim'], gold=row['gold']['label'],
                prediction=response['label'], correct=row['gold']['label'] == response['label'],
                diagnostic=row['id'] == diagnostic['id']))
        self.sources = dict(call=source_function(ROOT / 'src/local_llm_experiments.py', 'call'),
            prompt=source_function(self.folder / 'runner_source.py', 'prompt', 'src/experiment_ollama_fever.py (frozen inference source)'),
            parse=source_function(self.folder / 'runner_source.py', 'validate_response', 'src/experiment_ollama_fever.py (frozen inference source)'))
        helper = ROOT / 'src/local_llm_experiments.py'
        self.sources['call']['matches_saved_run'] = hashlib.sha256(helper.read_bytes()).hexdigest() == self.protocol['input_sha256'].get(str(helper))

    def listing(self, query='', outcome='all', page=1):
        if outcome not in ('all', 'correct', 'wrong') or page < 1 or page > 10000 or len(query) > 1000:
            raise ValueError('Invalid trace filter')
        query = query.casefold().strip()
        items = [r for r in self.catalog if (not query or query in r['claim'].casefold() or query == str(r['id']))
                 and (outcome == 'all' or r['correct'] == (outcome == 'correct'))]
        size = 20
        return dict(total=len(items), page=page, page_size=size, traces_total=len(self.catalog),
                    sample_total=300, diagnostic_total=1, items=items[(page - 1) * size:page * size])

    def detail(self, claim_id):
        row = self.inputs[str(claim_id)]
        response = self.responses[str(claim_id)]
        raw = response.get('raw', {})
        return dict(id=row['id'], claim=row['claim'], gold=row['gold']['label'],
            prediction=response['label'], reason=response.get('reason', ''), error=response.get('error'),
            diagnostic=row['id'] == 111897, model=self.protocol['model'], ollama=self.protocol['ollama'],
            baselines={name: LABELS[value] for name, value in row['baselines'].items()},
            sentences=row['sentences'], request=raw.get('request'), response=raw.get('response'),
            recorded_utc=raw.get('recorded_utc'), wall_seconds=response['wall_seconds'],
            parsed_output={'label': response['label'], 'reason': response.get('reason', '')},
            sources=self.sources, endpoint='http://127.0.0.1:11434/api/chat',
            provenance='Saved inference artifacts; this viewer makes no Ollama calls. Gold labels are shown for evaluation only and were not sent to the model.',
            explanation_note='The reason is generated explanatory text, not a faithful internal reasoning trace or a probability.')


@lru_cache(maxsize=1)
def trace_store():
    return TraceStore()
