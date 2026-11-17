"""Loopback-only saved test traces; never performs retrieval or inference."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def handler(out, port):
    inputs = [json.loads(line) for line in (out / 'inputs.jsonl').read_text().splitlines()]
    ids = {str(r['id']) for r in inputs}
    def load(name):
        path = out / name
        return json.loads(path.read_text()) if path.exists() else None
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send(self, payload, mime='application/json', code=200):
            body = json.dumps(payload, ensure_ascii=False).encode() if mime == 'application/json' else payload
            self.send_response(code)
            self.send_header('Content-Type', mime + ('; charset=utf-8' if mime.startswith('text/') or mime == 'application/json' else ''))
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.headers.get('Host') not in {'127.0.0.1:'+str(port), 'localhost:'+str(port)}:
                return self.send({'error': 'Loopback access only'}, code=403)
            route = urlparse(self.path).path
            if route == '/':
                return self.send((ROOT / 'web/evidence_explorer/parts-test.html').read_bytes(), 'text/html')
            if route == '/api/state':
                protocol = load('protocol.json')
                scored = {str(r['id']): r for r in (load('scored_cases.json') or [])}
                cases = []
                for row in inputs:
                    item = dict(row)
                    for condition in ['evidence_roberta', 'parts_rag']:
                        trace = load('traces/'+condition+'/'+str(row['id'])+'.json')
                        item[condition] = trace['label'] if trace else None
                    if str(row['id']) in scored:
                        item['gold'] = scored[str(row['id'])]['gold']
                    cases.append(item)
                return self.send(dict(progress=load('progress.json'), cases=cases, summary=load('summary.json'),
                                      settings=protocol['settings'], n=protocol['n'], model=protocol['model']['name']))
            if route.startswith('/api/trace/'):
                ident = route.rsplit('/', 1)[1]
                if ident not in ids:
                    return self.send({'error': 'Claim not in frozen sample'}, code=404)
                return self.send({c: load('traces/'+c+'/'+ident+'.json') for c in ['evidence_roberta', 'parts_rag']})
            assets = {'/REPORT.md': 'text/markdown', '/comparison.csv': 'text/csv', '/per_class.csv': 'text/csv',
                      '/predictions.csv': 'text/csv', '/comparison.png': 'image/png', '/confusion.png': 'image/png'}
            if route in assets and (out / route[1:]).is_file():
                source = out / route[1:]
                if route == '/REPORT.md' and (out / 'REVIEWED_ANALYSIS.md').is_file():
                    source = out / 'REVIEWED_ANALYSIS.md'
                return self.send(source.read_bytes(), assets[route])
            return self.send({'error': 'Not found or not generated yet'}, code=404)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(args.run.resolve(), args.port))
    print('Test trace viewer: http://127.0.0.1:%d/' % args.port, flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
