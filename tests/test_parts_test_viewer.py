"""Read-only viewer must not reveal reference labels before scoring."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from serve_parts_test import handler


class ViewerIsolation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        (self.out / 'inputs.jsonl').write_text(json.dumps({'id': 42, 'claim': 'A claim.'})+'\n')
        (self.out / 'reference_labels.json').write_text(json.dumps({'42': 'REFUTES'}))
        (self.out / 'protocol.json').write_text(json.dumps({'settings': {}, 'n': 1, 'model': {'name': 'test-model'}}))
        (self.out / 'progress.json').write_text(json.dumps({'status': 'running'}))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), lambda *a: None)
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = handler(self.out, self.port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def get(self, path, host=None):
        request = Request('http://127.0.0.1:%d%s' % (self.port, path))
        if host:
            request.add_header('Host', host)
        return urlopen(request)

    def test_unscored_viewer_does_not_expose_gold_or_reference_file(self):
        with self.get('/api/state') as response:
            data = json.load(response)
        self.assertNotIn('gold', data['cases'][0])
        self.assertIsNone(data['summary'])
        with self.assertRaises(HTTPError) as error:
            self.get('/reference_labels.json')
        self.assertEqual(error.exception.code, 404)

    def test_scored_labels_appear_only_from_completed_analysis(self):
        (self.out / 'scored_cases.json').write_text(json.dumps([{'id': 42, 'gold': 'REFUTES'}]))
        with self.get('/api/state') as response:
            self.assertEqual(json.load(response)['cases'][0]['gold'], 'REFUTES')

    def test_unknown_claim_and_untrusted_host_rejected(self):
        for path, host, status in [('/api/trace/43', None, 404), ('/api/state', 'example.com', 403)]:
            with self.subTest(path=path, host=host), self.assertRaises(HTTPError) as error:
                self.get(path, host)
            self.assertEqual(error.exception.code, status)


if __name__ == '__main__':
    unittest.main()
