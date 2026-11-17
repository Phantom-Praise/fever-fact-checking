"""Historical bug reproduction, not assertions that the old rules are correct."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from debug_symbolic_trace import AuditVerifier,replay_claim,SOURCE


class HistoricalSymbolicTrace(unittest.TestCase):
    def test_month_claim_ignores_a_birth_date_already_in_the_cached_response(self):
        verifier=AuditVerifier(SOURCE/'wikidata_snapshot')
        self.addCleanup(verifier.session.close)
        previous=sys.gettrace()
        result=replay_claim(verifier,'Jean-Martin Charcot was born in November 1825.')
        self.assertIs(sys.gettrace(),previous)
        self.assertEqual(result['reason'],'multiple_property_values')
        self.assertTrue(any(s['locals'].get('property_id')=='P19' and s['locals'].get('object_text')=='November 1825' for s in result['steps']))
        entity=next(q['response']['entities']['Q20710'] for q in result['queries'] if 'entities' in q['response'])
        self.assertEqual(entity['claims']['P569'][0]['mainsnak']['datavalue']['value']['time'],'+1825-11-29T00:00:00Z')
        self.assertEqual(verifier.calls,0)

    def test_historical_alias_miss_is_visible_in_query_response(self):
        verifier=AuditVerifier(SOURCE/'wikidata_snapshot')
        self.addCleanup(verifier.session.close)
        result=replay_claim(verifier,'Samuel Witwer was born in October.')
        self.assertEqual(result['reason'],'unresolved_or_ambiguous_subject')
        candidates=result['queries'][0]['response']['search']
        self.assertEqual(candidates[0]['label'],'Sam Witwer')
        self.assertIn('Samuel Witwer',candidates[0]['aliases'])

    def test_trace_is_restored_when_a_missing_snapshot_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            verifier=AuditVerifier(folder)
            self.addCleanup(verifier.session.close)
            previous=sys.gettrace()
            with self.assertRaises(RuntimeError):replay_claim(verifier,'Jean-Martin Charcot was born in November 1825.')
            self.assertIs(sys.gettrace(),previous)


if __name__=='__main__':unittest.main()
