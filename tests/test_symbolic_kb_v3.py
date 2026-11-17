import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from symbolic_kb_v3 import FactStore,parse_date,plan,compare_date,verify


def statement(value,**extra):return dict(mainsnak={'datavalue':{'value':value}},rank='normal',**extra)
def date(time='+1825-11-29T00:00:00Z',precision=11,**extra):
    value=dict(time=time,precision=precision,before=0,after=0,calendarmodel='http://www.wikidata.org/entity/Q1985727')
    value.update(extra)
    return statement(value)


class SymbolicV3(unittest.TestCase):
    def test_months_are_dates_and_compound_claim_keeps_both_assertions(self):
        self.assertEqual(plan('A Person was born in November 1825.')['parts'][0]['property'],'P569')
        self.assertEqual(plan('A Person was born in October.')['parts'][0]['expected'],{'month':10})
        self.assertEqual([p['property'] for p in plan('A Person was born in Texas on August 14, 1966.')['parts']],['P19','P569'])
        self.assertIsNone(plan('A Person was born before November 1825.'))
        self.assertIsNone(plan('A Person was born on Monday.'))

    def test_supported_wordings_preserve_conditions(self):
        self.assertIsNone(plan('A Person is not an actor.'))
        self.assertIsNone(plan('A Person is an actor and writer.'))
        self.assertIsNone(plan('A Person is only an actor.'))
        self.assertIsNone(plan('A Person is an American actor.'))
        self.assertIsNone(parse_date('February 30, 2000'))

    def test_date_comparison_support_and_direct_contradiction(self):
        self.assertEqual(compare_date([date()],{'year':1825,'month':11})[0],'SUPPORTS')
        self.assertEqual(compare_date([date()],{'year':1826})[0],'REFUTES')
        self.assertEqual(compare_date([date()],{'month':11})[0],'SUPPORTS')
        self.assertEqual(compare_date([date()],{'year':1820,'decade':True})[0],'SUPPORTS')

    def test_uncertain_missing_conflicting_and_low_precision_abstain(self):
        self.assertIsNone(compare_date([],{'year':1825})[0])
        self.assertIsNone(compare_date([date(precision=9)],{'month':11})[0])
        self.assertIsNone(compare_date([date(),date('+1826-11-29T00:00:00Z')],{'year':1825})[0])
        qualified=date();qualified['qualifiers']={'P1480':[]}
        self.assertIsNone(compare_date([qualified],{'year':1825})[0])
        self.assertIsNone(compare_date([date(calendarmodel='different')],{'year':1825})[0])

    def test_alias_resolution_preserves_ambiguity(self):
        with tempfile.TemporaryDirectory() as folder:
            store=FactStore(folder,offline=True)
            self.addCleanup(store.close)
            store.query=Mock(return_value={'search':[{'id':'Q1','label':'Short Name','aliases':['Long Name']}]})
            entity={'id':'Q1','labels':{'en':{'value':'Short Name'}},'aliases':{'en':[{'value':'Long Name'}]},'claims':{'P31':[statement({'id':'Q5'})]}}
            store.entity=Mock(return_value=entity)
            self.assertEqual(store.resolve('Long Name',human=True)[0]['id'],'Q1')
            entity['aliases']={}
            self.assertEqual(store.resolve('Long Name',human=True)[0]['id'],'Q1')
            store.query.return_value={'search':[{'id':'Q1','label':'Long Name'},{'id':'Q2','label':'Long Name'}]}
            store.entity.side_effect=lambda qid:{'id':qid,'labels':{'en':{'value':'Long Name'}},'claims':{'P31':[statement({'id':'Q5'})]}}
            self.assertIsNone(store.resolve('Long Name',human=True)[0])

    def test_facts_in_sqlite_come_from_response_not_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            store=FactStore(folder,offline=True);self.addCleanup(store.close)
            store.events.append({'source':'synthetic unit-test response'})
            store.query=Mock(return_value={'entities':{'Q1':{'id':'Q1','claims':{'P569':[date()]}}}})
            entity=store.entity('Q1')
            record=store.db.execute('SELECT payload FROM facts WHERE qid=? AND pid=?',('Q1','P569')).fetchone()
            self.assertEqual(json.loads(record[0])['mainsnak']['datavalue']['value']['time'],'+1825-11-29T00:00:00Z')
            self.assertEqual(store.entity('Q1'),entity)
            self.assertEqual(store.query.call_count,1)

    def test_unverified_compound_cannot_support_whole_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            store=FactStore(folder,offline=True);self.addCleanup(store.close)
            entity={'id':'Q1','claims':{'P569':[date()]}}
            store.resolve=Mock(return_value=(entity,'unique'))
            result=verify('A Person was born in Paris on November 29, 1825.',store)
            self.assertIsNone(result['label'])
            self.assertEqual(result['parts'][1]['verdict'],'SUPPORTS')

    def test_service_error_is_separate_from_missing_facts(self):
        with tempfile.TemporaryDirectory() as folder:
            store=FactStore(folder,offline=True);self.addCleanup(store.close)
            result=verify('A Person was born in November 1825.',store)
            self.assertEqual(result['reason'],'service_failure_fallback')
            self.assertIsNone(result['label'])

    def test_conflicting_places_abstain_even_if_one_label_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            store=FactStore(folder,offline=True);self.addCleanup(store.close)
            store.resolve=Mock(return_value=({'id':'Q1','claims':{'P19':[statement({'id':'Q2'}),statement({'id':'Q3'})]}},'unique'))
            store.entity=Mock(return_value={'labels':{'en':{'value':'Paris'}}})
            result=verify('A Person was born in Paris.',store)
            self.assertIsNone(result['label'])
            self.assertEqual(result['parts'][0]['reason'],'multiple_place_values')


if __name__=='__main__':unittest.main()
