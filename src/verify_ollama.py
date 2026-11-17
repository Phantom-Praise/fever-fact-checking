"""Check new claims with the existing Wikipedia retriever and a local Ollama LLM.

Uses installed models only. No model download, training or cloud API. The LLM's
brief explanation is generated text, not a verified internal reasoning trace.
"""
import argparse
import json
from pathlib import Path
import sqlite3

from experiment_ollama_fever import LABELS, SCHEMA, prompt, validate_response
from local_llm_experiments import connect, call
from retrieve_title_aware import retrieve

ROOT = Path(__file__).resolve().parents[1]


class OllamaVerifier:
    def __init__(self, model='llama3.1:8b', db=None):
        self.session, self.model, self.version = connect(model)
        self.db = Path(db) if db else ROOT / 'data/processed/wiki_fts.db'

    def predict(self, claims):
        if not claims or any(not isinstance(c,str) or not c.strip() for c in claims):
            raise ValueError('Provide nonempty claim strings')
        connection=sqlite3.connect(self.db.resolve().as_uri()+'?mode=ro',uri=True)
        results=[]
        try:
            for claim in claims:
                evidence=retrieve(connection,claim,5)
                raw=call(self.session,self.model['name'],prompt({'claim':claim,'sentences':evidence}),42,SCHEMA,0,128)
                result=dict(claim=claim,model=self.model['name'],model_digest=self.model['digest'],
                    retrieved_evidence=evidence,raw_trace=raw,status='experimental_local_llm')
                try:
                    parsed=validate_response(raw['response'])
                    if raw['response'].get('prompt_eval_count',0)+raw['response'].get('eval_count',0)>=4096:
                        raise ValueError('Context budget may have been exhausted')
                    result.update(label=parsed['label'],label_id=LABELS.index(parsed['label']),explanation=parsed['reason'])
                except (ValueError,KeyError,TypeError) as exc:
                    result.update(label='INVALID',label_id=None,explanation='',status='invalid_response',error=str(exc))
                results.append(result)
        finally:
            connection.close()
        return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claim',action='append',required=True)
    parser.add_argument('--model',default='llama3.1:8b')
    parser.add_argument('--db')
    parser.add_argument('--trace-file',help='Optional new JSON file containing full prompts/responses; existing files rejected')
    args=parser.parse_args()
    if args.trace_file and Path(args.trace_file).exists():raise FileExistsError(args.trace_file)
    verifier=OllamaVerifier(args.model,args.db);results=verifier.predict(args.claim)
    if args.trace_file:
        with Path(args.trace_file).open('x') as stream:json.dump(results,stream,indent=2)
    for result in results:
        print(json.dumps({k:v for k,v in result.items() if k!='raw_trace'},ensure_ascii=False))


if __name__=='__main__':main()
