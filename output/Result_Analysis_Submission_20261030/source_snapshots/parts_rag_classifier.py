"""Fresh-claim adapter for the frozen original parts prompt. No work on import."""
import json
from pathlib import Path
import tempfile
import time

from ephemeral_rag import (ClaimVectorDatabase, article_chunks, pack_context)
from fetch_wikipedia_articles import WikipediaArticleFetcher, selected_urls, sha
from replay_claim_parts import INSTRUCTION, validate


def prompt_for(payload):
    return INSTRUCTION + '\n\nINPUT JSON:\n' + json.dumps(payload, ensure_ascii=False)


def check_budget(raw_tokens, wrapper_tokens, context=4096, output=256):
    if not isinstance(raw_tokens, int) or raw_tokens <= 0:
        raise ValueError('Missing exact prompt-token count')
    expected = raw_tokens + wrapper_tokens
    if expected + output >= context:
        raise ValueError('Prompt plus output reserve exceeds frozen context; no passages removed')
    return expected


def validate_parts_response(raw, citations, expected_tokens):
    body = raw['response']
    if body.get('prompt_eval_count') != expected_tokens:
        raise ValueError('Reported prompt-token count differs from preflight; possible input truncation')
    if not isinstance(body.get('eval_count'), int) or body['eval_count'] < 1:
        raise ValueError('Missing output-token count')
    return validate(raw, citations)


def classify(claim, embedder, generate, count_tokens, wrapper_tokens, temp_root,
             save_prepared=lambda trace: None, fetcher_factory=WikipediaArticleFetcher,
             select=selected_urls):
    start = time.perf_counter()
    trace = dict(claim=claim, label='INVALID', model_label='INVALID', derived_label='INVALID',
                 status='failed', articles=[], fetch_errors=[], stage='selection',
                 retention='Full articles and vectors deleted; supplied passages and exact calls retained.')
    workspace = None
    db = None
    try:
        trace['selection'] = select(claim=claim)
        Path(temp_root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='claim-', dir=temp_root) as work:
            workspace = Path(work)
            trace['stage'] = 'article_fetch'
            fetcher = fetcher_factory(cache_dir=workspace / 'articles')
            chunks = []
            fetch_start = time.perf_counter()
            try:
                for selected in trace['selection']['selected_articles']:
                    try:
                        article = fetcher.fetch(selected['url'])['article']
                        trace['articles'].append({k: article[k] for k in (
                            'page_id', 'title', 'url', 'revision_url', 'revision_id',
                            'revision_timestamp', 'retrieved_utc')})
                        trace['articles'][-1].update(rank=selected['rank'], characters=len(article['text']),
                                                     source_sha256=article['artifact_sha256'])
                        chunks.extend(article_chunks(article, embedder.tokenizer))
                        del article
                    except Exception as exc:
                        trace['fetch_errors'].append(dict(url=selected['url'], error=repr(exc)))
                trace['fetch_requests'] = getattr(fetcher, 'requests_made', None)
            finally:
                fetcher.close()
            trace['fetch_seconds'] = time.perf_counter() - fetch_start
            if not chunks:
                raise ValueError('No full article passages; no snippet fallback')
            trace['stage'] = 'embedding'
            phase = time.perf_counter()
            vectors = embedder.encode([c['embedding_text'] for c in chunks] + [claim],
                                      temp_folder=workspace / 'embedding')
            trace['embedding_seconds'] = time.perf_counter() - phase
            db = ClaimVectorDatabase(workspace / 'database', chunks, vectors[:-1])
            trace['database'] = dict(chunks=len(chunks), dimensions=int(vectors.shape[1]),
                                    vector_sha256=sha((workspace / 'database/vectors.faiss').read_bytes()))
            trace['temporary_storage_bytes'] = sum(p.stat().st_size for p in workspace.rglob('*') if p.is_file())
            trace['stage'] = 'context'
            phase = time.perf_counter()
            payload, citations = pack_context(claim, db.search(vectors[-1:], k=30))
            trace.update(input=payload, supplied_evidence=citations, prompt=prompt_for(payload))
            trace['vector_search_pack_seconds'] = time.perf_counter() - phase
            if not citations:
                raise ValueError('No whole passages fit the frozen context budget')
            trace['stage'] = 'token_preflight'
            count = count_tokens(trace['prompt'])
            expected = check_budget(count['raw_prompt_tokens'], wrapper_tokens)
            trace['token_preflight'] = dict(**count, wrapper_tokens=wrapper_tokens,
                                            expected_prompt_tokens=expected, context=4096, output_reserve=256)
            trace['preparation_seconds'] = time.perf_counter() - start
            save_prepared(trace)
            trace['stage'] = 'generation'
            trace['raw'] = generate(trace['prompt'])
            trace['stage'] = 'validation'
            trace.update(validate_parts_response(trace['raw'], citations, expected))
            trace.update(label=trace['derived_label'], status='complete', stage='complete')
            ids = list(dict.fromkeys(i for part in trace['parts'] for i in part['evidence_ids']))
            trace['cited_evidence'] = [dict(id=i, **citations[i]) for i in ids]
    except Exception as exc:
        trace['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if db is not None:
            db.close()
        trace['cleanup'] = dict(workspace_created=workspace is not None,
                                articles_and_database_deleted=workspace is None or not workspace.exists())
        trace['wall_seconds'] = time.perf_counter() - start
    return trace
