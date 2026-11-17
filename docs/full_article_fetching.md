# Full Wikipedia article collection

Implemented in `src/fetch_wikipedia_articles.py`. This is the URL-to-content stage of the planned RAG classifier. It does not generate embeddings, call Llama, classify claims or restart the paused evaluation.

## Inputs

Use the existing title retriever (top ten candidates, retain the first five in its unchanged ranking):

```sh
.venv/bin/python src/fetch_wikipedia_articles.py \
  --claim 'Jack Falahee was born.' \
  --output output/NEW_FETCH_RUN
```

Or supply already selected URLs:

```sh
.venv/bin/python src/fetch_wikipedia_articles.py \
  --url 'https://en.wikipedia.org/wiki/Telemundo' \
  --output output/NEW_TELEMUNDO_RUN
```

Repeat `--url` for ranked URLs, or use `--urls-file` pointing to a JSON list. Up to ten candidate URLs are recorded; the first five distinct normalized URLs are selected by default. `--top-k` supports 1–5. No new article reranker is introduced. FEVER title markers such as `-LRB-` are converted back to URL characters.

## Saved output

Each new run directory receives:

- `manifest.json`: selection ranks, source URLs, successful articles, failures, content counts, cache hits, disambiguation flags, HTTP request count and source hashes.
- `fetcher_source.py` and `retriever_source.py`: source snapshots for that invocation.

The separate cache is `data/processed/wikipedia_full_articles/`. Each immutable article revision/render directory contains:

- `article.html`: complete HTML returned by Wikipedia's parse API, without an intro/section limit.
- `article.txt`: cleaned article text, including substantive sections, lists, infoboxes and data tables.
- `article.json`: ordered typed content blocks with section paths, page/revision IDs, retrieval and revision timestamps, canonical/revision URLs, links, image metadata, categories and API properties.
- `parse_response.json` and `metadata_response.json`: exact decoded API response objects.

Cleaning removes navigation, edit controls, reference markers/reference lists and scripts. Removed material remains available in the raw HTML/API record. Images and linked external/subarticles are not downloaded. This is a complete returned article render, not a claim to fetch every related page or multimedia object.

Existing revisions are never overwritten. A cached URL is reused without a network request; use `--refresh` to fetch current content again while retaining earlier revisions/renders. An `oldid` in the input URL requests that specific revision. Cached artifacts are checked against their recorded hashes.

Missing articles, API errors and unavailable content are explicit failures in the manifest. The collector does not substitute titles, snippets or FEVER sentences. Runs with any failed selected URL exit nonzero and record `partial` or `failed`. Choose a new run directory when retrying; successful cached articles are reused. Existing output directories are rejected.

## Verification

Eight offline tests in `tests/test_fetch_wikipedia_articles.py` cover non-lead body content, infobox/table extraction, URL validation, immutable revisions, cache integrity, explicit errors/retries, unchanged top-ten/top-five selection and protection of earlier output.

Live fetches on October 29, 2026 collected all five pages selected for the Jack Falahee claim and a separate Telemundo article. Source content is frozen in the cache:

| Article | Cleaned words | Content blocks | Headings |
|---|---:|---:|---:|
| Jack Falahee | 842 | 20 | 8 |
| Telemundo | 9,185 | 135 | 30 |

These are content-collection measurements, not classification results. Headings include endmatter headings where present. The unchanged Jack retrieval also selects unrelated Jack pages and a disambiguation page; metadata flags the latter rather than silently changing the ranking. Later relevance retrieval still needs implementation and evaluation.

See `output/full_article_fetch_20261029/` for first-fetch and cache-verification manifests. The verified repeat invocations required zero HTTP requests.

## Research boundary

This is a separate corpus of fetched Wikipedia revisions. It does not alter `wiki_fts.db`, the old FEVER sentence snapshot, model checkpoints, earlier predictions or the stopped 19K run. Current Wikipedia text may differ from FEVER's historical evidence and sentence IDs. Do not label citations from this new corpus as original FEVER evidence without a separate mapping and validation.

No package installation was needed: the implementation uses the existing virtual environment's requests library and Python's standard HTML parser.

API references: [Wikipedia article parsing](https://www.mediawiki.org/wiki/API:Parsing_wikitext), [revision metadata](https://www.mediawiki.org/wiki/API:Revisions), [page information](https://www.mediawiki.org/wiki/API:Info). A descriptive User-Agent is sent; a project contact can be supplied through `--user-agent`.
