# Full article fetch — verified examples

The collector has fetched the actual Wikipedia article HTML and extracted text beyond the introductory paragraphs.

| Article | Clean words | Blocks | Headings | Disambiguation |
|---|---:|---:|---:|---|
| Jack Falahee | 842 | 20 | 8 | False |
| Jack & Jack | 1784 | 55 | 19 | False |
| Jack-Jack Attack | 946 | 29 | 8 | False |
| Jack | 848 | 120 | 24 | True |
| Jack Allen (footballer, born 1891) | 465 | 14 | 4 | False |
| Telemundo | 9185 | 135 | 30 | False |

The Jack claim uses the existing top-ten candidate ranking and retains its first five pages. The collector does not silently replace unrelated candidates. Telemundo was fetched separately by URL.

All article artifacts include raw HTML, clean text, structured content blocks, revision metadata and API records in the separate data/processed/wikipedia_full_articles/ cache. First-fetch manifests and cached verification manifests remain in this directory. Cached verification required zero new HTTP requests.

Eight offline tests passed. This step makes no Llama or embedding calls, provides no new accuracy score, and leaves the 19K evaluation stopped.

Implementation and usage: docs/full_article_fetching.md.
