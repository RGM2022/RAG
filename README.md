# Local GraphRAG

A local, extractive GraphRAG-style search app built with Streamlit and a lightweight concept graph layered on top of BM25 ranking. It uses no LLM, embedding model, pretrained model, or hosted API. Results quote the matching source passages instead of generating an answer, so the displayed information stays grounded in the indexed files.

## Run with uv on Windows

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already installed. From PowerShell in this folder:

```powershell
uv sync
uv run streamlit run app.py
```

Upload files using **Upload documents**, place reusable files in `knowledge_base` and explicitly select them, or enter multiple public webpage URLs (one per line) and choose **Scrape and index webpages**. Each page is scraped independently; successful pages are indexed even if another URL fails. The scraper extracts readable text from each page (it does not crawl links or run JavaScript), excludes citation/reference sections, and keeps webpage passages within their heading sections to avoid mixing unrelated topics. Search results show a clickable source URL, linking to the matching section when the page provides heading IDs or to the page otherwise. For local PDFs, **Open PDF page** downloads a one-page PDF containing the cited page. Scraped pages remain available for the current session until cleared. Only uploaded files, selected knowledge-base files, and scraped webpages are searched; other files beside the app, including this README, are not indexed automatically. Scraping accepts public HTTP/HTTPS webpages up to 2 MB each. Type a question and select **Get answer**. The app shows up to five high-similarity passages, each with a short excerpt and its file, page, and passage citation; expand a passage to see the complete text. It uses extractive search, not generated answers, so it quotes information found in the selected documents. Documents are recursively split at paragraph, line, sentence, clause, and word boundaries. Chunk size and overlap are measured in lexical tokens (words and numbers), not characters; long individual words are kept intact. Supported file types are `.txt`, `.md`, `.csv`, `.json`, and text-based `.pdf`. Uploaded files are not saved to disk.

Retrieval first gathers BM25 candidates, then re-ranks them using query-term coverage, ordered phrase fit, term proximity, normalized BM25 score, and concept-graph overlap. The UI shows at most five chunks scoring at least 97% of the top reranked score, prioritizing distinct source pages. Excerpts are limited to 220 characters; complete chunks are available on demand. This deterministic re-ranking remains local and does not use a cross-encoder or hosted model.

PDFs must contain selectable text; scanned-page OCR is not supported. Search is lexical rather than semantic, so queries should use words that appear in the documents.

## Run tests

```powershell
uv run python -m unittest discover -s tests -v
```
