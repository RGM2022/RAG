# Local Document Search

A local, extractive search app built with Streamlit and a small BM25 search engine. It uses no LLM, embedding model, pretrained model, or hosted API. Results quote the matching source passages instead of generating an answer, so the displayed information stays grounded in the indexed files.

## Run with uv on Windows

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already installed. From PowerShell in this folder:

```powershell
uv sync
uv run streamlit run app.py
```

Upload a file, type a question, and select **Get answer**. The app displays one concise, relevant extract alongside the question and its source citation. It uses extractive search, not generated answers, so it quotes information found in the documents. Supported types are `.txt`, `.md`, `.csv`, `.json`, and text-based `.pdf`. For reusable files, place them directly in the app folder or in `knowledge_base`; the latest contents are indexed on every run. Uploaded files are not saved to disk.

PDFs must contain selectable text; scanned-page OCR is not supported. Search is lexical rather than semantic, so queries should use words that appear in the documents.

## Run tests

```powershell
uv run python -m unittest discover -s tests -v
```
