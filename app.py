from pathlib import Path

import streamlit as st

from search_engine import SUPPORTED_SUFFIXES, chunk_documents, find_relevant_excerpt, load_document, search


APP_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = APP_DIR / "knowledge_base"

st.set_page_config(page_title="Local Document Search", page_icon="🔎", layout="wide")
st.title("Local Document Search")
st.caption("Search your own files with BM25. No LLM, embeddings, pretrained model, or external API is used.")

with st.expander("How it works"):
    st.write(
        "Files are read again whenever you search, split into overlapping passages, and ranked with "
        "BM25 keyword matching. The highlighted result is the closest matching text from your files, "
        "not a generated paraphrase. Results include file and page citations. Add, remove, or edit "
        "files in this folder or in the knowledge base and search again to use the updated data."
    )
    st.write("Supported files: TXT, Markdown, CSV, JSON, and text-based PDF. Scanned PDFs are not OCR processed.")

KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
uploaded_files = st.file_uploader(
    "Add files for this session",
    type=[suffix.removeprefix(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
    accept_multiple_files=True,
    help="Uploaded files are indexed for this session. For reusable data, put files in the knowledge_base folder.",
)
uploaded_names = {file.name.casefold() for file in uploaded_files or []}
with st.form("question_form"):
    query = st.text_input(
        "Ask a question about your files",
        placeholder="e.g. What does debugging mean?",
    )
    submitted = st.form_submit_button("Get answer")

documents = []
load_errors = []
for file_path in sorted(KNOWLEDGE_BASE_DIR.rglob("*")):
    if file_path.is_file() and file_path.suffix.casefold() in SUPPORTED_SUFFIXES:
        try:
            documents.extend(
                load_document(
                    file_path.relative_to(KNOWLEDGE_BASE_DIR).as_posix(),
                    file_path.read_bytes(),
                )
            )
        except (OSError, ValueError, ImportError) as error:
            load_errors.append(f"{file_path.name}: {error}")

for file_path in sorted(APP_DIR.iterdir()):
    if (
        file_path.is_file()
        and file_path.suffix.casefold() in SUPPORTED_SUFFIXES
        and file_path.name.casefold() not in uploaded_names
    ):
        try:
            documents.extend(load_document(file_path.name, file_path.read_bytes()))
        except (OSError, ValueError, ImportError) as error:
            load_errors.append(f"{file_path.name}: {error}")

for uploaded_file in uploaded_files or []:
    try:
        documents.extend(load_document(uploaded_file.name, uploaded_file.getvalue()))
    except (OSError, ValueError, ImportError) as error:
        load_errors.append(f"{uploaded_file.name}: {error}")

for error in load_errors:
    st.error(error)

passages = chunk_documents(documents)
st.caption(f"Indexed {len(documents)} document page(s) into {len(passages)} passages.")

if submitted and query.strip():
    all_results = search(query, passages, top_k=len(passages))
    if not all_results:
        st.info("I couldn't find an answer in the uploaded files. Try rephrasing the question.")
    else:
        excerpt_match = find_relevant_excerpt(query, all_results)
        if excerpt_match:
            source_result, excerpt = excerpt_match
            st.subheader("Answer")
            st.markdown(f"**Question:** {query}")
            st.write(excerpt)
            st.caption(f"Source: {source_result.citation}")
        else:
            st.info("I found related text, but couldn't extract a reliable answer. Try rephrasing the question.")
