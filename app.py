from pathlib import Path

import streamlit as st

from search_engine import (
    SUPPORTED_SUFFIXES,
    chunk_documents,
    extract_pdf_page,
    find_relevant_excerpt,
    graph_rag_search,
    load_document,
    scrape_website,
)


APP_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = APP_DIR / "knowledge_base"

st.set_page_config(page_title="Local GraphRAG", page_icon="🕸️", layout="wide")
st.title("Local GraphRAG")
st.caption("A graph-enriched local search app that links related concepts across source passages while keeping results grounded in the original files.")

with st.expander("How it works"):
    st.write(
        "Files are read again whenever you search, split into overlapping passages, and indexed with a "
        "lightweight concept graph. Retrieval combines BM25 relevance with related-term links so the app "
        "can surface concept clusters instead of only exact keyword matches. The highlighted result is still "
        "the closest matching text from your files, not a generated paraphrase. Results include file and page citations."
    )
    st.write("Supported files: TXT, Markdown, CSV, JSON, and text-based PDF. Scanned PDFs are not OCR processed.")

KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
uploaded_files = st.file_uploader(
    "Upload documents",
    type=[suffix.removeprefix(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
    accept_multiple_files=True,
    help="Drag and drop documents here. To search reusable files, select them from knowledge_base below.",
)
knowledge_base_files = sorted(
    (
        file_path
        for file_path in KNOWLEDGE_BASE_DIR.rglob("*")
        if file_path.is_file() and file_path.suffix.casefold() in SUPPORTED_SUFFIXES
    ),
    key=lambda file_path: file_path.relative_to(KNOWLEDGE_BASE_DIR).as_posix().casefold(),
)
knowledge_base_sources = {
    file_path.relative_to(KNOWLEDGE_BASE_DIR).as_posix(): file_path
    for file_path in knowledge_base_files
}
selected_knowledge_base_sources = st.multiselect(
    "Select reusable documents from knowledge_base",
    options=list(knowledge_base_sources),
    help="Only selected files and documents uploaded above are searched.",
)

st.subheader("Scrape a website")
st.caption("Enter one public webpage URL per line. Each successfully scraped page is added to the documents searched below.")
with st.form("website_form"):
    source_urls_input = st.text_area(
        "Website URLs",
        key="website-urls",
        placeholder="https://example.com/document\nhttps://example.org/another-page",
    )
    scrape_submitted = st.form_submit_button("Scrape and index webpages")

if "scraped_documents" not in st.session_state:
    st.session_state["scraped_documents"] = []

if scrape_submitted:
    source_urls = list(dict.fromkeys(
        url.strip()
        for url in source_urls_input.splitlines()
        if url.strip()
    ))
    if not source_urls:
        st.error("Enter at least one website URL to scrape.")
    else:
        scraped_documents = st.session_state["scraped_documents"]
        scraped_by_url = {
            document.source_url: document
            for document in scraped_documents
        }
        scrape_errors = []
        newly_scraped_sources = []
        for source_url in source_urls:
            try:
                scraped_document = scrape_website(source_url)
                scraped_by_url[scraped_document.source_url] = scraped_document
                newly_scraped_sources.append(scraped_document.source)
            except ValueError as error:
                scrape_errors.append(f"{source_url}: {error}")
        st.session_state["scraped_documents"] = list(scraped_by_url.values())
        if newly_scraped_sources:
            st.success(
                f"Scraped and indexed {len(newly_scraped_sources)} webpage(s): "
                f"{', '.join(newly_scraped_sources)}"
            )
        for error in scrape_errors:
            st.error(error)

scraped_documents = st.session_state["scraped_documents"]
if scraped_documents:
    st.caption("Scraped webpages included in search:")
    for document in scraped_documents:
        st.markdown(f"- [{document.source}]({document.source_url})")
    if st.button("Clear scraped webpages"):
        st.session_state["scraped_documents"] = []
        st.rerun()

with st.form("question_form"):
    query = st.text_input(
        "Ask a question about your files",
        placeholder="e.g. What does debugging mean?",
    )
    submitted = st.form_submit_button("Get answer")

documents = []
load_errors = []
pdf_source_contents = {}
for source in selected_knowledge_base_sources:
    file_path = knowledge_base_sources[source]
    try:
        content = file_path.read_bytes()
        documents.extend(
            load_document(
                source,
                content,
            )
        )
        if file_path.suffix.casefold() == ".pdf":
            pdf_source_contents[source] = content
    except (OSError, ValueError, ImportError) as error:
        load_errors.append(f"{source}: {error}")

for uploaded_file in uploaded_files or []:
    try:
        content = uploaded_file.getvalue()
        documents.extend(
            load_document(
                uploaded_file.name,
                content,
            )
        )
        if Path(uploaded_file.name).suffix.casefold() == ".pdf":
            pdf_source_contents[uploaded_file.name] = content
    except (OSError, ValueError, ImportError) as error:
        load_errors.append(f"{uploaded_file.name}: {error}")

documents.extend(scraped_documents)

for error in load_errors:
    st.error(error)

passages = chunk_documents(documents)
st.caption(f"Indexed {len(documents)} document page(s) into {len(passages)} passages.")

if submitted and query.strip():
    if not passages:
        st.info("Upload or select documents before asking a question.")
    else:
        results = graph_rag_search(
            query,
            passages,
            top_k=5,
            min_relative_score=0.97,
        )
        if not results:
            st.info("I couldn't find a sufficiently relevant passage. Try rephrasing the question.")
        else:
            st.subheader("Best-matching passages")
            for result in results:
                excerpt_match = find_relevant_excerpt(query, [result])
                st.markdown(
                    f"**{result.citation}** · similarity {result.score:.3f}"
                )
                if result.source_url:
                    st.markdown(f"[Open source URL]({result.source_url})")
                if excerpt_match:
                    st.write(excerpt_match[1])
                else:
                    st.write(result.text[:220].rstrip() + ("..." if len(result.text) > 220 else ""))

                with st.expander("Show full passage"):
                    st.write(result.text)
                if result.page is not None and result.source in pdf_source_contents:
                    try:
                        page_pdf = extract_pdf_page(
                            pdf_source_contents[result.source],
                            result.page,
                        )
                    except (OSError, ValueError, ImportError) as error:
                        st.error(f"Could not prepare the matching PDF page: {error}")
                    else:
                        st.download_button(
                            f"Open PDF page {result.page} (passage {result.chunk_id})",
                            page_pdf,
                            file_name=f"{Path(result.source).stem}-page-{result.page}.pdf",
                            mime="application/pdf",
                            key=f"pdf-page-{result.source}-{result.page}-{result.chunk_id}",
                        )
