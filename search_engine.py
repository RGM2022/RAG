"""Local GraphRAG retrieval built on BM25 ranking and graph-aware concept linking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import ipaddress
from math import log
from pathlib import Path
import re
import socket
from typing import Iterable
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SUPPORTED_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".txt"}
MAX_WEBPAGE_BYTES = 2_000_000
WEB_REQUEST_TIMEOUT_SECONDS = 15
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", "")
BLOCK_HTML_TAGS = {
    "address", "article", "blockquote", "br", "dd", "div", "dl", "dt",
    "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
    "li", "main", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
}
IGNORED_HTML_TAGS = {
    "aside", "footer", "form", "iframe", "nav", "noscript", "script", "style",
    "svg", "template",
}
VOID_HTML_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "the", "this", "to", "what", "when", "where", "which", "who", "with",
}


@dataclass(frozen=True)
class SourceDocument:
    source: str
    text: str
    page: int | None = None
    source_url: str | None = None
    anchors: tuple[tuple[int, str], ...] = ()
    sections: tuple[tuple[int, str | None], ...] = ()


@dataclass(frozen=True)
class SearchResult:
    source: str
    page: int | None
    chunk_id: int
    text: str
    score: float
    source_url: str | None = None
    graph_terms: tuple[str, ...] = ()
    graph_edges: tuple[tuple[str, str], ...] = ()

    @property
    def citation(self) -> str:
        location = f", page {self.page}" if self.page is not None else ""
        return f"{self.source}{location}, passage {self.chunk_id}"


class _WebpageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._inside_title = False
        self._ignored_depth = 0
        self._heading_parts: list[str] | None = None
        self._heading_start_index: int | None = None
        self._heading_text_offset: int | None = None
        self._heading_anchor: str | None = None
        self.heading_anchors: list[tuple[int, str]] = []
        self.headings: list[tuple[int, str | None]] = []
        self._exclude_end_matter = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._exclude_end_matter:
            return
        if self._ignored_depth:
            if tag not in VOID_HTML_TAGS:
                self._ignored_depth += 1
        elif tag == "title":
            self._inside_title = True
        elif tag in IGNORED_HTML_TAGS or (
            tag == "a"
            and any(
                name.casefold() == "hidden"
                or (name.casefold() == "aria-hidden" and value == "true")
                for name, value in attrs
            )
        ):
            self._ignored_depth += 1
        else:
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self._heading_parts = []
                self._heading_start_index = len(self.text_parts)
                self._heading_text_offset = sum(map(len, self.text_parts))
                self._heading_anchor = next(
                    (value for name, value in attrs if name.casefold() == "id" and value),
                    None,
                )
            self.text_parts.append("\n" if tag in BLOCK_HTML_TAGS else " ")

    def handle_endtag(self, tag: str) -> None:
        if self._exclude_end_matter:
            return
        if self._ignored_depth:
            if tag not in VOID_HTML_TAGS:
                self._ignored_depth -= 1
        elif tag == "title":
            self._inside_title = False
        elif not self._inside_title:
            self.text_parts.append("\n" if tag in BLOCK_HTML_TAGS else " ")
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_parts is not None:
                heading_text = re.sub(
                    r"\s*#\s*$",
                    "",
                    " ".join("".join(self._heading_parts).split()),
                ).strip()
                heading = re.sub(
                    r"^\W*\d*\W*|\W+$",
                    "",
                    heading_text.casefold(),
                ).strip()
                if heading in {
                    "references",
                    "reference",
                    "reference list",
                    "bibliography",
                    "works cited",
                    "citations",
                    "citation",
                }:
                    assert self._heading_start_index is not None
                    del self.text_parts[self._heading_start_index :]
                    self._exclude_end_matter = True
                elif self._heading_anchor and heading_text:
                    assert self._heading_text_offset is not None
                    self.heading_anchors.append(
                        (self._heading_text_offset, self._heading_anchor)
                    )
                if not self._exclude_end_matter and heading_text:
                    assert self._heading_text_offset is not None
                    self.headings.append((self._heading_text_offset, self._heading_anchor))
                self._heading_parts = None
                self._heading_start_index = None
                self._heading_text_offset = None
                self._heading_anchor = None

    def handle_data(self, data: str) -> None:
        if self._exclude_end_matter:
            return
        if self._inside_title:
            self.title_parts.append(data)
        elif not self._ignored_depth:
            if self._heading_parts is not None:
                self._heading_parts.append(data)
            self.text_parts.append(data)


def _validate_public_http_url(url: str) -> None:
    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme.casefold() == "https" else 80)
    except ValueError as error:
        raise ValueError("Enter a valid public HTTP or HTTPS website URL.") from error

    if (
        parsed_url.scheme.casefold() not in {"http", "https"}
        or not parsed_url.netloc
        or not hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError("Enter a valid public HTTP or HTTPS website URL without credentials.")

    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise ValueError(f"Could not resolve website host {hostname}.") from error

    ip_addresses = {ipaddress.ip_address(address[4][0]) for address in addresses}
    if not ip_addresses or any(not address.is_global for address in ip_addresses):
        raise ValueError("Only publicly reachable websites can be scraped.")


class _PublicWebsiteRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        _validate_public_http_url(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


def scrape_website(url: str) -> SourceDocument:
    """Fetch one public HTML page and extract its readable text."""
    _validate_public_http_url(url)
    request = Request(
        url,
        headers={"User-Agent": "LocalGraphRAG/1.0 (webpage text extraction)"},
    )
    opener = build_opener(_PublicWebsiteRedirectHandler())
    try:
        with opener.open(request, timeout=WEB_REQUEST_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"The URL returned {content_type}, not an HTML page.")
            content = response.read(MAX_WEBPAGE_BYTES + 1)
            if len(content) > MAX_WEBPAGE_BYTES:
                raise ValueError("The webpage is larger than the 2 MB scraping limit.")
            page_url = response.geturl()
            charset = response.headers.get_content_charset() or "utf-8"
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise ValueError(f"Unable to fetch the webpage: {error}") from error

    try:
        html = content.decode(charset)
    except (LookupError, UnicodeError) as error:
        raise ValueError(f"The webpage uses an unsupported or invalid character encoding: {charset}.") from error

    parser = _WebpageTextParser()
    parser.feed(html)
    title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
    raw_text = "".join(parser.text_parts)

    def normalize_text(value: str) -> str:
        normalized = re.sub(r"[ \t\f\v]+", " ", value)
        normalized = re.sub(r" *\n *", "\n", normalized)
        return re.sub(r"\n{3,}", "\n\n", normalized)

    text = normalize_text(raw_text).strip()
    if not text:
        raise ValueError("No readable page text was found. The site may require JavaScript to display content.")

    source = title or urlsplit(page_url).hostname or page_url
    leading_whitespace = len(normalize_text(raw_text)) - len(normalize_text(raw_text).lstrip())

    def normalized_offset(raw_offset: int) -> int:
        return max(
            0,
            min(len(text), len(normalize_text(raw_text[:raw_offset])) - leading_whitespace),
        )

    anchors = tuple(
        (normalized_offset(offset), anchor)
        for offset, anchor in parser.heading_anchors
    )
    sections = tuple(
        (normalized_offset(offset), anchor)
        for offset, anchor in parser.headings
    )
    return SourceDocument(
        source=source,
        text=text,
        source_url=page_url,
        anchors=anchors,
        sections=sections,
    )


def tokenize(text: str) -> list[str]:
    """Split text into lowercase Unicode words and numbers."""
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold(), flags=re.UNICODE)


def _normalize_term(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _search_tokens(text: str) -> list[str]:
    terms = []
    for term in tokenize(text):
        if term in STOP_WORDS:
            continue
        terms.append(_normalize_term(term))
    return terms


def _search_terms(text: str) -> set[str]:
    return set(_search_tokens(text))


def _graph_terms(text: str) -> tuple[str, ...]:
    tokens = _search_tokens(text)
    term_frequencies: dict[str, int] = defaultdict(int)
    for token in tokens:
        if len(token) >= 3:
            term_frequencies[token] += 1

    for match in re.findall(r"\b[A-Z][A-Za-z0-9_/-]{2,}\b", text):
        normalized = _normalize_term(match.casefold())
        if normalized:
            term_frequencies[normalized] += 2

    ranked_terms = sorted(term_frequencies.items(), key=lambda item: (-item[1], item[0]))
    seen: set[str] = set()
    selected: list[str] = []
    for term, _ in ranked_terms:
        if term in seen or term in STOP_WORDS:
            continue
        seen.add(term)
        selected.append(term)
        if len(selected) >= 12:
            break
    return tuple(selected)


def _graph_edges(text: str) -> tuple[tuple[str, str], ...]:
    terms = list(_graph_terms(text))
    if len(terms) < 2:
        return ()
    edges: list[tuple[str, str]] = []
    for left_index, left in enumerate(terms):
        for right in terms[left_index + 1 : min(left_index + 6, len(terms))]:
            if left == right:
                continue
            edges.append((left, right))
    return tuple(edges)


def _longest_ordered_phrase(query_tokens: list[str], passage_tokens: list[str]) -> int:
    longest = 0
    for query_start in range(len(query_tokens)):
        for passage_start in range(len(passage_tokens)):
            matched = 0
            while (
                query_start + matched < len(query_tokens)
                and passage_start + matched < len(passage_tokens)
                and query_tokens[query_start + matched] == passage_tokens[passage_start + matched]
            ):
                matched += 1
            longest = max(longest, matched)
    return longest


def _looks_like_heading(line: str) -> bool:
    return (
        len(line) <= 100
        and len(line.split()) <= 6
        and line[:1].isupper()
        and not line.endswith((".", "!", "?", ":", ";"))
        and not any(character in line for character in "`<>{}[]")
    )


def load_document(
    filename: str,
    content: bytes,
    source_url: str | None = None,
) -> list[SourceDocument]:
    """Extract searchable text from one supported file."""
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported file type {suffix or '(no extension)'}. Use: {supported}.")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ImportError("PDF support requires pypdf. Install it with: pip install pypdf") from error

        from io import BytesIO

        reader = PdfReader(BytesIO(content))
        pages = [
            SourceDocument(
                source=filename,
                page=number,
                text=page.extract_text() or "",
                source_url=source_url,
            )
            for number, page in enumerate(reader.pages, start=1)
        ]
        if not any(document.text.strip() for document in pages):
            raise ValueError(
                f"No selectable text found in {filename}. Scanned PDFs need OCR, which this app does not use."
            )
        return pages

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{filename} is not valid UTF-8 text.") from error

    if not text.strip():
        raise ValueError(f"{filename} is empty.")
    return [SourceDocument(source=filename, text=text, source_url=source_url)]


def extract_pdf_page(content: bytes, page_number: int) -> bytes:
    """Return a PDF containing only the requested one-based source page."""
    try:
        from io import BytesIO
        from pypdf import PdfReader, PdfWriter
    except ImportError as error:
        raise ImportError("PDF support requires pypdf. Install it with: pip install pypdf") from error

    reader = PdfReader(BytesIO(content))
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError(f"PDF page {page_number} is outside the document's page range.")

    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _split_on_separator(text: str, separator: str) -> list[str]:
    if not separator:
        return list(text)

    pieces = []
    start = 0
    for match in re.finditer(re.escape(separator), text):
        pieces.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def _token_count(text: str) -> int:
    return len(tokenize(text))


def _merge_splits(splits: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current: list[str] = []
    current_length = 0

    for split in splits:
        split_length = _token_count(split)
        if current_length + split_length > chunk_size and current:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)

            while current and (
                current_length > overlap or current_length + split_length > chunk_size
            ):
                current_length -= _token_count(current.pop(0))

        current.append(split)
        current_length += split_length

    chunk = "".join(current).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _recursive_split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: tuple[str, ...] = CHUNK_SEPARATORS,
) -> list[str]:
    if _token_count(text) <= chunk_size:
        return [text]

    separator_index = next(
        (index for index, separator in enumerate(separators) if not separator or separator in text),
        len(separators) - 1,
    )
    separator = separators[separator_index]
    splits = _split_on_separator(text, separator)
    smaller_separators = separators[separator_index + 1 :]
    recursively_split: list[str] = []

    for split in splits:
        if _token_count(split) > chunk_size and smaller_separators:
            recursively_split.extend(
                _recursive_split_text(split, chunk_size, overlap, smaller_separators)
            )
        else:
            recursively_split.append(split)

    return _merge_splits(recursively_split, chunk_size, overlap)


def chunk_documents(
    documents: Iterable[SourceDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[SearchResult]:
    """Split documents into overlapping token-sized passages, preserving citations."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[SearchResult] = []
    for document in documents:
        text = document.text.strip()
        chunk_id = 0
        sections = sorted(
            {
                max(0, min(offset, len(text))): anchor
                for offset, anchor in document.sections
                if 0 <= offset < len(text)
            }.items()
        )
        section_ranges = []
        section_start = 0
        section_anchor = None
        for offset, anchor in sections:
            if offset == 0:
                section_anchor = anchor
                continue
            section_ranges.append((section_start, offset, section_anchor))
            section_start = offset
            section_anchor = anchor
        section_ranges.append((section_start, len(text), section_anchor))

        for start, end, section_anchor in section_ranges:
            section_text = text[start:end]
            search_from = 0
            for passage in _recursive_split_text(section_text, chunk_size, overlap):
                if not passage:
                    continue
                chunk_id += 1
                passage_start = section_text.find(passage, search_from)
                if passage_start < 0:
                    passage_start = search_from
                else:
                    search_from = passage_start + 1
                anchor = (
                    section_anchor
                    if document.sections
                    else next(
                        (
                            anchor
                            for anchor_offset, anchor in reversed(document.anchors)
                            if anchor_offset <= start + passage_start
                        ),
                        None,
                    )
                )
                source_url = document.source_url
                if source_url and anchor:
                    parsed_url = urlsplit(source_url)
                    source_url = urlunsplit(
                        parsed_url._replace(fragment=quote(anchor, safe="-._~"))
                    )
                graph_terms = _graph_terms(passage)
                graph_edges = _graph_edges(passage)
                chunks.append(
                    SearchResult(
                        source=document.source,
                        page=document.page,
                        chunk_id=chunk_id,
                        text=passage,
                        score=0.0,
                        source_url=source_url,
                        graph_terms=graph_terms,
                        graph_edges=graph_edges,
                    )
                )
    return chunks


def graph_rag_index(passages: Iterable[SearchResult]) -> dict[str, list[str]]:
    """Create a lightweight in-memory concept graph for the indexed passages."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for passage in passages:
        terms = list(passage.graph_terms)
        for left_index, left in enumerate(terms):
            for right in terms[left_index + 1 :]:
                if left == right:
                    continue
                adjacency[left].add(right)
                adjacency[right].add(left)
        for left, right in passage.graph_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return {term: sorted(neighbors) for term, neighbors in sorted(adjacency.items())}


def build_graph_summary(query: str, passages: Iterable[SearchResult], top_k: int = 6) -> list[str]:
    """Return the most relevant concept terms from the graph-aware retrieval results."""
    query_terms = _search_terms(query)
    if not query_terms:
        return []

    scores: dict[str, float] = defaultdict(float)
    for passage in passages:
        matched = set(passage.graph_terms) & query_terms
        if not matched:
            continue
        for term in passage.graph_terms:
            scores[term] += 1.0 if term in matched else 0.35
        for left, right in passage.graph_edges:
            if left in matched or right in matched:
                scores[left] += 0.25
                scores[right] += 0.25

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:top_k]]


def search(query: str, passages: Iterable[SearchResult], top_k: int = 5) -> list[SearchResult]:
    """Return the highest-scoring passages using BM25 (no fitted or pretrained model)."""
    query_terms = _search_terms(query)
    corpus = list(passages)
    if not query_terms or not corpus or top_k <= 0:
        return []

    tokenized = [_search_tokens(passage.text) for passage in corpus]
    document_frequencies: dict[str, int] = {}
    for tokens in tokenized:
        for term in set(tokens):
            document_frequencies[term] = document_frequencies.get(term, 0) + 1

    document_count = len(corpus)
    average_length = sum(map(len, tokenized)) / document_count or 1.0
    k1 = 1.5
    b = 0.75
    ranked: list[SearchResult] = []

    for passage, tokens in zip(corpus, tokenized):
        term_frequencies: dict[str, int] = {}
        for term in tokens:
            term_frequencies[term] = term_frequencies.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue
            doc_frequency = document_frequencies[term]
            inverse_document_frequency = log(
                1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            length_normalizer = 1 - b + b * len(tokens) / average_length
            score += inverse_document_frequency * (
                frequency * (k1 + 1) / (frequency + k1 * length_normalizer)
            )

        if score > 0:
            ranked.append(
                SearchResult(
                    source=passage.source,
                    page=passage.page,
                    chunk_id=passage.chunk_id,
                    text=passage.text,
                    score=score,
                    source_url=passage.source_url,
                    graph_terms=passage.graph_terms,
                    graph_edges=passage.graph_edges,
                )
            )

    ranked.sort(key=lambda result: (-result.score, result.source, result.page or 0, result.chunk_id))
    return ranked[:top_k]


def graph_rag_search(
    query: str,
    passages: Iterable[SearchResult],
    top_k: int = 5,
    min_relative_score: float = 0.95,
) -> list[SearchResult]:
    """Re-rank BM25 candidates by query coverage, phrase fit, proximity, and graph overlap."""
    query_tokens = _search_tokens(query)
    query_terms = set(query_tokens)
    corpus = list(passages)
    if not query_terms or not corpus or top_k <= 0:
        return []
    if not 0.0 <= min_relative_score <= 1.0:
        raise ValueError("min_relative_score must be between 0 and 1.")

    base_results = search(query, corpus, top_k=len(corpus))
    if not base_results:
        return []

    max_bm25_score = max(result.score for result in base_results) or 1.0
    enriched: list[SearchResult] = []
    for result in base_results:
        passage_tokens = _search_tokens(result.text)
        passage_terms = set(passage_tokens)
        matched_terms = query_terms & passage_terms
        coverage = len(matched_terms) / len(query_terms)

        phrase_fit = (
            _longest_ordered_phrase(query_tokens, passage_tokens) / len(query_tokens)
        )
        positions = [
            index
            for index, token in enumerate(passage_tokens)
            if token in matched_terms
        ]
        proximity = (
            len(matched_terms) / (max(positions) - min(positions) + 1)
            if positions
            else 0.0
        )
        graph_fit = len(query_terms & set(result.graph_terms)) / len(query_terms)
        normalized_bm25 = result.score / max_bm25_score

        score = (
            0.40 * coverage
            + 0.25 * phrase_fit
            + 0.20 * proximity
            + 0.10 * normalized_bm25
            + 0.05 * graph_fit
        )
        enriched.append(
            SearchResult(
                source=result.source,
                page=result.page,
                chunk_id=result.chunk_id,
                text=result.text,
                score=score,
                source_url=result.source_url,
                graph_terms=result.graph_terms,
                graph_edges=result.graph_edges,
            )
        )
    enriched.sort(
        key=lambda result: (
            -result.score,
            result.source,
            result.page or 0,
            result.chunk_id,
        )
    )
    score_cutoff = enriched[0].score * min_relative_score
    enriched = [result for result in enriched if result.score >= score_cutoff]

    selected: list[SearchResult] = []
    selected_ids: set[int] = set()
    seen_pages: set[tuple[str, int | None]] = set()
    for result in enriched:
        page_key = (result.source, result.page)
        if page_key in seen_pages:
            continue
        selected.append(result)
        selected_ids.add(id(result))
        seen_pages.add(page_key)
        if len(selected) == top_k:
            return selected

    for result in enriched:
        if id(result) in selected_ids:
            continue
        selected.append(result)
        if len(selected) == top_k:
            break
    return selected


def find_relevant_excerpt(
    query: str,
    results: Iterable[SearchResult],
    max_characters: int = 220,
) -> tuple[SearchResult, str] | None:
    """Pick the most query-specific nearby lines from the retrieved passages."""
    query_tokens = _search_tokens(query)
    query_terms = set(query_tokens)
    if not query_terms:
        return None

    candidates: list[
        tuple[tuple[int, float, int, int, float], SearchResult, list[str], int]
    ] = []
    for result in results:
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in result.text.splitlines()
            if line.strip()
            and not re.fullmatch(r"(?:page\s+\d+|docker cheat sheet|springframework\.guru)", line.strip(), re.I)
        ]
        if not lines:
            lines = [re.sub(r"\s+", " ", result.text).strip()]

        # Nearby lines often form one entry in cheat sheets: a heading followed by its command.
        for window_size in range(1, min(3, len(lines)) + 1):
            for start in range(len(lines) - window_size + 1):
                excerpt = " ".join(lines[start : start + window_size])
                matched = query_terms & _search_terms(excerpt)
                if not matched:
                    continue
                coverage = len(matched) / len(query_terms)
                ordered_phrase = _longest_ordered_phrase(query_tokens, _search_tokens(excerpt))
                definition = excerpt.split(":", 1)[0]
                definition_terms = _search_terms(definition)
                is_definition = int(
                    ":" in excerpt
                    and bool(definition_terms)
                    and definition_terms.issubset(query_terms)
                )
                rank = (is_definition, coverage, ordered_phrase, len(matched), result.score)
                candidates.append((rank, result, lines, start))

    if not candidates:
        return None

    _, result, lines, start = max(candidates, key=lambda candidate: candidate[0])
    context = [lines[start]]
    selected_line_is_heading = _looks_like_heading(lines[start])
    for line in lines[start + 1 : start + 3]:
        if _looks_like_heading(line) and not (selected_line_is_heading and len(context) == 1):
            break
        context.append(line)
    excerpt = " ".join(context)
    if len(excerpt) > max_characters:
        excerpt = excerpt[:max_characters].rsplit(" ", 1)[0].rstrip(" ,;:-") + "..."
    return result, excerpt
