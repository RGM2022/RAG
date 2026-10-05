"""Local, model-free document retrieval using BM25 keyword ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
import re
from typing import Iterable


SUPPORTED_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".txt"}
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120
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


@dataclass(frozen=True)
class SearchResult:
    source: str
    page: int | None
    chunk_id: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        location = f", page {self.page}" if self.page is not None else ""
        return f"{self.source}{location}, passage {self.chunk_id}"


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
        and line[:1].isupper()
        and not line.endswith((".", "!", "?", ":", ";"))
        and not any(character in line for character in "`<>{}[]")
    )


def load_document(filename: str, content: bytes) -> list[SourceDocument]:
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
            SourceDocument(source=filename, page=number, text=page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
        if not any(document.text.strip() for document in pages):
            raise ValueError(f"No selectable text found in {filename}. Scanned PDFs need OCR, which this app does not use.")
        return pages

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{filename} is not valid UTF-8 text.") from error

    if not text.strip():
        raise ValueError(f"{filename} is empty.")
    return [SourceDocument(source=filename, text=text)]


def chunk_documents(
    documents: Iterable[SourceDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[SearchResult]:
    """Split documents into overlapping passages while preserving citations."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[SearchResult] = []
    for document in documents:
        text = document.text.strip()
        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            passage = text[start:end].strip()
            if passage:
                chunk_id += 1
                chunks.append(
                    SearchResult(
                        source=document.source,
                        page=document.page,
                        chunk_id=chunk_id,
                        text=passage,
                        score=0.0,
                    )
                )
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


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
                )
            )

    ranked.sort(key=lambda result: (-result.score, result.source, result.page or 0, result.chunk_id))
    return ranked[:top_k]


def find_relevant_excerpt(
    query: str,
    results: Iterable[SearchResult],
    max_characters: int = 360,
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
    for line in lines[start + 1 : start + 3]:
        if _looks_like_heading(line):
            break
        context.append(line)
    excerpt = " ".join(context)
    if len(excerpt) > max_characters:
        excerpt = excerpt[:max_characters].rsplit(" ", 1)[0].rstrip(" ,;:-") + "..."
    return result, excerpt
