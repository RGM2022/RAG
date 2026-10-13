from email.message import Message
import socket
import unittest
from unittest.mock import patch

from search_engine import (
    SourceDocument,
    chunk_documents,
    extract_pdf_page,
    find_relevant_excerpt,
    graph_rag_search,
    load_document,
    scrape_website,
    search,
    tokenize,
)


def _public_dns_result():
    return (
        socket.AF_INET,
        socket.SOCK_STREAM,
        6,
        "",
        ("93.184.216.34", 443),
    )


class _FakeWebResponse:
    def __init__(self, content: bytes, url: str, content_type: str = "text/html") -> None:
        self.content = content
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size: int) -> bytes:
        return self.content[:size]

    def geturl(self) -> str:
        return self.url


class SearchEngineTests(unittest.TestCase):
    def test_tokenize_is_case_insensitive_and_handles_punctuation(self):
        self.assertEqual(tokenize("Returns, RETURNS! user's"), ["returns", "returns", "user's"])

    def test_chunks_preserve_source_and_page(self):
        passage = chunk_documents(
            [SourceDocument(source="guide.pdf", page=4, text="alpha " * 30)],
            chunk_size=10,
            overlap=2,
        )
        self.assertGreater(len(passage), 1)
        self.assertTrue(all(item.source == "guide.pdf" and item.page == 4 for item in passage))

    def test_chunking_prefers_paragraph_boundaries(self):
        text = "First paragraph is a complete thought.\n\nSecond paragraph is another thought."
        passages = chunk_documents(
            [SourceDocument("guide.txt", text)],
            chunk_size=8,
            overlap=0,
        )
        self.assertEqual(
            [passage.text for passage in passages],
            [
                "First paragraph is a complete thought.",
                "Second paragraph is another thought.",
            ],
        )

    def test_chunking_size_is_measured_in_tokens_not_characters(self):
        long_word = "a" * 75
        text = f"{long_word} alpha beta gamma delta epsilon"
        passages = chunk_documents(
            [SourceDocument("guide.txt", text)],
            chunk_size=1,
            overlap=0,
        )
        self.assertEqual(passages[0].text, long_word)
        self.assertTrue(all(len(tokenize(passage.text)) <= 1 for passage in passages))
        self.assertEqual(
            [token for passage in passages for token in tokenize(passage.text)],
            tokenize(text),
        )

    def test_chunking_recurses_from_sentence_to_word_boundary(self):
        sentence = "A short sentence ends here. "
        text = sentence + ("x" * 60)
        passages = chunk_documents(
            [SourceDocument("guide.txt", text)],
            chunk_size=5,
            overlap=0,
        )
        self.assertEqual(passages[0].text, sentence.strip())
        self.assertTrue(all(len(tokenize(passage.text)) <= 5 for passage in passages))
        self.assertEqual(sum(len(tokenize(passage.text)) for passage in passages), len(tokenize(text)))

    def test_chunking_overlap_preserves_source_and_bounded_chunk_size(self):
        passages = chunk_documents(
            [SourceDocument("guide.txt", "alpha beta gamma delta epsilon zeta eta theta")],
            chunk_size=4,
            overlap=1,
        )
        self.assertGreater(len(passages), 1)
        self.assertTrue(all(len(tokenize(passage.text)) <= 4 for passage in passages))
        self.assertTrue(all(passage.source == "guide.txt" for passage in passages))
        self.assertEqual(
            set(tokenize(passages[0].text)) & set(tokenize(passages[1].text)),
            {"delta"},
        )

    def test_search_ranks_relevant_passage_and_returns_citation(self):
        passages = chunk_documents(
            [
                SourceDocument("returns.txt", "Our return policy allows returns within thirty days."),
                SourceDocument("shipping.txt", "Orders ship within three business days."),
            ]
        )
        result = search("return policy", passages)
        self.assertEqual(result[0].source, "returns.txt")
        self.assertIn("returns.txt", result[0].citation)
        self.assertGreater(result[0].score, 0)

    def test_source_url_is_preserved_from_document_load_through_search(self):
        source_url = "https://example.com/guide"
        documents = load_document(
            "guide.txt",
            b"Find the troubleshooting guide.",
            source_url=source_url,
        )
        results = search("troubleshooting guide", chunk_documents(documents))

        self.assertEqual(documents[0].source_url, source_url)
        self.assertTrue(results)
        self.assertEqual(results[0].source_url, source_url)

    def test_scrape_website_extracts_readable_text_and_preserves_page_url(self):
        response = _FakeWebResponse(
            b"<html><head><title>Useful Guide</title><style>hidden style</style></head>"
            b"<body><nav>navigation links</nav><main><h1 id='network-guide'>Network Guide</h1>"
            b"<p>Configure the network gateway.</p><script>hidden script</script>"
            b"</main></body></html>",
            "https://example.com/guide",
        )
        with (
            patch(
                "search_engine.socket.getaddrinfo",
                return_value=[_public_dns_result()],
            ),
            patch("search_engine.build_opener") as build_opener,
        ):
            build_opener.return_value.open.return_value = response
            document = scrape_website("https://example.com/guide")

        self.assertEqual(document.source, "Useful Guide")
        self.assertEqual(document.source_url, "https://example.com/guide")
        self.assertEqual(document.anchors, ((0, "network-guide"),))
        self.assertIn("Network Guide", document.text)
        self.assertIn("Configure the network gateway.", document.text)
        self.assertNotIn("navigation links", document.text)
        self.assertNotIn("hidden style", document.text)
        self.assertNotIn("hidden script", document.text)
        results = graph_rag_search("network gateway", chunk_documents([document]))
        self.assertTrue(results)
        self.assertEqual(results[0].source_url, "https://example.com/guide#network-guide")

    def test_search_results_link_to_the_matching_website_heading(self):
        response = _FakeWebResponse(
            b"<main><h2 id='planning-section'>Planning</h2>"
            b"<p>Goal decomposition supports long term planning.</p>"
            b"<h2 id='memory-section'>Memory</h2>"
            b"<p>Memory stores records for agent reasoning tasks.</p></main>",
            "https://example.com/agents",
        )
        with (
            patch(
                "search_engine.socket.getaddrinfo",
                return_value=[_public_dns_result()],
            ),
            patch("search_engine.build_opener") as build_opener,
        ):
            build_opener.return_value.open.return_value = response
            document = scrape_website("https://example.com/agents")

        passages = chunk_documents([document], chunk_size=7, overlap=0)
        results = graph_rag_search("goal decomposition", passages)

        self.assertTrue(results)
        self.assertIn("Goal decomposition", results[0].text)
        self.assertEqual(
            results[0].source_url,
            "https://example.com/agents#planning-section",
        )

    def test_scraped_passages_do_not_cross_article_heading_boundaries(self):
        response = _FakeWebResponse(
            b"<main><div>Generative Agents Simulation</div>"
            b"<h2 id='scientific-agent'>Scientific Discovery Agent"
            b"<a hidden class='anchor'>#</a></h2>"
            b"<p>It is instructed to answer a user-given prompt using tools.</p>"
            b"<p>ChemCrow outperforms GPT-4 in expert evaluations.</p>"
            b"<h2 id='generative-agents'>Generative Agents Simulation"
            b"<a hidden class='anchor'>#</a></h2>"
            b"<p>Generative agents create believable behavior using memory and planning.</p>"
            b"<p>Each agent records observations in a long-term memory stream.</p></main>",
            "https://example.com/agents",
        )
        with (
            patch(
                "search_engine.socket.getaddrinfo",
                return_value=[_public_dns_result()],
            ),
            patch("search_engine.build_opener") as build_opener,
        ):
            build_opener.return_value.open.return_value = response
            document = scrape_website("https://example.com/agents")

        first_heading_offset, first_heading_anchor = document.sections[0]
        self.assertEqual(first_heading_anchor, "scientific-agent")
        self.assertIn(
            "Scientific Discovery Agent",
            document.text[first_heading_offset : first_heading_offset + 40],
        )
        second_heading_offset, second_heading_anchor = document.sections[1]
        self.assertEqual(second_heading_anchor, "generative-agents")
        self.assertGreater(
            second_heading_offset,
            document.text.index("Generative Agents Simulation"),
        )
        self.assertIn(
            "Generative Agents Simulation",
            document.text[second_heading_offset : second_heading_offset + 40],
        )
        self.assertNotIn("#", document.text)
        passages = chunk_documents([document])
        results = graph_rag_search("generative agents memory stream", passages)

        self.assertTrue(results)
        self.assertTrue(
            all(
                "Scientific Discovery Agent" not in result.text
                and "ChemCrow" not in result.text
                for result in results
            )
        )
        self.assertEqual(
            results[0].source_url,
            "https://example.com/agents#generative-agents",
        )

    def test_scrape_website_excludes_citation_and_reference_sections(self):
        page_bodies = [
            (
                b"<main><h2>Challenges #</h2><p>Long-term planning remains challenging.</p>"
                b"<h2>Citation #</h2><p>Cited as: Example article.</p>"
                b"<h2>References #</h2><ol><li>Wei et al. Chain of thought prompting.</li></ol></main>",
                "Long-term planning remains challenging.",
            ),
            (
                b"<main><p>Long-term planning remains challenging.</p>"
                b"<h2>References #</h2><ol><li>Wei et al. Chain of thought prompting.</li></ol></main>",
                "Long-term planning remains challenging.",
            ),
        ]
        for html, expected_content in page_bodies:
            with self.subTest(html=html):
                response = _FakeWebResponse(html, "https://example.com/guide")
                with (
                    patch(
                        "search_engine.socket.getaddrinfo",
                        return_value=[_public_dns_result()],
                    ),
                    patch("search_engine.build_opener") as build_opener,
                ):
                    build_opener.return_value.open.return_value = response
                    document = scrape_website("https://example.com/guide")

                self.assertIn(expected_content, document.text)
                self.assertNotIn("Cited as", document.text)
                self.assertNotIn("Wei et al.", document.text)
                passages = chunk_documents([document])
                self.assertTrue(passages)
                self.assertTrue(all("Wei et al." not in passage.text for passage in passages))

    def test_scrape_website_rejects_private_network_addresses(self):
        private_dns_result = (
            socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            ("127.0.0.1", 80),
        )
        with (
            patch("search_engine.socket.getaddrinfo", return_value=[private_dns_result]),
            patch("search_engine.build_opener") as build_opener,
        ):
            with self.assertRaisesRegex(ValueError, "publicly reachable"):
                scrape_website("http://localhost/")
        build_opener.assert_not_called()

    def test_scrape_website_rejects_non_html_responses(self):
        response = _FakeWebResponse(b"plain text", "https://example.com/file.txt", "text/plain")
        with (
            patch(
                "search_engine.socket.getaddrinfo",
                return_value=[_public_dns_result()],
            ),
            patch("search_engine.build_opener") as build_opener,
        ):
            build_opener.return_value.open.return_value = response
            with self.assertRaisesRegex(ValueError, "not an HTML page"):
                scrape_website("https://example.com/file.txt")

    def test_search_returns_matching_chunks_from_multiple_pages(self):
        passages = chunk_documents(
            [
                SourceDocument("guide.pdf", "Troubleshooting guide for network errors.", page=2),
                SourceDocument("guide.pdf", "Network troubleshooting steps for servers.", page=5),
            ]
        )

        results = search("network troubleshooting", passages)

        self.assertEqual({result.page for result in results}, {2, 5})

    def test_graph_rag_search_covers_matching_pages_before_repeating_a_page(self):
        passages = chunk_documents(
            [
                SourceDocument(
                    "guide.pdf",
                    "retrieval network alpha. retrieval network beta. retrieval network gamma.",
                    page=1,
                ),
                SourceDocument("guide.pdf", "retrieval network delta.", page=2),
                SourceDocument("guide.pdf", "retrieval network epsilon.", page=3),
            ],
            chunk_size=4,
            overlap=0,
        )

        results = graph_rag_search("retrieval network", passages, top_k=4)

        self.assertEqual({result.page for result in results[:3]}, {1, 2, 3})
        self.assertEqual(len(results), 4)
        self.assertEqual(sum(result.page == 1 for result in results), 2)

    def test_graph_rag_reranks_full_query_match_above_repeated_single_term(self):
        passages = chunk_documents(
            [
                SourceDocument(
                    "guide.txt",
                    "Network troubleshooting protocol details and recovery steps.",
                ),
                SourceDocument(
                    "guide.txt",
                    "network network network network network network network network",
                ),
            ]
        )

        results = graph_rag_search(
            "network troubleshooting protocol",
            passages,
            top_k=2,
            min_relative_score=0.0,
        )

        self.assertIn("troubleshooting protocol", results[0].text)
        self.assertGreater(results[0].score, results[1].score)

    def test_graph_rag_similarity_cutoff_keeps_multiple_close_matches(self):
        passages = chunk_documents(
            [
                SourceDocument(
                    "guide.pdf",
                    "A mobile device includes smartphones and tablet computers.",
                    page=1,
                ),
                SourceDocument(
                    "guide.pdf",
                    "Mobile devices such as smartphones and tablets support learning.",
                    page=2,
                ),
                SourceDocument(
                    "guide.pdf",
                    "Mobile devices can also be used in classroom discussions.",
                    page=3,
                ),
            ]
        )

        results = graph_rag_search(
            "what is mobile device",
            passages,
            top_k=5,
            min_relative_score=0.97,
        )

        self.assertGreaterEqual(len(results), 2)
        self.assertLessEqual(len(results), 3)
        self.assertEqual(len({result.page for result in results}), len(results))

    def test_search_uses_current_documents_and_returns_no_false_match(self):
        old_data = chunk_documents([SourceDocument("old.txt", "The old policy mentions blue widgets.")])
        new_data = chunk_documents([SourceDocument("new.txt", "The changed policy mentions green gadgets.")])
        self.assertEqual(search("green gadgets", new_data)[0].source, "new.txt")
        self.assertEqual(search("green gadgets", old_data), [])

    def test_load_document_supports_utf8_text(self):
        documents = load_document("notes.md", "A useful note.".encode("utf-8"))
        self.assertEqual(documents, [SourceDocument("notes.md", "A useful note.")])

    def test_load_document_rejects_unsupported_file_type(self):
        with self.assertRaises(ValueError):
            load_document("notes.docx", b"not supported")

    def test_extract_pdf_page_returns_only_the_requested_page(self):
        from io import BytesIO
        from pypdf import PdfReader, PdfWriter

        source = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.add_blank_page(width=100, height=100)
        writer.write(source)

        extracted = extract_pdf_page(source.getvalue(), 2)

        self.assertEqual(len(PdfReader(BytesIO(extracted)).pages), 1)
        with self.assertRaisesRegex(ValueError, "outside the document"):
            extract_pdf_page(source.getvalue(), 3)

    def test_empty_and_nonmatching_queries_return_no_results(self):
        passages = chunk_documents([SourceDocument("guide.txt", "Travel plans and schedules.")])
        self.assertEqual(search("   ", passages), [])
        self.assertEqual(search("volcano", passages), [])

    def test_excerpt_prefers_specific_heading_and_command(self):
        passages = chunk_documents(
            [
                SourceDocument(
                    "docker.pdf",
                    "Use Docker Compose to Start a Group of Containers\n"
                    "Use this command from the compose directory.\n"
                    "docker-compose up -d\n"
                    "This fetches the latest image from the repository.",
                    page=3,
                )
            ]
        )
        results = search("How do I start a group of containers with Docker Compose?", passages)
        excerpt_match = find_relevant_excerpt(
            "How do I start a group of containers with Docker Compose?", results
        )
        self.assertIsNotNone(excerpt_match)
        assert excerpt_match is not None
        self.assertIn("docker-compose up -d", excerpt_match[1])
        self.assertEqual(excerpt_match[0].page, 3)

    def test_plural_queries_match_singular_terms(self):
        passages = chunk_documents([SourceDocument("docker.pdf", "Stop a Docker Container.")])
        self.assertEqual(search("stop docker containers", passages)[0].source, "docker.pdf")

    def test_definition_question_prefers_glossary_entry(self):
        passages = chunk_documents(
            [
                SourceDocument(
                    "python.pdf",
                    "Debugging can be frustrating, but it is a valuable skill.\n"
                    "debugging: The process of finding and removing errors from a program.",
                    page=30,
                )
            ]
        )
        query = "What is debugging?"
        excerpt_match = find_relevant_excerpt(query, search(query, passages))
        self.assertIsNotNone(excerpt_match)
        assert excerpt_match is not None
        self.assertIn("debugging: The process of finding", excerpt_match[1])

    def test_definition_question_includes_paragraph_after_matching_heading(self):
        passages = chunk_documents(
            [
                SourceDocument(
                    "report.pdf",
                    "C. Data Extraction\n"
                    "Given the methodological and contextual heterogeneity of the studies including "
                    "intervention types, outcomes, and designs, a narrative synthesis approach was "
                    "used to analyze and report findings.\n"
                    "This approach involved organizing studies into conceptual categories such as "
                    "training delivery modes, content focus, behavioral outcomes, and residual risks.\n"
                    "The emphasis was on summarizing patterns and describing trends",
                    page=3,
                )
            ]
        )
        query = "what is Data Extraction"
        excerpt_match = find_relevant_excerpt(query, search(query, passages))

        self.assertIsNotNone(excerpt_match)
        assert excerpt_match is not None
        self.assertIn("C. Data Extraction", excerpt_match[1])
        self.assertIn("C. Data Extraction", excerpt_match[1])
        self.assertLessEqual(len(excerpt_match[1]), 220)
        self.assertEqual(excerpt_match[0].page, 3)


if __name__ == "__main__":
    unittest.main()
