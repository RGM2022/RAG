import unittest

from search_engine import (
    SourceDocument,
    chunk_documents,
    find_relevant_excerpt,
    load_document,
    search,
    tokenize,
)


class SearchEngineTests(unittest.TestCase):
    def test_tokenize_is_case_insensitive_and_handles_punctuation(self):
        self.assertEqual(tokenize("Returns, RETURNS! user's"), ["returns", "returns", "user's"])

    def test_chunks_preserve_source_and_page(self):
        passage = chunk_documents(
            [SourceDocument(source="guide.pdf", page=4, text="alpha " * 30)],
            chunk_size=40,
            overlap=8,
        )
        self.assertGreater(len(passage), 1)
        self.assertTrue(all(item.source == "guide.pdf" and item.page == 4 for item in passage))

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


if __name__ == "__main__":
    unittest.main()
