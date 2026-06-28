import unittest

from rag.generator import parse_json_response
from rag.ingestion import chunk_text
from rag.retriever import RetrievedChunk, estimate_confidence, keyword_overlap_score, tokenize


class CoreRagTests(unittest.TestCase):
    def test_chunk_text_uses_overlap(self):
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_text(text, chunk_size=5, overlap=2)

        self.assertEqual(chunks[0], "one two three four five")
        self.assertEqual(chunks[1], "four five six seven eight")
        self.assertEqual(chunks[2], "seven eight nine ten")

    def test_tokenize_removes_common_stopwords(self):
        self.assertEqual(tokenize("What is the refund policy?"), {"refund", "policy"})

    def test_keyword_overlap_prefers_matching_chunks(self):
        matching = keyword_overlap_score("refund policy", "The refund policy allows returns.")
        unrelated = keyword_overlap_score("refund policy", "Employees receive annual leave.")

        self.assertGreater(matching, unrelated)

    def test_parse_json_response_handles_fenced_json(self):
        parsed = parse_json_response('```json\n{"answer": "ok", "answer_found": true}\n```')

        self.assertEqual(parsed["answer"], "ok")
        self.assertTrue(parsed["answer_found"])

    def test_confidence_is_zero_without_chunks(self):
        self.assertEqual(estimate_confidence([]), 0.0)

    def test_confidence_uses_supporting_sources(self):
        chunks = [
            RetrievedChunk(
                text="Refunds are allowed within 30 days.",
                metadata={"document": "Policy.pdf", "page": 1},
                semantic_score=0.8,
                keyword_score=0.2,
                combined_score=0.8,
            ),
            RetrievedChunk(
                text="Refund requests must include a receipt.",
                metadata={"document": "Policy.pdf", "page": 2},
                semantic_score=0.75,
                keyword_score=0.2,
                combined_score=0.76,
            ),
        ]

        self.assertGreater(estimate_confidence(chunks), 0.8)


if __name__ == "__main__":
    unittest.main()
