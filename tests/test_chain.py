import unittest
from unittest.mock import Mock, patch
import numpy as np

from mas_consensus.chain import ChunkProcessor, ChainOfAgents
from mas_consensus.config import TextChunk, ChainOfAgentsConfig, ProcessingMode
from mas_consensus.tasks import TaskType


class TestChunkProcessor(unittest.TestCase):

    def setUp(self):
        self.mock_llm = Mock()
        self.mock_llm.tokenizer = Mock()
        self.mock_llm.tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        self.config = ChainOfAgentsConfig(max_tokens_per_chunk=100)
        self.processor = ChunkProcessor(self.mock_llm, self.config)

    def test_calculate_entropy(self):
        # Test with a simple text
        text = "the the the and and or"
        entropy = self.processor.calculate_entropy(text)
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0)

    def test_split_into_chunks_small_text(self):
        # Text smaller than max_tokens_per_chunk should not be split
        small_text = "This is a small text."
        self.mock_llm.tokenizer.encode.return_value = [1, 2, 3]  # 3 tokens

        chunks = self.processor.split_into_chunks(small_text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, small_text)
        self.assertEqual(chunks[0].chunk_id, "0")

    @patch("mas_consensus.chain.SENTENCE_SPLIT_PATTERN")
    def test_split_into_chunks_large_text(self, mock_pattern):
        # Text larger than max_tokens_per_chunk should be split
        large_text = "Sentence one. Sentence two. Sentence three. Sentence four."
        self.mock_llm.tokenizer.encode.return_value = list(
            range(150)
        )  # 150 tokens (> 100)
        mock_pattern.split.return_value = [
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
            "Sentence four.",
        ]

        chunks = self.processor.split_into_chunks(large_text)

        # Should be split into multiple chunks
        self.assertGreater(len(chunks), 1)
        # All chunks should have the same text as the original (simplified test)
        combined_text = "".join(chunk.text for chunk in chunks)
        self.assertIn("Sentence one", combined_text)
        self.assertIn("Sentence four", combined_text)


class TestChainOfAgents(unittest.TestCase):

    def setUp(self):
        self.mock_llm = Mock()
        self.chunks = [
            TextChunk(text="Chunk 1", chunk_id="1"),
            TextChunk(text="Chunk 2", chunk_id="2"),
            TextChunk(text="Chunk 3", chunk_id="3"),
        ]
        self.config = ChainOfAgentsConfig()
        self.chain = ChainOfAgents(self.mock_llm, self.chunks, self.config, TaskType.QA)

    def test_get_chunk_order_ltr(self):
        self.config.processing_mode = ProcessingMode.LTR
        ordered_chunks = self.chain._get_chunk_order()

        self.assertEqual(ordered_chunks[0].chunk_id, "1")
        self.assertEqual(ordered_chunks[1].chunk_id, "2")
        self.assertEqual(ordered_chunks[2].chunk_id, "3")

    def test_get_chunk_order_rtl(self):
        self.config.processing_mode = ProcessingMode.RTL
        ordered_chunks = self.chain._get_chunk_order()

        self.assertEqual(ordered_chunks[0].chunk_id, "3")
        self.assertEqual(ordered_chunks[1].chunk_id, "2")
        self.assertEqual(ordered_chunks[2].chunk_id, "1")

    @patch("mas_consensus.chain.random")
    def test_get_chunk_order_random(self, mock_random):
        mock_random.sample.return_value = [
            self.chunks[2],
            self.chunks[0],
            self.chunks[1],
        ]
        self.config.processing_mode = ProcessingMode.RAND
        ordered_chunks = self.chain._get_chunk_order()

        self.assertEqual(ordered_chunks[0].chunk_id, "3")
        self.assertEqual(ordered_chunks[1].chunk_id, "1")
        self.assertEqual(ordered_chunks[2].chunk_id, "2")


if __name__ == "__main__":
    unittest.main()
