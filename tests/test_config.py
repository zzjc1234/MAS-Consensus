import unittest
from mas_consensus.config import (
    ProcessingMode,
    TextChunk,
    ChainOfAgentsConfig,
    TaskConfig,
)


class TestProcessingMode(unittest.TestCase):

    def test_processing_mode_values(self):
        self.assertEqual(ProcessingMode.LTR.value, "left_to_right")
        self.assertEqual(ProcessingMode.RTL.value, "right_to_left")
        self.assertEqual(ProcessingMode.RAND.value, "random")


class TestTextChunk(unittest.TestCase):

    def test_text_chunk_creation(self):
        chunk = TextChunk(text="Test text", chunk_id="1", token_count=100)

        self.assertEqual(chunk.text, "Test text")
        self.assertEqual(chunk.chunk_id, "1")
        self.assertEqual(chunk.token_count, 100)
        self.assertIsNone(chunk.left_child)
        self.assertIsNone(chunk.right_child)
        self.assertEqual(chunk.depth, 0)

    def test_text_chunk_with_children(self):
        left_child = TextChunk(text="Left text", chunk_id="1.L")
        right_child = TextChunk(text="Right text", chunk_id="1.R")

        parent_chunk = TextChunk(
            text="Parent text",
            chunk_id="1",
            left_child=left_child,
            right_child=right_child,
            depth=1,
            token_count=200,
        )

        self.assertEqual(parent_chunk.left_child, left_child)
        self.assertEqual(parent_chunk.right_child, right_child)
        self.assertEqual(parent_chunk.depth, 1)
        self.assertEqual(parent_chunk.token_count, 200)


class TestChainOfAgentsConfig(unittest.TestCase):

    def test_default_config(self):
        config = ChainOfAgentsConfig()

        self.assertEqual(config.worker_context_window, 16384)
        self.assertEqual(config.manager_context_window, 16384)
        self.assertEqual(config.max_tokens_per_chunk, 8192)
        self.assertEqual(config.processing_mode, ProcessingMode.LTR)
        self.assertEqual(config.split_threshold, 1.1)
        self.assertEqual(config.sensitivity_curve, 0.3)
        self.assertEqual(config.min_tokens_to_split, 512)

    def test_custom_config(self):
        config = ChainOfAgentsConfig(
            max_tokens_per_chunk=4096, processing_mode=ProcessingMode.RTL
        )

        self.assertEqual(config.max_tokens_per_chunk, 4096)
        self.assertEqual(config.processing_mode, ProcessingMode.RTL)


class TestTaskConfig(unittest.TestCase):

    def test_task_config_creation(self):
        config = TaskConfig(
            first_worker_instruction="First worker instruction",
            worker_instruction="Worker instruction",
            manager_instruction="Manager instruction",
            multi_summary_instruction="Multi summary instruction",
        )

        self.assertEqual(config.first_worker_instruction, "First worker instruction")
        self.assertEqual(config.worker_instruction, "Worker instruction")
        self.assertEqual(config.manager_instruction, "Manager instruction")
        self.assertEqual(config.multi_summary_instruction, "Multi summary instruction")


if __name__ == "__main__":
    unittest.main()
