import unittest
from unittest.mock import Mock, patch
import torch

# Import the actual classes to understand their structure
from mas_consensus.agents import WorkerAgent, ManagerAgent
from mas_consensus.config import TextChunk
from mas_consensus.tasks import TaskType
from mas_consensus.llm import HuggingFaceLLM


class TestWorkerAgent(unittest.TestCase):

    def setUp(self):
        # Create a proper mock for the LLM that matches the expected interface
        self.mock_llm = Mock(spec=HuggingFaceLLM)
        self.mock_llm.invoke.return_value = "Test response from worker"

    @patch("mas_consensus.agents.logging")
    def test_process_chunk(self, mock_logging):
        worker = WorkerAgent(self.mock_llm, "test_chunk")
        chunk = TextChunk(text="Test chunk content", chunk_id="test_chunk")
        result = worker.process_chunk(
            chunk=chunk,
            previous_summary="Previous summary",
            query="Test query",
            instruction="Process chunk: {chunk_text} with summary: {previous_summary} and query: {question}",
        )

        # Verify the result
        self.assertEqual(result, "Test response from worker")

        # Verify the LLM was called (exact call content might vary based on implementation)
        self.mock_llm.invoke.assert_called_once()


class TestManagerAgent(unittest.TestCase):

    def setUp(self):
        # Create a proper mock for the LLM that matches the expected interface
        self.mock_llm = Mock(spec=HuggingFaceLLM)
        self.mock_llm.invoke.return_value = "<answer>Test final answer</answer>"

    @patch("mas_consensus.agents.logging")
    def test_generate_response(self, mock_logging):
        manager = ManagerAgent(self.mock_llm)
        result = manager.generate_response(
            summary="Test summary",
            query="Test query",
            instruction="Generate response from summary: {last_summary} and query: {question}",
            task_type=TaskType.QA,
        )

        # Verify the result (tags should be removed)
        self.assertEqual(result, "Test final answer")

        # Verify the LLM was called
        self.mock_llm.invoke.assert_called_once()

    def test_remove_answer_tags_qa(self):
        # Test directly with the class method since it's static
        response = "<answer>This is the answer</answer>"
        result = ManagerAgent.remove_answer_tags(response, TaskType.QA)
        self.assertEqual(result, "This is the answer")

    def test_remove_answer_tags_summarization(self):
        # Test directly with the class method since it's static
        response = "<summary>This is the summary</summary>"
        result = ManagerAgent.remove_answer_tags(response, TaskType.SUMMARIZATION)
        self.assertEqual(result, "This is the summary")

    def test_remove_answer_tags_no_tags(self):
        # Test directly with the class method since it's static
        response = "This is the response without tags"
        result = ManagerAgent.remove_answer_tags(response, TaskType.QA)
        self.assertEqual(result, "This is the response without tags")


if __name__ == "__main__":
    unittest.main()
