import re
from typing import Optional

from phi.assistant import Assistant

from .llm import HuggingFaceLLM
from .config import TextChunk
from .tasks import TaskType


class WorkerAgent(Assistant):
    """A worker agent that processes a chunk of text."""

    def __init__(self, llm: HuggingFaceLLM, chunk_id: str):
        super().__init__(
            name=f"worker_{chunk_id}",
            llm=llm
        )

    def process_chunk(
        self,
        chunk: TextChunk,
        previous_summary: Optional[str],
        query: Optional[str],
        instruction: str
    ) -> str:
        """Processes a single chunk of text and returns a summary."""
        formatted_instruction = instruction.format(
            chunk_text=chunk.text,
            previous_summary=previous_summary or "No previous summary",
            question=query
        )

        prompt = self.llm.format_prompt(formatted_instruction)

        print(f"\n=== Worker Agent {chunk.chunk_id} Processing ===")
        response = self.llm.complete(prompt)
        print(f"Worker Response:\n{response}\n")

        return response


class ManagerAgent(Assistant):
    """A manager agent that synthesizes summaries into a final response."""

    def __init__(self, llm: HuggingFaceLLM):
        super().__init__(
            name="manager",
            llm=llm
        )

    @staticmethod
    def remove_answer_tags(response: str, task_type: TaskType) -> str:
        """Removes answer/summary tags from the response."""
        tag = "answer" if task_type == TaskType.QA else "summary"
        pattern = f"<({tag})>(.*?)</\1>"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(2).strip()
        return response.strip()

    def generate_response(
        self,
        summary: str,
        query: Optional[str],
        instruction: str,
        task_type: TaskType
    ) -> str:
        """Generates the final response from the summary."""
        formatted_instruction = instruction.format(
            last_summary=summary,
            question=query
        )

        prompt = self.llm.format_prompt(formatted_instruction)

        self.logger.info("Manager Agent processing...")
        response = self.llm.complete(prompt)
        self.logger.info(f"Manager response length: {len(response)}")
        return self.remove_answer_tags(response, task_type)
