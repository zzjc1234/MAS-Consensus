import re
import logging
from typing import Optional

from phi.assistant.assistant import Assistant
from pydantic import PrivateAttr

from .llm import HuggingFaceLLM
from .config import TextChunk
from .tasks import TaskType


class WorkerAgent(Assistant):
    """A worker agent that processes a chunk of text."""

    _logger: logging.Logger = PrivateAttr()

    def __init__(self, llm: HuggingFaceLLM, chunk_id: str):
        super().__init__(
            name=f"worker_{chunk_id}",
            llm=llm,
            tools=[],  # Add empty tools list to satisfy Assistant requirements
            run_id=f"worker_{chunk_id}_run",
        )
        self._logger = logging.getLogger(__name__)

    def process_chunk(
        self,
        chunk: TextChunk,
        previous_summary: Optional[str],
        query: Optional[str],
        instruction: str,
    ) -> str:
        """Processes a single chunk of text and returns a summary."""
        formatted_instruction = instruction.format(
            chunk_text=chunk.text,
            previous_summary=previous_summary or "No previous summary",
            question=query,
        )

        # Use the LLM's format_prompt method if available, otherwise use the instruction directly
        prompt = formatted_instruction
        if hasattr(self.llm, "format_prompt"):
            try:
                prompt = self.llm.format_prompt(formatted_instruction)  # type: ignore
            except Exception:
                prompt = formatted_instruction

        self._logger.info(f"Worker Agent {chunk.chunk_id} processing...")
        # Use the LLM's invoke method if available, otherwise return empty string
        response = ""
        try:
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt) or ""  # type: ignore
            elif hasattr(self.llm, "response"):
                response = self.llm.response(prompt) or ""  # type: ignore
            elif hasattr(self.llm, "complete"):
                response = self.llm.complete(prompt) or ""  # type: ignore
            else:
                response = ""
        except Exception:
            response = ""
        self._logger.info(f"Worker {chunk.chunk_id} response length: {len(response)}")

        return response


class ManagerAgent(Assistant):
    """A manager agent that synthesizes summaries into a final response."""

    _logger: logging.Logger = PrivateAttr()

    def __init__(self, llm: HuggingFaceLLM):
        super().__init__(
            name="manager",
            llm=llm,
            tools=[],  # Add empty tools list to satisfy Assistant requirements
            run_id="manager_run",
        )
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def remove_answer_tags(response: str, task_type: TaskType) -> str:
        """Removes answer/summary tags from the response."""
        tag = "answer" if task_type == TaskType.QA else "summary"
        pattern = f"<({tag})>(.*?)</\\1>"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(2).strip()
        return response.strip()

    def generate_response(
        self, summary: str, query: Optional[str], instruction: str, task_type: TaskType
    ) -> str:
        """Generates the final response from the summary."""
        formatted_instruction = instruction.format(last_summary=summary, question=query)

        # Use the LLM's format_prompt method if available, otherwise use the instruction directly
        prompt = formatted_instruction
        if hasattr(self.llm, "format_prompt"):
            try:
                prompt = self.llm.format_prompt(formatted_instruction)  # type: ignore
            except Exception:
                prompt = formatted_instruction

        self._logger.info("Manager Agent processing...")
        # Use the LLM's invoke method if available, otherwise return empty string
        response = ""
        try:
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt) or ""  # type: ignore
            elif hasattr(self.llm, "response"):
                response = self.llm.response(prompt) or ""  # type: ignore
            elif hasattr(self.llm, "complete"):
                response = self.llm.complete(prompt) or ""  # type: ignore
            else:
                response = ""
        except Exception:
            response = ""
        self._logger.info(f"Manager response length: {len(response)}")
        return self.remove_answer_tags(response, task_type)
