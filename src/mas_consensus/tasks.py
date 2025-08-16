from enum import Enum
from .config import TaskConfig


class TaskType(Enum):
    """Enum for the type of task."""
    QA = "qa"
    SUMMARIZATION = "summarization"


class TaskFactory:
    """Factory for creating task configurations."""

    @staticmethod
    def create_task(task_type: TaskType) -> TaskConfig:
        """Creates a task configuration based on the task type."""
        if task_type == TaskType.QA:
            return TaskConfig(
                first_worker_instruction="""
                [CONTEXT]
                {chunk_text}
                [QUESTION]
                {question}
                [TASK]
                List several facts from the provided context that might help to answer the question:
                - [Fact 1]
                - [Fact 2]
                ...
                Provide a [SUMMARY] by summarizing all the relevant information related to the question.
                Keep the existing structure in focus. Do not answer the question.
                """,
                worker_instruction="""
                [SUMMARY]
                {previous_summary}
                [CONTEXT]
                {chunk_text}
                [QUESTION]
                {question}
                [TASK]
                List several facts from the provided context that might help to answer the question:
                - [Fact 1]
                - [Fact 2]
                ...
                Prioritize all relevant information to the question and then refine the current summary by including the new additional information in [REVISED SUMMARY].
                Do not answer the question.
                """,
                manager_instruction="""
                [SUMMARY]
                {last_summary}
                [QUESTION]
                {question}
                [TASK]
                Using all context available, resolve any contradictions, and provide a comprehensive answer below.
                Answer format: <answer>...</answer>
                """,
                multi_summary_instruction="""
                [SUMMARIES]
                {last_summary}
                [QUESTION]
                {question}
                [TASK]
                Integrate all informations from every summary, resolve contradictions, and provide a comprehensive answer.
                Answer format: <answer>[Combined response using all relevant facts]</answer>
                """
            )
        elif task_type == TaskType.SUMMARIZATION:
            return TaskConfig(
                first_worker_instruction="""
                [CONTEXT]
                {chunk_text}
                [TASK]
                Create a lengthy summary by incorporating key information from this context.
                Focus on main ideas, findings, and conclusions. Maintain a coherent narrative.
                """,
                worker_instruction="""
                [PREVIOUS SUMMARY]
                {previous_summary}
                [CONTEXT]
                {chunk_text}
                [TASK]
                Refine the existing summary by incorporating key information from this context.
                Focus on main ideas, findings, and conclusions. Maintain a coherent narrative.
                """,
                manager_instruction="""
                [SUMMARY]
                {last_summary}
                [TASK]
                Using all context available, resolve any contradictions and only provide a single lengthy summary using simple, everyday language.
                Format: <summary>...</summary>
                """,
                multi_summary_instruction="""
                [SUMMARIES]
                {last_summary}
                [TASK]
                Integrate all information from the summaries into a single coherent summary.
                Resolve any contradictions and ensure the logical flow of ideas.
                """
            )
