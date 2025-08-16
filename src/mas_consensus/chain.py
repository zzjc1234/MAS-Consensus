import random
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm import HuggingFaceLLM
from .config import ChainOfAgentsConfig, TextChunk, ProcessingMode
from .agents import WorkerAgent
from .tasks import TaskFactory, TaskType
from .text_processing import SENTENCE_SPLIT_PATTERN


class ChunkProcessor:
    """Processes and splits text chunks."""

    def __init__(self, llm: HuggingFaceLLM, config: ChainOfAgentsConfig):
        self.llm = llm
        self.config = config
        self.vectorizer = TfidfVectorizer()
        self.mean_score: float = 0.0
        self.std_score: float = 0.0

    def calculate_entropy(self, text: str) -> float:
        """Calculates the Shannon entropy of the text."""
        words = text.split()
        if not words:
            return 0.0
        _, counts = np.unique(words, return_counts=True)
        probabilities = counts / len(words)
        return -np.sum(probabilities * np.log2(probabilities))

    def calculate_priority_score(self, chunk: TextChunk, query: str) -> float:
        """Calculates a priority score for a chunk based on entropy and query similarity."""
        entropy = self.calculate_entropy(chunk.text)
        if not query:
            return entropy

        vectors = self.vectorizer.fit_transform([chunk.text, query])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        return (0.7 * entropy * np.log1p(entropy)) + (0.3 * similarity ** 2)

    def _needs_split(self, chunk: TextChunk, query: Optional[str]) -> bool:
        """Determines if a chunk needs to be split."""
        if chunk.token_count < self.config.min_tokens_to_split:
            return False

        score = self.calculate_priority_score(chunk, query) if query else self.calculate_entropy(chunk.text)
        distribution_factor = 1 - np.exp(-self.config.sensitivity_curve * self.std_score)
        dynamic_threshold = self.mean_score + (self.config.split_threshold * distribution_factor * self.std_score)

        return score > dynamic_threshold

    def split_into_chunks(self, text: str) -> List[TextChunk]:
        """Splits a large text into smaller chunks."""
        tokens = self.llm.tokenizer.encode(text)
        total_tokens = len(tokens)
        del tokens

        if total_tokens <= self.config.max_tokens_per_chunk:
            return [TextChunk(text=text, chunk_id="0", token_count=total_tokens)]

        segments = text.split('\n')
        if len(segments) <= 1:
            segments = SENTENCE_SPLIT_PATTERN.split(text)

        chunks = []
        current_chunk_text = ""
        current_token_count = 0
        chunk_id_counter = 0

        for segment in segments:
            segment_token_count = len(self.llm.tokenizer.encode(segment))
            if current_token_count + segment_token_count > self.config.max_tokens_per_chunk:
                if current_chunk_text:
                    chunks.append(TextChunk(
                        text=current_chunk_text,
                        chunk_id=str(chunk_id_counter),
                        token_count=current_token_count
                    ))
                    chunk_id_counter += 1
                current_chunk_text = segment
                current_token_count = segment_token_count
            else:
                current_chunk_text += "\n" + segment
                current_token_count += segment_token_count

        if current_chunk_text:
            chunks.append(TextChunk(
                text=current_chunk_text,
                chunk_id=str(chunk_id_counter),
                token_count=current_token_count
            ))

        return chunks


class ChainOfAgents:
    """Orchestrates the chain of agents for processing text."""

    def __init__(
        self,
        llm: HuggingFaceLLM,
        chunks: List[TextChunk],
        config: ChainOfAgentsConfig,
        task_type: TaskType,
    ):
        self.llm = llm
        self.chunks = chunks
        self.config = config
        self.task_type = task_type
        self.task_config = TaskFactory.create_task(task_type)
        self.chunk_processor = ChunkProcessor(llm, config)
        self.is_first_chunk = True

    def _get_chunk_order(self) -> List[TextChunk]:
        """Gets the order in which to process the chunks."""
        if self.config.processing_mode == ProcessingMode.LTR:
            return self.chunks
        elif self.config.processing_mode == ProcessingMode.RTL:
            return self.chunks[::-1]
        else:
            return random.sample(self.chunks, len(self.chunks))

    def process(self, query: Optional[str] = None) -> str:
        """Processes the text through the chain of agents."""
        current_summary = None
        self.is_first_chunk = True

        initial_scores = [
            self.chunk_processor.calculate_priority_score(chunk, query) if query else self.chunk_processor.calculate_entropy(chunk.text)
            for chunk in self.chunks
        ]
        self.chunk_processor.mean_score = np.mean(initial_scores) if initial_scores else 0.0
        self.chunk_processor.std_score = np.std(initial_scores) if initial_scores else 0.0

        ordered_chunks = self._get_chunk_order()
        for chunk in ordered_chunks:
            current_summary = self._process_chunk_recursively(chunk, current_summary, query)

        return current_summary

    def _process_chunk_recursively(self, chunk: TextChunk, current_summary: str, query: Optional[str]) -> str:
        """Processes a chunk, splitting it recursively if necessary."""
        if not self.chunk_processor._needs_split(chunk, query):
            worker = WorkerAgent(self.llm, chunk.chunk_id)
            instruction = self.task_config.first_worker_instruction if self.is_first_chunk else self.task_config.worker_instruction
            self.is_first_chunk = False
            return worker.process_chunk(
                chunk=chunk,
                previous_summary=current_summary,
                query=query,
                instruction=instruction
            )

        sentences = SENTENCE_SPLIT_PATTERN.split(chunk.text)
        if len(sentences) < 2:
            # Cannot split further, process as is
            worker = WorkerAgent(self.llm, chunk.chunk_id)
            instruction = self.task_config.first_worker_instruction if self.is_first_chunk else self.task_config.worker_instruction
            self.is_first_chunk = False
            return worker.process_chunk(
                chunk=chunk,
                previous_summary=current_summary,
                query=query,
                instruction=instruction
            )

        mid = len(sentences) // 2
        left_text = " ".join(sentences[:mid])
        right_text = " ".join(sentences[mid:])

        left_chunk = TextChunk(
            text=left_text,
            chunk_id=f"{chunk.chunk_id}.L",
            token_count=len(self.llm.tokenizer.encode(left_text))
        )
        right_chunk = TextChunk(
            text=right_text,
            chunk_id=f"{chunk.chunk_id}.R",
            token_count=len(self.llm.tokenizer.encode(right_text))
        )

        summary = self._process_chunk_recursively(left_chunk, current_summary, query)
        return self._process_chunk_recursively(right_chunk, summary, query)
