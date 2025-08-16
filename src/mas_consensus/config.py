from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for a task, defining the instructions for agents."""
    first_worker_instruction: str
    worker_instruction: str
    manager_instruction: str
    multi_summary_instruction: str


class ProcessingMode(Enum):
    """Enum for the processing mode of the chain of agents."""
    LTR = "left_to_right"
    RTL = "right_to_left"
    RAND = "random"


@dataclass
class TextChunk:
    """Represents a chunk of text for processing."""
    text: str
    chunk_id: str = "-1"
    left_child: Optional['TextChunk'] = None
    right_child: Optional['TextChunk'] = None
    depth: int = 0
    token_count: int = 0


@dataclass
class ChainOfAgentsConfig:
    """Configuration for the chain of agents."""
    worker_context_window: int = 16384
    manager_context_window: int = 16384
    max_tokens_per_chunk: int = 8192
    processing_mode: ProcessingMode = ProcessingMode.LTR
    split_threshold: float = 1.1
    sensitivity_curve: float = 0.3
    min_tokens_to_split: int = 512
