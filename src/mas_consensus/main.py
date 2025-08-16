"""
Main module for the MAS Consensus system.

This module implements a multi-agent system for consensus-based text processing.
It orchestrates a chain of worker agents to analyze text chunks and a manager
agent to synthesize the final response. The system supports both question
answering and summarization tasks.

Example usage:
    mas_consensus --model google/flan-t5-small --model_type seq2seq --instruction_format t5 --file_path paper.pdf --task qa --query "What is the main contribution of the paper?"

For users in China or those experiencing network issues with Hugging Face:
    HF_ENDPOINT=https://hf-mirror.com mas_consensus [OPTIONS]
"""

import argparse
import gc
import logging
import os

import requests
import torch

from .agents import ManagerAgent
from .chain import ChainOfAgents, ChunkProcessor
from .config import ChainOfAgentsConfig, ProcessingMode
from .llm import HuggingFaceLLM
from .tasks import TaskType
from .text_processing import extract_text


def download_file(url: str, save_path: str):
    """Downloads a file from a URL."""
    if os.path.exists(save_path):
        logging.info(f"{save_path} already exists. Skipping the download")
    else:
        logging.info(f"Downloading file from {url} to {save_path}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"File downloaded successfully and saved to {save_path}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading file: {e}")
            raise


def main():
    """Main function to run the chain of agents."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run the chain of agents.")
    parser.add_argument(
        "--task",
        type=str,
        default="qa",
        choices=["qa", "summarization"],
        help="Task to perform.",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="paper.pdf",
        help="Path to the input file (PDF or TXT).",
    )
    parser.add_argument(
        "--download_url",
        type=str,
        default="https://openreview.net/pdf?id=LuCLf4BJsr",
        help="URL to download the file from.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="List all the datasets used in the paper.",
        help="Query for the QA task.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Hugging Face model to use.",
    )
    parser.add_argument(
        "--instruction_format",
        type=str,
        default="phi",
        help="Instruction format for the model.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="causal",
        help="Type of the model, either 'causal' or 'seq2seq'.",
    )
    parser.add_argument(
        "--max_tokens_per_chunk", type=int, default=4096, help="Max tokens per chunk."
    )
    parser.add_argument(
        "--processing_modes",
        nargs="+",
        default=["left_to_right"],
        help="List of processing modes to run.",
    )

    args = parser.parse_args()
    logger.info(f"Starting chain of agents with args: {args}")

    download_file(args.download_url, args.file_path)

    logger.info(f"Loading model: {args.model}")
    llm = HuggingFaceLLM(
        model=args.model,
        max_tokens_response=1024,
        context_window=16384,
        instruction_format=args.instruction_format,
        model_type=args.model_type,
    )

    logger.info(f"Extracting text from {args.file_path}")
    text = extract_text(args.file_path)
    task_type = TaskType(args.task)

    config = ChainOfAgentsConfig(
        max_tokens_per_chunk=args.max_tokens_per_chunk,
    )

    logger.info("Splitting text into chunks")
    chunk_processor = ChunkProcessor(llm, config)
    initial_chunks = chunk_processor.split_into_chunks(text)
    logger.info(f"Created {len(initial_chunks)} chunks")

    final_summaries = []
    coa = None  # Initialize coa to prevent unbound variable error
    for mode_str in args.processing_modes:
        # Convert string to ProcessingMode enum
        if mode_str.upper() in ProcessingMode.__members__:
            mode = ProcessingMode[mode_str.upper()]
        else:
            # Try to match by value
            mode = next(
                (
                    m
                    for m in ProcessingMode
                    if m.value == mode_str.lower().replace("-", "_")
                ),
                ProcessingMode.LTR,
            )
        logger.info(f"Processing with {mode.value} mode")
        config.processing_mode = mode
        coa = ChainOfAgents(llm, initial_chunks, config, task_type)
        chain_summary = coa.process(
            query=args.query if task_type == TaskType.QA else None
        )
        final_summaries.append(chain_summary)

    logger.info("Generating final response with manager agent")
    manager = ManagerAgent(llm)
    if len(final_summaries) > 1:
        combined_summaries = "\n\n".join(
            f"Summary n°{i+1}:\n{summary}" for i, summary in enumerate(final_summaries)
        )
        final_response = manager.generate_response(
            summary=combined_summaries,
            query=args.query if task_type == TaskType.QA else None,
            instruction=(
                coa.task_config.multi_summary_instruction if coa else ""
            ),  # Handle potential None coa
            task_type=task_type,
        )
    else:
        final_response = manager.generate_response(
            summary=final_summaries[0] if final_summaries else "",
            query=args.query if task_type == TaskType.QA else None,
            instruction=(
                coa.task_config.manager_instruction if coa else ""
            ),  # Handle potential None coa
            task_type=task_type,
        )

    logger.info("Final Manager Response:")
    print(final_response)

    # Clean up memory
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU memory has been cleared.")


if __name__ == "__main__":
    main()
