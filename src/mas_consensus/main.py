import argparse
import gc
import torch
import requests

from .llm import HuggingFaceLLM
from .config import ChainOfAgentsConfig, ProcessingMode
from .tasks import TaskType
from .text_processing import extract_text
from .chain import ChainOfAgents, ChunkProcessor
from .agents import ManagerAgent


def download_file(url: str, save_path: str):
    """Downloads a file from a URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded successfully and saved to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        raise


def main():
    """Main function to run the chain of agents."""
    parser = argparse.ArgumentParser(description="Run the chain of agents.")
    parser.add_argument("--task", type=str, default="qa", choices=["qa", "summarization"], help="Task to perform.")
    parser.add_argument("--file_path", type=str, default="paper.pdf", help="Path to the input file (PDF or TXT).")
    parser.add_argument("--download_url", type=str, default="https://openreview.net/pdf?id=LuCLf4BJsr", help="URL to download the file from.")
    parser.add_argument("--query", type=str, default="List all the datasets used in the paper.", help="Query for the QA task.")
    parser.add_argument("--model", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct", help="Hugging Face model to use.")
    parser.add_argument("--max_tokens_per_chunk", type=int, default=4096, help="Max tokens per chunk.")
    parser.add_argument("--processing_modes", nargs='+', default=['left_to_right'], help="List of processing modes to run.")

    args = parser.parse_args()

    download_file(args.download_url, args.file_path)

    llm = HuggingFaceLLM(
        model=args.model,
        max_tokens_response=1024,
        context_window=16384,
        instruction_format=args.instruction_format,
    )

    text = extract_text(args.file_path)
    task_type = TaskType(args.task)

    config = ChainOfAgentsConfig(
        max_tokens_per_chunk=args.max_tokens_per_chunk,
    )

    chunk_processor = ChunkProcessor(llm, config)
    initial_chunks = chunk_processor.split_into_chunks(text)

    final_summaries = []
    for mode_str in args.processing_modes:
        mode = ProcessingMode(mode_str.lower().replace('-', '_'))
        print(f"\n=== Processing with {mode.value} mode ===")
        config.processing_mode = mode
        coa = ChainOfAgents(llm, initial_chunks, config, task_type)
        chain_summary = coa.process(query=args.query if task_type == TaskType.QA else None)
        final_summaries.append(chain_summary)

    manager = ManagerAgent(llm)
    if len(final_summaries) > 1:
        combined_summaries = "\n\n".join(
            f"Summary n°{i+1}:\n{summary}"
            for i, summary in enumerate(final_summaries)
        )
        final_response = manager.generate_response(
            summary=combined_summaries,
            query=args.query if task_type == TaskType.QA else None,
            instruction=coa.task_config.multi_summary_instruction,
            task_type=task_type
        )
    else:
        final_response = manager.generate_response(
            summary=final_summaries[0],
            query=args.query if task_type == TaskType.QA else None,
            instruction=coa.task_config.manager_instruction,
            task_type=task_type
        )

    print("\n=== Final Manager Response ===")
    print(final_response)

    # Clean up memory
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory has been cleared.")


if __name__ == "__main__":
    main()
