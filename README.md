# MAS Consensus

A multi-agent system for consensus-based text processing, implementing the "Chain of Agents" approach for analyzing long-context tasks with large language models.

Based on the paper: [Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://openreview.net/pdf?id=LuCLf4BJsr)

## Installation

1.  **Clone the repository:**

    ```sh
    git clone git@github.com:zzjc1234/MAS-Consensus.git
    cd MAS_Consensus
    ```

2.  **Install dependencies:**

    This project uses `uv` for package management. Install the dependencies, including the development dependencies, by running:

    ```sh
    uv pip install -e .[dev]
    ```

## Usage

You can run the application using the `mas_consensus` command-line script.

```sh
mas_consensus [OPTIONS]
```

### Example

Basic QA task with a PDF file:

```sh
mas_consensus --model google/flan-t5-small --model_type seq2seq --instruction_format t5 --file_path paper.pdf --task qa --query "What is the main contribution of the paper?"
```

Summarization task:

```sh
mas_consensus --model google/flan-t5-small --model_type seq2seq --instruction_format t5 --file_path paper.pdf --task summarization
```

For users in China or those experiencing network issues with Hugging Face, you can use a mirror endpoint:

```sh
HF_ENDPOINT=https://hf-mirror.com mas_consensus [OPTIONS]
```

### Key Arguments

- `--task`: The task to perform (`qa` or `summarization`).
- `--file_path`: Path to the input file (PDF or TXT).
- `--download_url`: URL to download the file from (default: https://openreview.net/pdf?id=LuCLf4BJsr).
- `--query`: The query for the QA task.
- `--model`: The Hugging Face model to use (default: microsoft/phi-2).
- `--model_type`: The type of model (`causal` or `seq2seq`).
- `--instruction_format`: The prompt format for the model (`phi`, `llama`, `mistral`, `t5`, etc.).
- `--max_tokens_per_chunk`: Maximum tokens per chunk (default: 4096).
- `--processing_modes`: List of processing modes to run (LTR, RTL, RAND).

## How It Works

MAS Consensus implements a multi-agent system where:

1. Text is split into chunks and processed by worker agents
2. Each worker agent analyzes their assigned chunk
3. A manager agent synthesizes the final response from all worker outputs

The system supports different processing modes:

- Left-to-right (LTR)
- Right-to-left (RTL)
- Random (RAND)

## Running Tests

To run the tests, use `pytest`:

```sh
pytest
```

The project includes comprehensive tests for all modules:
- `test_agents.py`: Tests for worker and manager agents
- `test_chain.py`: Tests for the chain of agents and chunk processing
- `test_config.py`: Tests for configuration classes and enums
- `test_llm.py`: Tests for the HuggingFace LLM wrapper
- `test_main.py`: Tests for the main module functions
- `test_tasks.py`: Tests for task configurations
- `test_text_processing.py`: Tests for text extraction and processing

## Baseline Implementation

This repository contains a baseline implementation in the `src/baseline` directory. This is a snapshot of the original MAS Consensus implementation that serves as a reference point for future versions. The baseline is fully functional and can be run independently.

To use the baseline implementation, you can directly run:

```sh
python -m baseline.main [OPTIONS]
```

## Pre-commit Hooks

This project uses pre-commit hooks to enforce code quality. To enable them, run the following command once after cloning the repository:

```sh
git config core.hooksPath .githooks
```
