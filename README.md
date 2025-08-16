# MAS Consensus

A multi-agent system for consensus-based text processing.

## Installation

1.  **Clone the repository:**

    ```sh
    git clone <repository-url>
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

```sh
mas_consensus --model google/flan-t5-small --model_type seq2seq --instruction_format t5 --file_path paper.pdf --task qa --query "What is the main contribution of the paper?"
```

### Key Arguments

-   `--task`: The task to perform (`qa` or `summarization`).
-   `--file_path`: Path to the input file (PDF or TXT).
-   `--download_url`: URL to download the file from.
-   `--query`: The query for the QA task.
-   `--model`: The Hugging Face model to use.
-   `--model_type`: The type of model (`causal` or `seq2seq`).
-   `--instruction_format`: The prompt format for the model (`llama`, `mistral`, `t5`, etc.).

## Running Tests

To run the tests, use `pytest`:

```sh
pytest
```

## Pre-commit Hooks

This project uses pre-commit hooks to enforce code quality. To enable them, run the following command once after cloning the repository:

```sh
git config core.hooksPath .githooks
```