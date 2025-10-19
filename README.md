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

    This project uses uv for package management.

    - To install `uv`, check [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
    - Install the dependencies, including the development dependencies, by running:

    ```sh
    uv sync --extra dev # install required packages for dev
    source .venv/bin/activate # activate the virtual env for this prj
    ```

    To install the pre-commit hook, run

    ```bash
    pre-commit install
    ```

## Usage

To run current code, go to src and run

```bash
python3 -m mas_consensus.run
```
