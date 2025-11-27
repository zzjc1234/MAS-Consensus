# MAS Consensus

A multi-agent system for consensus-based text processing, implementing the "NetSafe" approach for analyzing how reviewer avoid malicious nodes during the tasks with large language models.

Based on the paper: [Netsafe](https://anonymous.4open.science/r/NetSafe-B726/README.md)

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

See `docs/experiment_guide.md` for experiment runner details.

### Experiments and evaluation

- Run experiments from the repo root with `./run_experiments.sh`; it reads scenarios from `experiment.psv` and stores outputs under `src/output/`.
- Evaluate aggregated results after an experiment run with:

```bash
python -m src.mas_consensus.evaluate --evaluation dynamic_MJA --model gemini-2.5-flash --graph_types chain circle complete star tree --agent_num 6 --attacker_num 0 --sample_ids 3 --aggregate_by_topology --datasets csqa
```

- To evaluate files individually instead, use `./evaluate.sh`.
