# How to Run Experiments

This document provides instructions on how to run the various experiments for the Multi-Agent System Consensus project.

## Prerequisites

1.  **Install Dependencies**: Ensure all required Python packages are installed.
    ```bash
    pip install -r requirements.txt # Or based on pyproject.toml/uv.lock
    ```
2.  **API Key**: Make sure your OpenAI API key is configured as an environment variable.
    ```bash
    export OPENAI_API_KEY="your_api_key_here"
    ```
3.  **Datasets**: The required datasets should be present in the `src/dataset/` directory.

---

## 1. Running Basic, Single-Dataset Experiments

These scripts are designed to run experiments for a specific dataset. You can configure parameters like the number of attackers directly within the script.

To run an experiment, execute the script as a module from the project's root directory.

**Example (running the CSQA experiment):**
```bash
python -m src.mas_consensus.run_csqa
```

**Available Scripts:**
- `run_csqa.py`
- `run_gsm8k.py`
- `run_fact.py`
- `run_bias.py`
- `run_adv.py`

### Configuration

Open the script you want to run (e.g., `src/mas_consensus/run_csqa.py`) and modify the variables in the `if __name__ == "__main__":` block:
- `attacker_nums`: A list of attacker counts to iterate through (e.g., `[0, 1, 2]`).
- `graph_type`: Network topology (e.g., `"complete"`, `"star"`).
- `num_agents`: Total number of agents.

---

## 2. Running Advanced Analysis Experiments

These scripts are designed for comparative analysis and are now highly flexible. They accept command-line arguments to specify the dataset and number of attackers.

**Key Arguments:**
- `--dataset`: The name of the dataset to run the analysis on.
  - **Options**: `csqa`, `gsm8k`, `fact`, `bias`, `adv`.
- `--attacker_num`: The number of malicious agents to include in the scenario.

### Example: Defense Comparison

This compares a baseline (0 attackers) vs. an attacked scenario with `N` attackers vs. a defended scenario with `N` attackers.

```bash
# Run defense comparison on the GSM8K dataset with 2 attackers
python -m src.mas_consensus.run_defense_comparison --dataset gsm8k --attacker_num 2
```

### Example: Chain Analysis

This analyzes a chain topology with `N` attackers at the head of the chain.

```bash
# Run chain analysis on the FACT dataset with 1 attacker
python -m src.mas_consensus.run_chain_analysis --dataset fact --attacker_num 1
```

### Other Analysis Scripts

The `run_efficiency_analysis.py` and `run_malicious_behavior_analysis.py` scripts also accept the `--dataset` argument. However, they are designed to run a fixed set of internal scenarios with specific attacker counts and types, so the `--attacker_num` argument is not applicable to them.

```bash
# Run efficiency analysis on the BIAS dataset
python -m src.mas_consensus.run_efficiency_analysis --dataset bias
```

---

## 3. Running Batch Experiments with `run.py`

The `run.py` script is a dispatcher for running a large batch of experiments across multiple configurations.

### How to Use:

1.  **Open `src/mas_consensus/run.py`**.
2.  **Edit the configuration lists** at the top of the `if __name__ == "__main__":` block:
    - `datasets`: A list of dataset names to run (e.g., `["csqa", "gsm8k"]`).
    - `graph_types`: A list of network topologies.
    - `num_agents_list`: A list of total agent counts.
    - `attacker_nums`: A list of attacker counts.
3.  **Run the script** from the root directory:
    ```bash
    python -m src.mas_consensus.run
    ```

The script will iterate through every combination of the parameters you specified.

---

## Troubleshooting

- **API Rate Limits**: If you encounter errors, try reducing the number of parallel processes (`p` parameter) in the script you are running.
- **Long Runtimes**: For quicker tests, reduce `num_agents` or the number of discussion rounds (`reg_turn`).
