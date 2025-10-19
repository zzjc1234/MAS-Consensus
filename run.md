# Running Experiments: MAS Consensus with Auditing Defense

This document provides detailed instructions on how to run the various experiments related to the auditing and voting defense mechanisms in the MAS Consensus system.

## Prerequisites

Before running the experiments, ensure you have:

1. The required Python packages installed (see `pyproject.toml`)
2. Access to OpenAI API (or other LLM provider) with proper API keys configured
3. Dataset files in the `./dataset/` directory
4. Proper environment setup (Python 3.10+)

## Available Experiment Scripts

### 1. Defense Mechanism Comparison (`run_defense_comparison.py`)

This script compares three scenarios:
- **Baseline**: No malicious agents
- **Attacked**: 1 Type 1 malicious agent without defense
- **Defended**: 1 Type 1 malicious agent with audit/vote defense mechanism

#### Running the Defense Comparison Experiment:

```bash
cd src/mas_consensus
python run_defense_comparison.py
```

#### What this experiment does:
- Runs three scenarios with the same parameters
- Evaluates each scenario using SAA (Single Agent Accuracy) metrics
- Creates a comparative plot showing accuracy recovery
- Saves results to the output directory

#### Expected output:
- Output files in `./output/{model}/{dataset}/{sample_id}/`
- Console output showing accuracy for each scenario
- Plot comparing Baseline vs. Attacked vs. Defended performance

### 2. Chain Topology Analysis (`run_chain_analysis.py`)

This experiment analyzes the performance of individual agents in a chain topology with 1 faulty agent and the defense mechanism enabled.

#### Running the Chain Analysis Experiment:

```bash
cd src/mas_consensus
python run_chain_analysis.py
```

#### What this experiment does:
- Sets up a chain topology with 6 agents
- Introduces 1 malicious agent
- Enables the audit/vote defense mechanism
- Evaluates both MJA (Multi-agent Joint Accuracy) and SAA (Single Agent Accuracy)
- Creates visualization plots for both metrics

#### Expected output:
- Execution time comparison
- Performance plots for each agent
- Accuracy metrics for intermediate agents

### 3. Malicious Behavior Impact Analysis (`run_malicious_behavior_analysis.py`)

This experiment measures the specific impact of Type 2 (malicious auditing) and Type 3 (malicious voting) agents on system accuracy.

#### Running the Malicious Behavior Analysis:

```bash
cd src/mas_consensus
python run_malicious_behavior_analysis.py
```

#### What this experiment does:
- Runs four scenarios: Type 1 only, Type 2 only, Type 3 only, and combined
- Evaluates each scenario for system accuracy impact
- Creates comparative visualization of malicious behavior impacts
- Provides analysis of different attack vectors

#### Expected output:
- Output files for each malicious behavior type
- Comparative accuracy plots
- Impact analysis summary

### 4. System Efficiency Analysis (`run_efficiency_analysis.py`)

This experiment measures system efficiency (execution time) when malicious auditing and voting agents are present.

#### Running the Efficiency Analysis:

```bash
cd src/mas_consensus
python run_efficiency_analysis.py
```

#### What this experiment does:
- Measures execution time for each malicious behavior scenario
- Compares efficiency ratios against baseline
- Analyzes performance degradation due to malicious agents
- Creates efficiency comparison visualizations

#### Expected output:
- Execution time measurements for each scenario
- Efficiency ratios compared to baseline
- Performance degradation analysis

## Parameters and Configuration

Most experiments have configurable parameters that can be modified in the script:

- `dataset`: Dataset to use (default: "csqa")
- `sample_id`: Sample ID for the experiment (default: 3)
- `graph_type`: Topology type (options: "chain", "circle", "tree", "star", "complete")
- `model`: LLM model to use (default: "gpt-4o-mini")
- `p`: Number of parallel processes (default: 16)
- `num_agents`: Number of agents (default: 6)
- `turn`: Number of discussion turns (default: 9)

## Running All Experiments Sequentially

To run all experiments in sequence:

```bash
cd src/mas_consensus
echo "Running Defense Comparison Experiment..."
python run_defense_comparison.py

echo "Running Chain Analysis Experiment..."
python run_chain_analysis.py

echo "Running Malicious Behavior Analysis..."
python run_malicious_behavior_analysis.py

echo "Running Efficiency Analysis..."
python run_efficiency_analysis.py
```

## Running Tests

To verify that all functionality works correctly, run the test suite:

```bash
cd /path/to/project
python -m pytest tests/ -v
```

## Custom Experiment Setup

To create custom experiments, you can modify the parameters in any of the run scripts or create new scripts that follow the same pattern, using the utility functions in `util.py`.

## Expected Results

After running the experiments:

1. Output files will be stored in `./output/{model}/{dataset}/{sample_id}/`
2. All experiments will generate plots showing the results
3. Console logs will provide numerical results
4. The defense mechanisms should show accuracy recovery compared to the attacked scenario

## Troubleshooting

- If API rate limits are reached, try reducing the number of parallel processes (`p` parameter)
- Ensure your API keys are properly configured
- If experiments take too long, reduce `num_agents` or `turn` parameters
- Check that dataset files exist in the `./dataset/` directory
