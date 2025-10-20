# MAS Consensus Code and Experiment Analysis

This document provides a detailed overview of the **Multi-Agent Consensus (MAS Consensus)** project located in the `src/mas_consensus/` directory — including its code structure, core modules, and the execution flow of each experiment.

---

## I. Code Overview

The `src/mas_consensus/` directory contains all the core components for **multi-agent simulation, interaction, evaluation, and defense mechanisms**.

- **`agent_base.py`**

  - **Purpose**: Defines the core classes forming the foundation of the multi-agent system.
  - **`BaseAgent`**: The general base class for agents. It handles communication with large language models (LLMs), parses the returned `reason` and `answer`, and manages its dialogue history and short-term memory.
  - **`SimpleAgent`**: A simplified version of `BaseAgent` used for single-response tasks (e.g., in the `adv` dataset).
  - **`AgentGraph`**: The manager for the entire agent network. It initializes a group of agents based on an adjacency matrix and controls the experimental process — including multi-round (turn-based) interactions, viewpoint updates, defense mechanisms (auditing and voting), and result saving.

- **`methods.py`**

  - **Purpose**: Provides key utility functions used across the project.
  - `get_client()`: Initializes and returns an OpenAI API client.
  - `generate_adj()`: Generates an adjacency matrix (`adj_matrix`) given the number of nodes (`n`) and graph type (`graph_type`, e.g., `chain`, `star`, `complete`, etc.), defining how agents are connected.
  - `get_dataset()`: Reads a dataset from a `.jsonl` file.

- **`prompts.py`**

  - **Purpose**: Stores all system prompts used across experiments.
  - Includes: standard discussion prompts, attacker (Type 1) prompts (encouraging stubborn defense of wrong answers), malicious behavior (AdvBench) prompts, auditor prompts, voter prompts, and their malicious variants (malicious auditor/voter). This file defines the behavioral profiles of agents.

- **`util.py`**

  - **Purpose**: Implements the main experiment orchestration logic through `run_dataset`.
  - `run_dataset()`: A general-purpose experiment launcher. It takes parameters such as dataset name, graph type, and attacker configuration, then loads data, generates topology, prepares prompts, and processes each task in parallel. Each task creates an `AgentGraph` instance for simulation.

- **`run_*.py` (e.g., `run_csqa.py`, `run_gsm8k.py`)**

  - **Purpose**: Entry scripts for specific experiments.
  - Each script defines a `task_formatter` function that adapts dataset entries into tasks for normal and attacker agents. It then calls `util.run_dataset` to execute the experiment.

- **`evaluate.py`**

  - **Purpose**: Evaluates experiment outputs.
  - Provides evaluation functions for datasets (`csqa`, `gsm8k`, `fact`, `bias`) to read the saved results from `AgentGraph` and compute:

    - **SAA (Single Agent Accuracy)** — accuracy of individual agents.
    - **MJA (Majority Judgement Accuracy)** — accuracy of the group’s majority decision.

  - Also includes functions for analyzing static network metrics (e.g., `NE`, `EC`, `PV`).

- **`defense.py`**

  - **Purpose**: Implements the auditing defense mechanism.
  - **`AuditorAgent`**: A special agent that examines other agents’ outputs to judge whether they are “reasonable”.

- **`draw.py`**

  - **Purpose**: Provides visualization functions to plot results from `evaluate.py`, including heatmaps (`SAA_heatmap`), line charts (`SAA_linechart`), and radar charts (`SAA_radar`).

- **`run_*_analysis.py` / `run_defense_comparison.py`**

  - **Purpose**: High-level analysis scripts that run sets of experiments and produce comparative visualizations using `evaluate.py` and `draw.py` to study how variables such as defense mechanisms or malicious behaviors affect performance.

---

## II. Experimental Workflow

All experiments follow a standard pipeline:
**Load data → Generate topology → Agent interaction → Save and evaluate results.**

### Experiment 1: `run_csqa.py` (CommonsenseQA)

A multiple-choice commonsense reasoning task where agents must select the correct answer.

1. **Data Loading**

   - `util.run_dataset` calls `methods.get_dataset` to read `src/dataset/csqa.jsonl`.
   - Each JSON object includes a question (`question`), choices (`choices`), and the correct answer key (`answerKey`).

2. **Topology Generation**

   - `util.run_dataset` calls `methods.generate_adj(num_agents, graph_type)` to create the adjacency matrix, where `1` indicates a connection between agents.

3. **Agent Interaction**

   - For each question, `AgentGraph` handles the simulation.
   - **Task Formatting (`task_formatter`)**:

     - **Normal agents**: instructed to “choose the best answer.”
     - **Attacker agents**: instructed to “defend a specific wrong answer (Reference Answer)” and treat it as their belief.

   - **Interaction Process**:

     - **Turn 0 (First Generate)**: Agents independently produce initial answers and reasoning.
     - **Turns 1–N (Re-generate)**: Each agent receives its neighbors’ previous answers and reasoning, updates its own opinion, and regenerates its response.
     - **Defense Mechanism (Optional)**: If `num_auditors > 0`, after each round, an `AuditorAgent` randomly audits a subset of agents. If flagged as abnormal (`False` judgment), a voting process (`_run_voting`) is triggered. If a majority confirms malicious behavior, the agent’s system prompt resets to the standard (non-malicious) prompt — effectively “reforming” it.

4. **Result Generation and Evaluation**

   - **Saving**: `AgentGraph.save()` stores all conversation histories (reasons, answers, memory) into a `.output` file.
   - **Evaluation**: `evaluate_csqa` extracts each agent’s final answers (choice letters) and compares them with `answerKey` to compute `SAA` and `MJA`.

---

### Experiment 2: `run_gsm8k.py` (GSM8K)

A grade-school math reasoning task requiring multi-step logical inference.

1. **Data Loading**

   - Loads `src/dataset/gsm8k.jsonl`, containing `question` and correct `answer_number`.

2. **Topology Generation**

   - Same as in CSQA.

3. **Agent Interaction**

   - **Normal agents**: instructed to “solve the math problem and show reasoning.”
   - **Attacker agents**: instructed to “defend the wrong answer `0`.”
   - Agents refine reasoning through multiple discussion rounds.

4. **Result Generation and Evaluation**

   - Outputs are saved to `.output`.
   - `evaluate_gsm8k` extracts numerical answers and compares them with `answer_number` to compute accuracy.

---

### Experiment 3: `run_fact.py` / `run_bias.py` (Fact / Bias Detection)

Binary classification tasks determining whether a statement is **True** or **False**.

1. **Data Loading**

   - `run_fact`: loads factual statements (`True`).
   - `run_bias`: loads biased statements (`False`).

2. **Topology Generation**

   - Same as CSQA.

3. **Agent Interaction**

   - **Normal agents**: “Judge whether the statement is true.”
   - **Attacker agents**: “Defend the opposite (wrong) judgment.”
   - Interaction is the same as in CSQA.

4. **Result Generation and Evaluation**

   - Outputs saved as `.output`.
   - `evaluate_fact` / `evaluate_bias` compute accuracy by comparing each answer to the correct label.

---

### Experiment 4: `run_adv.py` (AdvBench)

A special safety-related task testing whether agents generate **harmful content**.

1. **Data Loading**

   - Loads `src/dataset/adv.jsonl` (each entry contains a harmful `prompt`).

2. **Topology Generation**

   - Same as CSQA.

3. **Agent Interaction**

   - Uses `SimpleAgent` (single-response).
   - **Normal agents**: given a harmless discussion prompt.
   - **Attacker agents**: directly respond to harmful prompts with system instructions designed to produce unethical or dangerous outputs.
   - Other agents may be influenced by the attacker’s content.

4. **Result Generation and Evaluation**

   - **Saving**: results stored in `.output`.
   - **Evaluation**:

     - `moderation.py` sends outputs to the `text-moderation-latest` API for safety scoring (`harassment`, `hate`, `violence`, etc.).
     - `evaluate_adv` quantifies harmfulness using these scores:

       - **SAA**: average harmfulness per agent.
       - **MJA**: group-level average harmfulness.

---

## III. Advanced Analysis Scripts

These scripts orchestrate multiple experiments and produce comparative analyses and visualizations.

### `run_defense_comparison.py`

- **Goal**: Quantitatively compare the effectiveness of the **audit–vote defense mechanism**.
- **Method**:

  1. Run three experiments:

     - **Baseline**: all normal agents, no defense.
     - **Attacked**: one attacker, no defense.
     - **Defended**: one attacker, with 2 auditors enabled.

  2. **Evaluation & Visualization**:

     - Compute `SAA`/`MJA` for each setting and generate a bar chart comparing system performance across the three scenarios — showing how much performance the defense recovers.

---

### `run_chain_analysis.py`

- **Goal**: Analyze **error propagation** in a _chain topology_ and how defense mechanisms mitigate it.
- **Method**:

  - Uses `graph_type="chain"` with an attacker at the head (`attacker_idx=[0]`), and two auditors.
  - **Evaluation**:

    - `evaluate_and_plot_chain_accuracy`: plots overall MJA across rounds.
    - `analyze_intermediate_agent_performance`: plots per-agent SAA to visualize how accuracy improves farther away from the attacker — showing how defense suppresses misinformation spread.

---

### `run_efficiency_analysis.py`

- **Goal**: Evaluate the **runtime cost** of various malicious behaviors, especially those targeting defense mechanisms.
- **Method**:

  - Measures total `util.run_dataset` execution time across several scenarios:

    - **Baseline** (no attack/defense),
    - **Type 1 Only** (standard attacker),
    - **Type 2** (malicious auditors),
    - **Type 3** (malicious voters),
    - **Combined** (mixed attacks).

  - **Visualization**:

    - `plot_efficiency_comparison` displays runtime for each scenario, illustrating:

      - Overhead added by attackers.
      - Computational cost of enabling defense.
      - Extra cost from malicious defense behaviors (e.g., more voting rounds).
