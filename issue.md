# Project Tasks: MAS Consensus with Auditing

This document breaks down the project plan into actionable development tasks. The existing codebase has already implemented the multi-agent simulation framework, malicious agent behaviors (Type 1), and the experimental setup for running simulations and evaluating results.

The main remaining work is to implement the proposed defense mechanism: **Auditing and Voting**.

## Epic 1: Core Defense Mechanism Implementation (Completed)

- [x] **Auditing Mechanism:**
  - [x] Create a dedicated audit expert group or enable agents to audit each other.
  - [x] After a task step, randomly select agents to perform an audit.
  - [x] Implement the logic for auditors to receive the inputs/outputs of the audited agent and provide a T/F judgment on the result's reasonableness.
- [x] **System-wide Voting:**
  - [x] Implement a trigger for a system-wide vote when an audit fails (returns 'F').
  - [x] Implement the mechanism to share the audited agent's data with all voting agents.
  - [x] Implement the voting logic to identify and confirm a malicious agent.
- [x] **Task Reassignment / Correction:**
  - [x] Strategy 1: Replace the malicious agent's prompt with a normal one and have it retry the task.

## Epic 2: Malicious Agent Simulation (Completed)

- [x] **Type 1 (Malicious Attack):** Implemented via attacker-specific prompts.
- [x] **Type 2 (Malicious Auditing):**
  - [x] Create prompts/logic for auditing agents to deliberately invert audit results (e.g., approve bad results, reject good results).
  - [x] Integrate into experiments to test the defense mechanism's resilience.
- [x] **Type 3 (Malicious Voting):**
  - [x] Create logic for agents to vote maliciously during a system-wide consensus.
  - [x] Integrate into experiments to test the voting mechanism's resilience.

## Epic 3: Experimental Setup & Execution

- [x] **Baseline & Control Experiments:** Framework is in place.
- [x] **Defense Mechanism Evaluation:**
  - [x] **Task 3:** With 1 Type 1 malicious agent present, enable the **new** audit/vote defense mechanism and measure the accuracy recovery. Create a plot for Baseline vs. Attacked vs. Defended accuracy.
  - [x] **Task 4:** For a chain of 6 agents (1 faulty), plot the accuracy of intermediate agents' answers after adding the new defense mechanism.
- [x] **Scalability & Robustness Testing:** Framework is in place.
- [x] **Model & Performance Analysis:** Framework is in place.
- [x] **Advanced Malicious Behavior Analysis:**
  - [x] **Task 9:** Design and run experiments to measure the specific impact of **new** Type 2 (malicious auditing) and Type 3 (malicious voting) agents on system accuracy.
  - [x] **Task 10:** Measure system efficiency (time) when malicious auditing and voting agents are present.

## Epic 4: Infrastructure & General Tasks (Completed)

- [x] **Topology Framework:** `methods.generate_adj` creates various topologies.
- [x] **Data Logging:** `AgentGraph.save` writes dialogue history to output files.
- [x] **Visualization:** `draw.py` contains plotting functions for analysis.
- [x] **Dataset Handling:** `generate_dataset.py` and `methods.get_dataset` handle dataset preparation.
- [x] **Evaluation Metrics:** `evaluate.py` calculates SAA, MJA, and other metrics.
