# Plan

## Overview

When multi-agent systems collaborate to solve problems (because the inputs and outputs of different agents have causal relationships), if there exists a malicious agent or an agent that makes a mistake at some step and propagates incorrect outputs, the accumulation of those errors can very likely cause system failure.

Solution: stepwise random auditing and voting. Specifically:

1. Each agent possesses a public/private key pair and signs its messages with digital signatures.
2. Each agent randomly audits a certain number of other agents' work results, and—using the encrypted inputs and outputs—rationally judges whether the audited agent's result is reasonable (output is T/F). (Public/private key encryption ensures message integrity.) (Add audit steps at some critical nodes, first audit with a small number of agents; if anomalies are found then escalate to a full vote. Use KV Cache techniques during audits to improve audit efficiency.)
3. If an unreasonable result is found, request a system-wide vote. Share the audited agent's input/output information with other agents; use voting to eliminate malicious agents, then reassign the malicious agents' work to honest agents to complete the task. Performing small-scale audits first and only escalating to collective voting when issues are detected saves resources and improves efficiency.

## Abnormal (malicious) agent behaviors

- Type 1: Malicious attack: causes system task failure (or probabilistic errors can be grouped with malicious attacks—examples should be given).

  - Attack method: according to context, the malicious agent inputs or propagates seemingly reasonable instructions that mislead other agents.

  For example, under normal circumstances, given the context text, agents 1, 2, 3's answers should be:

  > 1：Robert Aldrich...Vera
  > Cruz, Kiss Me Deadly

  > 2：Mary Astor...
  > a cameo role as

  > 3： Mary Astor

  If agent 1 is maliciously attacked, it will find content similar to zhong1 in the text to mislead; through chained transmission, the attack goal is ultimately achieved. Example:

  > 1: Vera Cruz, "Hush...Hush, Sweet Charlotte, Robert Aldrich, a cameo role as, Kiss Me Deadly
  > 2: Robert Aldrich, Kiss Me Deadly
  > 3: Robert Aldrich

  Attack idea: use content within the materials to mislead subtly, keeping some stealth while propagating through chains to achieve the attack effect.

- Type 2: Malicious auditing: when acting as an auditing agent, deliberately invert audit results, causing the system to accept malicious consensus or to perform meaningless votes on correct results.

- Type 3: Malicious voting: during required network-wide consensus, vote maliciously to mislead the system's judgment of results, causing final task failure or degrading system efficiency.

### Experimental requirements

Run separate experiments: only Type 1 malicious agents exist; only Type 2 malicious agents exist; only Type 3 malicious agents exist.

## Implementation details

Based on the current code under `src`, run experiments on agent collaboration architectures under different topologies (chain, mesh, decentralized, hierarchical, etc.) — the framework already exists.

### Modifications

1. Add prompts to the original framework and change some task-executing agents into malicious agents.
2. Add an audit expert group for auditing. (Alternatively, other agents could perform audits—this is slightly more complex.)
3. Separate experiments: run experiments for each of the three malicious-agent types independently (only Type 1 exists; only Type 2 exists; only Type 3 exists).

#### Specific audit process:

1. Design a dedicated audit expert group composed of several agents.
2. After each subtask is completed, randomly select several agents; those agents provide the audited group's members with all inputs and outputs from the task flow.
3. The auditors examine whether the subtask inputs and outputs are reasonable and give a T/F judgement.
4. If any unreasonable results are found, request a system-wide vote: pass the audited agent's inputs and outputs to other agents and use voting to eliminate malicious agents.
5. If an audited agent is found malicious, reassign that agent's work to honest agents and complete the task. Conduct small-scale audits first and only call a collective vote if problems are found—this conserves resources and improves efficiency. (You may avoid changing system topology; simply replace the malicious agent's prompt with a normal prompt so the original agent answers again correctly.)

#### Experimental schedule (start with chain-structured agent collaboration):

(Replace the malicious agent with a safe agent after auditing: i.e., replace the original agent's attacking prompt with a normal prompt.)

1. (6 agents, 1 faulty) Accuracy of each architecture without malicious agents.
2. Add malicious agents and measure each architecture's accuracy.
3. With malicious agents present, add the defense mechanism and measure each architecture's accuracy. (Table 1) (Plot: baseline accuracy, accuracy drop after adding malicious agents, and recovery after adding the defense mechanism.)
4. (6 agents, 1 faulty) Accuracy of intermediate agents' answers before and after adding defense mechanisms. (Figure 1 — 5 subplots) (Plot lines: attacker accuracy, honest agents' accuracy, and both after adding the defense.) Run for 8 rounds.
5. For 10 agents with varying numbers of faulty agents, measure average accuracy with and without defense (0 faulty, 1 faulty, 2 faulty), across five topologies and three datasets (Table 2).
6. Use different models: GPT-3.5, GPT-4o, deepseek-v3, Qianwen (千问) (Table 3).
   Core method: achieve safety and answer-quality assurance in distributed multi-agent systems using random audits and consensus voting.
7. Compare task completion time for the original system and after adding audits. (Compare malicious vs. normal states.) Single-case measurement is sufficient.
8. Comparison of audit cost (time) across different topologies.
9. Impact of malicious voting and malicious auditing agents on the system.
10. Efficiency (time) under malicious auditing and malicious voting.
    Architecture: centralized system that performs unified random scheduling and assignment, integrates agents' work, then agents perform distributed subtasking.
