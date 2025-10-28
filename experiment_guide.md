# Experiment Runner Overview

This document explains how to configure and launch the batch runner that reads `experiment.psv`, generates multi-agent consensus simulations, and collects their outputs.

## 1. Input: `experiment.psv`

The runner expects a pipe-separated file with a header row. Each non-empty line expands into the Cartesian product of its datasets and graph topologies, and can enable up to four scenarios per combination.

Ensure the referenced datasets exist at `src/dataset/<name>.jsonl`.

| Column | Description |
| --- | --- |
| `datasets` | A space-separated list of dataset aliases (`csqa`, `gsm8k`, `fact`, `bias`, `adv`) or `all` to include every dataset. |
| `graphs` | Space-separated graph topologies (`chain`, `circle`, `tree`, `star`, `complete`) or `all`. |
| `num_agents` | Total number of worker agents in each run. |
| `attacker_num` | How many workers act maliciously. Use `default` to fall back to `0`. |
| `malicious_auditor_num` | Number of malicious auditors (`default` → `0`). |
| `num_auditors` | Total auditors selected from the worker pool (`default` → `0`). |
| `run_baseline`, `type_one_attack`, `type_two_attack`, `both_attacks` | Scenario toggles (`0`/`1`). Multiple flags can be enabled on the same line. |
| `reg_turn`, `sample_id`, `threads`, `model` | Optional overrides. `default` maps to `reg_turn=9`, `sample_id=3`, `threads=16`, `model=gpt-4o-mini`. |

Minimal example:

```text
datasets|graphs|num_agents|attacker_num|malicious_auditor_num|num_auditors|run_baseline|type_one_attack|type_two_attack|both_attacks|reg_turn|sample_id|threads|model
csqa|chain circle|6|default|default|2|1|1|0|0|default|default|default|default
```

## 2. Launching the batch script

From the repository root, run:

```bash
bash run_experiments.sh
```

The script iterates through each line of `experiment.psv` (skipping the header) and dispatches `python -m src.mas_consensus.run_experiment` for every dataset/graph/scenario combination.

## 3. What happens during a run

1. The shell script creates a unique folder per line, dataset, and graph under the chosen log root.
2. The Python entrypoint expands attack settings, selects auditors, and spins up a multi-threaded agent graph where workers answer questions and auditors audit/vote.
3. Each dataset item is processed in parallel batches (`--threads` controls the batch size). Workers respond, auditors audit, votes may reclassify agents, and the full dialogue record is saved.

## 4. Outputs

- **Aggregate transcripts**: `src/output/<model>/<dataset>/<sample_id>/<dataset>_<graph>_<num_agents>_<attackers>[<suffix>].output`
  - Contains the serialized dialogue records for all questions processed under that run.
- **Program logs**: `<log_root>/line_<N>/<dataset>/<graph>/<graph>_<agents>_<attackers>[<suffix>]/program.log`
  - Captures structured INFO-level logging for the scenario, including audit/vote events, question boundaries, and completion notices.
- **Per-question responses**: Same folder as the log file, inside `responses/`.
  - One file per processed dataset entry (JSON when `--json_format` is set), useful for inspecting individual conversations or feeding downstream evaluators.

The console mirrors the same log stream that is written to `program.log`.

## 5. Running a single experiment manually

You can bypass the batch script by invoking the entrypoint directly:

```bash
python -m src.mas_consensus.run_experiment \
  --dataset csqa \
  --graph_type chain \
  --num_agents 6 \
  --threads 4 \
  --sample_id 3 \
  --model gpt-4o-mini \
  --log_dir ./logs/debug/csqa_chain \
  --scenario baseline
```

Every CLI flag matches the columns in `experiment.psv`. The `--log_dir` parameter controls where `program.log` and the `responses/` files are written.
