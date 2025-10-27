#!/bin/bash

# This script reads experiments from experiment.psv and executes them.

# Check if experiment.psv exists
if [ ! -f experiment.psv ]; then
    echo "experiment.psv not found!"
    exit 1
fi

# Read and execute experiments, skipping the header line
tail -n +2 experiment.psv | while IFS='|' read -r experiments datasets graphs num_agents attack_id malicious_auditor_num num_auditors reg_turn sample_id threads model log_dir; do
    # Skip empty lines that might exist at the end of the file
    if [ -z "$experiments" ]; then
        continue
    fi

    # Handle 'default' values
    reg_turn_val=$([ "$reg_turn" == "default" ] && echo 9 || echo "$reg_turn")
    sample_id_val=$([ "$sample_id" == "default" ] && echo 3 || echo "$sample_id")
    threads_val=$([ "$threads" == "default" ] && echo 16 || echo "$threads")
    model_val=$([ -z "$model" ] && echo "gpt-4o-mini" || echo "$model")

    # Construct the command
    cmd="python -m src.mas_consensus.run_defense_comparison \
        --dataset $datasets \
        --graph_type $graphs \
        --num_agents $num_agents \
        --attacker_num $attack_id \
        --malicious_auditor_num $malicious_auditor_num \
        --num_auditors $num_auditors \
        --reg_turn $reg_turn_val \
        --sample_id $sample_id_val \
        --threads $threads_val \
        --model $model_val"

    echo "======================================================================"
    echo "Executing experiment:"
    echo "$cmd"
    echo "======================================================================"

    # Run the command
    eval $cmd
done

echo "All experiments have been started."
