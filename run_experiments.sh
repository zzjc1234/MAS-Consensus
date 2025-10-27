#!/bin/bash

# This script reads experiment parameters from experiment.psv and executes them.
#
# NODE ARCHITECTURE:
#   - Workers: Answer questions and update responses (don't vote)
#   - Auditors: Audit and vote (don't answer questions)
#
# ATTACK TYPES:
#   - Type 1: Malicious workers (give wrong answers)
#   - Type 2: Malicious auditors (audit and vote maliciously)
#
# Each line can trigger up to 4 scenarios controlled by flags:
#   - run_baseline:  Workers only, all honest, no auditors
#   - type_one_attack:  Malicious workers + honest auditors (Type 1 attack with defense)
#   - type_two_attack:  Honest workers + malicious auditors (Type 2 attack)
#   - both_attacks:      Malicious workers + malicious auditors (both attacks)
#
# See EXPERIMENT_GUIDE.md for documentation.

# Check if experiment.psv exists
if [ ! -f experiment.psv ]; then
    echo "experiment.psv not found!"
    exit 1
fi

# Read and execute experiments, skipping the header line
tail -n +2 experiment.psv | while IFS='|' read -r datasets graphs num_agents attacker_num malicious_auditor_num num_auditors run_baseline type_one_attack type_two_attack both_attacks reg_turn sample_id threads model; do
    # Skip empty lines
    if [ -z "$datasets" ]; then
        continue
    fi

    # Handle 'default' values
    reg_turn_val=$([ "$reg_turn" == "default" ] && echo 9 || echo "$reg_turn")
    sample_id_val=$([ "$sample_id" == "default" ] && echo 3 || echo "$sample_id")
    threads_val=$([ "$threads" == "default" ] && echo 16 || echo "$threads")
    model_val=$([ -z "$model" ] || [ "$model" == "default" ] && echo "gpt-4o-mini" || echo "$model")
    num_auditors_val=$([ "$num_auditors" == "default" ] && echo 0 || echo "$num_auditors")
    malicious_auditor_num_val=$([ "$malicious_auditor_num" == "default" ] && echo 0 || echo "$malicious_auditor_num")
    attacker_num_val=$([ "$attacker_num" == "default" ] && echo 0 || echo "$attacker_num")

    # Expand dataset list
    if [ "$datasets" == "all" ]; then
        dataset_list="csqa gsm8k fact bias adv"
    else
        dataset_list="$datasets"
    fi

    # Expand graph list
    if [ "$graphs" == "all" ]; then
        graph_list="chain circle tree star complete"
    else
        graph_list="$graphs"
    fi

    # Execute for each dataset and graph combination
    for dataset in $dataset_list; do
        for graph in $graph_list; do
            
            # BASELINE: Honest agents only, no attacks, no auditors
            if [ "$run_baseline" == "1" ]; then
                echo "======================================================================"
                echo "BASELINE: dataset=$dataset, graph=$graph"
                echo "  Agents: $num_agents (all honest)"
                echo "  Auditors: 0"
                echo "======================================================================"
                
                python -m src.mas_consensus.run_experiment \
                    --dataset "$dataset" \
                    --graph_type "$graph" \
                    --num_agents "$num_agents" \
                    --attacker_num 0 \
                    --malicious_auditor_num 0 \
                    --num_auditors 0 \
                    --reg_turn "$reg_turn_val" \
                    --sample_id "$sample_id_val" \
                    --threads "$threads_val" \
                    --model "$model_val"
            fi
            
            # TYPE 1 ATTACK: Agents (some malicious) + auditors (all honest)
            # Type 1 attack: malicious agents give wrong answers
            if [ "$type_one_attack" == "1" ]; then
                echo "======================================================================"
                echo "TYPE 1 ATTACK: dataset=$dataset, graph=$graph"
                echo "  Agents: $num_agents ($attacker_num_val malicious - Type 1 attack)"
                echo "  Auditors: $num_auditors_val (all honest - defending)"
                echo "======================================================================"
                
                python -m src.mas_consensus.run_experiment \
                    --dataset "$dataset" \
                    --graph_type "$graph" \
                    --num_agents "$num_agents" \
                    --attacker_num "$attacker_num_val" \
                    --malicious_auditor_num 0 \
                    --num_auditors "$num_auditors_val" \
                    --reg_turn "$reg_turn_val" \
                    --sample_id "$sample_id_val" \
                    --threads "$threads_val" \
                    --model "$model_val" \
                    --output_suffix "_type1"
            fi
            
            # TYPE 2 ONLY: Honest agents + malicious auditors
            # Type 2 attack only: malicious auditors (audit + vote maliciously)
            if [ "$type_two_attack" == "1" ]; then
                echo "======================================================================"
                echo "TYPE 2 ATTACK: dataset=$dataset, graph=$graph"
                echo "  Agents: $num_agents (all honest)"
                echo "  Auditors: $num_auditors_val ($malicious_auditor_num_val malicious - Type 2 attack)"
                echo "======================================================================"
                
                python -m src.mas_consensus.run_experiment \
                    --dataset "$dataset" \
                    --graph_type "$graph" \
                    --num_agents "$num_agents" \
                    --attacker_num 0 \
                    --malicious_auditor_num "$malicious_auditor_num_val" \
                    --num_auditors "$num_auditors_val" \
                    --reg_turn "$reg_turn_val" \
                    --sample_id "$sample_id_val" \
                    --threads "$threads_val" \
                    --model "$model_val" \
                    --output_suffix "_type2"
            fi
            
            # BOTH ATTACKS: Malicious agents + malicious auditors
            # Type 1 attack: malicious agents
            # Type 2 attack: malicious auditors
            if [ "$both_attacks" == "1" ]; then
                echo "======================================================================"
                echo "BOTH ATTACKS: dataset=$dataset, graph=$graph"
                echo "  Agents: $num_agents ($attacker_num_val malicious - Type 1 attack)"
                echo "  Auditors: $num_auditors_val ($malicious_auditor_num_val malicious - Type 2 attack)"
                echo "======================================================================"
                
                python -m src.mas_consensus.run_experiment \
                    --dataset "$dataset" \
                    --graph_type "$graph" \
                    --num_agents "$num_agents" \
                    --attacker_num "$attacker_num_val" \
                    --malicious_auditor_num "$malicious_auditor_num_val" \
                    --num_auditors "$num_auditors_val" \
                    --reg_turn "$reg_turn_val" \
                    --sample_id "$sample_id_val" \
                    --threads "$threads_val" \
                    --model "$model_val" \
                    --output_suffix "_both"
            fi
            
        done
    done
done

echo "All experiments have been completed."
