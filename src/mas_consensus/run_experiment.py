"""
Unified experiment runner - a thin wrapper around util.run_dataset()

Node Architecture:
  - Agents: Respond to questions and update their answers (don't vote)
  - Auditors: Audit agents and vote (don't answer questions)

Attack Types:
  - Type 1: Malicious agents (give wrong answers)
  - Type 2: Malicious auditors (audit and vote maliciously)

All experiment logic is controlled through command-line arguments.
See EXPERIMENT_GUIDE.md for examples.
"""
import os
import argparse
import random

from . import util
from . import experiment_config

# Set random seed for reproducibility
random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run consensus experiments with separate agent and auditor nodes."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to use (e.g., csqa, gsm8k, fact, bias, adv)",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        required=True,
        help="Graph topology to use (e.g., chain, circle, tree, star, complete)",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        required=True,
        help="Number of agents (agents respond to questions, don't vote)",
    )
    parser.add_argument(
        "--attacker_num",
        type=int,
        default=0,
        help="Number of malicious agents (Type 1 attack: give wrong answers). Default: 0",
    )
    parser.add_argument(
        "--malicious_auditor_num",
        type=int,
        default=0,
        help="Number of malicious auditors (Type 2 attack: audit and vote maliciously). Default: 0",
    )
    parser.add_argument(
        "--num_auditors",
        type=int,
        default=0,
        help="Number of auditor agents (auditors audit and vote, don't answer questions). Default: 0",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=3,
        help="Sample ID to use from the dataset. Default: 3",
    )
    parser.add_argument(
        "--reg_turn",
        type=int,
        default=9,
        help="Number of regulation turns. Default: 9",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of threads. Default: 16",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use. Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix to add to output filename (e.g., '_defended', '_baseline'). Default: empty",
    )
    parser.add_argument(
        "--json_format",
        action="store_true",
        help="Save output in JSON format instead of default format",
    )

    args = parser.parse_args()

    # Get dataset-specific configuration
    config = experiment_config.get_dataset_config(args.dataset)
    
    # Prepare attacker indices (sequential from agents)
    attacker_idx = list(range(args.attacker_num))
    
    # Prepare malicious auditor indices (random selection from all agents)
    # Auditors are selected from the agent pool, so malicious auditor indices
    # should be random unique numbers that don't exceed num_agents
    if args.malicious_auditor_num > 0:
        if args.malicious_auditor_num > args.num_auditors:
            raise ValueError(
                f"malicious_auditor_num ({args.malicious_auditor_num}) cannot exceed num_auditors ({args.num_auditors})"
            )
        if args.num_auditors > args.num_agents:
            raise ValueError(
                f"num_auditors ({args.num_auditors}) cannot exceed num_agents ({args.num_agents})"
            )
        # Randomly select which agents become malicious auditors
        malicious_auditor_idx = random.sample(range(args.num_agents), args.malicious_auditor_num)
    else:
        malicious_auditor_idx = None

    # Create output directory
    output_dir = f"./output/{args.model}/{args.dataset}/{args.sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Run the experiment
    print(f"Running experiment: dataset={args.dataset}, graph={args.graph_type}")
    print(f"  Agents: {args.num_agents} total, {args.attacker_num} malicious at indices {attacker_idx} (Type 1)")
    if args.num_auditors > 0:
        if malicious_auditor_idx:
            print(f"  Auditors: {args.num_auditors} total, {args.malicious_auditor_num} malicious at agent indices {malicious_auditor_idx} (Type 2)")
        else:
            print(f"  Auditors: {args.num_auditors} total, all honest")
    else:
        print(f"  Auditors: 0")
    
    util.run_dataset(
        ds_name=args.dataset,
        sample_id=args.sample_id,
        attacker_idx=attacker_idx,
        graph_type=args.graph_type,
        model=args.model,
        p=args.threads,
        num_agents=args.num_agents,
        json_format=args.json_format,
        turn=args.reg_turn,
        agent_class=config.agent_class,
        task_formatter=config.task_formatter,
        num_auditors=args.num_auditors,
        malicious_auditor_idx=malicious_auditor_idx,
        mode_suffix=args.output_suffix,
    )

    # Compute and display output path
    output_path = f"{output_dir}/{args.dataset}_{args.graph_type}_{args.num_agents}_{args.attacker_num}{args.output_suffix}.output"
    print(f"Experiment completed. Output saved to: {output_path}")
