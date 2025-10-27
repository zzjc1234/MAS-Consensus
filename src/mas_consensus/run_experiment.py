"""
Unified experiment runner - a thin wrapper around util.run_dataset()

Node Architecture:
  - Workers: Respond to questions and update their answers (don't vote)
  - Auditors: Audit workers and vote (don't answer questions)

Attack Types:
  - Type 1: Malicious workers (give wrong answers)
  - Type 2: Malicious auditors (audit and vote maliciously)

All experiment logic is controlled through command-line arguments.
See EXPERIMENT_GUIDE.md for examples.
"""
import os
import argparse

from . import util
from . import experiment_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run consensus experiments with separate worker and auditor nodes."
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
        help="Number of worker agents (workers respond to questions, don't vote)",
    )
    parser.add_argument(
        "--attacker_num",
        type=int,
        default=0,
        help="Number of malicious workers (Type 1 attack: give wrong answers). Default: 0",
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
    
    # Prepare attacker and auditor indices
    attacker_idx = list(range(args.attacker_num))
    malicious_auditor_idx = list(range(args.malicious_auditor_num)) if args.malicious_auditor_num > 0 else None

    # Create output directory
    output_dir = f"./output/{args.model}/{args.dataset}/{args.sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Run the experiment
    print(f"Running experiment: dataset={args.dataset}, graph={args.graph_type}")
    print(f"  Workers: {args.num_agents} total, {args.attacker_num} malicious (Type 1)")
    print(f"  Auditors: {args.num_auditors} total, {args.malicious_auditor_num} malicious (Type 2)")
    
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
