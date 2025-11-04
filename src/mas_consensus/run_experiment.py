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

from . import util
from . import experiment_config
from . import logging_config

# Get console logger for high-level progress
progress_logger = logging_config.get_console_logger()


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
        help="Number of malicious agents (Type 1 attack: give wrong answers). Assigned deterministically after auditors. Default: 0",
    )
    parser.add_argument(
        "--malicious_auditor_num",
        type=int,
        default=0,
        help="Number of malicious auditors (Type 2 attack: audit and vote maliciously). Assigned as first N auditors (indices 0 to N-1). Default: 0",
    )
    parser.add_argument(
        "--num_auditors",
        type=int,
        default=0,
        help="Number of auditor agents (auditors audit and vote, don't answer questions). Assigned starting from index 0. Default: 0",
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
        default="openai/gpt-4o-mini",
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

    # Validation
    if args.num_auditors > args.num_agents:
        raise ValueError(
            f"num_auditors ({args.num_auditors}) cannot exceed num_agents ({args.num_agents})"
        )
    if args.malicious_auditor_num > args.num_auditors:
        raise ValueError(
            f"malicious_auditor_num ({args.malicious_auditor_num}) cannot exceed num_auditors ({args.num_auditors})"
        )
    if args.attacker_num + args.num_auditors > args.num_agents:
        raise ValueError(
            f"attacker_num ({args.attacker_num}) + num_auditors ({args.num_auditors}) = {args.attacker_num + args.num_auditors} "
            f"cannot exceed num_agents ({args.num_agents}). "
            f"Malicious agents must remain in discussion, and auditors are selected from remaining agents."
        )

    # Prepare deterministic index assignment:
    # - Auditors start from index 0 (if any auditors exist)
    # - Malicious auditors are the first N auditors (indices 0 to malicious_auditor_num-1)
    # - Attackers follow after all auditors (starting from num_auditors)
    # - If no auditors, attackers start from 0
    
    if args.num_auditors > 0:
        # Auditors occupy indices [0, 1, 2, ..., num_auditors-1]
        auditor_idx = list(range(args.num_auditors))
        
        # Malicious auditors are the first malicious_auditor_num auditors
        if args.malicious_auditor_num > 0:
            malicious_auditor_idx = list(range(args.malicious_auditor_num))
        else:
            malicious_auditor_idx = None
        
        # Attackers start after all auditors
        if args.attacker_num > 0:
            attacker_idx = list(range(args.num_auditors, args.num_auditors + args.attacker_num))
        else:
            attacker_idx = []
    else:
        # No auditors, so attackers start from 0
        auditor_idx = None
        malicious_auditor_idx = None
        
        if args.attacker_num > 0:
            attacker_idx = list(range(args.attacker_num))
        else:
            attacker_idx = []

    # With deterministic assignment, auditors and attackers never overlap by design

    # Create output directory
    output_dir = f"./src/output/{args.model}/{args.dataset}/{args.sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Run the experiment
    progress_logger.info("=" * 80)
    progress_logger.info(f"EXPERIMENT: {args.dataset}/{args.graph_type}")
    progress_logger.info(f"  Total agents: {args.num_agents}")
    if args.attacker_num > 0:
        progress_logger.info(
            f"  Malicious agents (Type 1): {args.attacker_num} at indices {attacker_idx} → DISCUSSION"
        )
    else:
        progress_logger.info("  Malicious agents (Type 1): 0 - all honest")

    if args.num_auditors > 0:
        progress_logger.info(
            f"  Auditors: {args.num_auditors} at indices {auditor_idx} → AUDIT & VOTE"
        )
        if malicious_auditor_idx:
            progress_logger.info(
                f"    - Malicious auditors (Type 2): {args.malicious_auditor_num} at indices {malicious_auditor_idx}"
            )
        else:
            progress_logger.info("    - All auditors honest")
    else:
        progress_logger.info("  Auditors: 0")

    progress_logger.info(
        f"  Discussion group: {args.num_agents - args.num_auditors} agents"
    )
    progress_logger.info("=" * 80)

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
        auditor_idx=auditor_idx,
        malicious_auditor_idx=malicious_auditor_idx,
        mode_suffix=args.output_suffix,
    )

    # Compute and display output path
    output_path = f"{output_dir}/{args.dataset}_{args.graph_type}_{args.num_agents}_{args.attacker_num}{args.output_suffix}.output"
    progress_logger.info(f"✓ Experiment completed. Output: {output_path}")
    progress_logger.info("")
