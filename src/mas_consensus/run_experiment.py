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
from . import logging_config

# Set random seed for reproducibility
random.seed(42)

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
    
    # Prepare attacker indices (randomly select malicious agents)
    if args.attacker_num > 0:
        attacker_idx = random.sample(range(args.num_agents), args.attacker_num)
    else:
        attacker_idx = []
    
    # Prepare auditor selection:
    # Step 1: Randomly select which agents become auditors (excluding malicious agents)
    # Step 2: From those selected auditors, randomly choose which ones are malicious
    if args.num_auditors > 0:
        # Malicious agents (attacker_idx) must remain in discussion, so exclude them from auditor selection
        # Available agents for auditor selection = all agents except malicious ones
        available_for_auditors = [i for i in range(args.num_agents) if i not in attacker_idx]
        
        if len(available_for_auditors) < args.num_auditors:
            raise ValueError(
                f"Cannot select {args.num_auditors} auditors from {len(available_for_auditors)} available agents "
                f"(excluding {args.attacker_num} malicious agents that must remain in discussion)"
            )
        
        # Step 1: Randomly select auditor indices from available agents (not including attackers)
        auditor_indices = random.sample(available_for_auditors, args.num_auditors)
        
        # Step 2: From selected auditors, randomly choose which are malicious
        if args.malicious_auditor_num > 0:
            malicious_auditor_idx = random.sample(auditor_indices, args.malicious_auditor_num)
        else:
            malicious_auditor_idx = None
        
        # Verify constraint: auditor indices must not overlap with attacker indices
        overlap = set(auditor_indices) & set(attacker_idx)
        if overlap:
            raise ValueError(
                f"Internal error: Auditors {overlap} overlap with malicious agents! "
                "Malicious agents must stay in discussion."
            )
        
        # Verify malicious auditors are subset of auditors
        if malicious_auditor_idx:
            if not set(malicious_auditor_idx).issubset(set(auditor_indices)):
                raise ValueError(
                    f"Internal error: Malicious auditors {malicious_auditor_idx} not subset of auditors {auditor_indices}"
                )
        
        # Pass auditor_indices to util.run_dataset
        auditor_idx = auditor_indices
    else:
        auditor_idx = None
        malicious_auditor_idx = None
    
    # Final validation: Ensure we have the expected configuration
    if args.attacker_num > 0:
        # Verify malicious agents will be in discussion (not in auditor group)
        if auditor_idx and any(idx in auditor_idx for idx in attacker_idx):
            raise ValueError(
                f"Configuration error: Malicious agents {attacker_idx} cannot be auditors. "
                "They must remain in discussion group to execute Type 1 attack."
            )

    # Create output directory
    output_dir = f"./src/output/{args.model}/{args.dataset}/{args.sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Run the experiment
    progress_logger.info("=" * 80)
    progress_logger.info(f"EXPERIMENT: {args.dataset}/{args.graph_type}")
    progress_logger.info(f"  Total agents: {args.num_agents}")
    if args.attacker_num > 0:
        progress_logger.info(f"  Malicious agents (Type 1): {args.attacker_num} randomly selected at {attacker_idx} → DISCUSSION")
    else:
        progress_logger.info(f"  Malicious agents (Type 1): 0 - all honest")
    
    if args.num_auditors > 0:
        progress_logger.info(f"  Auditors: {args.num_auditors} randomly selected from non-malicious agents at {auditor_idx} → AUDIT & VOTE")
        if malicious_auditor_idx:
            progress_logger.info(f"    - Malicious auditors (Type 2): {args.malicious_auditor_num} at {malicious_auditor_idx}")
        else:
            progress_logger.info(f"    - All auditors honest")
    else:
        progress_logger.info(f"  Auditors: 0")
    
    progress_logger.info(f"  Discussion group: {args.num_agents - args.num_auditors} agents")
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
