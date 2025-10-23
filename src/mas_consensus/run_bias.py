import argparse

from . import util
from . import experiment_config


def run_dataset(
    ds_name,
    sample_id,
    attacker_idx,
    graph_type,
    model,
    p,
    num_agents,
    json_format,
    turn,
    num_auditors=0,
    malicious_auditor_idx=None,
):
    config = experiment_config.get_dataset_config(ds_name)
    util.run_dataset(
        ds_name,
        sample_id,
        attacker_idx,
        graph_type,
        model,
        p,
        num_agents,
        json_format,
        turn,
        agent_class=config.agent_class,
        task_formatter=config.task_formatter,
        num_auditors=num_auditors,
        malicious_auditor_idx=malicious_auditor_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bias detection experiments.")
    parser.add_argument(
        "--sample_id",
        type=int,
        default=3,
        help="Sample ID to use from the dataset. Default: 3",
    )
    parser.add_argument(
        "--attacker_num",
        type=int,
        default=0,
        help="Number of malicious agents. Default: 0",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="complete",
        help="Graph topology (e.g., complete, chain, circle, tree, star). Default: complete",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use. Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=8,
        help="Number of agents in the simulation. Default: 8",
    )
    parser.add_argument(
        "--reg_turn", type=int, default=9, help="Number of regulation turns. Default: 9"
    )
    parser.add_argument(
        "--num_auditors",
        type=int,
        default=2,
        help="Number of auditor agents (set to 0 to disable auditing). Default: 2",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of threads. Default: 8"
    )
    args = parser.parse_args()

    dataset = "bias"
    sample_ids = [args.sample_id]
    graph_type = args.graph_type
    model = args.model
    json_format = False
    p = args.threads
    reg_turn = args.reg_turn
    num_agents = args.num_agents
    num_auditors = args.num_auditors
    malicious_auditor_idx = [0]
    attacker_nums = [args.attacker_num]

    for sample_id in sample_ids:
        for attacker_num in attacker_nums:
            attacker_idx = list(range(attacker_num))
            print("Attacker Idx:", attacker_idx)
            run_dataset(
                dataset,
                sample_id,
                attacker_idx,
                graph_type,
                model,
                p,
                num_agents,
                json_format,
                reg_turn,
                num_auditors=num_auditors,
                malicious_auditor_idx=malicious_auditor_idx,
            )
