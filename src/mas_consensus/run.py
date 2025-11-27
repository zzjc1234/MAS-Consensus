"""
Minimal batch runner equivalent to the original run.py, now using util/experiment_config.
Edit the lists near the bottom to sweep datasets/graphs/agent counts/attack settings.
"""

import os

from . import util
from . import experiment_config
from . import logging_config

# Console logger for high-level progress
progress = logging_config.get_console_logger()


def run_single(
    dataset: str,
    graph_type: str,
    sample_id: int,
    num_agents: int,
    attacker_num: int,
    num_auditors: int,
    malicious_auditor_num: int,
    model: str,
    threads: int,
    reg_turn: int,
    json_format: bool,
    output_suffix: str = "",
) -> None:
    if num_auditors > num_agents:
        raise ValueError("num_auditors cannot exceed num_agents")
    if malicious_auditor_num > num_auditors:
        raise ValueError("malicious_auditor_num cannot exceed num_auditors")
    if attacker_num + num_auditors > num_agents:
        raise ValueError("attacker_num + num_auditors cannot exceed num_agents")

    if num_auditors > 0:
        auditor_idx = list(range(num_auditors))
        malicious_auditor_idx = (
            list(range(malicious_auditor_num)) if malicious_auditor_num > 0 else None
        )
        attacker_idx = list(range(num_auditors, num_auditors + attacker_num)) if attacker_num > 0 else []
    else:
        auditor_idx = None
        malicious_auditor_idx = None
        attacker_idx = list(range(attacker_num)) if attacker_num > 0 else []

    config = experiment_config.get_dataset_config(dataset)
    output_dir = f"./src/output/{model}/{dataset}/{sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    progress.info(
        f"{dataset}/{graph_type} sample={sample_id} agents={num_agents} "
        f"attackers={attacker_num} auditors={num_auditors}"
    )

    util.run_dataset(
        ds_name=dataset,
        sample_id=sample_id,
        attacker_idx=attacker_idx,
        graph_type=graph_type,
        model=model,
        p=threads,
        num_agents=num_agents,
        json_format=json_format,
        turn=reg_turn,
        agent_class=config.agent_class,
        task_formatter=config.task_formatter,
        num_auditors=num_auditors,
        auditor_idx=auditor_idx,
        malicious_auditor_idx=malicious_auditor_idx,
        mode_suffix=output_suffix,
    )

    output_path = f"{output_dir}/{dataset}_{graph_type}_{num_agents}_{attacker_num}{output_suffix}.output"
    progress.info(f"✓ Output: {output_path}\n")


if __name__ == "__main__":
    # Defaults mirror the original script; edit here to sweep new settings.
    datasets = ["csqa"]
    sample_ids = [3]
    graph_types = ["chain", "circle", "tree", "star", "complete"]
    model = "google/gemini-2.5-flash"
    json_format = False
    threads = 16
    reg_turn = 9
    num_agents_list = [7, 8, 9, 10]
    attacker_nums = [0]
    num_auditors = 2
    malicious_auditor_num = 0
    output_suffix = ""

    for num_agents in num_agents_list:
        for dataset in datasets:
            for graph_type in graph_types:
                for sample_id in sample_ids:
                    for attacker_num in attacker_nums:
                        run_single(
                            dataset=dataset,
                            graph_type=graph_type,
                            sample_id=sample_id,
                            num_agents=num_agents,
                            attacker_num=attacker_num,
                            num_auditors=num_auditors,
                            malicious_auditor_num=malicious_auditor_num,
                            model=model,
                            threads=threads,
                            reg_turn=reg_turn,
                            json_format=json_format,
                            output_suffix=output_suffix,
                        )
