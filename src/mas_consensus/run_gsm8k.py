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
    dataset = "gsm8k"
    sample_ids = [3]
    graph_type = "complete"
    model = "gpt-4o-mini"
    json_format = False
    p = 16
    reg_turn = 9
    num_agents = 6
    num_auditors = 2  # Number of auditor agents (set to 0 to disable auditing)
    malicious_auditor_idx = [0]
    attacker_nums = [0, 1, 2]

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
