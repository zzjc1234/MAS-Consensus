from . import run_adv
from . import run_bias
from . import run_csqa
from . import run_fact
from . import run_gsm8k

if __name__ == "__main__":
    datasets = ["bias"]
    sample_ids = [3]
    graph_types = ["chain", "circle", "tree", "star", "complete"][:-1]
    model = "gpt-4o-mini"
    json_format = False
    p = 16  # Number of threads to process the dataset
    reg_turn = 9
    num_agents_list = [7, 8, 9, 10]
    # attacker_idx = [0, 1]
    attacker_nums = [0]
    num_auditors = 2  # Number of auditor agents (set to 0 to disable auditing)
    for num_agents in num_agents_list:
        for dataset in datasets:
            for graph_type in graph_types:
                for sample_id in sample_ids:
                    for attacker_num in attacker_nums:
                        attacker_idx = list(range(attacker_num))
                        print(
                            f"Dataset: {dataset}, Graph: {graph_type}, Sample: {sample_id}, Attacker Idx: {attacker_idx}"
                        )
                        if "fact" in dataset:
                            run_fact.run_dataset(
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
                            )
                        if "csqa" in dataset:
                            run_csqa.run_dataset(
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
                            )
                        if "gsm8k" in dataset:
                            run_gsm8k.run_dataset(
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
                            )
                        if "bias" in dataset:
                            run_bias.run_dataset(
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
                            )
                        if "adv" in dataset:
                            run_adv.run_dataset(
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
                            )
