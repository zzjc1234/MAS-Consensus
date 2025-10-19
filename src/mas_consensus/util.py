import threading
from tqdm import tqdm
from . import agent_base
from . import methods
from . import prompts


def _process_item(
    data,
    ds_name,
    num_agents,
    turn,
    adj_matrix,
    system_prompt,
    attacker_system_prompt,
    attacker_idx,
    model,
    json_format,
    mode,
    sample_id,
    agent_class,
    task_formatter,
    num_auditors,
    malicious_auditor_idx,
):
    tasks, task_id = task_formatter(data, attacker_idx, num_agents)

    system_prompts = [
        attacker_system_prompt if i in attacker_idx else system_prompt
        for i in range(num_agents)
    ]

    graph = agent_base.AgentGraph(
        num_agents,
        adj_matrix,
        system_prompts,
        tasks,
        task_id,
        agent_class,
        model,
        num_auditors=num_auditors,
        attacker_idx=attacker_idx,
        malicious_auditor_idx=malicious_auditor_idx,
    )
    graph.run(turn)
    output_path = f"./output/{model}/{ds_name}/{sample_id}/{ds_name}_{mode}.output"
    graph.save(output_path, json_format)


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
    agent_class,
    task_formatter,
    num_auditors=0,
    malicious_auditor_idx=None,
):
    adj_matrix = methods.generate_adj(num_agents, graph_type)
    mode = f"{graph_type}_{num_agents}_{len(attacker_idx)}"

    if ds_name == "adv":
        system_prompt = prompts.discussion_prompt["system_prompt"]
        attacker_system_prompt = prompts.discussion_prompt[
            "attacker_system_prompt_harm"
        ]
    else:
        system_prompt = prompts.discussion_prompt["system_prompt"]
        attacker_system_prompt = prompts.discussion_prompt["attacker_system_prompt"]

    methods.create_directory(f"./output/{model}/{ds_name}/{sample_id}")
    dataset = methods.get_dataset(f"./dataset/{ds_name}.jsonl")

    threads = []
    for data in tqdm(dataset):
        thread = threading.Thread(
            target=_process_item,
            args=(
                data,
                ds_name,
                num_agents,
                turn,
                adj_matrix,
                system_prompt,
                attacker_system_prompt,
                attacker_idx,
                model,
                json_format,
                mode,
                sample_id,
                agent_class,
                task_formatter,
                num_auditors,
                malicious_auditor_idx,
            ),
        )
        threads.append(thread)
        if len(threads) >= p:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            threads = []

    for t in threads:
        t.start()
    for t in threads:
        t.join()
