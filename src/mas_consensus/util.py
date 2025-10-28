import threading
from pathlib import Path
from tqdm import tqdm
from . import agent_base
from . import methods
from . import prompts
from . import logging_config


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
    auditor_idx,
    malicious_auditor_idx,
    log_dir,
):
    tasks, task_id = task_formatter(data, attacker_idx, num_agents)

    system_prompts = [
        attacker_system_prompt if i in attacker_idx else system_prompt
        for i in range(num_agents)
    ]
    
    # Create task-specific log directory
    if log_dir:
        task_log_dir = Path(log_dir) / f"task_{task_id}"
        task_log_dir.mkdir(parents=True, exist_ok=True)
    else:
        task_log_dir = None

    graph = agent_base.AgentGraph(
        num_agents,
        adj_matrix,
        system_prompts,
        tasks,
        task_id,
        agent_class,
        model,
        num_auditors=num_auditors,
        auditor_idx=auditor_idx,
        attacker_idx=attacker_idx,
        malicious_auditor_idx=malicious_auditor_idx,
        log_dir=task_log_dir,
    )
    graph.run(turn)
    output_path = f"./src/output/{model}/{ds_name}/{sample_id}/{ds_name}_{mode}.output"
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
    auditor_idx=None,
    malicious_auditor_idx=None,
    mode_suffix="",
):
    adj_matrix = methods.generate_adj(num_agents, graph_type)
    mode = f"{graph_type}_{num_agents}_{len(attacker_idx)}{mode_suffix}"

    if ds_name == "adv":
        system_prompt = prompts.discussion_prompt["system_prompt"]
        attacker_system_prompt = prompts.discussion_prompt[
            "attacker_system_prompt_harm"
        ]
    else:
        system_prompt = prompts.discussion_prompt["system_prompt"]
        attacker_system_prompt = prompts.discussion_prompt["attacker_system_prompt"]

    methods.create_directory(f"./src/output/{model}/{ds_name}/{sample_id}")
    dataset = methods.get_dataset(f"./src/dataset/{ds_name}.jsonl")
    
    # Set up logging for this experiment run
    log_dir = logging_config.setup_experiment_logging(
        model, ds_name, sample_id, graph_type, num_agents, len(attacker_idx), mode_suffix
    )
    
    # Get console logger for high-level progress
    progress = logging_config.get_console_logger()
    progress.info(f"Processing {len(dataset)} items from {ds_name} dataset...")
    progress.info(f"Logs: {log_dir}")

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
                auditor_idx,
                malicious_auditor_idx,
                log_dir,
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
    
    progress.info(f"✓ Completed processing {len(dataset)} items")
