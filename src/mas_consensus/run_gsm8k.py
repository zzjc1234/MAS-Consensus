import random
import threading


from .agent_base import create_default_agent_graph

random.seed(42)
write_lock = threading.Lock()  # Lock for thread-safe file writing
assignment_lock = threading.Lock()  # Lock for thread-safe variable assignment


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
):
    # Use the default function with the default agent graph factory
    from .agent_base import run_dataset as base_run_dataset

    base_run_dataset(
        ds_name,
        sample_id,
        attacker_idx,
        graph_type,
        model,
        p,
        num_agents,
        json_format,
        turn,
        create_default_agent_graph,  # Pass the factory function
    )
