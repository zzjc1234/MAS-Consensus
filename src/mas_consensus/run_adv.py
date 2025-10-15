import random
import threading

from .agent_base import create_discussion_agent_graph, run_adv_dataset

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
    # Use the specialized function for adversarial datasets
    run_adv_dataset(
        ds_name,
        sample_id,
        attacker_idx,
        graph_type,
        model,
        p,
        num_agents,
        json_format,
        turn,
        create_discussion_agent_graph,  # Pass the factory function
    )
