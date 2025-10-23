import time
import os
import argparse

from . import util
from . import experiment_config


def run_efficiency_experiments(
    ds_name,
    sample_id,
    graph_type,
    model,
    p,
    num_agents,
    json_format,
    turn,
    output_dir,
    task_formatter,
    agent_class,
    num_auditors=2,
):
    """
    Run efficiency experiments comparing different scenarios:
    1. No malicious agents (baseline)
    2. Type 1 malicious agents only
    3. Type 2 malicious auditing agents
    4. Type 3 malicious voting agents
    5. Combined malicious behaviors
    """
    results = {}

    # Scenario 1: No malicious agents (baseline)
    print("Running baseline (no malicious agents) efficiency experiment...")
    start_time = time.time()
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=[],  # No malicious agents
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=0,  # No auditing
        malicious_auditor_idx=None,
    )
    baseline_time = time.time() - start_time
    results["baseline"] = baseline_time

    print(f"Baseline execution time: {baseline_time:.2f} seconds")

    # Scenario 2: Type 1 malicious agents only
    print("Running Type 1 (regular malicious) efficiency experiment...")
    start_time = time.time()
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=[0],  # 1 malicious agent at index 0
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=0,  # No auditing for Type 1 comparison
        malicious_auditor_idx=None,
    )
    type1_time = time.time() - start_time
    results["type1_only"] = type1_time

    print(f"Type 1 execution time: {type1_time:.2f} seconds")

    # Scenario 3: Type 2 malicious auditing
    print("Running Type 2 (malicious auditing) efficiency experiment...")
    start_time = time.time()
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=[0],  # 1 malicious agent at index 0
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=num_auditors,  # Enable auditing
        malicious_auditor_idx=[0],  # Malicious auditors
    )
    type2_time = time.time() - start_time
    results["type2_only"] = type2_time

    print(f"Type 2 execution time: {type2_time:.2f} seconds")

    # Scenario 4: Type 3 malicious voting
    print("Running Type 3 (malicious voting) efficiency experiment...")
    start_time = time.time()
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=[0],  # 1 malicious agent at index 0
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=num_auditors,  # Enable auditing to trigger voting
        malicious_auditor_idx=None,  # Honest auditors, but voting will be malicious
    )
    type3_time = time.time() - start_time
    results["type3_only"] = type3_time

    print(f"Type 3 execution time: {type3_time:.2f} seconds")

    # Scenario 5: Combined malicious behaviors
    print("Running combined malicious behaviors efficiency experiment...")
    start_time = time.time()
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=[0, 1],  # Multiple malicious agents
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=num_auditors,  # Enable auditing
        malicious_auditor_idx=[0],  # Malicious auditors
    )
    combined_time = time.time() - start_time
    results["combined"] = combined_time

    print(f"Combined execution time: {combined_time:.2f} seconds")

    return results


def plot_efficiency_comparison(execution_times):
    """
    Plot comparison of execution times under different malicious behavior scenarios
    """
    import matplotlib.pyplot as plt

    plt.rc("font", family="Comic Sans MS")
    plt.figure(figsize=(12, 6))

    # Prepare labels and values
    labels = [
        "Baseline (No Malicious)",
        "Type 1 (Regular)",
        "Type 2 (Malicious Auditing)",
        "Type 3 (Malicious Voting)",
        "Combined",
    ]
    values = [
        execution_times["baseline"],
        execution_times["type1_only"],
        execution_times["type2_only"],
        execution_times["type3_only"],
        execution_times["combined"],
    ]

    # Colors to distinguish the types
    colors = ["green", "blue", "orange", "red", "purple"]

    bars = plt.bar(
        labels, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.2
    )

    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.title(
        "System Efficiency: Execution Time Under Different Malicious Behaviors",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Execution Time (seconds)", fontsize=14, fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate efficiency ratios compared to baseline
    baseline_time = execution_times["baseline"]
    efficiency_ratios = {
        label: time_val / baseline_time for label, time_val in zip(labels, values)
    }

    print("\nEfficiency Ratios (compared to baseline):")
    for label, ratio in efficiency_ratios.items():
        print(f"{label}: {ratio:.2f}x")

    return labels, values, efficiency_ratios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run efficiency analysis experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="csqa",
        help="Dataset to use (e.g., csqa, gsm8k, fact, bias, adv). Default: csqa",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=3,
        help="Sample ID to use from the dataset. Default: 3",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="complete",
        help="Graph topology. Default: complete (recommended for efficiency analysis)",
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
        default=4,
        help="Number of agents in the simulation. Default: 4 (smaller for faster testing)",
    )
    parser.add_argument(
        "--reg_turn",
        type=int,
        default=3,
        help="Number of regulation turns. Default: 3 (fewer for efficiency testing)",
    )
    parser.add_argument(
        "--num_auditors",
        type=int,
        default=2,
        help="Number of auditor agents (set to 0 to disable auditing). Default: 2",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads. Default: 4 (fewer for clearer timing)",
    )
    args = parser.parse_args()

    # Define experiment parameters (use a smaller sample for efficiency test)
    sample_id = args.sample_id
    graph_type = args.graph_type
    model = args.model
    json_format = False
    p = args.threads
    reg_turn = args.reg_turn
    num_agents = args.num_agents
    num_auditors = args.num_auditors

    # Get dataset-specific configuration
    config = experiment_config.get_dataset_config(args.dataset)

    # Create output directory if it doesn't exist
    output_dir = f"./output/{model}/{args.dataset}/{sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Run efficiency experiments
    execution_times = run_efficiency_experiments(
        ds_name=args.dataset,
        sample_id=sample_id,
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=reg_turn,
        output_dir=output_dir,
        task_formatter=config.task_formatter,
        agent_class=config.agent_class,
        num_auditors=num_auditors,
    )

    # Create comparison plot
    labels, values, efficiency_ratios = plot_efficiency_comparison(execution_times)

    print("\nDetailed Execution Times:")
    for label, value in zip(labels, values):
        print(f"{label}: {value:.2f} seconds")
