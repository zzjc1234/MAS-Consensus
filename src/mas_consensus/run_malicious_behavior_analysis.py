import numpy as np
import os
import argparse

from . import util
from . import evaluate
from . import experiment_config


def run_malicious_behavior_experiments(
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
):
    """
    Run experiments for different malicious behaviors:
    1. Type 1: Regular malicious agent (for comparison)
    2. Type 2: Malicious auditing (honest agents, malicious auditors)
    3. Type 3: Malicious voting (honest agents, malicious voters)
    4. Combined: Type 1 + Type 2 + Type 3
    """
    results = {}

    # Scenario 1: Type 1 only (for baseline comparison)
    print("Running Type 1 (regular malicious) experiment...")
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
        num_auditors=2,  # Enable auditing
        malicious_auditor_idx=None,  # Honest auditors
    )

    results["type1_only"] = f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_1.output"

    # Scenario 2: Type 2 (malicious auditing) - honest agents, malicious auditors
    print("Running Type 2 (malicious auditing) experiment...")
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
        num_auditors=2,  # Enable auditing
        malicious_auditor_idx=[0, 1],  # Malicious auditors at indices 0, 1
    )

    results["type2_only"] = (
        f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_1_malicious_auditors.output"
    )

    # Scenario 3: Type 3 (malicious voting) - honest agents, malicious voters during voting
    print("Running Type 3 (malicious voting) experiment...")
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
        num_auditors=2,  # Enable auditing
        malicious_auditor_idx=None,  # Honest auditors but malicious voters
    )

    results["type3_only"] = (
        f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_1_malicious_voters.output"
    )

    # Scenario 4: Combined malicious behaviors
    print("Running combined malicious behaviors experiment...")
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=[0, 1],  # 2 malicious agents at indices 0, 1
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=2,  # Enable auditing
        malicious_auditor_idx=[0],  # Some malicious auditors
    )

    results["combined"] = (
        f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_2_malicious_combined.output"
    )

    return results


def evaluate_malicious_behavior_impact(
    dataset_path, experiment_results, evaluation_type="SAA"
):
    """
    Evaluate and compare the impact of different types of malicious behavior on system accuracy
    """
    print("Evaluating impact of different malicious behaviors...")

    accuracies = {}

    # Evaluate Type 1 (regular malicious)
    print("Evaluating Type 1 (regular malicious)...")
    accuracies["type1_only"] = evaluate.evaluate(
        dataset_path, experiment_results["type1_only"], 1, evaluation_type
    )
    print(f"Type 1 accuracy: {np.mean(accuracies['type1_only']):.4f}")

    # Evaluate Type 2 (malicious auditing)
    print("Evaluating Type 2 (malicious auditing)...")
    accuracies["type2_only"] = evaluate.evaluate(
        dataset_path, experiment_results["type2_only"], 1, evaluation_type
    )
    print(f"Type 2 accuracy: {np.mean(accuracies['type2_only']):.4f}")

    # Evaluate Type 3 (malicious voting)
    print("Evaluating Type 3 (malicious voting)...")
    accuracies["type3_only"] = evaluate.evaluate(
        dataset_path, experiment_results["type3_only"], 1, evaluation_type
    )
    print(f"Type 3 accuracy: {np.mean(accuracies['type3_only']):.4f}")

    # Evaluate combined
    print("Evaluating combined malicious behaviors...")
    accuracies["combined"] = evaluate.evaluate(
        dataset_path, experiment_results["combined"], 2, evaluation_type
    )
    print(f"Combined accuracy: {np.mean(accuracies['combined']):.4f}")

    return accuracies


def plot_malicious_behavior_comparison(accuracies):
    """
    Plot comparison of different types of malicious behavior impact
    """
    import matplotlib.pyplot as plt

    plt.rc("font", family="Comic Sans MS")
    plt.figure(figsize=(12, 6))

    # Prepare labels and values
    labels = [
        "Type 1 (Regular)",
        "Type 2 (Malicious Auditing)",
        "Type 3 (Malicious Voting)",
        "Combined",
    ]
    values = [
        np.mean(accuracies["type1_only"]),
        np.mean(accuracies["type2_only"]),
        np.mean(accuracies["type3_only"]),
        np.mean(accuracies["combined"]),
    ]

    # Colors to distinguish the types
    colors = ["blue", "orange", "red", "purple"]

    bars = plt.bar(
        labels, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.2
    )

    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.title(
        "Impact of Different Malicious Behaviors on System Accuracy",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Performance (SAA)", fontsize=14, fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Set y-axis to go from 0 to 1 for accuracy
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return labels, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run malicious behavior analysis experiments."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="csqa",
        help="Dataset to use (e.g., csqa, gsm8k, fact, bias, adv)",
    )
    args = parser.parse_args()

    # Define experiment parameters
    sample_id = 3
    graph_type = "complete"  # Using complete graph for better interaction
    model = "gpt-4o-mini"
    json_format = False
    p = 16
    reg_turn = 9
    num_agents = 6

    # Get dataset-specific configuration
    config = experiment_config.get_dataset_config(args.dataset)

    # Create output directory if it doesn't exist
    output_dir = f"./output/{model}/{args.dataset}/{sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Run experiments for different malicious behaviors
    experiment_results = run_malicious_behavior_experiments(
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
    )

    # Evaluate and plot results
    dataset_path = f"./dataset/{args.dataset}.jsonl"
    accuracies = evaluate_malicious_behavior_impact(
        dataset_path=dataset_path,
        experiment_results=experiment_results,
        evaluation_type="SAA",
    )

    # Create comparison plot
    labels, values = plot_malicious_behavior_comparison(accuracies)

    print("Experiment completed. Impact of malicious behaviors:")
    for label, value in zip(labels, values):
        print(f"{label}: {value:.4f}")
