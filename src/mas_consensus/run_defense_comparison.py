import numpy as np
import os
import argparse

from . import util
from . import evaluate
from . import experiment_config


def run_defense_comparison(
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
    attacker_idx,
):
    """
    Run three scenarios: Baseline, Attacked, and Defended
    """

    # Scenario 1: Baseline (no malicious agents)
    print("Running Baseline (no malicious agents)...")
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=[],
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=0,  # No auditing in baseline
        malicious_auditor_idx=None,
    )

    baseline_output_path = f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_0.output"

    # Scenario 2: Attacked (with N malicious agents)
    print(f"Running Attacked scenario ({len(attacker_idx)} malicious agent(s))...")
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=attacker_idx,
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=0,  # No auditing in attacked scenario
        malicious_auditor_idx=None,
    )

    attacked_output_path = (
        f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_{len(attacker_idx)}.output"
    )

    # Scenario 3: Defended (with N malicious agents + audit/vote defense mechanism)
    print(
        f"Running Defended scenario ({len(attacker_idx)} malicious agent(s) + defense mechanism)..."
    )
    util.run_dataset(
        ds_name=ds_name,
        sample_id=sample_id,
        attacker_idx=attacker_idx,
        graph_type=graph_type,
        model=model,
        p=p,
        num_agents=num_agents,
        json_format=json_format,
        turn=turn,
        agent_class=agent_class,
        task_formatter=task_formatter,
        num_auditors=2,  # Enable auditing with 2 auditors
        malicious_auditor_idx=None,  # Honest auditors in this scenario
    )

    defended_output_path = f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_{len(attacker_idx)}_defended.output"

    return baseline_output_path, attacked_output_path, defended_output_path


def evaluate_and_plot_comparison(
    dataset_path,
    baseline_output_path,
    attacked_output_path,
    defended_output_path,
    attacker_num,
    evaluation_type="SAA",
):
    """
    Evaluate all three scenarios and create a plot for comparison
    """
    print("Evaluating Baseline...")
    baseline_accuracy = evaluate.evaluate(
        dataset_path, baseline_output_path, 0, evaluation_type
    )

    print("Evaluating Attacked...")
    attacked_accuracy = evaluate.evaluate(
        dataset_path, attacked_output_path, attacker_num, evaluation_type
    )

    print("Evaluating Defended...")
    defended_accuracy = evaluate.evaluate(
        dataset_path, defended_output_path, attacker_num, evaluation_type
    )

    print(f"Baseline {evaluation_type}: {np.mean(baseline_accuracy):.4f}")
    print(f"Attacked {evaluation_type}: {np.mean(attacked_accuracy):.4f}")
    print(f"Defended {evaluation_type}: {np.mean(defended_accuracy):.4f}")

    # Create comparison plot
    import matplotlib.pyplot as plt

    plt.rc("font", family="Comic Sans MS")
    plt.figure(figsize=(10, 6))

    # Plot MJA over iterations if that's what we're evaluating
    if evaluation_type == "MJA":
        iterations = range(1, len(baseline_accuracy) + 1)
        plt.plot(
            iterations,
            baseline_accuracy,
            label="Baseline",
            linewidth=3,
            marker="o",
            markersize=10,
            color="blue",
        )
        plt.plot(
            iterations,
            attacked_accuracy,
            label="Attacked",
            linewidth=3,
            marker="s",
            markersize=10,
            color="red",
        )
        plt.plot(
            iterations,
            defended_accuracy,
            label="Defended",
            linewidth=3,
            marker="^",
            markersize=10,
            color="green",
        )

        plt.title(
            f"{evaluation_type} Comparison: Baseline vs Attacked vs Defended",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Iterations", fontsize=14, fontweight="bold")
        plt.ylabel(f"Performance ({evaluation_type})", fontsize=14, fontweight="bold")
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
    else:
        # For single accuracy values, use a bar chart
        labels = ["Baseline", "Attacked", "Defended"]
        values = [
            np.mean(baseline_accuracy),
            np.mean(attacked_accuracy),
            np.mean(defended_accuracy),
        ]
        colors = ["blue", "red", "green"]

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
                fontsize=14,
                fontweight="bold",
            )

        plt.title(
            f"{evaluation_type} Comparison: Baseline vs Attacked vs Defended",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel(f"Performance ({evaluation_type})", fontsize=14, fontweight="bold")

        # Set y-axis to go from 0 to 1 for accuracy
        plt.ylim(0, 1)
        plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return baseline_accuracy, attacked_accuracy, defended_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run defense comparison experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="csqa",
        help="Dataset to use (e.g., csqa, gsm8k, fact, bias, adv)",
    )
    parser.add_argument(
        "--attacker_num",
        type=int,
        default=1,
        help="Number of malicious agents for the attacked/defended scenarios.",
    )
    args = parser.parse_args()

    # Define experiment parameters
    sample_id = 3
    graph_type = "complete"
    model = "gpt-4o-mini"
    json_format = False
    p = 16
    reg_turn = 9
    num_agents = 6

    # Get dataset-specific configuration
    config = experiment_config.get_dataset_config(args.dataset)
    attacker_idx = list(range(args.attacker_num))

    # Create output directory if it doesn't exist
    output_dir = f"./output/{model}/{args.dataset}/{sample_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Run the comparison experiment
    baseline_path, attacked_path, defended_path = run_defense_comparison(
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
        attacker_idx=attacker_idx,
    )

    # Evaluate and plot results
    dataset_path = f"./dataset/{args.dataset}.jsonl"
    evaluate_and_plot_comparison(
        dataset_path=dataset_path,
        baseline_output_path=baseline_path,
        attacked_output_path=attacked_path,
        defended_output_path=defended_path,
        attacker_num=args.attacker_num,
        evaluation_type="SAA",  # You can change to "MJA" for different evaluation
    )
