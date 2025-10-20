import numpy as np
import os
import argparse

from . import util
from . import evaluate
from . import experiment_config


def run_chain_experiment(
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
    Run experiment with chain topology and N faulty agents + defense mechanism
    """
    print(
        f"Running chain experiment with {num_agents} agents, {len(attacker_idx)} faulty agent(s) + defense mechanism..."
    )

    # Run with N malicious agents and defense mechanism enabled
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
        malicious_auditor_idx=None,  # Honest auditors
    )

    output_path = f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_{len(attacker_idx)}_defended_chain.output"

    return output_path


def evaluate_and_plot_chain_accuracy(
    dataset_path,
    output_path,
    attacker_num,
    evaluation_type="MJA",  # MJA shows accuracy across iterations which is what we want
):
    """
    Evaluate the chain experiment and plot accuracy of intermediate agents
    """
    print("Evaluating chain experiment...")
    accuracy = evaluate.evaluate(
        dataset_path, output_path, attacker_num, evaluation_type
    )

    print(f"Chain experiment {evaluation_type}: {np.mean(accuracy):.4f}")

    # Create a plot showing the accuracy over iterations for the chain
    import matplotlib.pyplot as plt

    plt.rc("font", family="Comic Sans MS")
    plt.figure(figsize=(12, 6))

    # Plot MJA over iterations
    if evaluation_type == "MJA":
        iterations = range(1, len(accuracy) + 1)
        plt.plot(
            iterations,
            accuracy,
            label="Chain with Defense",
            linewidth=3,
            marker="o",
            markersize=10,
            color="green",
        )

        plt.title(
            f"{evaluation_type} for Chain Topology ({attacker_num} faulty + defense)",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Iterations", fontsize=14, fontweight="bold")
        plt.ylabel(f"Performance ({evaluation_type})", fontsize=14, fontweight="bold")
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add annotations for important iterations
        max_idx = np.argmax(accuracy)
        plt.annotate(
            f"Max: {accuracy[max_idx]:.3f}",
            xy=(max_idx + 1, accuracy[max_idx]),
            xytext=(max_idx + 1, accuracy[max_idx] + 0.1),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=12,
            color="red",
            fontweight="bold",
        )
    else:
        # For single accuracy values, use a bar chart
        labels = ["Chain with Defense"]
        values = [np.mean(accuracy)]
        colors = ["green"]

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
            f"{evaluation_type} for Chain Topology ({attacker_num} faulty + defense)",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel(f"Performance ({evaluation_type})", fontsize=14, fontweight="bold")

        # Set y-axis to go from 0 to 1 for accuracy
        plt.ylim(0, 1)
        plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return accuracy


def analyze_intermediate_agent_performance(dataset_path, output_path, attacker_num):
    """
    Analyze and plot the performance of individual agents in the chain
    """
    print("Analyzing individual agent performance...")
    # This will be SAA evaluation (per-agent accuracy)
    saa_accuracy = evaluate.evaluate(dataset_path, output_path, attacker_num, "SAA")

    import matplotlib.pyplot as plt

    plt.rc("font", family="Comic Sans MS")
    plt.figure(figsize=(10, 6))

    # Plot accuracy per agent
    agent_ids = [f"Agent {i}" for i in range(len(saa_accuracy))]
    attacker_colors = [
        "red" if i < attacker_num else "blue" for i in range(len(saa_accuracy))
    ]
    bars = plt.bar(
        agent_ids,
        saa_accuracy,
        color=attacker_colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add value labels on top of bars
    for bar, value in zip(bars, saa_accuracy):
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
        f"SAA Performance by Agent in Chain Topology (w/ {attacker_num} Attacker(s) + Defense)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Agent ID", fontsize=14, fontweight="bold")
    plt.ylabel("Performance (SAA)", fontsize=14, fontweight="bold")

    # Set y-axis to go from 0 to 1 for accuracy
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return saa_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chain analysis experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="csqa",
        help="Dataset to use (e.g., csqa, gsm8k, fact, bias, adv)",
    )
    parser.add_argument(
        "--attacker_num", type=int, default=1, help="Number of malicious agents."
    )
    args = parser.parse_args()

    # Define experiment parameters
    sample_id = 3
    graph_type = "chain"  # Chain topology for this task
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

    # Run the chain experiment
    output_path = run_chain_experiment(
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

    # Plot MJA (accuracy across iterations)
    mja_accuracy = evaluate_and_plot_chain_accuracy(
        dataset_path=dataset_path,
        output_path=output_path,
        attacker_num=args.attacker_num,
        evaluation_type="MJA",
    )

    # Analyze and plot SAA (accuracy per agent)
    saa_accuracy = analyze_intermediate_agent_performance(
        dataset_path=dataset_path,
        output_path=output_path,
        attacker_num=args.attacker_num,
    )

    print(f"MJA accuracy across iterations: {mja_accuracy}")
    print(f"SAA accuracy per agent: {saa_accuracy}")
