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
    num_auditors=2,
    malicious_auditor_idx=None,
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
        num_auditors=num_auditors,  # Enable auditing
        malicious_auditor_idx=malicious_auditor_idx,
        mode_suffix="_defended",
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
        help="Dataset to use (e.g., csqa, gsm8k, fact, bias, adv), or 'all' to run all datasets. Default: csqa",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="complete",
        help="Graph topology to use, or 'all' to run all. Default: complete",
    )
    parser.add_argument(
        "--attacker_num",
        type=int,
        default=1,
        help="Number of malicious agents for the attacked/defended scenarios. Default: 1",
    )
    parser.add_argument(
        "--malicious_auditor_num",
        type=int,
        default=0,
        help="Number of malicious auditors for the defended scenario. Default: 0",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller parameters for a quick test run (turn=1, num_agents=3, sample_id=0).",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=3,
        help="Sample ID to use from the dataset. Default: 3",
    )
    parser.add_argument(
        "--reg_turn", type=int, default=9, help="Number of regulation turns. Default: 9"
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=6,
        help="Number of agents in the simulation. Default: 6",
    )
    parser.add_argument(
        "--num_auditors",
        type=int,
        default=2,
        help="Number of auditor agents (set to 0 to disable auditing). Default: 2",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only run the simulation and skip evaluation.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Only run evaluation on existing results.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of threads. Default: 16",
    )

    args = parser.parse_args()

    if args.evaluation_only and args.skip_evaluation:
        print("Error: --evaluation-only and --skip-evaluation cannot be used together.")
        exit(1)

    # Define experiment parameters
    sample_id = args.sample_id
    model = "gpt-4o-mini"
    json_format = False
    p = args.threads
    reg_turn = args.reg_turn
    num_agents = args.num_agents
    num_auditors = args.num_auditors

    if args.fast:
        reg_turn = 1
        num_agents = 3
        sample_id = 0

    if args.dataset == "all":
        datasets = ["csqa", "gsm8k", "fact", "bias", "adv"]
    else:
        datasets = [args.dataset]

    if args.graph_type == "all":
        graph_types = ["chain", "circle", "tree", "star", "complete"]
    else:
        graph_types = [args.graph_type]

    for ds_name in datasets:
        for graph_type in graph_types:
            print(
                f"--- Running experiment for dataset: {ds_name}, graph: {graph_type} ---"
            )
            # Get dataset-specific configuration
            config = experiment_config.get_dataset_config(ds_name)
            attacker_idx = list(range(args.attacker_num))
            malicious_auditor_idx = list(range(args.malicious_auditor_num))

            output_dir = f"./output/{model}/{ds_name}/{sample_id}"

            if not args.evaluation_only:
                print("Running simulation...")
                os.makedirs(output_dir, exist_ok=True)
                baseline_path, attacked_path, defended_path = run_defense_comparison(
                    ds_name=ds_name,
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
                    num_auditors=num_auditors,
                    malicious_auditor_idx=malicious_auditor_idx,
                )
            else:
                print("Skipping simulation. Assuming result files exist.")
                baseline_path = (
                    f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_0.output"
                )
                attacked_path = f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_{args.attacker_num}.output"
                defended_path = f"{output_dir}/{ds_name}_{graph_type}_{num_agents}_{args.attacker_num}_defended.output"

                if not all(
                    os.path.exists(p)
                    for p in [baseline_path, attacked_path, defended_path]
                ):
                    print(
                        f"Error: Not all output files found in {output_dir} for dataset {ds_name} and graph {graph_type}. Cannot run evaluation."
                    )
                    print(
                        f"Missing one of: \n{baseline_path}\n{attacked_path}\n{defended_path}"
                    )
                    continue

            if not args.skip_evaluation:
                print("Running evaluation...")
                # Evaluate and plot results
                dataset_path = f"./dataset/{ds_name}.jsonl"
                evaluate_and_plot_comparison(
                    dataset_path=dataset_path,
                    baseline_output_path=baseline_path,
                    attacked_output_path=attacked_path,
                    defended_output_path=defended_path,
                    attacker_num=args.attacker_num,
                    evaluation_type="SAA",  # You can change to "MJA" for different evaluation
                )
