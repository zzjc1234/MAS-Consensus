import os
import json
import argparse
import networkx as nx
from tqdm import tqdm
import numpy as np
from . import methods
from .evaluation import (
    evaluate_csqa,
    evaluate_gsm8k,
    evaluate_fact,
    evaluate_bias,
    evaluate_adv,
)


def evaluate(dataset_path, output_path, attacker_num, auditor_num, type):
    if "csqa" in dataset_path:
        accuracy = evaluate_csqa(
            dataset_path, output_path, attacker_num, auditor_num, type
        )
    if "fact" in dataset_path:
        accuracy = evaluate_fact(dataset_path, output_path, attacker_num, type)
    if "bias" in dataset_path:
        accuracy = evaluate_bias(dataset_path, output_path, attacker_num, type)
    if "gsm8k" in dataset_path:
        accuracy = evaluate_gsm8k(dataset_path, output_path, attacker_num, type)
    if "adv" in dataset_path:
        accuracy = evaluate_adv(output_path, attacker_num, type)
    return accuracy


def static_evaluate(adj_matrix, attacker_idx, type):
    if type == "NE":
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
        length_dict = dict(nx.all_pairs_shortest_path_length(G))
        n = len(G.nodes)
        total_efficiency = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    d_ij = length_dict[i].get(j, float("inf"))
                    if d_ij != float("inf"):  # 确保节点之间是可达的
                        total_efficiency += 1 / d_ij
        efficiency = total_efficiency / (n * (n - 1))
        return efficiency
    if type == "EC":
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight=None)

        return eigenvector_centrality
    if type == "PV":
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        n = len(G.nodes)
        apv_count = 0
        for s in G.nodes:
            for t in G.nodes:
                if s != t:
                    try:
                        paths = list(nx.all_shortest_paths(G, source=s, target=t))
                        for path in paths:
                            for i in range(len(path) - 1):
                                if path[i] in attacker_idx:
                                    apv_count += 1
                                    break
                    except nx.NetworkXNoPath:
                        continue
        apv = apv_count / (n * (n - 1))
        return apv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument(
        "--evaluation",
        type=str,
        default="dynamic_MJA",
        help="Evaluation type: dynamic_SAA, dynamic_MJA, or static_*",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="csqa",
        help="Dataset name: csqa, gsm8k, fact, bias, adv",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of datasets to aggregate across (e.g., csqa gsm8k fact bias adv). If omitted and --aggregate_by_topology is set, will auto-detect from src/output/{model}/*",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults based on dataset)",
    )
    parser.add_argument(
        "--sample_ids",
        type=int,
        nargs="+",
        default=[3],
        help="Sample IDs to evaluate",
    )
    parser.add_argument(
        "--graph_types",
        type=str,
        nargs="+",
        default=["chain"],
        help="Graph types to evaluate",
    )
    parser.add_argument(
        "--agent_num",
        type=int,
        default=6,
        help="Number of agents",
    )
    parser.add_argument(
        "--attacker_num",
        type=int,
        default=1,
        help="Number of attackers",
    )
    parser.add_argument(
        "--auditor_num",
        type=int,
        default=2,
        help="Number of auditors",
    )
    parser.add_argument(
        "--type",
        type=int,
        default=None,
        help="Output file type suffix (e.g., 1 for _type1, None for baseline)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Custom suffix for output files (e.g., _both, _test). If provided, overrides --type",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output LaTeX format",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Direct path to output file to evaluate (bypasses automatic path construction)",
    )
    parser.add_argument(
        "--aggregate_by_topology",
        action="store_true",
        help="When set, compute per-turn averages for each topology across multiple datasets",
    )
    
    args = parser.parse_args()
    
    evaluation = args.evaluation
    latex = args.latex
    file_path = args.file_path
    suffix = args.suffix
    
    # If file_path is provided, evaluate that file directly
    if file_path:
        dataset = args.dataset
        attacker_num = args.attacker_num
        auditor_num = args.auditor_num
        dataset_path = f"src/dataset/{dataset}.jsonl"
        eval_type = evaluation.split("_")[-1]
        
        print(f"Evaluating file: {file_path}")
        try:
            accuracy = evaluate(dataset_path, file_path, attacker_num, auditor_num, eval_type)
            # Wrap in array to match normal mode structure (1 sample)
            metrics = np.array([accuracy])  # shape: (1, num_turns)
            mean = np.mean(metrics, axis=0)  # same as accuracy, but consistent with normal mode
            variance = np.var(metrics, axis=0)  # variance of 1 sample = 0
            if dataset != "adv":
                mean = np.round(100 * mean, 2)
            change = np.round(mean[:-1] - mean[1:], 2) if len(mean) > 1 else np.array([])
            print("Mean", mean)
            if len(change) > 0:
                print("Change", change)
            print("Variance", variance)
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
        except Exception as e:
            print(f"Error evaluating file: {e}")
            import traceback
            traceback.print_exc()
        exit(0)
    
    # Otherwise, use automatic path construction
    sample_ids = args.sample_ids
    dataset = args.dataset
    model = args.model
    if model is None:
        if dataset == "adv":
            model = "gpt-3.5-turbo"
        else:
            model = "gpt-4o-mini"
    graph_types = args.graph_types
    agent_num = args.agent_num
    attacker_num = args.attacker_num
    auditor_num = args.auditor_num
    output_type = args.type
    dataset_path = f"src/dataset/{dataset}.jsonl"
    
    # Aggregate per-turn averages across datasets for each topology
    if args.aggregate_by_topology and "static" not in evaluation:
        eval_type = evaluation.split("_")[-1]

        # Determine dataset list
        datasets_to_use = args.datasets
        if datasets_to_use is None:
            model_root = os.path.join("src", "output", model)
            try:
                datasets_to_use = sorted(
                    [
                        name
                        for name in os.listdir(model_root)
                        if os.path.isdir(os.path.join(model_root, name))
                    ]
                )
            except FileNotFoundError:
                print(f"Output path not found for model: {model_root}")
                datasets_to_use = []

        # Known datasets in this repo
        known_datasets = {"csqa", "gsm8k", "fact", "bias", "adv"}
        datasets_to_use = [d for d in datasets_to_use if d in known_datasets]
        if len(datasets_to_use) == 0:
            print("No valid datasets found to aggregate. Provide --datasets or check output directory.")
            exit(1)

        print(f"Aggregating across datasets: {datasets_to_use}")

        for graph_type in graph_types:
            print(f"\n[Aggregate] Graph: {graph_type}_{agent_num}, Attacker Number: {attacker_num}")
            per_dataset_means = []
            min_turns = None

            for ds in datasets_to_use:
                if ds == "adv" and eval_type == "MJA":
                    # MJA not implemented for adv
                    print(f"Skipping dataset '{ds}' for MJA aggregation (unsupported).")
                    continue

                ds_metrics = []
                for sample_id in sample_ids:
                    base_filename = f"{ds}_{graph_type}_{agent_num}_{attacker_num}"
                    if suffix is not None:
                        base_filename += suffix
                    elif output_type is not None:
                        base_filename += f"_type{output_type}"
                    base_filename += ".output"
                    output_path = f"src/output/{model}/{ds}/{sample_id}/{base_filename}"
                    tqdm.write("evaluating file: " + output_path)
                    try:
                        ds_dataset_path = f"src/dataset/{ds}.jsonl"
                        acc = evaluate(ds_dataset_path, output_path, attacker_num, auditor_num, eval_type)
                        ds_metrics.append(acc)
                    except FileNotFoundError:
                        print(f"File not found, skipping: {output_path}")
                    except Exception as e:
                        print(f"Error evaluating {output_path}: {e}")

                if len(ds_metrics) == 0:
                    print(f"No metrics collected for dataset '{ds}', skipping.")
                    continue

                ds_metrics = np.array(ds_metrics)
                ds_mean = np.mean(ds_metrics, axis=0)

                # Normalize scaling (percentage) for non-adv datasets
                if ds != "adv":
                    ds_mean = 100.0 * ds_mean

                # Track min turn count to align
                if min_turns is None:
                    try:
                        min_turns = ds_mean.shape[0]
                    except Exception:
                        min_turns = 1
                else:
                    try:
                        min_turns = min(min_turns, ds_mean.shape[0])
                    except Exception:
                        min_turns = min(min_turns, 1)

                per_dataset_means.append(ds_mean)

            if len(per_dataset_means) == 0:
                print("No datasets produced metrics for this topology.")
                continue

            # Trim all arrays to min_turns for safe stacking
            trimmed = []
            for arr in per_dataset_means:
                if np.ndim(arr) == 0:
                    trimmed.append(np.array([arr])[:min_turns])
                else:
                    trimmed.append(np.array(arr)[:min_turns])
            stacked = np.stack(trimmed, axis=0)  # shape: (num_datasets, min_turns, ...)

            # Average across datasets (axis=0)
            topo_mean = np.mean(stacked, axis=0)

            # If the result is per-turn vector, compute change
            if topo_mean.ndim >= 1:
                try:
                    change = np.round(topo_mean[:-1] - topo_mean[1:], 2)
                except Exception:
                    change = None
            else:
                change = None

            try:
                topo_mean_display = np.round(topo_mean, 2)
            except Exception:
                topo_mean_display = topo_mean

            print("Mean (per turn across datasets)", topo_mean_display)
            if change is not None and getattr(change, "size", 0) > 0:
                print("Change", change)

            # Write results to file per graph type
            try:
                eval_dir = os.path.join("src", "evaluation")
                os.makedirs(eval_dir, exist_ok=True)
                outfile = os.path.join(
                    eval_dir,
                    f"{model}_{evaluation}_aggregate_{graph_type}_{agent_num}_{attacker_num}.json",
                )
                payload = {
                    "model": model,
                    "evaluation": evaluation,
                    "graph_type": graph_type,
                    "agent_num": agent_num,
                    "attacker_num": attacker_num,
                    "datasets": datasets_to_use,
                    "sample_ids": sample_ids,
                    "mean": (topo_mean_display.tolist() if hasattr(topo_mean_display, "tolist") else topo_mean_display),
                    "change": (change.tolist() if change is not None and hasattr(change, "tolist") else change),
                    "note": "Mean per turn across datasets",
                }
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"Wrote aggregate results to: {outfile}")
            except Exception as e:
                print(f"Failed to write aggregate results file: {e}")

        # End aggregation run
        exit(0)
    
    for graph_type in graph_types:
        print(f"Graph: {graph_type}_{agent_num}, Attacker Number: {attacker_num}")
        if "static" in evaluation:
            eval_type = evaluation.split("_")[-1]
            metric = static_evaluate(
                methods.generate_adj(agent_num, graph_type),
                list(range(attacker_num + 1)),
                eval_type,
            )
            print(f"Metric ({eval_type}):", metric)
        else:
            eval_type = evaluation.split("_")[-1]
            metrics = []
            for sample_id in sample_ids:
                # Build output path with suffix or type suffix if provided
                base_filename = f"{dataset}_{graph_type}_{agent_num}_{attacker_num}"
                if suffix is not None:
                    base_filename += suffix
                elif output_type is not None:
                    base_filename += f"_type{output_type}"
                base_filename += ".output"
                
                # All datasets use src/output directory (including adv)
                output_path = f"src/output/{model}/{dataset}/{sample_id}/{base_filename}"
                tqdm.write("evaluating file: " + output_path)
                accuracy = evaluate(
                    dataset_path, output_path, attacker_num, auditor_num, eval_type
                )
                metrics.append(accuracy)
            metrics = np.array(metrics)
            mean = np.mean(metrics, axis=0)
            variance = np.var(metrics, axis=0)
            if dataset != "adv":
                mean = np.round(100 * mean, 2)
            change = np.round(mean[:-1] - mean[1:], 2)
            print("Mean", mean)
            print("Change", change)
            print("Variance", variance)

            # Write results to file per graph type (normal mode)
            try:
                eval_dir = os.path.join("src", "evaluation")
                os.makedirs(eval_dir, exist_ok=True)
                outfile = os.path.join(
                    eval_dir,
                    f"{model}_{evaluation}_{dataset}_{graph_type}_{agent_num}_{attacker_num}.json",
                )
                payload = {
                    "model": model,
                    "evaluation": evaluation,
                    "dataset": dataset,
                    "graph_type": graph_type,
                    "agent_num": agent_num,
                    "attacker_num": attacker_num,
                    "sample_ids": sample_ids,
                    "mean": (mean.tolist() if hasattr(mean, "tolist") else mean),
                    "change": (change.tolist() if hasattr(change, "tolist") else change),
                    "variance": (variance.tolist() if hasattr(variance, "tolist") else variance),
                }
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"Wrote results to: {outfile}")
            except Exception as e:
                print(f"Failed to write results file: {e}")
            if latex:
                temp = ""
                color = "gray"
                graph_type_display = graph_type[0].upper() + graph_type[1:]
                temp += f"\\rowcolor<{color}!10>\n{graph_type_display} &\n"
                for i in range(mean.shape[0]):
                    if i == 0:
                        temp += f"${mean[i]}$ &\n"
                    else:
                        if change[i - 1] > 0:
                            arrow_type = "down"
                        if change[i - 1] < 0:
                            arrow_type = "up"
                        if change[i - 1] == 0:
                            arrow_type = "right"
                        temp += f"${mean[i]}_<\\textcolor<{arrow_type}><\\{arrow_type}arrow {np.abs(change[i - 1])}>>$ "
                        temp += "\\\\" if i == mean.shape[0] - 1 else "&\n"
                temp = temp.replace("<", "{").replace(">", "}")
                print(temp)
                print("-" * 50)
