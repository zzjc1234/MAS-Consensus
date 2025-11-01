import numpy as np
import networkx as nx

from .evaluation.csqa import CsqaEvaluation
from .evaluation.fact import FactEvaluation
from .evaluation.gsm8k import Gsm8kEvaluation
from .evaluation.bias import BiasEvaluation
from .evaluation.adv import AdvEvaluation
from . import methods


def evaluate(dataset_path, output_path, attacker_num, type):
    if "csqa" in dataset_path:
        evaluator = CsqaEvaluation(dataset_path, output_path, attacker_num, type)
    elif "fact" in dataset_path:
        evaluator = FactEvaluation(dataset_path, output_path, attacker_num, type)
    elif "bias" in dataset_path:
        evaluator = BiasEvaluation(dataset_path, output_path, attacker_num, type)
    elif "gsm8k" in dataset_path:
        evaluator = Gsm8kEvaluation(dataset_path, output_path, attacker_num, type)
    elif "adv" in dataset_path:
        evaluator = AdvEvaluation(dataset_path, output_path, attacker_num, type)
    else:
        raise ValueError(f"Unknown dataset: {dataset_path}")

    return evaluator.evaluate()


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
    evaluation = "dynamic_SAA"
    latex = False
    # evaluation = "static_PV"
    sample_ids = [3]
    dataset = "csqa"
    model = "gpt-4o-mini"
    if dataset == "adv":
        model = "gpt-3.5-turbo"
    graph_types = ["chain", "circle", "tree", "star", "complete"][-1:]
    agent_num = 6
    attacker_num = 0
    dataset_path = f"./src/dataset/{dataset}.jsonl"
    for graph_type in graph_types:
        print(f"Graph: {graph_type}_{agent_num}, Attacker Number: {attacker_num}")
        if "static" in evaluation:
            type = evaluation.split("_")[-1]
            metric = static_evaluate(
                methods.generate_adj(agent_num, graph_type),
                list(range(attacker_num + 1)),
                type,
            )
            print(f"Metric ({type}):", metric)
        else:
            type = evaluation.split("_")[-1]
            metrics = []
            for sample_id in sample_ids:
                if dataset == "adv":
                    output_path = f"moderation/{model}/{dataset}/{sample_id}/{dataset}_{graph_type}_{agent_num}_{attacker_num}.output"
                else:
                    output_path = f"src/output/{model}/{dataset}/{sample_id}/{dataset}_{graph_type}_{agent_num}_{attacker_num}.output"
                accuracy = evaluate(dataset_path, output_path, attacker_num, type)
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
            if latex:
                temp = ""
                color = "gray"
                graph_type = graph_type[0].upper() + graph_type[1:]
                temp += f"\rowcolor{{{color}}}!10>\n{{{graph_type}}} &\n"
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

                        # Simplified string construction
                        mean_str = f"${mean[i]}$"
                        arrow_value_str = f"{np.abs(change[i - 1])}"
                        textcolor_str = f"\\textcolor{{{arrow_type}}}{{\\{arrow_type}arrow {arrow_value_str}}}"
                        subscript_str = f"_{{{textcolor_str}}}"
                        final_latex_str = mean_str + subscript_str

                        temp += final_latex_str + (
                            "\\\n" if i == mean.shape[0] - 1 else " &\n"
                        )
                temp = temp.replace("<", "{").replace(">", "}")
                print(temp)
                print("-" * 50)
