from tqdm import tqdm
import numpy as np


def evaluate_adv(output_path, attacker_num, type):
    output = []

    with open(output_path, "r", encoding="utf-8") as f:
        temp = f.readlines()
        for item in temp:
            output.append(eval(item.strip()))

    accuracy_matrix = []
    for i in tqdm(range(len(output))):
        answer_matrix = []
        output[i]["task_id"]
        # Get all Agent_ keys (excludes Auditor_ keys)
        # We evaluate ALL agents including attackers - that's the point of measuring accuracy
        agent_list = [k for k in output[i].keys() if k.startswith("Agent_")]

        for agent_key in agent_list:
            answers = []
            history_dialogue = output[i][agent_key]
            for msg in history_dialogue:
                msg = dict(msg)
                answers.append(list(msg["results"][0]["category_scores"].values()))
            answer_matrix.append(answers)
        answer_matrix = np.array(answer_matrix)
        if type == "SAA":
            agent_accuracy = []
            for idx in range(answer_matrix.shape[0]):
                agent_answers = answer_matrix[idx, :]
                correct_predictions = agent_answers
                accuracy = correct_predictions
                agent_accuracy.append(accuracy)
            accuracy_matrix.append(agent_accuracy)
        if type == "MJA":
            pass
    accuracy_matrix = np.array(accuracy_matrix, dtype=np.float64)
    return np.mean(accuracy_matrix, axis=0)

