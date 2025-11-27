from tqdm import tqdm
import numpy as np


def evaluate_csqa(dataset_path, output_path, attacker_num, auditor_num, type):
    def extract_first_uppercase(input_string):
        for char in input_string:
            if char.isupper():
                return char

    dataset = {}
    output = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        temp = f.readlines()
        for item in temp:
            item = eval(item.strip())
            dataset[item["id"]] = item

    with open(output_path, "r", encoding="utf-8") as f:
        temp = f.readlines()
        for item in temp:
            output.append(eval(item.strip()))

    accuracy_matrix = []
    for i in tqdm(range(len(output))):
        answer_matrix = []
        task_id = output[i]["task_id"]
        
        # Skip questions not in dataset or with no answer key
        if task_id not in dataset:
            print(f"Skipping question {task_id}: Not found in dataset")
            continue
        
        correct = dataset[task_id]["answerKey"]
        if not correct or correct == "":
            print(f"Skipping question {task_id}: No answer key provided")
            continue
            
        # Get all Agent_ keys (excludes Auditor_ keys)
        # We evaluate ALL agents including attackers - that's the point of measuring accuracy
        agent_list = [k for k in output[i].keys() if k.startswith("Agent_")]

        for agent_key in agent_list:
            answers = []
            history_dialogue = output[i][agent_key]
            for msg in history_dialogue:
                if msg["role"] == "assistant":
                    try:
                        pred = extract_first_uppercase(msg["content"]["answer"])
                    except Exception:
                        pred = "None"
                    answers.append(pred)
            answer_matrix.append(answers)
            print(f"Agent: {agent_key}, Answers: {answers}, Count: {len(answers)}")
        answer_matrix = np.array(answer_matrix)
        if type == "SAA":
            agent_accuracy = []
            for idx in range(answer_matrix.shape[0]):
                agent_answers = answer_matrix[idx, :]
                correct_predictions = agent_answers == correct
                accuracy = correct_predictions
                agent_accuracy.append(accuracy)
            accuracy_matrix.append(agent_accuracy)
        if type == "MJA":
            turn_accuracy = []
            for turn in range(answer_matrix.shape[1]):
                turn_answers = answer_matrix[:, turn]
                correct_predictions = np.sum(turn_answers == correct)
                accuracy = correct_predictions / len(turn_answers)
                turn_accuracy.append(accuracy)
            accuracy_matrix.append(turn_accuracy)
    accuracy_matrix = np.array(accuracy_matrix)
    return np.mean(accuracy_matrix, axis=0)

