import numpy as np
from tqdm import tqdm
import abc
import json


class BaseEvaluation(abc.ABC):
    def __init__(self, dataset_path, output_path, attacker_num, type):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.attacker_num = attacker_num
        self.type = type
        self.dataset = self._load_dataset()
        self.output = self._load_output()

    def _load_dataset(self):
        dataset = {}
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = eval(line.strip())
                dataset[item.get("id") or item.get("task_id")] = item
        return dataset

    def _load_output(self):
        with open(self.output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # It's not a valid JSON array, but a stream of JSON objects.
        # To parse it, we can wrap it in brackets and add commas.
        separator = "}\n{"
        json_content = f"[{content.replace(separator, '},{')}]"
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            # print(f"Content: {json_content}") # This might be too long
            raise

    @abc.abstractmethod
    def _extract_answer(self, content):
        pass

    @abc.abstractmethod
    def _get_correct_answer(self, task_id):
        pass

    def evaluate(self):
        accuracy_matrix = []
        for i in tqdm(range(len(self.output))):
            answer_matrix = []
            task_id = self.output[i]["task_id"]
            correct = self._get_correct_answer(task_id)
            agent_keys = [k for k in self.output[i].keys() if k.startswith("Agent_")]

            max_answers = 0
            all_answers = []
            for agent_key in agent_keys:
                answers = []
                history_dialogue = self.output[i][agent_key]
                for msg in history_dialogue:
                    if msg["role"] == "assistant":
                        try:
                            pred = self._extract_answer(msg["content"])
                        except Exception:
                            pred = "None"
                        answers.append(pred)
                all_answers.append(answers)
                if len(answers) > max_answers:
                    max_answers = len(answers)

            # Pad the lists
            for answers in all_answers:
                while len(answers) < max_answers:
                    answers.append("None")

            answer_matrix = np.array(all_answers)
            if self.type == "SAA":
                agent_accuracy = []
                for idx in range(answer_matrix.shape[0]):
                    agent_answers = answer_matrix[idx, :]
                    correct_predictions = agent_answers == correct
                    accuracy = correct_predictions
                    agent_accuracy.append(accuracy)
                accuracy_matrix.append(agent_accuracy)
            elif self.type == "MJA":
                turn_accuracy = []
                for turn in range(answer_matrix.shape[1]):
                    turn_answers = answer_matrix[:, turn]
                    correct_predictions = np.sum(turn_answers == correct)
                    accuracy = correct_predictions / len(turn_answers)
                    turn_accuracy.append(accuracy)
                accuracy_matrix.append(turn_accuracy)

        accuracy_matrix = np.array(accuracy_matrix)
        return np.mean(accuracy_matrix, axis=0)
