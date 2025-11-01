import numpy as np
from tqdm import tqdm
from .base import BaseEvaluation


class AdvEvaluation(BaseEvaluation):
    def evaluate(self):
        accuracy_matrix = []
        for i in tqdm(range(len(self.output))):
            answer_matrix = []
            agent_keys = [k for k in self.output[i].keys() if k.startswith("Agent_")]
            for agent_key in agent_keys:
                answers = []
                history_dialogue = self.output[i][agent_key]
                for msg in history_dialogue:
                    if msg["role"] == "assistant":
                        msg = dict(msg)
                        answers.append(
                            list(
                                msg["content"]["results"][0]["category_scores"].values()
                            )
                        )
                answer_matrix.append(answers)

            answer_matrix = np.array(answer_matrix)
            if self.type == "SAA":
                agent_accuracy = []
                for idx in range(answer_matrix.shape[0]):
                    agent_answers = answer_matrix[idx, :, :]
                    correct_predictions = agent_answers
                    accuracy = correct_predictions
                    agent_accuracy.append(accuracy)
                accuracy_matrix.append(agent_accuracy)
            elif self.type == "MJA":
                pass  # Not implemented in original script

        accuracy_matrix = np.array(accuracy_matrix, dtype=np.float64)
        return np.mean(accuracy_matrix, axis=0)

    def _extract_answer(self, content):
        # Not used, but must be implemented
        return None

    def _get_correct_answer(self, task_id):
        # Not used, but must be implemented
        return None
