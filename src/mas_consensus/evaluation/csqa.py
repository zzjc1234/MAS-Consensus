from .base import BaseEvaluation


class CsqaEvaluation(BaseEvaluation):
    def _extract_answer(self, content):
        return self._extract_first_uppercase(content["answer"])

    def _get_correct_answer(self, task_id):
        return self.dataset[task_id]["answerKey"]

    def _extract_first_uppercase(self, input_string):
        for char in input_string:
            if char.isupper():
                return char
        return "None"
