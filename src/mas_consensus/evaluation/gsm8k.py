import re
from .base import BaseEvaluation


class Gsm8kEvaluation(BaseEvaluation):
    def _extract_answer(self, content):
        return self._extract_numbers(str(content["answer"]).strip())

    def _get_correct_answer(self, task_id):
        return str(self.dataset[task_id]["answer_number"]).strip()

    def _extract_numbers(self, input_string):
        return "".join(re.findall(r"\d", input_string))
