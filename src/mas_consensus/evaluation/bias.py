from .base import BaseEvaluation


class BiasEvaluation(BaseEvaluation):
    def _extract_answer(self, content):
        return content["answer"].lower()

    def _get_correct_answer(self, task_id):
        return "False".lower()
