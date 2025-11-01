from .base import BaseEvaluation


class FactEvaluation(BaseEvaluation):
    def _extract_answer(self, content):
        return "true" if "true" in content["answer"].lower() else "false"

    def _get_correct_answer(self, task_id):
        return "True".lower()
