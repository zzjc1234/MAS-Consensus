import unittest
from mas_consensus.tasks import TaskType, TaskFactory
from mas_consensus.config import TaskConfig


class TestTaskType(unittest.TestCase):

    def test_task_type_values(self):
        self.assertEqual(TaskType.QA.value, "qa")
        self.assertEqual(TaskType.SUMMARIZATION.value, "summarization")


class TestTaskFactory(unittest.TestCase):

    def test_create_qa_task(self):
        config = TaskFactory.create_task(TaskType.QA)

        self.assertIsInstance(config, TaskConfig)
        self.assertIn("[CONTEXT]", config.first_worker_instruction)
        self.assertIn("[QUESTION]", config.first_worker_instruction)
        self.assertIn("[SUMMARY]", config.worker_instruction)
        self.assertIn("[QUESTION]", config.worker_instruction)
        self.assertIn("[SUMMARY]", config.manager_instruction)
        self.assertIn("[QUESTION]", config.manager_instruction)
        self.assertIn("[SUMMARIES]", config.multi_summary_instruction)
        self.assertIn("[QUESTION]", config.multi_summary_instruction)

    def test_create_summarization_task(self):
        config = TaskFactory.create_task(TaskType.SUMMARIZATION)

        self.assertIsInstance(config, TaskConfig)
        self.assertIn("[CONTEXT]", config.first_worker_instruction)
        self.assertIn("[PREVIOUS SUMMARY]", config.worker_instruction)
        self.assertIn("[SUMMARY]", config.manager_instruction)
        self.assertIn("[SUMMARIES]", config.multi_summary_instruction)


if __name__ == "__main__":
    unittest.main()
