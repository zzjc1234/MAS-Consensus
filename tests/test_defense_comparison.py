"""
Tests for the defense comparison functionality
"""

import pytest
import numpy as np
from unittest.mock import patch
from src.mas_consensus import run_defense_comparison


def test_task_formatter():
    """Test the task formatter function"""
    data = {
        "question": "What is 2+2?",
        "choices": {"label": ["A", "B"], "text": ["4", "5"]},
        "answerKey": "A",
        "id": "test_id",
    }

    # Test with no attackers
    tasks, task_id = run_defense_comparison.task_formatter(data, [], 2)
    assert len(tasks) == 2
    assert task_id == "test_id"
    assert "Choose the best answer" in tasks[0]
    assert "Choose the best answer" in tasks[1]

    # Test with one attacker
    tasks, task_id = run_defense_comparison.task_formatter(data, [0], 2)
    assert len(tasks) == 2
    assert "defend the given reference answer" in tasks[0]  # Attacker task
    assert "Choose the best answer" in tasks[1]  # Normal task


@patch("src.mas_consensus.util.run_dataset")
def test_run_defense_comparison(mock_run_dataset):
    """Test the run_defense_comparison function"""
    # Mock return values for run_dataset
    mock_run_dataset.return_value = None

    # Test the function with sample parameters
    baseline_path, attacked_path, defended_path = (
        run_defense_comparison.run_defense_comparison(
            ds_name="test_dataset",
            sample_id=3,
            graph_type="complete",
            model="gpt-4o-mini",
            p=16,
            num_agents=6,
            json_format=False,
            turn=9,
            output_dir="./test_output",
        )
    )

    # Verify that run_dataset was called 3 times (for baseline, attacked, and defended)
    assert mock_run_dataset.call_count == 3

    # Verify first call was baseline (no attackers, no auditors)
    args, kwargs = mock_run_dataset.call_args_list[0]
    assert kwargs["attacker_idx"] == []
    assert kwargs["num_auditors"] == 0

    # Verify second call was attacked (with 1 attacker, no auditors)
    args, kwargs = mock_run_dataset.call_args_list[1]
    assert kwargs["attacker_idx"] == [0]
    assert kwargs["num_auditors"] == 0

    # Verify third call was defended (with 1 attacker and 2 auditors)
    args, kwargs = mock_run_dataset.call_args_list[2]
    assert kwargs["attacker_idx"] == [0]
    assert kwargs["num_auditors"] == 2


@patch("src.mas_consensus.evaluate.evaluate")
@patch(
    "matplotlib.pyplot.show"
)  # Mock the show function to prevent plots from displaying during tests
def test_evaluate_and_plot_comparison(mock_plt_show, mock_evaluate):
    """Test the evaluation and plotting function with mock data"""
    # Mock the evaluate function to return some test data
    mock_evaluate.return_value = np.array([0.8, 0.7, 0.9, 0.85, 0.75])

    # Test the evaluation and plotting function
    baseline_acc, attacked_acc, defended_acc = (
        run_defense_comparison.evaluate_and_plot_comparison(
            dataset_path="./test_dataset.jsonl",
            baseline_output_path="./baseline.output",
            attacked_output_path="./attacked.output",
            defended_output_path="./defended.output",
            attacker_num=1,
            evaluation_type="SAA",
        )
    )

    # Verify that evaluate was called 3 times
    assert mock_evaluate.call_count == 3

    # Verify that the returned values are numpy arrays
    assert isinstance(baseline_acc, np.ndarray)
    assert isinstance(attacked_acc, np.ndarray)
    assert isinstance(defended_acc, np.ndarray)


def test_imports():
    """Test that all necessary modules can be imported"""
    from src.mas_consensus import run_defense_comparison

    assert hasattr(run_defense_comparison, "task_formatter")
    assert hasattr(run_defense_comparison, "run_defense_comparison")
    assert hasattr(run_defense_comparison, "evaluate_and_plot_comparison")


if __name__ == "__main__":
    pytest.main([__file__])
