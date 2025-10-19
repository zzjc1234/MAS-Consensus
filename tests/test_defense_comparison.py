"""
Tests for the defense comparison functionality
"""

import pytest
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

    # Verify first call was baseline (no attackers)
    args, kwargs = mock_run_dataset.call_args_list[0]
    assert kwargs["attacker_idx"] == []

    # Verify second call was attacked (with 1 attacker)
    args, kwargs = mock_run_dataset.call_args_list[1]
    assert kwargs["attacker_idx"] == [0]

    # Verify third call was defended (with 1 attacker and auditors)
    args, kwargs = mock_run_dataset.call_args_list[2]
    assert kwargs["attacker_idx"] == [0]
    assert kwargs["num_auditors"] == 2


def test_evaluate_and_plot_comparison():
    """Test the evaluation and plotting function with mock data"""
    # This is a mock test since we can't run the full evaluation without actual data files
    # In a real scenario, we'd need to create mock data files or mock the evaluation function
    pass


if __name__ == "__main__":
    pytest.main([__file__])
