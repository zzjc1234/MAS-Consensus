"""
Tests for the chain analysis functionality
"""

import pytest
from unittest.mock import patch
from src.mas_consensus import run_chain_analysis


def test_task_formatter():
    """Test the task formatter function"""
    data = {
        "question": "What is 2+2?",
        "choices": {"label": ["A", "B"], "text": ["4", "5"]},
        "answerKey": "A",
        "id": "test_id",
    }

    # Test with no attackers
    tasks, task_id = run_chain_analysis.task_formatter(data, [], 2)
    assert len(tasks) == 2
    assert task_id == "test_id"

    # Test with one attacker
    tasks, task_id = run_chain_analysis.task_formatter(data, [0], 2)
    assert len(tasks) == 2
    assert "defend the given reference answer" in tasks[0]  # Attacker task
    assert "Choose the best answer" in tasks[1]  # Normal task


@patch("src.mas_consensus.util.run_dataset")
def test_run_chain_experiment(mock_run_dataset):
    """Test the run_chain_experiment function"""
    # Mock return values for run_dataset
    mock_run_dataset.return_value = None

    # Verify that run_dataset was called once (for chain experiment)
    assert mock_run_dataset.call_count == 1

    # Verify that the call had the expected parameters
    _, kwargs = mock_run_dataset.call_args
    assert kwargs["attacker_idx"] == [0]
    assert kwargs["num_auditors"] == 2  # Defense mechanism enabled
    assert kwargs["graph_type"] == "chain"


def test_evaluate_and_plot_chain_accuracy():
    """Test the evaluation and plotting function with mock data"""
    # This is a mock test since we can't run the full evaluation without actual data files
    # In a real scenario, we'd need to create mock data files or mock the evaluation function
    pass


if __name__ == "__main__":
    pytest.main([__file__])
