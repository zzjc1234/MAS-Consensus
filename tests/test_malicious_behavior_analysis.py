"""
Tests for the malicious behavior analysis functionality
"""

import pytest
from unittest.mock import patch
from src.mas_consensus import run_malicious_behavior_analysis


def test_task_formatter():
    """Test the task formatter function"""
    data = {
        "question": "What is 2+2?",
        "choices": {"label": ["A", "B"], "text": ["4", "5"]},
        "answerKey": "A",
        "id": "test_id",
    }

    # Test with no attackers
    tasks, task_id = run_malicious_behavior_analysis.task_formatter(data, [], 2)
    assert len(tasks) == 2
    assert task_id == "test_id"

    # Test with one attacker
    tasks, task_id = run_malicious_behavior_analysis.task_formatter(data, [0], 2)
    assert len(tasks) == 2
    assert "defend the given reference answer" in tasks[0]  # Attacker task
    assert "Choose the best answer" in tasks[1]  # Normal task


@patch("src.mas_consensus.util.run_dataset")
def test_run_malicious_behavior_experiments(mock_run_dataset):
    """Test the run_malicious_behavior_experiments function"""
    # Mock return values for run_dataset
    mock_run_dataset.return_value = None

    # Test the function with sample parameters
    _ = run_malicious_behavior_analysis.run_malicious_behavior_experiments(
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

    # Verify that run_dataset was called 4 times (for each scenario)
    assert mock_run_dataset.call_count == 4

    # Check the call arguments for each scenario
    call_args_list = mock_run_dataset.call_args_list

    # First call: Type 1 only
    args, kwargs = call_args_list[0]
    assert kwargs["attacker_idx"] == [0]
    assert kwargs["malicious_auditor_idx"] is None

    # Second call: Type 2 only (malicious auditing)
    args, kwargs = call_args_list[1]
    assert kwargs["attacker_idx"] == [0]
    assert kwargs["malicious_auditor_idx"] == [0, 1]

    # Third call: Type 3 only (malicious voting)
    args, kwargs = call_args_list[2]
    assert kwargs["attacker_idx"] == [0]
    assert (
        kwargs["malicious_auditor_idx"] is None
    )  # Voting happens during the decision process

    # Fourth call: Combined
    args, kwargs = call_args_list[3]
    assert kwargs["attacker_idx"] == [0, 1]
    assert kwargs["malicious_auditor_idx"] == [0]


def test_evaluate_malicious_behavior_impact():
    """Test the evaluation function with mock data"""
    # This is a mock test since we can't run the full evaluation without actual data files
    pass


if __name__ == "__main__":
    pytest.main([__file__])
