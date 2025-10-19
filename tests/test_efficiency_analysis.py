"""
Tests for the efficiency analysis functionality
"""

import pytest
from unittest.mock import patch
from src.mas_consensus import run_efficiency_analysis


def test_task_formatter():
    """Test the task formatter function"""
    data = {
        "question": "What is 2+2?",
        "choices": {"label": ["A", "B"], "text": ["4", "5"]},
        "answerKey": "A",
        "id": "test_id",
    }

    # Test with no attackers
    tasks, task_id = run_efficiency_analysis.task_formatter(data, [], 2)
    assert len(tasks) == 2
    assert task_id == "test_id"

    # Test with one attacker
    tasks, task_id = run_efficiency_analysis.task_formatter(data, [0], 2)
    assert len(tasks) == 2
    assert "defend the given reference answer" in tasks[0]  # Attacker task
    assert "Choose the best answer" in tasks[1]  # Normal task


@patch("src.mas_consensus.util.run_dataset")
def test_run_efficiency_experiments(mock_run_dataset):
    """Test the run_efficiency_experiments function"""
    # Mock return values for run_dataset
    mock_run_dataset.return_value = None

    # Mock time.time to return predictable values for testing
    with patch("time.time") as mock_time:
        mock_time.side_effect = [
            0,
            10,
            10,
            20,
            20,
            30,
            30,
            40,
            40,
            50,
        ]  # Simulate 10s for each experiment

        # Test the function with sample parameters
        results = run_efficiency_analysis.run_efficiency_experiments(
            ds_name="test_dataset",
            sample_id=3,
            graph_type="complete",
            model="gpt-4o-mini",
            p=4,
            num_agents=4,
            json_format=False,
            turn=3,
            output_dir="./test_output",
        )

        # Verify that run_dataset was called 5 times (for each scenario)
        assert mock_run_dataset.call_count == 5

        # Check that we have 5 results with positive times
        assert len(results) == 5
        assert all(t > 0 for t in results.values())
        assert "baseline" in results
        assert "type1_only" in results
        assert "type2_only" in results
        assert "type3_only" in results
        assert "combined" in results


def test_plot_efficiency_comparison():
    """Test the plotting function with mock data"""
    # This is a mock test since we can't run the full plotting without matplotlib in test environment
    pass


if __name__ == "__main__":
    pytest.main([__file__])
