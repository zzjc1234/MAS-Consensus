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
    assert "Choose the best answer" in tasks[0]
    assert "Choose the best answer" in tasks[1]

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
        # Simulate time progression for each of the 5 experiments
        # Each experiment takes ~2 seconds (e.g., 0->2, 2->4, 4->6, 6->8, 8->10)
        mock_time.side_effect = [0, 2, 2, 4, 4, 6, 6, 8, 8, 10]

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


@patch(
    "matplotlib.pyplot.show"
)  # Mock the show function to prevent plots from displaying during tests
def test_plot_efficiency_comparison(mock_plt_show):
    """Test the plotting function with mock data"""
    # Create mock execution time data
    mock_execution_times = {
        "baseline": 10.5,
        "type1_only": 11.2,
        "type2_only": 15.8,
        "type3_only": 14.3,
        "combined": 18.7,
    }

    # Test the plotting function
    labels, values, efficiency_ratios = (
        run_efficiency_analysis.plot_efficiency_comparison(mock_execution_times)
    )

    # Verify that we have 5 labels, values, and ratios
    assert len(labels) == 5
    assert len(values) == 5
    assert len(efficiency_ratios) == 5

    # Verify baseline ratio is 1.0 (baseline compared to itself)
    assert efficiency_ratios["Baseline (No Malicious)"] == 1.0

    # Verify all labels are present
    expected_labels = [
        "Baseline (No Malicious)",
        "Type 1 (Regular)",
        "Type 2 (Malicious Auditing)",
        "Type 3 (Malicious Voting)",
        "Combined",
    ]
    for label in expected_labels:
        assert label in labels
        # Verify that the corresponding value matches the execution time for that label
        # Mapping from labels back to execution_times keys
        time_keys_map = {
            "Baseline (No Malicious)": "baseline",
            "Type 1 (Regular)": "type1_only",
            "Type 2 (Malicious Auditing)": "type2_only",
            "Type 3 (Malicious Voting)": "type3_only",
            "Combined": "combined",
        }
        time_key = time_keys_map[label]
        assert values[labels.index(label)] == mock_execution_times[time_key]


def test_imports():
    """Test that all necessary modules can be imported"""
    from src.mas_consensus import run_efficiency_analysis

    assert hasattr(run_efficiency_analysis, "task_formatter")
    assert hasattr(run_efficiency_analysis, "run_efficiency_experiments")
    assert hasattr(run_efficiency_analysis, "plot_efficiency_comparison")


if __name__ == "__main__":
    pytest.main([__file__])
