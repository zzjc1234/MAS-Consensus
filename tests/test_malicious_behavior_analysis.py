"""
Tests for the malicious behavior analysis functionality
"""

import pytest
import numpy as np
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
    assert "Choose the best answer" in tasks[0]
    assert "Choose the best answer" in tasks[1]

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


@patch("src.mas_consensus.evaluate.evaluate")
@patch(
    "matplotlib.pyplot.show"
)  # Mock the show function to prevent plots from displaying during tests
def test_evaluate_malicious_behavior_impact(mock_plt_show, mock_evaluate):
    """Test the evaluation function with mock data"""
    # Mock the evaluate function to return test data for each scenario
    mock_evaluate.side_effect = [
        np.array([0.8, 0.7, 0.9]),  # type1_only
        np.array([0.7, 0.6, 0.8]),  # type2_only
        np.array([0.75, 0.65, 0.85]),  # type3_only
        np.array([0.6, 0.5, 0.7]),  # combined
    ]

    # Create mock results dictionary
    mock_results = {
        "type1_only": "./type1.output",
        "type2_only": "./type2.output",
        "type3_only": "./type3.output",
        "combined": "./combined.output",
    }

    # Test the evaluation function
    accuracies = run_malicious_behavior_analysis.evaluate_malicious_behavior_impact(
        dataset_path="./test_dataset.jsonl",
        experiment_results=mock_results,
        evaluation_type="SAA",
    )

    # Verify that evaluate was called 4 times
    assert mock_evaluate.call_count == 4

    # Verify that all scenarios are in the results
    assert "type1_only" in accuracies
    assert "type2_only" in accuracies
    assert "type3_only" in accuracies
    assert "combined" in accuracies


@patch(
    "matplotlib.pyplot.show"
)  # Mock the show function to prevent plots from displaying during tests
def test_plot_malicious_behavior_comparison(mock_plt_show):
    """Test the plotting function with mock data"""
    # Create mock accuracy data
    mock_accuracies = {
        "type1_only": np.array([0.8, 0.7, 0.9]),
        "type2_only": np.array([0.7, 0.6, 0.8]),
        "type3_only": np.array([0.75, 0.65, 0.85]),
        "combined": np.array([0.6, 0.5, 0.7]),
    }

    # Test the plotting function
    labels, values = run_malicious_behavior_analysis.plot_malicious_behavior_comparison(
        mock_accuracies
    )

    # Verify that we have 4 labels and values
    assert len(labels) == 4
    assert len(values) == 4
    assert "Type 1 (Regular)" in labels
    assert "Type 2 (Malicious Auditing)" in labels
    assert "Type 3 (Malicious Voting)" in labels
    assert "Combined" in labels


def test_imports():
    """Test that all necessary modules can be imported"""
    from src.mas_consensus import run_malicious_behavior_analysis

    assert hasattr(run_malicious_behavior_analysis, "task_formatter")
    assert hasattr(
        run_malicious_behavior_analysis, "run_malicious_behavior_experiments"
    )
    assert hasattr(
        run_malicious_behavior_analysis, "evaluate_malicious_behavior_impact"
    )
    assert hasattr(
        run_malicious_behavior_analysis, "plot_malicious_behavior_comparison"
    )


if __name__ == "__main__":
    pytest.main([__file__])
