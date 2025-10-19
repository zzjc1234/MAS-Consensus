"""
Tests for the chain analysis functionality
"""

import pytest
import numpy as np
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
    assert "Choose the best answer" in tasks[0]
    assert "Choose the best answer" in tasks[1]

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

    # Test the function with sample parameters
    _ = run_chain_analysis.run_chain_experiment(
        ds_name="test_dataset",
        sample_id=3,
        graph_type="chain",
        model="gpt-4o-mini",
        p=16,
        num_agents=6,
        json_format=False,
        turn=9,
        output_dir="./test_output",
    )

    # Verify that run_dataset was called once (for chain experiment)
    assert mock_run_dataset.call_count == 1

    # Verify that the call had the expected parameters
    args, kwargs = mock_run_dataset.call_args
    assert kwargs["attacker_idx"] == [0]
    assert kwargs["num_auditors"] == 2  # Defense mechanism enabled
    assert kwargs["graph_type"] == "chain"


@patch("src.mas_consensus.evaluate.evaluate")
@patch(
    "matplotlib.pyplot.show"
)  # Mock the show function to prevent plots from displaying during tests
def test_evaluate_and_plot_chain_accuracy(mock_plt_show, mock_evaluate):
    """Test the evaluation and plotting function with mock data"""
    # Mock the evaluate function to return some test data
    mock_evaluate.return_value = np.array([0.8, 0.7, 0.9, 0.85, 0.75])

    # Test the evaluation and plotting function
    accuracy = run_chain_analysis.evaluate_and_plot_chain_accuracy(
        dataset_path="./test_dataset.jsonl",
        output_path="./chain.output",
        attacker_num=1,
        evaluation_type="MJA",
    )

    # Verify that evaluate was called once
    assert mock_evaluate.call_count == 1

    # Verify that the returned value is a numpy array
    assert isinstance(accuracy, np.ndarray)


@patch("src.mas_consensus.evaluate.evaluate")
@patch(
    "matplotlib.pyplot.show"
)  # Mock the show function to prevent plots from displaying during tests
def test_analyze_intermediate_agent_performance(mock_plt_show, mock_evaluate):
    """Test the intermediate agent analysis function with mock data"""
    # Mock the evaluate function to return some test data for SAA
    mock_evaluate.return_value = np.array([0.8, 0.7, 0.9, 0.85, 0.75, 0.82])

    # Test the analysis function
    saa_accuracy = run_chain_analysis.analyze_intermediate_agent_performance(
        dataset_path="./test_dataset.jsonl",
        output_path="./chain.output",
        attacker_num=1,
    )

    # Verify that evaluate was called once with SAA
    assert mock_evaluate.call_count == 1

    # Verify that the returned value is a numpy array
    assert isinstance(saa_accuracy, np.ndarray)
    # Verify it has the expected number of agents
    assert len(saa_accuracy) == 6  # 6 agents


def test_imports():
    """Test that all necessary modules can be imported"""
    from src.mas_consensus import run_chain_analysis

    assert hasattr(run_chain_analysis, "task_formatter")
    assert hasattr(run_chain_analysis, "run_chain_experiment")
    assert hasattr(run_chain_analysis, "evaluate_and_plot_chain_accuracy")
    assert hasattr(run_chain_analysis, "analyze_intermediate_agent_performance")


if __name__ == "__main__":
    pytest.main([__file__])
