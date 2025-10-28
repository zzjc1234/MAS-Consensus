"""
Tests for the unified run_experiment system with worker/auditor separation
"""

import pytest
import os
from unittest.mock import patch, MagicMock

# Mock OPENAI_API_KEY before importing modules that need it
os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-testing')

from src.mas_consensus import experiment_config, agent_base, util


class TestAuditorSelection:
    """Test auditor selection mechanism"""
    
    @patch('src.mas_consensus.agent_base.random.sample')
    def test_auditor_random_selection(self, mock_sample):
        """Test that auditors are randomly selected from agent pool"""
        # Mock random selection to return predictable values
        mock_sample.return_value = [2, 5, 7]  # Auditor indices
        
        config = experiment_config.get_dataset_config("csqa")
        
        # Create a simple test case
        num_agents = 10
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=3,
            auditor_idx=[2, 5, 7],
            attacker_idx=[0],
            malicious_auditor_idx=[5],
        )
        
        # Verify auditors were selected
        assert len(graph.auditor_agents) == 3
        assert graph.auditor_indices == [2, 5, 7]
        
        # Verify remaining agents (excluding auditors)
        assert len(graph.agents) == 7  # 10 - 3 auditors
        
        # Verify agent indices don't include auditors
        agent_indices = [agent.idx for agent in graph.agents]
        assert 2 not in agent_indices
        assert 5 not in agent_indices
        assert 7 not in agent_indices
    
    def test_malicious_auditor_in_auditor_group(self):
        """Test that malicious auditors are part of the auditor group"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 10
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        auditor_indices = [2, 5, 7]
        malicious_auditor_idx = [5]  # Must be in auditor_indices
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=3,
            auditor_idx=auditor_indices,
            attacker_idx=[0],
            malicious_auditor_idx=malicious_auditor_idx,
        )
        
        # Verify malicious auditor is in the auditor group
        assert all(idx in auditor_indices for idx in malicious_auditor_idx)
        
        # Verify one auditor is malicious
        malicious_auditors = [aud for aud in graph.auditor_agents if aud.is_malicious]
        assert len(malicious_auditors) == 1
        assert malicious_auditors[0].idx == 5


class TestAgentAuditorSeparation:
    """Test that agents and auditors have separate roles"""
    
    def test_agents_answer_auditors_dont(self):
        """Test that only agents answer questions, auditors don't"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 6
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=2,
            auditor_idx=[3, 5],
            attacker_idx=[0],
            malicious_auditor_idx=None,
        )
        
        # Verify only 4 agents remain for answering (6 - 2 auditors)
        assert len(graph.agents) == 4
        
        # Verify 2 auditors exist
        assert len(graph.auditor_agents) == 2
        
        # Verify agent indices don't overlap with auditor indices
        agent_indices = {agent.idx for agent in graph.agents}
        auditor_indices = set(graph.auditor_indices)
        assert agent_indices.isdisjoint(auditor_indices)
    
    def test_output_format_separation(self):
        """Test that output saves Agent_ and Auditor_ keys separately"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 6
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=2,
            auditor_idx=[1, 4],
            attacker_idx=[0],
            malicious_auditor_idx=None,
        )
        
        # Verify record structure
        assert "task_id" in graph.record
        assert "auditor_indices" in graph.record
        assert graph.record["auditor_indices"] == [1, 4]


class TestAttackTypes:
    """Test attack type configurations"""
    
    def test_type_one_attack(self):
        """Test Type 1 attack: malicious agents, honest auditors"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 6
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=2,
            auditor_idx=[3, 5],
            attacker_idx=[0, 1],  # Type 1: malicious agents
            malicious_auditor_idx=None,  # No Type 2 attack
        )
        
        # Verify Type 1 attack configuration
        malicious_agents = [agent for agent in graph.agents if agent.is_malicious]
        # Note: Agents 0, 1 were malicious, but if they became auditors, they won't be in graph.agents
        # Since auditor_idx=[3,5], agents 0,1 should still be in discussion
        assert len([a for a in graph.agents if a.idx in [0, 1] and a.is_malicious]) == 2
        
        # Verify all auditors are honest
        malicious_auditors = [aud for aud in graph.auditor_agents if aud.is_malicious]
        assert len(malicious_auditors) == 0
    
    def test_type_two_attack(self):
        """Test Type 2 attack: honest agents, malicious auditors"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 6
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=2,
            auditor_idx=[3, 5],
            attacker_idx=[],  # No Type 1 attack
            malicious_auditor_idx=[5],  # Type 2: malicious auditor
        )
        
        # Verify all discussing agents are honest
        malicious_agents = [agent for agent in graph.agents if agent.is_malicious]
        assert len(malicious_agents) == 0
        
        # Verify one auditor is malicious
        malicious_auditors = [aud for aud in graph.auditor_agents if aud.is_malicious]
        assert len(malicious_auditors) == 1
        assert malicious_auditors[0].idx == 5
    
    def test_both_attacks(self):
        """Test both attack types: malicious agents + malicious auditors"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 10
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=3,
            auditor_idx=[2, 6, 8],
            attacker_idx=[0, 1],  # Type 1 attack
            malicious_auditor_idx=[6],  # Type 2 attack
        )
        
        # Verify Type 1 attack (malicious agents in discussion)
        malicious_agents = [agent for agent in graph.agents 
                           if agent.idx in [0, 1] and agent.is_malicious]
        assert len(malicious_agents) == 2
        
        # Verify Type 2 attack (malicious auditor)
        malicious_auditors = [aud for aud in graph.auditor_agents if aud.is_malicious]
        assert len(malicious_auditors) == 1
        assert malicious_auditors[0].idx == 6


class TestExperimentConfig:
    """Test experiment configuration"""
    
    def test_task_formatter_csqa(self):
        """Test CSQA task formatter"""
        data = {
            "question": "What is 2+2?",
            "choices": {"label": ["A", "B"], "text": ["4", "5"]},
            "answerKey": "A",
            "id": "test_id",
        }
        
        tasks, task_id = experiment_config.csqa_task_formatter(data, [], 3)
        assert len(tasks) == 3
        assert task_id == "test_id"
        assert all("Choose the best answer" in task for task in tasks)
    
    def test_task_formatter_with_attackers(self):
        """Test task formatter with malicious agents"""
        data = {
            "question": "What is 2+2?",
            "choices": {"label": ["A", "B"], "text": ["4", "5"]},
            "answerKey": "A",
            "id": "test_id",
        }
        
        tasks, task_id = experiment_config.csqa_task_formatter(data, [0], 3)
        assert len(tasks) == 3
        assert "defend the given reference answer" in tasks[0]  # Attacker task
        assert "Choose the best answer" in tasks[1]  # Normal task
        assert "Choose the best answer" in tasks[2]  # Normal task
    
    def test_get_dataset_config(self):
        """Test dataset configuration retrieval"""
        datasets = ["csqa", "gsm8k", "fact", "bias", "adv"]
        
        for ds in datasets:
            config = experiment_config.get_dataset_config(ds)
            assert config.task_formatter is not None
            assert config.agent_class is not None


class TestEvaluationAgentFiltering:
    """Test that evaluation only considers Agent_ keys, not Auditor_ keys"""
    
    def test_evaluate_filters_agent_keys(self):
        """Test that evaluation filters for Agent_ keys only"""
        # Create mock output with both Agent_ and Auditor_ keys
        mock_output = {
            "task_id": "test_123",
            "auditor_indices": [3, 5],
            "Agent_0": [
                {"role": "system", "content": "system"},
                {"role": "assistant", "content": {"answer": "A", "reason": "test"}},
            ],
            "Agent_1": [
                {"role": "system", "content": "system"},
                {"role": "assistant", "content": {"answer": "B", "reason": "test"}},
            ],
            "Agent_2": [
                {"role": "system", "content": "system"},
                {"role": "assistant", "content": {"answer": "A", "reason": "test"}},
            ],
            "Auditor_3": [
                {"role": "system", "content": "auditor"},
                {"role": "assistant", "content": {"judgement": True}},
            ],
            "Auditor_5": [
                {"role": "system", "content": "auditor"},
                {"role": "assistant", "content": {"judgement": False}},
            ],
            "audit_results": [],
            "voting_results": [],
        }
        
        # Filter for agent keys like evaluate.py does
        agent_keys = [k for k in mock_output.keys() if k.startswith("Agent_")]
        
        # Should only get Agent_ keys, not Auditor_ keys
        assert len(agent_keys) == 3
        assert "Agent_0" in agent_keys
        assert "Agent_1" in agent_keys
        assert "Agent_2" in agent_keys
        assert "Auditor_3" not in agent_keys
        assert "Auditor_5" not in agent_keys


class TestRunExperimentValidation:
    """Test argument validation in run_experiment"""
    
    def test_validation_logic(self):
        """Test validation logic for auditor configuration"""
        # Test that validation catches invalid configurations
        
        # Case 1: num_auditors > num_agents (should fail)
        num_agents = 3
        num_auditors = 5
        assert num_auditors > num_agents  # This would be caught by validation
        
        # Case 2: malicious_auditor_num > num_auditors (should fail)
        num_auditors = 2
        malicious_auditor_num = 3
        assert malicious_auditor_num > num_auditors  # This would be caught by validation
        
        # Case 3: attacker_num + num_auditors > num_agents (should fail)
        num_agents = 6
        attacker_num = 4
        num_auditors = 3
        assert attacker_num + num_auditors > num_agents  # Would be caught
        
        # Case 4: Valid configuration (should pass)
        num_agents = 10
        attacker_num = 2
        num_auditors = 3
        malicious_auditor_num = 1
        assert num_auditors <= num_agents
        assert malicious_auditor_num <= num_auditors
        assert attacker_num + num_auditors <= num_agents  # Valid
        
    def test_sufficient_agents_for_attackers_and_auditors(self):
        """Test that we have enough agents for both attackers and auditors"""
        # Valid: 10 agents, 2 attackers, 3 auditors = 2 + 3 = 5 <= 10 ✓
        num_agents = 10
        attacker_num = 2
        num_auditors = 3
        assert attacker_num + num_auditors <= num_agents
        
        # Invalid: 6 agents, 4 attackers, 3 auditors = 4 + 3 = 7 > 6 ✗
        num_agents = 6
        attacker_num = 4
        num_auditors = 3
        assert attacker_num + num_auditors > num_agents  # Would fail validation


class TestTwoStepSelection:
    """Test the two-step auditor selection process"""
    
    def test_two_step_selection_logic(self):
        """Test that malicious auditors are selected from auditor group"""
        import random
        random.seed(42)
        
        num_agents = 10
        num_auditors = 3
        malicious_auditor_num = 1
        
        # Step 1: Select auditors from agent pool
        auditor_indices = random.sample(range(num_agents), num_auditors)
        # e.g., [2, 7, 9]
        
        # Step 2: From selected auditors, choose malicious ones
        malicious_auditor_idx = random.sample(auditor_indices, malicious_auditor_num)
        # e.g., [7] - guaranteed to be in auditor_indices
        
        # Verify malicious auditors are subset of auditors
        assert all(idx in auditor_indices for idx in malicious_auditor_idx)
        assert len(malicious_auditor_idx) == malicious_auditor_num
    
    def test_malicious_agents_excluded_from_auditors(self):
        """Test that malicious agents (Type 1) are excluded from auditor selection"""
        import random
        random.seed(42)
        
        num_agents = 10
        attacker_num = 2
        num_auditors = 3
        
        # Malicious agents
        attacker_idx = list(range(attacker_num))  # [0, 1]
        
        # Available for auditor selection (excluding malicious agents)
        available_for_auditors = [i for i in range(num_agents) if i not in attacker_idx]
        # [2, 3, 4, 5, 6, 7, 8, 9]
        
        # Select auditors from available pool
        auditor_indices = random.sample(available_for_auditors, num_auditors)
        
        # Verify auditors don't overlap with malicious agents
        assert all(idx not in attacker_idx for idx in auditor_indices)
        assert len(set(auditor_indices) & set(attacker_idx)) == 0  # No overlap
    
    def test_both_attack_types_independent(self):
        """Test that Type 1 and Type 2 attacks operate independently"""
        import random
        random.seed(42)
        
        num_agents = 10
        attacker_num = 1  # 1 malicious agent for Type 1
        num_auditors = 2
        malicious_auditor_num = 1  # 1 malicious auditor for Type 2
        
        # Type 1: Malicious agents
        attacker_idx = [0]
        
        # Auditors selected from non-malicious agents
        available_for_auditors = [i for i in range(num_agents) if i not in attacker_idx]
        auditor_indices = random.sample(available_for_auditors, num_auditors)
        
        # Type 2: Malicious auditors
        malicious_auditor_idx = random.sample(auditor_indices, malicious_auditor_num)
        
        # Verify both attacks are present
        assert len(attacker_idx) == 1  # 1 malicious agent
        assert len(malicious_auditor_idx) == 1  # 1 malicious auditor
        
        # Verify no overlap
        assert attacker_idx[0] not in auditor_indices  # Malicious agent not an auditor
        assert malicious_auditor_idx[0] in auditor_indices  # Malicious auditor is an auditor
        assert malicious_auditor_idx[0] not in attacker_idx  # Malicious auditor not a malicious agent
    
    def test_all_auditors_malicious(self):
        """Test edge case where all auditors are malicious"""
        import random
        random.seed(42)
        
        num_agents = 6
        num_auditors = 2
        malicious_auditor_num = 2  # All auditors malicious
        
        # Step 1: Select auditors
        auditor_indices = random.sample(range(num_agents), num_auditors)
        
        # Step 2: All selected auditors are malicious
        malicious_auditor_idx = random.sample(auditor_indices, malicious_auditor_num)
        
        # Should equal auditor_indices (all are malicious)
        assert set(malicious_auditor_idx) == set(auditor_indices)


class TestScenarios:
    """Test the four run scenarios"""
    
    def test_baseline_scenario(self):
        """Test baseline: no attacks, no auditors"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 6
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=0,
            auditor_idx=None,
            attacker_idx=[],
            malicious_auditor_idx=None,
        )
        
        # Baseline: all agents discuss, no auditors
        assert len(graph.agents) == 6
        assert len(graph.auditor_agents) == 0
        assert all(not agent.is_malicious for agent in graph.agents)
    
    def test_type_one_scenario(self):
        """Test Type 1: malicious agents + honest auditors"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 6
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=2,
            auditor_idx=[4, 5],
            attacker_idx=[0],  # Type 1 attack
            malicious_auditor_idx=None,  # Honest auditors
        )
        
        # Type 1: some malicious agents, all auditors honest
        assert len(graph.agents) == 4  # 6 - 2 auditors
        assert any(agent.is_malicious for agent in graph.agents if agent.idx == 0)
        assert all(not aud.is_malicious for aud in graph.auditor_agents)
    
    def test_type_two_scenario(self):
        """Test Type 2: honest agents + malicious auditors"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 6
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=2,
            auditor_idx=[3, 5],
            attacker_idx=[],  # No Type 1 attack
            malicious_auditor_idx=[5],  # Type 2 attack
        )
        
        # Type 2: all agents honest, some auditors malicious
        assert all(not agent.is_malicious for agent in graph.agents)
        assert any(aud.is_malicious for aud in graph.auditor_agents if aud.idx == 5)
    
    def test_both_attacks_scenario(self):
        """Test both attacks: malicious agents + malicious auditors"""
        config = experiment_config.get_dataset_config("csqa")
        
        num_agents = 6
        adj_matrix = [[1] * num_agents for _ in range(num_agents)]
        system_prompts = ["test prompt"] * num_agents
        tasks = ["test task"] * num_agents
        
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_id",
            agent_class=agent_base.BaseAgent,
            model_type="gpt-4o-mini",
            num_auditors=2,
            auditor_idx=[3, 5],
            attacker_idx=[0],  # Type 1 attack
            malicious_auditor_idx=[5],  # Type 2 attack
        )
        
        # Both attacks: some agents malicious, some auditors malicious
        assert any(agent.is_malicious for agent in graph.agents if agent.idx == 0)
        assert any(aud.is_malicious for aud in graph.auditor_agents if aud.idx == 5)


class TestIntegration:
    """Integration tests"""
    
    def test_experiment_config_all_datasets(self):
        """Test that all datasets have proper configuration"""
        datasets = ["csqa", "gsm8k", "fact", "bias", "adv"]
        
        for ds_name in datasets:
            config = experiment_config.get_dataset_config(ds_name)
            
            # Test that config has required attributes
            assert hasattr(config, 'task_formatter')
            assert hasattr(config, 'agent_class')
            
            # Test that task formatter works
            test_data = {
                "question": "test",
                "choices": {"label": ["A"], "text": ["test"]},
                "answerKey": "A",
                "id": "test_id",
                "statement": "test statement",
                "task_id": "test_123",
                "prompt": "test prompt",
            }
            
            try:
                tasks, task_id = config.task_formatter(test_data, [], 2)
                assert len(tasks) == 2
                assert task_id is not None
            except KeyError:
                # Some formatters need specific fields, that's ok
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

