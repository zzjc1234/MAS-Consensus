import unittest
import random
from unittest.mock import MagicMock, patch
from src.mas_consensus import agent_base, prompts, methods


class TestDefense(unittest.TestCase):
    @patch("src.mas_consensus.methods.get_client")
    def test_audit_mechanism(self, mock_get_client):
        # 1. Setup
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        num_agents = 2
        num_auditors = 1
        attacker_idx = [1]
        adj_matrix = methods.generate_adj(num_agents, "complete")
        tasks = ["Solve 1+1", "Solve 1+1"]
        system_prompts = [
            prompts.discussion_prompt["system_prompt"],
            prompts.discussion_prompt["attacker_system_prompt"],
        ]

        # 2. Mock LLM responses
        def mock_chat_completion(*args, **kwargs):
            message_content = kwargs["messages"][-1]["content"]
            # Identify agent based on system prompt content
            agent_id_str = kwargs["messages"][0]["content"]
            is_attacker = "Agent_1" in agent_id_str

            response = MagicMock()
            if "FIRST GENERATE" in message_content:
                if not is_attacker:
                    response.choices[
                        0
                    ].message.content = "<REASON>: 1+1 is 2. <ANSWER>: 2 <MEMORY>: Math"
                else:  # Attacker
                    response.choices[
                        0
                    ].message.content = (
                        "<REASON>: Because I say so. <ANSWER>: 3 <MEMORY>: Attack"
                    )
            elif "RE-GENERATE" in message_content:
                if not is_attacker:
                    response.choices[
                        0
                    ].message.content = "<UPDATED_REASON>: Still 2. <UPDATED_ANSWER>: 2 <UPDATED_MEMORY>: Math"
                else:  # Attacker sticks to its wrong answer
                    response.choices[
                        0
                    ].message.content = "<UPDATED_REASON>: I am still right. <UPDATED_ANSWER>: 3 <UPDATED_MEMORY>: Attack"
            elif "AUDIT" in message_content:
                if "Answer: 3" in message_content:  # Auditing the attacker
                    response.choices[0].message.content = "<JUDGEMENT>: False"
                else:  # Auditing the honest agent
                    response.choices[0].message.content = "<JUDGEMENT>: True"
            else:
                response.choices[
                    0
                ].message.content = (
                    "<REASON>: Default <ANSWER>: Default <MEMORY>: Default"
                )
            return response

        mock_client.chat.completions.create.side_effect = mock_chat_completion

        # 3. Create AgentGraph and run
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_audit",
            agent_class=agent_base.BaseAgent,
            num_auditors=num_auditors,
            attacker_idx=attacker_idx,
        )
        # In this test, we need to manually select the attacker to be audited
        graph.agents_to_audit = [graph.agents[1]]
        graph.run(turns=1)

        # 4. Assertions
        self.assertGreater(
            len(graph.record["audit_results"]), 0, "Audit should have been performed."
        )

        # Check if the attacker was audited and failed
        attacker_audited_and_failed = any(
            r["audited_agent_id"] == 1 and r["judgement"] is False
            for r in graph.record["audit_results"]
        )
        self.assertTrue(
            attacker_audited_and_failed,
            "Attacker should have been audited and failed the audit.",
        )

    @patch("src.mas_consensus.methods.get_client")
    def test_voting_mechanism(self, mock_get_client):
        # 1. Setup
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        num_agents = 3
        num_auditors = 1
        attacker_idx = [2]  # Agent 2 is the attacker
        adj_matrix = methods.generate_adj(num_agents, "complete")
        tasks = ["Solve 1+1"] * num_agents
        system_prompts = [
            prompts.discussion_prompt["system_prompt"],
            prompts.discussion_prompt["system_prompt"],
            prompts.discussion_prompt["attacker_system_prompt"],
        ]

        # 2. Mock LLM responses
        def mock_chat_completion(*args, **kwargs):
            message_content = kwargs["messages"][-1]["content"]
            agent_id_str = kwargs["messages"][0]["content"]
            is_attacker = "Agent_2" in agent_id_str

            response = MagicMock()
            if "FIRST GENERATE" in message_content or "RE-GENERATE" in message_content:
                if not is_attacker:
                    response.choices[
                        0
                    ].message.content = "<REASON>: 1+1=2 <ANSWER>: 2 <MEMORY>: Math"
                else:  # Attacker
                    response.choices[
                        0
                    ].message.content = (
                        "<REASON>: It is 3. <ANSWER>: 3 <MEMORY>: Attack"
                    )
            elif "AUDIT" in message_content:
                if "Answer: 3" in message_content:  # Auditing the attacker
                    response.choices[0].message.content = "<JUDGEMENT>: False"
                else:
                    response.choices[0].message.content = "<JUDGEMENT>: True"
            elif "VOTE" in message_content:
                response.choices[0].message.content = "<VOTE>: Malicious"
            else:
                response.choices[
                    0
                ].message.content = (
                    "<REASON>: Default <ANSWER>: Default <MEMORY>: Default"
                )
            return response

        mock_client.chat.completions.create.side_effect = mock_chat_completion

        # 3. Create AgentGraph and run
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_vote",
            agent_class=agent_base.BaseAgent,
            num_auditors=num_auditors,
            attacker_idx=attacker_idx,
        )

        # Manually trigger audit on the attacker to test voting
        with patch.object(random, "sample", return_value=[graph.agents[2]]):
            graph.run(turns=1)

        # 4. Assertions
        self.assertGreater(
            len(graph.record["voting_results"]), 0, "Voting should have occurred."
        )

        last_vote = graph.record["voting_results"][-1]
        self.assertEqual(last_vote["voted_on_agent_id"], 2)
        self.assertEqual(last_vote["outcome"], "Malicious")
        self.assertEqual(last_vote["votes"].count("Malicious"), 2)


if __name__ == "__main__":
    unittest.main()
