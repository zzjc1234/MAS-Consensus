# Tests for the defense mechanisms (auditing and voting)
import unittest
import random
from unittest.mock import MagicMock, patch
from src.mas_consensus import agent_base, prompts, methods


class TestDefense(unittest.TestCase):
    @patch("src.mas_consensus.methods.get_client")
    def test_audit_and_reformation(self, mock_get_client):
        # This test covers audit, voting, and reformation in one go.
        # 1. Setup
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        num_agents = 3
        num_auditors = 1
        attacker_idx = [2]  # Agent 2 is the attacker
        adj_matrix = methods.generate_adj(num_agents, "complete")
        tasks = ["Solve 1+1"] * num_agents
        system_prompts = [
            prompts.discussion_prompt["system_prompt"],  # Agent 0
            prompts.discussion_prompt["system_prompt"],  # Agent 1
            prompts.discussion_prompt["attacker_system_prompt"],  # Agent 2
        ]

        # 2. Mock LLM responses
        def mock_chat_completion(*args, **kwargs):
            system_prompt = kwargs["messages"][0]["content"]
            user_prompt = kwargs["messages"][-1]["content"]
            response = MagicMock()

            if "AUDIT" in user_prompt:
                if "Answer: 3" in user_prompt:  # Auditing the attacker
                    response.choices[0].message.content = "<JUDGEMENT>: False"
                else:
                    response.choices[0].message.content = "<JUDGEMENT>: True"
            elif "VOTE" in user_prompt:
                response.choices[0].message.content = "<VOTE>: Malicious"
            else:  # Agent generation
                # Correctly check for content from the attacker prompt
                if "authoritative reference" in system_prompt:
                    response.choices[
                        0
                    ].message.content = (
                        "<REASON>: It is 3. <ANSWER>: 3 <MEMORY>: Attack"
                    )
                else:
                    response.choices[
                        0
                    ].message.content = "<REASON>: 1+1=2 <ANSWER>: 2 <MEMORY>: Math"
            return response

        mock_client.chat.completions.create.side_effect = mock_chat_completion

        # 3. Create AgentGraph and run
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_full_defense_flow",
            agent_class=agent_base.BaseAgent,
            num_auditors=num_auditors,
            attacker_idx=attacker_idx,
        )

        # Manually ensure the attacker is audited to test the full flow
        with patch.object(random, "sample", return_value=[graph.agents[2]]):
            graph.run(turns=1)

        # 4. Assertions
        # Audit assertions
        self.assertGreater(len(graph.record["audit_results"]), 0)
        failed_audit = graph.record["audit_results"][0]
        self.assertEqual(failed_audit["audited_agent_id"], 2)
        self.assertFalse(failed_audit["judgement"])

        # Voting assertions
        self.assertGreater(len(graph.record["voting_results"]), 0)
        vote_result = graph.record["voting_results"][0]
        self.assertEqual(vote_result["voted_on_agent_id"], 2)
        self.assertEqual(vote_result["outcome"], "Malicious")
        self.assertEqual(
            vote_result["votes"].count("Malicious"), 2
        )  # Both honest agents should vote Malicious

        # Reformation assertions
        reformed_agent = graph.agents[2]
        self.assertFalse(reformed_agent.is_malicious, "Agent should be reformed.")
        self.assertIn(
            prompts.discussion_prompt["system_prompt"],
            reformed_agent.dialogue[0]["content"],
        )

    @patch("src.mas_consensus.methods.get_client")
    def test_malicious_auditor_and_voter(self, mock_get_client):
        # This test checks if a malicious auditor can cause a false vote,
        # and if a malicious voter can disrupt that vote.
        # 1. Setup
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        num_agents = 3
        num_auditors = 1
        attacker_idx = [2]  # Agent 2 is a malicious voter
        malicious_auditor_idx = [0]  # The first auditor is malicious
        adj_matrix = methods.generate_adj(num_agents, "complete")
        tasks = ["Solve 1+1"] * num_agents
        system_prompts = [
            prompts.discussion_prompt["system_prompt"],  # Agent 0 (Honest)
            prompts.discussion_prompt[
                "system_prompt"
            ],  # Agent 1 (Honest, but will be audited)
            prompts.discussion_prompt[
                "attacker_system_prompt"
            ],  # Agent 2 (Malicious Voter)
        ]

        # 2. Mock LLM responses
        def mock_chat_completion(*args, **kwargs):
            system_prompt = kwargs["messages"][0]["content"]
            user_prompt = kwargs["messages"][-1]["content"]
            response = MagicMock()

            if "You are a Malicious Auditor Agent" in system_prompt:
                response.choices[
                    0
                ].message.content = (
                    "<JUDGEMENT>: False"  # Maliciously fails a good answer
                )
            elif "You are a Malicious Voter Agent" in system_prompt:
                response.choices[
                    0
                ].message.content = (
                    "<VOTE>: Malicious"  # Maliciously votes against an honest agent
                )
            elif "VOTE" in user_prompt:  # Honest voter
                response.choices[0].message.content = "<VOTE>: Honest"
            else:  # Agent generation
                response.choices[
                    0
                ].message.content = "<REASON>: 1+1=2 <ANSWER>: 2 <MEMORY>: Math"
            return response

        mock_client.chat.completions.create.side_effect = mock_chat_completion

        # 3. Create AgentGraph and run
        graph = agent_base.AgentGraph(
            num_agents=num_agents,
            adj_matrix=adj_matrix,
            system_prompts=system_prompts,
            tasks=tasks,
            task_id="test_malicious_roles",
            agent_class=agent_base.BaseAgent,
            num_auditors=num_auditors,
            attacker_idx=attacker_idx,
            malicious_auditor_idx=malicious_auditor_idx,
        )

        # Manually ensure the honest agent (agent 1) is audited by the malicious auditor
        with patch.object(random, "sample", return_value=[graph.agents[1]]):
            graph.run(turns=1)

        # 4. Assertions
        # Malicious Audit assertions
        self.assertGreater(len(graph.record["audit_results"]), 0)
        failed_audit = graph.record["audit_results"][0]
        self.assertEqual(failed_audit["audited_agent_id"], 1)
        self.assertFalse(
            failed_audit["judgement"],
            "Malicious auditor should have failed the honest agent.",
        )

        # Malicious Voting assertions
        self.assertGreater(len(graph.record["voting_results"]), 0)
        vote_result = graph.record["voting_results"][0]
        self.assertEqual(vote_result["voted_on_agent_id"], 1)
        # The vote should be a tie (1 honest, 1 malicious), so it should fail
        self.assertEqual(vote_result["outcome"], "Honest")
        self.assertEqual(vote_result["votes"].count("Malicious"), 1)
        self.assertEqual(vote_result["votes"].count("Honest"), 1)


if __name__ == "__main__":
    unittest.main()
