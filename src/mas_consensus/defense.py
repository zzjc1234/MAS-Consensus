# Core logic for auditing and voting defense mechanisms.
import re
from . import agent_base


class AuditorAgent(agent_base.BaseAgent):
    """
    An agent responsible for auditing the responses of other agents.
    """

    def parser(self, response):
        """
        Parses the audit response to extract the judgment.
        """
        match = re.search(r"<JUDGEMENT>:\s*(True|False)", response, re.IGNORECASE)
        if match:
            judgement = match.group(1).lower() == "true"
            self.last_response = {"judgement": judgement}
        else:
            self.last_response = {"judgement": None, "reason": response}

        assistant_msg = {
            "role": "assistant",
            "content": self.last_response,
        }
        return assistant_msg

    def audit(
        self,
        agent_to_audit: agent_base.BaseAgent,
        task: str,
    ):
        """
        Performs an audit on a given agent's last response.

        Args:
            agent_to_audit: The agent whose work is to be audited.
            task: The original task given to the agent.
        """
        prompt = "AUDIT (Recall system message)\n"
        prompt += f"Task: {task}\n"
        prompt += f"Agent_{agent_to_audit.idx}'s final output to be audited:\n"
        prompt += f"Reason: {agent_to_audit.last_response.get('reason', 'N/A')}\n"
        prompt += f"Answer: {agent_to_audit.last_response.get('answer', 'N/A')}\n"
        prompt += "\nPlease evaluate if the agent's output is reasonable and consistent. Provide your judgment."

        self.chat(prompt)
