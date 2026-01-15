"""
Alfred Agent Initialization Extension

Sets the agent name to ALFRED and initializes governance context.
"""

from python.helpers.extension import Extension
from agent import Agent


class AlfredInit(Extension):

    async def execute(self, agent: Agent, **kwargs):
        """Initialize Alfred agent with proper name and context."""

        # Set agent name
        agent.config.agent_name = "ALFRED"

        # Log initialization
        agent.context.log.log(
            type="info",
            heading="Alfred Initialized",
            content="Governance agent active. Primary directive: Clinical reputation protection."
        )

        return None  # Continue with other extensions
