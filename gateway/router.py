"""
Message router connecting platforms to Agent Zero (Alfred).

This module handles:
- Routing incoming messages to Alfred for processing
- Managing the connection to Agent Zero
- Handling responses and errors
"""
import asyncio
import logging
import sys
from typing import Callable, Optional
from pathlib import Path


class AlfredRouter:
    """Routes messages between messaging platforms and Agent Zero."""

    def __init__(self, config: dict):
        """
        Initialize the router.

        Args:
            config: Full gateway configuration
        """
        self.config = config
        self.logger = logging.getLogger("Alfred.Router")
        self.agent = None
        self._mode = "echo"  # Default to echo mode until connected
        self._init_agent_zero()

    def _init_agent_zero(self):
        """Initialize connection to Agent Zero."""
        az_config = self.config.get("agent_zero", {})
        mode = az_config.get("mode", "direct")

        if mode == "direct":
            self._init_direct_mode(az_config)
        elif mode == "api":
            self._init_api_mode(az_config)
        else:
            self.logger.warning(f"Unknown mode '{mode}', using echo mode")
            self._mode = "echo"

    def _init_direct_mode(self, az_config: dict):
        """Initialize Agent Zero in direct import mode."""
        az_path = az_config.get("agent_zero_path", "/opt/alfred/agent-zero1")
        agent_name = az_config.get("agent_name", "alfred")

        # Check if path exists
        if not Path(az_path).exists():
            self.logger.warning(f"Agent Zero path not found: {az_path}")
            self.logger.info("Using echo mode for testing")
            self._mode = "echo"
            return

        # Add to Python path
        python_path = str(Path(az_path) / "python")
        if python_path not in sys.path:
            sys.path.insert(0, az_path)
            sys.path.insert(0, python_path)

        try:
            # Try to import Agent Zero
            # Note: This import structure depends on Agent Zero's actual API
            from python.helpers import settings

            # Load agent configuration
            settings.set_agent(agent_name)

            # Try to import the Agent class
            from agent import Agent

            self.agent = Agent(agent_name=agent_name)
            self._mode = "direct"
            self.logger.info(f"Agent Zero ({agent_name}) initialized in direct mode")

        except ImportError as e:
            self.logger.warning(f"Could not import Agent Zero: {e}")
            self.logger.info("Using echo mode for testing")
            self._mode = "echo"

        except Exception as e:
            self.logger.error(f"Error initializing Agent Zero: {e}")
            self._mode = "echo"

    def _init_api_mode(self, az_config: dict):
        """Initialize Agent Zero in API mode."""
        self.api_url = az_config.get("api_url", "http://localhost:50001/api/chat")
        self._mode = "api"
        self.logger.info(f"Agent Zero configured in API mode: {self.api_url}")

    async def process_message(
        self,
        message,
        send_reply: Callable,
        send_typing: Callable
    ) -> str:
        """
        Process an incoming message and return Alfred's response.

        Args:
            message: IncomingMessage object
            send_reply: Callback to send reply
            send_typing: Callback to send typing indicator

        Returns:
            Alfred's response text
        """
        # Show typing indicator
        await send_typing(message.chat_id)

        self.logger.info(f"Processing [{message.platform}]: {message.text[:100]}...")

        try:
            if self._mode == "direct":
                response = await self._process_direct(message.text)
            elif self._mode == "api":
                response = await self._process_api(message.text)
            else:
                response = await self._process_echo(message)

            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"I apologize, sir. I encountered an error: {str(e)[:200]}"

    async def _process_direct(self, text: str) -> str:
        """Process message via direct Agent Zero call."""
        if not self.agent:
            return "Agent Zero is not initialized."

        try:
            # Run in thread to avoid blocking
            response = await asyncio.to_thread(
                self.agent.chat,
                text
            )
            return response

        except Exception as e:
            self.logger.error(f"Agent Zero error: {e}")
            raise

    async def _process_api(self, text: str) -> str:
        """Process message via Agent Zero API."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json={"message": text},
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "No response received")
                    else:
                        return f"API error: {resp.status}"

        except aiohttp.ClientError as e:
            self.logger.error(f"API request failed: {e}")
            raise

    async def _process_echo(self, message) -> str:
        """Echo mode for testing without Agent Zero."""
        # Simulate some processing time
        await asyncio.sleep(0.5)

        return (
            f"[Echo Mode - Agent Zero not connected]\n\n"
            f"Platform: {message.platform}\n"
            f"Your message: {message.text}\n\n"
            f"Configure Agent Zero in config.yaml to enable full functionality."
        )

    @property
    def mode(self) -> str:
        """Get current operation mode."""
        return self._mode

    @property
    def is_connected(self) -> bool:
        """Check if connected to Agent Zero."""
        return self._mode in ("direct", "api")
