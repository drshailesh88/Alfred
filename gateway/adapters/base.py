"""Base adapter class for all messaging platforms."""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IncomingMessage:
    """Normalized incoming message from any platform."""
    platform: str           # "telegram", "discord", "signal"
    user_id: str           # Platform-specific user ID
    chat_id: str           # Chat/channel ID
    text: str              # Message content
    timestamp: float       # Unix timestamp
    raw: Any               # Original message object
    metadata: dict = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    """Normalized outgoing message to any platform."""
    platform: str
    chat_id: str
    text: str
    reply_to: Optional[str] = None


class BaseAdapter(ABC):
    """
    Base class for messaging platform adapters.

    Each adapter handles one messaging platform (Telegram, Discord, Signal)
    and normalizes messages to/from a common format.
    """

    def __init__(self, config: dict, owner_id: str, on_message: Callable):
        """
        Initialize the adapter.

        Args:
            config: Platform-specific configuration
            owner_id: User(s) allowed to interact (comma-separated for multiple)
            on_message: Callback for incoming messages
        """
        self.config = config
        # Support multiple owner IDs (comma-separated)
        self.owner_ids = [str(oid.strip()) for oid in str(owner_id).split(",") if oid.strip()]
        self.owner_id = self.owner_ids[0] if self.owner_ids else ""  # Primary owner for backwards compat
        self.on_message = on_message
        self.logger = logging.getLogger(f"Alfred.{self.__class__.__name__}")
        self._running = False
        self._reconnect_attempts = 0
        self._started_at: Optional[datetime] = None

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return platform identifier (telegram, discord, signal)."""
        pass

    @abstractmethod
    async def start(self) -> bool:
        """
        Start the adapter and begin listening for messages.
        Returns True if started successfully.
        """
        pass

    @abstractmethod
    async def stop(self):
        """Stop the adapter gracefully."""
        pass

    @abstractmethod
    async def send_message(self, chat_id: str, text: str, reply_to: Optional[str] = None):
        """Send a message to the specified chat."""
        pass

    @abstractmethod
    async def send_typing(self, chat_id: str):
        """Send typing indicator."""
        pass

    def is_owner(self, user_id: str) -> bool:
        """Check if message is from an authorized owner."""
        return str(user_id) in self.owner_ids

    async def handle_message(self, message: IncomingMessage):
        """
        Process incoming message if from owner.
        Non-owner messages are silently ignored.
        """
        if not self.is_owner(message.user_id):
            self.logger.debug(f"Ignored message from non-owner: {message.user_id}")
            return

        self.logger.info(f"[{self.platform_name}] Message: {message.text[:80]}...")

        try:
            await self.on_message(message)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            # Try to notify user of error
            try:
                await self.send_message(
                    message.chat_id,
                    f"I apologize, sir. An error occurred: {str(e)[:200]}"
                )
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        """Check if adapter is currently running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if not self._started_at:
            return 0
        return (datetime.now() - self._started_at).total_seconds()
