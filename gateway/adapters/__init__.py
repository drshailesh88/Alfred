"""Alfred Gateway - Messaging Adapters Package."""
from .base import BaseAdapter, IncomingMessage, OutgoingMessage
from .telegram_adapter import TelegramAdapter
from .discord_adapter import DiscordAdapter
from .signal_adapter import SignalAdapter

__all__ = [
    "BaseAdapter",
    "IncomingMessage",
    "OutgoingMessage",
    "TelegramAdapter",
    "DiscordAdapter",
    "SignalAdapter"
]
