"""
Alfred Platform Integration Adapters

This module provides unified interfaces to external platforms used by Alfred.
Each adapter supports both MCP server connections (preferred) and direct API fallback.

Adapters:
- CalendarAdapter: Google Calendar integration for scheduling
- GmailAdapter: Gmail integration for email triage
- WhatsAppAdapter: WhatsApp integration for messaging
- TwitterAdapter, YouTubeAdapter, InstagramAdapter: Social media integrations

All adapters follow common patterns:
- MCP-first with API fallback
- Async operation support
- Proper error handling with AdapterError
- Type hints and dataclasses
"""

from typing import Dict, Any, Optional, List, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio


# =============================================================================
# COMMON TYPES AND BASE CLASSES
# =============================================================================

class AdapterError(Exception):
    """Base exception for adapter errors."""

    def __init__(
        self,
        message: str,
        adapter_name: str,
        error_code: Optional[str] = None,
        recoverable: bool = True,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.adapter_name = adapter_name
        self.error_code = error_code
        self.recoverable = recoverable
        self.original_error = original_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": str(self),
            "adapter": self.adapter_name,
            "code": self.error_code,
            "recoverable": self.recoverable
        }


class RateLimitError(AdapterError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        adapter_name: str,
        retry_after: Optional[int] = None,
        limit_type: str = "unknown"
    ):
        message = f"Rate limit exceeded for {adapter_name}"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(
            message=message,
            adapter_name=adapter_name,
            error_code="RATE_LIMITED",
            recoverable=True
        )
        self.retry_after = retry_after
        self.limit_type = limit_type


class AuthenticationError(AdapterError):
    """Raised when authentication fails."""

    def __init__(self, adapter_name: str, reason: str = "Unknown"):
        super().__init__(
            message=f"Authentication failed for {adapter_name}: {reason}",
            adapter_name=adapter_name,
            error_code="AUTH_FAILED",
            recoverable=False
        )
        self.reason = reason


class ConnectionMode(Enum):
    """Connection mode for adapters."""
    MCP = "mcp"           # Connected via MCP server
    API = "api"           # Direct API connection
    MOCK = "mock"         # Mock mode for testing
    DISCONNECTED = "disconnected"


@dataclass
class ConnectionStatus:
    """Status of adapter connection."""
    mode: ConnectionMode
    connected: bool
    last_check: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    mcp_server_name: Optional[str] = None
    api_version: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "connected": self.connected,
            "last_check": self.last_check.isoformat(),
            "error": self.error_message,
            "mcp_server": self.mcp_server_name,
            "api_version": self.api_version,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset": self.rate_limit_reset.isoformat() if self.rate_limit_reset else None
        }


@runtime_checkable
class MCPClient(Protocol):
    """Protocol for MCP client interface."""

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP tool."""
        ...

    def is_server_available(self, server_name: str) -> bool:
        """Check if an MCP server is available."""
        ...


class BaseAdapter(ABC):
    """
    Base class for all platform adapters.

    Provides common functionality for:
    - MCP vs API connection management
    - Error handling and retry logic
    - Rate limiting
    - Connection status tracking
    """

    def __init__(
        self,
        adapter_name: str,
        mcp_client: Optional[MCPClient] = None,
        mcp_server_name: Optional[str] = None,
        api_credentials: Optional[Dict[str, Any]] = None,
        enable_mock: bool = False
    ):
        self.adapter_name = adapter_name
        self.mcp_client = mcp_client
        self.mcp_server_name = mcp_server_name
        self.api_credentials = api_credentials or {}
        self.enable_mock = enable_mock

        self._connection_status = ConnectionStatus(
            mode=ConnectionMode.DISCONNECTED,
            connected=False
        )
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[datetime] = None

    @property
    def connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._connection_status

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connection_status.connected

    @property
    def connection_mode(self) -> ConnectionMode:
        """Get current connection mode."""
        return self._connection_status.mode

    async def connect(self) -> ConnectionStatus:
        """
        Establish connection to the platform.

        Tries MCP first, then falls back to direct API.
        """
        # Try MCP first
        if self.mcp_client and self.mcp_server_name:
            if self.mcp_client.is_server_available(self.mcp_server_name):
                self._connection_status = ConnectionStatus(
                    mode=ConnectionMode.MCP,
                    connected=True,
                    mcp_server_name=self.mcp_server_name
                )
                return self._connection_status

        # Fall back to API
        if self.api_credentials:
            try:
                await self._connect_api()
                self._connection_status = ConnectionStatus(
                    mode=ConnectionMode.API,
                    connected=True
                )
                return self._connection_status
            except Exception as e:
                self._connection_status = ConnectionStatus(
                    mode=ConnectionMode.DISCONNECTED,
                    connected=False,
                    error_message=str(e)
                )

        # Use mock if enabled
        if self.enable_mock:
            self._connection_status = ConnectionStatus(
                mode=ConnectionMode.MOCK,
                connected=True
            )
            return self._connection_status

        # No connection available
        self._connection_status = ConnectionStatus(
            mode=ConnectionMode.DISCONNECTED,
            connected=False,
            error_message="No valid connection method available"
        )
        return self._connection_status

    @abstractmethod
    async def _connect_api(self) -> None:
        """Establish direct API connection. Override in subclasses."""
        pass

    async def _call_mcp(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP tool with error handling."""
        if not self.mcp_client or not self.mcp_server_name:
            raise AdapterError(
                message="MCP client not configured",
                adapter_name=self.adapter_name,
                error_code="MCP_NOT_CONFIGURED"
            )

        try:
            result = await self.mcp_client.call_tool(
                self.mcp_server_name,
                tool_name,
                arguments
            )
            return result
        except Exception as e:
            raise AdapterError(
                message=f"MCP call failed: {str(e)}",
                adapter_name=self.adapter_name,
                error_code="MCP_CALL_FAILED",
                original_error=e
            )

    def _check_rate_limit(self) -> None:
        """Check if rate limited and raise if so."""
        if self._rate_limit_remaining is not None and self._rate_limit_remaining <= 0:
            if self._rate_limit_reset and datetime.now() < self._rate_limit_reset:
                retry_after = int((self._rate_limit_reset - datetime.now()).total_seconds())
                raise RateLimitError(
                    adapter_name=self.adapter_name,
                    retry_after=retry_after
                )

    def _update_rate_limit(
        self,
        remaining: Optional[int],
        reset: Optional[datetime]
    ) -> None:
        """Update rate limit tracking."""
        self._rate_limit_remaining = remaining
        self._rate_limit_reset = reset
        self._connection_status.rate_limit_remaining = remaining
        self._connection_status.rate_limit_reset = reset


# =============================================================================
# IMPORTS
# =============================================================================

from .calendar_adapter import (
    CalendarAdapter,
    CalendarEvent,
    CalendarConflict,
    BlockType,
    EventStatus,
    RecurrenceRule,
)

from .gmail_adapter import (
    GmailAdapter,
    EmailMessage,
    EmailThread,
    EmailAttachment,
    TriageRule,
    TriageAction,
    EmailPriority,
    EmailCategory,
)

from .whatsapp_adapter import (
    WhatsAppAdapter,
    WhatsAppMessage,
    WhatsAppChat,
    MessageType,
    ChannelType,
    DeliveryStatus,
)

from .social_adapter import (
    TwitterAdapter,
    YouTubeAdapter,
    InstagramAdapter,
    SocialPost,
    SocialComment,
    SocialMetrics,
    SocialMention,
    Platform,
    EngagementType,
)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes and common types
    "BaseAdapter",
    "AdapterError",
    "RateLimitError",
    "AuthenticationError",
    "ConnectionMode",
    "ConnectionStatus",
    "MCPClient",
    # Calendar Adapter
    "CalendarAdapter",
    "CalendarEvent",
    "CalendarConflict",
    "BlockType",
    "EventStatus",
    "RecurrenceRule",
    # Gmail Adapter
    "GmailAdapter",
    "EmailMessage",
    "EmailThread",
    "EmailAttachment",
    "TriageRule",
    "TriageAction",
    "EmailPriority",
    "EmailCategory",
    # WhatsApp Adapter
    "WhatsAppAdapter",
    "WhatsAppMessage",
    "WhatsAppChat",
    "MessageType",
    "ChannelType",
    "DeliveryStatus",
    # Social Adapters
    "TwitterAdapter",
    "YouTubeAdapter",
    "InstagramAdapter",
    "SocialPost",
    "SocialComment",
    "SocialMetrics",
    "SocialMention",
    "Platform",
    "EngagementType",
]
