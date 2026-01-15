"""
WhatsApp Adapter for Alfred

WhatsApp integration for message handling and channel routing.
Supports both WhatsApp Business API and personal WhatsApp via unofficial APIs.

MCP Servers Supported:
- whatsapp-mcp (unofficial)
- whatsapp-business-mcp

Direct API:
- WhatsApp Business API
- WhatsApp Cloud API (Meta)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import asyncio
import re

from . import (
    BaseAdapter,
    AdapterError,
    RateLimitError,
    AuthenticationError,
    ConnectionMode,
    ConnectionStatus,
    MCPClient,
)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class MessageType(Enum):
    """WhatsApp message types."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    STICKER = "sticker"
    LOCATION = "location"
    CONTACT = "contact"
    TEMPLATE = "template"        # Business API templates
    INTERACTIVE = "interactive"  # Buttons, lists
    REACTION = "reaction"
    UNKNOWN = "unknown"


class DeliveryStatus(Enum):
    """Message delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


class ChannelType(Enum):
    """Channel/chat categorization for routing."""
    PERSONAL = "personal"        # Family, close friends
    WORK = "work"                # Professional contacts
    GROUP_PERSONAL = "group_personal"  # Personal group chats
    GROUP_WORK = "group_work"    # Work group chats
    BROADCAST = "broadcast"      # Broadcast lists
    BUSINESS = "business"        # Business accounts
    UNKNOWN = "unknown"


class ChatType(Enum):
    """Type of WhatsApp chat."""
    INDIVIDUAL = "individual"
    GROUP = "group"
    BROADCAST = "broadcast"
    STATUS = "status"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MediaContent:
    """Represents media content in a message."""
    media_type: MessageType
    media_id: Optional[str] = None
    url: Optional[str] = None
    mime_type: Optional[str] = None
    filename: Optional[str] = None
    file_size: Optional[int] = None
    sha256: Optional[str] = None
    caption: Optional[str] = None

    # For downloaded media
    data: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.media_type.value,
            "id": self.media_id,
            "url": self.url,
            "mimeType": self.mime_type,
            "filename": self.filename,
            "fileSize": self.file_size,
            "caption": self.caption
        }


@dataclass
class Contact:
    """WhatsApp contact information."""
    phone_number: str  # In international format: +1234567890
    display_name: Optional[str] = None
    push_name: Optional[str] = None  # Name set by user
    profile_picture_url: Optional[str] = None
    is_business: bool = False
    business_name: Optional[str] = None

    # Alfred routing
    channel_type: ChannelType = ChannelType.UNKNOWN
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def name(self) -> str:
        """Best available name for this contact."""
        return (
            self.display_name or
            self.push_name or
            self.business_name or
            self.phone_number
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phone": self.phone_number,
            "displayName": self.display_name,
            "pushName": self.push_name,
            "isBusiness": self.is_business,
            "channelType": self.channel_type.value,
            "tags": self.tags
        }


@dataclass
class WhatsAppMessage:
    """Represents a WhatsApp message."""
    message_id: str
    chat_id: str  # Phone number or group ID
    timestamp: datetime

    # Sender info
    from_number: str
    from_name: Optional[str] = None
    is_from_me: bool = False

    # Content
    message_type: MessageType = MessageType.TEXT
    text: str = ""
    media: Optional[MediaContent] = None

    # Reply context
    is_reply: bool = False
    reply_to_id: Optional[str] = None
    quoted_text: Optional[str] = None

    # Group context
    is_group: bool = False
    group_name: Optional[str] = None
    mentions: List[str] = field(default_factory=list)

    # Status
    status: DeliveryStatus = DeliveryStatus.SENT
    is_forwarded: bool = False
    forward_count: int = 0

    # Alfred routing
    channel_type: ChannelType = ChannelType.UNKNOWN
    priority: str = "normal"  # urgent, high, normal, low
    requires_response: bool = False
    response_deadline: Optional[datetime] = None

    @property
    def is_media(self) -> bool:
        """Check if this is a media message."""
        return self.message_type in [
            MessageType.IMAGE,
            MessageType.VIDEO,
            MessageType.AUDIO,
            MessageType.DOCUMENT,
            MessageType.STICKER
        ]

    @property
    def content_preview(self) -> str:
        """Short preview of message content."""
        if self.text:
            return self.text[:100] + "..." if len(self.text) > 100 else self.text
        elif self.media:
            return f"[{self.message_type.value}]"
        return "[empty]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.message_id,
            "chatId": self.chat_id,
            "timestamp": self.timestamp.isoformat(),
            "from": self.from_number,
            "fromName": self.from_name,
            "isFromMe": self.is_from_me,
            "type": self.message_type.value,
            "text": self.text,
            "media": self.media.to_dict() if self.media else None,
            "isReply": self.is_reply,
            "replyToId": self.reply_to_id,
            "isGroup": self.is_group,
            "groupName": self.group_name,
            "status": self.status.value,
            "channelType": self.channel_type.value,
            "priority": self.priority,
            "requiresResponse": self.requires_response
        }

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "WhatsAppMessage":
        """Create from WhatsApp API response."""
        # Parse message type and content
        msg_type = MessageType.TEXT
        text = ""
        media = None

        if "text" in data:
            msg_type = MessageType.TEXT
            text = data["text"].get("body", "")
        elif "image" in data:
            msg_type = MessageType.IMAGE
            media = MediaContent(
                media_type=msg_type,
                media_id=data["image"].get("id"),
                caption=data["image"].get("caption"),
                mime_type=data["image"].get("mime_type")
            )
        elif "video" in data:
            msg_type = MessageType.VIDEO
            media = MediaContent(
                media_type=msg_type,
                media_id=data["video"].get("id"),
                caption=data["video"].get("caption"),
                mime_type=data["video"].get("mime_type")
            )
        elif "audio" in data:
            msg_type = MessageType.AUDIO
            media = MediaContent(
                media_type=msg_type,
                media_id=data["audio"].get("id"),
                mime_type=data["audio"].get("mime_type")
            )
        elif "document" in data:
            msg_type = MessageType.DOCUMENT
            media = MediaContent(
                media_type=msg_type,
                media_id=data["document"].get("id"),
                filename=data["document"].get("filename"),
                caption=data["document"].get("caption"),
                mime_type=data["document"].get("mime_type")
            )
        elif "sticker" in data:
            msg_type = MessageType.STICKER
            media = MediaContent(
                media_type=msg_type,
                media_id=data["sticker"].get("id")
            )

        # Parse timestamp
        timestamp = datetime.now()
        if "timestamp" in data:
            try:
                timestamp = datetime.fromtimestamp(int(data["timestamp"]))
            except (ValueError, TypeError):
                pass

        # Check for reply context
        is_reply = "context" in data
        reply_to_id = data.get("context", {}).get("message_id")
        quoted_text = data.get("context", {}).get("quoted_content")

        return cls(
            message_id=data.get("id", ""),
            chat_id=data.get("from", ""),
            timestamp=timestamp,
            from_number=data.get("from", ""),
            from_name=data.get("profile", {}).get("name"),
            message_type=msg_type,
            text=text,
            media=media,
            is_reply=is_reply,
            reply_to_id=reply_to_id,
            quoted_text=quoted_text
        )


@dataclass
class WhatsAppChat:
    """Represents a WhatsApp chat/conversation."""
    chat_id: str
    chat_type: ChatType = ChatType.INDIVIDUAL
    name: str = ""
    description: str = ""

    # For individual chats
    contact: Optional[Contact] = None

    # For group chats
    participants: List[Contact] = field(default_factory=list)
    admins: List[str] = field(default_factory=list)  # Phone numbers
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None

    # State
    is_muted: bool = False
    muted_until: Optional[datetime] = None
    is_archived: bool = False
    is_pinned: bool = False
    unread_count: int = 0
    last_message: Optional[WhatsAppMessage] = None
    last_activity: Optional[datetime] = None

    # Alfred routing
    channel_type: ChannelType = ChannelType.UNKNOWN
    auto_respond: bool = False
    response_template: Optional[str] = None

    @property
    def is_group(self) -> bool:
        return self.chat_type == ChatType.GROUP

    @property
    def participant_count(self) -> int:
        return len(self.participants) if self.participants else (1 if self.contact else 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chat_id,
            "type": self.chat_type.value,
            "name": self.name,
            "description": self.description,
            "isGroup": self.is_group,
            "participantCount": self.participant_count,
            "isMuted": self.is_muted,
            "isArchived": self.is_archived,
            "isPinned": self.is_pinned,
            "unreadCount": self.unread_count,
            "lastActivity": self.last_activity.isoformat() if self.last_activity else None,
            "channelType": self.channel_type.value,
            "autoRespond": self.auto_respond
        }


@dataclass
class ChannelRoutingRule:
    """Rule for routing messages to appropriate channels."""
    rule_id: str
    name: str
    enabled: bool = True
    priority: int = 0

    # Match conditions
    match_numbers: Optional[List[str]] = None  # Phone numbers/patterns
    match_groups: Optional[List[str]] = None   # Group IDs/names
    match_contacts: Optional[List[str]] = None  # Contact tags
    match_keywords: Optional[List[str]] = None  # Content keywords

    # Routing
    route_to: ChannelType = ChannelType.UNKNOWN
    set_priority: str = "normal"
    requires_response: bool = False
    response_deadline_hours: Optional[int] = None

    def matches(self, message: WhatsAppMessage, chat: Optional[WhatsAppChat] = None) -> bool:
        """Check if a message matches this rule."""
        # Check phone numbers
        if self.match_numbers:
            if not any(
                pattern in message.from_number
                for pattern in self.match_numbers
            ):
                return False

        # Check groups
        if self.match_groups and message.is_group:
            if not any(
                pattern in (message.group_name or message.chat_id)
                for pattern in self.match_groups
            ):
                return False

        # Check keywords in text
        if self.match_keywords and message.text:
            text_lower = message.text.lower()
            if not any(
                keyword.lower() in text_lower
                for keyword in self.match_keywords
            ):
                return False

        return True


# =============================================================================
# WHATSAPP ADAPTER
# =============================================================================

class WhatsAppAdapter(BaseAdapter):
    """
    WhatsApp adapter with MCP and direct API support.

    Provides:
    - Message sending and receiving
    - Chat management
    - Channel routing (personal, work, groups)
    - Media handling
    - Contact management
    """

    # MCP server names
    MCP_WHATSAPP = "whatsapp-mcp"
    MCP_WHATSAPP_BUSINESS = "whatsapp-business-mcp"

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        api_credentials: Optional[Dict[str, Any]] = None,
        phone_number_id: Optional[str] = None,  # For Business API
        enable_mock: bool = False
    ):
        """
        Initialize WhatsApp adapter.

        Args:
            mcp_client: MCP client for server communication
            api_credentials: API credentials (access token, etc.)
            phone_number_id: WhatsApp Business phone number ID
            enable_mock: Enable mock mode for testing
        """
        # Determine MCP server
        mcp_server = None
        if mcp_client:
            if mcp_client.is_server_available(self.MCP_WHATSAPP):
                mcp_server = self.MCP_WHATSAPP
            elif mcp_client.is_server_available(self.MCP_WHATSAPP_BUSINESS):
                mcp_server = self.MCP_WHATSAPP_BUSINESS

        super().__init__(
            adapter_name="WhatsAppAdapter",
            mcp_client=mcp_client,
            mcp_server_name=mcp_server,
            api_credentials=api_credentials,
            enable_mock=enable_mock
        )

        self.phone_number_id = phone_number_id
        self._routing_rules: List[ChannelRoutingRule] = []
        self._contacts_cache: Dict[str, Contact] = {}

    async def _connect_api(self) -> None:
        """Establish direct WhatsApp API connection."""
        if not self.api_credentials:
            raise AuthenticationError(
                self.adapter_name,
                "No API credentials provided"
            )
        # Validate access token
        access_token = self.api_credentials.get("access_token")
        if not access_token:
            raise AuthenticationError(
                self.adapter_name,
                "Missing access_token in credentials"
            )

    # -------------------------------------------------------------------------
    # CHANNEL ROUTING
    # -------------------------------------------------------------------------

    def add_routing_rule(self, rule: ChannelRoutingRule) -> None:
        """Add a channel routing rule."""
        self._routing_rules.append(rule)
        self._routing_rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_routing_rule(self, rule_id: str) -> bool:
        """Remove a routing rule by ID."""
        for i, rule in enumerate(self._routing_rules):
            if rule.rule_id == rule_id:
                del self._routing_rules[i]
                return True
        return False

    def apply_routing(
        self,
        message: WhatsAppMessage,
        chat: Optional[WhatsAppChat] = None
    ) -> WhatsAppMessage:
        """
        Apply routing rules to categorize a message.

        Args:
            message: Message to route
            chat: Optional chat context

        Returns:
            Message with routing fields populated
        """
        for rule in self._routing_rules:
            if not rule.enabled:
                continue

            if rule.matches(message, chat):
                message.channel_type = rule.route_to
                message.priority = rule.set_priority
                message.requires_response = rule.requires_response

                if rule.response_deadline_hours:
                    message.response_deadline = (
                        datetime.now() +
                        timedelta(hours=rule.response_deadline_hours)
                    )
                break

        # Default routing based on chat type
        if message.channel_type == ChannelType.UNKNOWN:
            if message.is_group:
                message.channel_type = ChannelType.GROUP_PERSONAL
            else:
                message.channel_type = ChannelType.PERSONAL

        return message

    def set_contact_channel(self, phone_number: str, channel_type: ChannelType) -> None:
        """Set the channel type for a contact."""
        if phone_number in self._contacts_cache:
            self._contacts_cache[phone_number].channel_type = channel_type
        else:
            self._contacts_cache[phone_number] = Contact(
                phone_number=phone_number,
                channel_type=channel_type
            )

    # -------------------------------------------------------------------------
    # MESSAGE RETRIEVAL
    # -------------------------------------------------------------------------

    async def get_messages(
        self,
        chat_id: Optional[str] = None,
        limit: int = 50,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        channel_filter: Optional[ChannelType] = None
    ) -> List[WhatsAppMessage]:
        """
        Retrieve messages.

        Args:
            chat_id: Specific chat to retrieve from (None for all)
            limit: Maximum number of messages
            before: Get messages before this time
            after: Get messages after this time
            channel_filter: Filter by channel type

        Returns:
            List of WhatsAppMessage objects
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_messages_mcp(chat_id, limit, before, after)
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_messages_api(chat_id, limit, before, after)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_messages_mock()
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _get_messages_mcp(
        self,
        chat_id: Optional[str],
        limit: int,
        before: Optional[datetime],
        after: Optional[datetime]
    ) -> List[WhatsAppMessage]:
        """Get messages via MCP server."""
        args: Dict[str, Any] = {"limit": limit}
        if chat_id:
            args["chatId"] = chat_id
        if before:
            args["before"] = before.isoformat()
        if after:
            args["after"] = after.isoformat()

        try:
            result = await self._call_mcp("get_messages", args)

            messages = []
            for msg_data in result.get("messages", []):
                message = WhatsAppMessage.from_api_response(msg_data)
                message = self.apply_routing(message)
                messages.append(message)

            return messages
        except Exception as e:
            raise AdapterError(
                f"Failed to get messages: {str(e)}",
                self.adapter_name,
                "GET_MESSAGES_FAILED",
                original_error=e
            )

    async def _get_messages_api(
        self,
        chat_id: Optional[str],
        limit: int,
        before: Optional[datetime],
        after: Optional[datetime]
    ) -> List[WhatsAppMessage]:
        """Get messages via direct API."""
        # Note: WhatsApp Business API doesn't support message history retrieval
        # This would require webhook integration for incoming messages
        return []

    def _get_messages_mock(self) -> List[WhatsAppMessage]:
        """Get mock messages for testing."""
        now = datetime.now()
        return [
            WhatsAppMessage(
                message_id="mock_wa_1",
                chat_id="+1234567890",
                timestamp=now - timedelta(minutes=5),
                from_number="+1234567890",
                from_name="John Doe",
                message_type=MessageType.TEXT,
                text="Hey, are we still meeting tomorrow?",
                channel_type=ChannelType.WORK,
                priority="normal",
                requires_response=True
            ),
            WhatsAppMessage(
                message_id="mock_wa_2",
                chat_id="+0987654321",
                timestamp=now - timedelta(minutes=30),
                from_number="+0987654321",
                from_name="Mom",
                message_type=MessageType.TEXT,
                text="Don't forget dinner on Sunday!",
                channel_type=ChannelType.PERSONAL,
                priority="normal"
            ),
            WhatsAppMessage(
                message_id="mock_wa_3",
                chat_id="group_123",
                timestamp=now - timedelta(hours=1),
                from_number="+1111111111",
                from_name="Alice",
                message_type=MessageType.TEXT,
                text="@everyone Meeting notes are uploaded",
                is_group=True,
                group_name="Team Alpha",
                channel_type=ChannelType.GROUP_WORK,
                mentions=["everyone"]
            ),
        ]

    # -------------------------------------------------------------------------
    # CHAT MANAGEMENT
    # -------------------------------------------------------------------------

    async def get_chats(
        self,
        include_archived: bool = False,
        channel_filter: Optional[ChannelType] = None
    ) -> List[WhatsAppChat]:
        """
        Retrieve chat list.

        Args:
            include_archived: Include archived chats
            channel_filter: Filter by channel type

        Returns:
            List of WhatsAppChat objects
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_chats_mcp(include_archived)
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_chats_api(include_archived)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_chats_mock()
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _get_chats_mcp(
        self,
        include_archived: bool
    ) -> List[WhatsAppChat]:
        """Get chats via MCP server."""
        try:
            result = await self._call_mcp("get_chats", {
                "includeArchived": include_archived
            })

            chats = []
            for chat_data in result.get("chats", []):
                chat = WhatsAppChat(
                    chat_id=chat_data.get("id", ""),
                    chat_type=ChatType(chat_data.get("type", "individual")),
                    name=chat_data.get("name", ""),
                    is_muted=chat_data.get("isMuted", False),
                    is_archived=chat_data.get("isArchived", False),
                    is_pinned=chat_data.get("isPinned", False),
                    unread_count=chat_data.get("unreadCount", 0)
                )

                # Apply routing based on cached contact info
                if chat.chat_id in self._contacts_cache:
                    chat.channel_type = self._contacts_cache[chat.chat_id].channel_type

                chats.append(chat)

            return chats
        except Exception as e:
            raise AdapterError(
                f"Failed to get chats: {str(e)}",
                self.adapter_name,
                "GET_CHATS_FAILED",
                original_error=e
            )

    async def _get_chats_api(
        self,
        include_archived: bool
    ) -> List[WhatsAppChat]:
        """Get chats via direct API."""
        # Business API doesn't support chat listing
        return []

    def _get_chats_mock(self) -> List[WhatsAppChat]:
        """Get mock chats for testing."""
        now = datetime.now()
        return [
            WhatsAppChat(
                chat_id="+1234567890",
                chat_type=ChatType.INDIVIDUAL,
                name="John Doe",
                unread_count=2,
                last_activity=now - timedelta(minutes=5),
                channel_type=ChannelType.WORK
            ),
            WhatsAppChat(
                chat_id="+0987654321",
                chat_type=ChatType.INDIVIDUAL,
                name="Mom",
                unread_count=0,
                last_activity=now - timedelta(minutes=30),
                channel_type=ChannelType.PERSONAL,
                is_pinned=True
            ),
            WhatsAppChat(
                chat_id="group_123",
                chat_type=ChatType.GROUP,
                name="Team Alpha",
                unread_count=5,
                last_activity=now - timedelta(hours=1),
                channel_type=ChannelType.GROUP_WORK
            ),
        ]

    # -------------------------------------------------------------------------
    # SENDING MESSAGES
    # -------------------------------------------------------------------------

    async def send_message(
        self,
        to: str,
        text: Optional[str] = None,
        media: Optional[MediaContent] = None,
        reply_to: Optional[str] = None,
        preview_url: bool = True
    ) -> Optional[str]:
        """
        Send a message.

        Args:
            to: Recipient phone number or group ID
            text: Text message content
            media: Media content to send
            reply_to: Message ID to reply to
            preview_url: Enable link previews

        Returns:
            Message ID of sent message, or None on failure
        """
        self._check_rate_limit()

        if not text and not media:
            raise AdapterError(
                "Either text or media must be provided",
                self.adapter_name,
                "INVALID_MESSAGE"
            )

        if self.connection_mode == ConnectionMode.MCP:
            return await self._send_message_mcp(to, text, media, reply_to, preview_url)
        elif self.connection_mode == ConnectionMode.API:
            return await self._send_message_api(to, text, media, reply_to, preview_url)
        elif self.connection_mode == ConnectionMode.MOCK:
            return f"mock_sent_{datetime.now().timestamp()}"
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _send_message_mcp(
        self,
        to: str,
        text: Optional[str],
        media: Optional[MediaContent],
        reply_to: Optional[str],
        preview_url: bool
    ) -> Optional[str]:
        """Send message via MCP server."""
        args: Dict[str, Any] = {"to": to}

        if text:
            args["text"] = {"body": text, "preview_url": preview_url}
        if media:
            args[media.media_type.value] = {
                "id": media.media_id,
                "link": media.url,
                "caption": media.caption
            }
        if reply_to:
            args["context"] = {"message_id": reply_to}

        try:
            result = await self._call_mcp("send_message", args)
            return result.get("messages", [{}])[0].get("id")
        except Exception as e:
            raise AdapterError(
                f"Failed to send message: {str(e)}",
                self.adapter_name,
                "SEND_FAILED",
                original_error=e
            )

    async def _send_message_api(
        self,
        to: str,
        text: Optional[str],
        media: Optional[MediaContent],
        reply_to: Optional[str],
        preview_url: bool
    ) -> Optional[str]:
        """Send message via direct API."""
        # Placeholder for WhatsApp Cloud API implementation
        # This would use requests or aiohttp to call the API
        return None

    async def send_template(
        self,
        to: str,
        template_name: str,
        language_code: str = "en_US",
        components: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Send a template message (Business API only).

        Args:
            to: Recipient phone number
            template_name: Name of approved template
            language_code: Template language
            components: Template component parameters

        Returns:
            Message ID or None on failure
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            try:
                result = await self._call_mcp("send_template", {
                    "to": to,
                    "template": {
                        "name": template_name,
                        "language": {"code": language_code},
                        "components": components or []
                    }
                })
                return result.get("messages", [{}])[0].get("id")
            except Exception as e:
                raise AdapterError(
                    f"Failed to send template: {str(e)}",
                    self.adapter_name,
                    "TEMPLATE_FAILED",
                    original_error=e
                )
        elif self.connection_mode == ConnectionMode.MOCK:
            return f"mock_template_{datetime.now().timestamp()}"
        else:
            return None

    # -------------------------------------------------------------------------
    # MEDIA HANDLING
    # -------------------------------------------------------------------------

    async def download_media(
        self,
        media_id: str
    ) -> Optional[bytes]:
        """
        Download media content.

        Args:
            media_id: Media ID from message

        Returns:
            Media content as bytes
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            try:
                result = await self._call_mcp("get_media", {"mediaId": media_id})
                # MCP server returns base64 encoded data
                import base64
                return base64.b64decode(result.get("data", ""))
            except Exception:
                return None
        elif self.connection_mode == ConnectionMode.API:
            # Would use access token to fetch from media URL
            return None
        else:
            return None

    async def upload_media(
        self,
        file_data: bytes,
        mime_type: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload media for sending.

        Args:
            file_data: File content
            mime_type: MIME type of file
            filename: Optional filename

        Returns:
            Media ID for use in send_message
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            try:
                import base64
                result = await self._call_mcp("upload_media", {
                    "data": base64.b64encode(file_data).decode("utf-8"),
                    "mimeType": mime_type,
                    "filename": filename
                })
                return result.get("id")
            except Exception:
                return None
        else:
            return None

    # -------------------------------------------------------------------------
    # CHAT ACTIONS
    # -------------------------------------------------------------------------

    async def mark_read(self, message_id: str) -> bool:
        """Mark a message as read."""
        if self.connection_mode == ConnectionMode.MCP:
            try:
                await self._call_mcp("mark_read", {"messageId": message_id})
                return True
            except Exception:
                return False
        elif self.connection_mode == ConnectionMode.API:
            # Would call messages endpoint with status = read
            return True
        elif self.connection_mode == ConnectionMode.MOCK:
            return True
        return False

    async def archive_chat(self, chat_id: str) -> bool:
        """Archive a chat."""
        if self.connection_mode == ConnectionMode.MCP:
            try:
                await self._call_mcp("archive_chat", {"chatId": chat_id})
                return True
            except Exception:
                return False
        elif self.connection_mode == ConnectionMode.MOCK:
            return True
        return False

    async def mute_chat(
        self,
        chat_id: str,
        mute_until: Optional[datetime] = None
    ) -> bool:
        """Mute a chat."""
        if self.connection_mode == ConnectionMode.MCP:
            try:
                args: Dict[str, Any] = {"chatId": chat_id}
                if mute_until:
                    args["until"] = mute_until.isoformat()
                await self._call_mcp("mute_chat", args)
                return True
            except Exception:
                return False
        elif self.connection_mode == ConnectionMode.MOCK:
            return True
        return False

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    def format_phone_number(self, number: str) -> str:
        """
        Format a phone number to WhatsApp format.

        Args:
            number: Phone number in any format

        Returns:
            Phone number in international format (+1234567890)
        """
        # Remove all non-digit characters except leading +
        cleaned = re.sub(r'[^\d+]', '', number)

        # Ensure it starts with +
        if not cleaned.startswith('+'):
            # Assume US if no country code
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned
            elif not cleaned.startswith('1') and len(cleaned) == 11:
                cleaned = '+' + cleaned
            else:
                cleaned = '+' + cleaned

        return cleaned

    def get_pending_responses(self) -> List[WhatsAppMessage]:
        """
        Get messages that require a response.

        Returns:
            List of messages marked as requiring response
        """
        # This would query cached messages or a database
        # For now, return empty list
        return []
