"""
Gmail Adapter for Alfred

Gmail integration with MCP server support and direct API fallback.
Manages email messages, threads, triage rules, and attachments.

MCP Servers Supported:
- mcp-gsuite (gmail module)
- gmail-mcp

Direct API:
- Gmail API v1
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
import asyncio
import base64
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase

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

class EmailPriority(Enum):
    """Email priority levels for triage."""
    URGENT = "urgent"            # Requires immediate attention
    HIGH = "high"                # Important, handle today
    NORMAL = "normal"            # Standard priority
    LOW = "low"                  # Can wait
    ARCHIVE = "archive"          # Auto-archive, no action needed


class EmailCategory(Enum):
    """Email categories for organization."""
    CLINICAL = "clinical"        # Patient-related
    BUSINESS = "business"        # Business operations
    PERSONAL = "personal"        # Personal correspondence
    NEWSLETTER = "newsletter"    # Subscriptions, newsletters
    TRANSACTIONAL = "transactional"  # Receipts, confirmations
    SOCIAL = "social"            # Social media notifications
    PROMOTIONAL = "promotional"  # Marketing, promotions
    SPAM = "spam"                # Unwanted email
    UNKNOWN = "unknown"


class TriageAction(Enum):
    """Actions that can be taken by triage rules."""
    LABEL = "label"              # Apply label
    ARCHIVE = "archive"          # Archive message
    STAR = "star"                # Star message
    MARK_READ = "mark_read"      # Mark as read
    MARK_UNREAD = "mark_unread"  # Mark as unread
    FORWARD = "forward"          # Forward to another address
    MOVE = "move"                # Move to folder
    DELETE = "delete"            # Move to trash
    NOTIFY = "notify"            # Send notification
    FLAG_URGENT = "flag_urgent"  # Flag as urgent


class AttachmentType(Enum):
    """Types of email attachments."""
    DOCUMENT = "document"        # PDF, DOC, etc.
    SPREADSHEET = "spreadsheet"  # XLS, CSV, etc.
    IMAGE = "image"              # JPG, PNG, etc.
    VIDEO = "video"              # MP4, MOV, etc.
    ARCHIVE = "archive"          # ZIP, RAR, etc.
    OTHER = "other"


# Extension mappings
ATTACHMENT_TYPE_MAP: Dict[str, AttachmentType] = {
    ".pdf": AttachmentType.DOCUMENT,
    ".doc": AttachmentType.DOCUMENT,
    ".docx": AttachmentType.DOCUMENT,
    ".txt": AttachmentType.DOCUMENT,
    ".rtf": AttachmentType.DOCUMENT,
    ".xls": AttachmentType.SPREADSHEET,
    ".xlsx": AttachmentType.SPREADSHEET,
    ".csv": AttachmentType.SPREADSHEET,
    ".jpg": AttachmentType.IMAGE,
    ".jpeg": AttachmentType.IMAGE,
    ".png": AttachmentType.IMAGE,
    ".gif": AttachmentType.IMAGE,
    ".mp4": AttachmentType.VIDEO,
    ".mov": AttachmentType.VIDEO,
    ".avi": AttachmentType.VIDEO,
    ".zip": AttachmentType.ARCHIVE,
    ".rar": AttachmentType.ARCHIVE,
    ".7z": AttachmentType.ARCHIVE,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int
    attachment_type: AttachmentType = AttachmentType.OTHER

    # Content (loaded on demand)
    data: Optional[bytes] = None

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def extension(self) -> str:
        """File extension."""
        if "." in self.filename:
            return "." + self.filename.rsplit(".", 1)[1].lower()
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.attachment_id,
            "filename": self.filename,
            "mimeType": self.mime_type,
            "size": self.size_bytes,
            "type": self.attachment_type.value
        }

    @classmethod
    def from_gmail_part(cls, part: Dict[str, Any]) -> "EmailAttachment":
        """Create from Gmail API part data."""
        filename = part.get("filename", "attachment")
        ext = ""
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[1].lower()

        return cls(
            attachment_id=part.get("body", {}).get("attachmentId", ""),
            filename=filename,
            mime_type=part.get("mimeType", "application/octet-stream"),
            size_bytes=part.get("body", {}).get("size", 0),
            attachment_type=ATTACHMENT_TYPE_MAP.get(ext, AttachmentType.OTHER)
        )


@dataclass
class EmailAddress:
    """Represents an email address with optional name."""
    email: str
    name: Optional[str] = None

    def __str__(self) -> str:
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email

    @classmethod
    def parse(cls, value: str) -> "EmailAddress":
        """Parse from 'Name <email>' or 'email' format."""
        match = re.match(r'^(.+?)\s*<(.+?)>$', value)
        if match:
            return cls(email=match.group(2), name=match.group(1).strip())
        return cls(email=value.strip())


@dataclass
class EmailMessage:
    """Represents an email message."""
    message_id: str
    thread_id: str
    subject: str
    snippet: str

    # Addresses
    from_address: EmailAddress
    to_addresses: List[EmailAddress] = field(default_factory=list)
    cc_addresses: List[EmailAddress] = field(default_factory=list)
    bcc_addresses: List[EmailAddress] = field(default_factory=list)
    reply_to: Optional[EmailAddress] = None

    # Content
    body_plain: str = ""
    body_html: str = ""

    # Metadata
    date: Optional[datetime] = None
    labels: List[str] = field(default_factory=list)
    is_read: bool = False
    is_starred: bool = False
    is_important: bool = False
    is_draft: bool = False

    # Attachments
    attachments: List[EmailAttachment] = field(default_factory=list)
    has_attachments: bool = False

    # Triage fields (populated by Alfred)
    priority: EmailPriority = EmailPriority.NORMAL
    category: EmailCategory = EmailCategory.UNKNOWN
    triage_notes: List[str] = field(default_factory=list)

    # Raw data for reference
    raw_headers: Dict[str, str] = field(default_factory=dict)

    @property
    def is_unread(self) -> bool:
        return not self.is_read

    @property
    def word_count(self) -> int:
        """Approximate word count of plain text body."""
        return len(self.body_plain.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.message_id,
            "threadId": self.thread_id,
            "subject": self.subject,
            "snippet": self.snippet,
            "from": str(self.from_address),
            "to": [str(a) for a in self.to_addresses],
            "cc": [str(a) for a in self.cc_addresses],
            "date": self.date.isoformat() if self.date else None,
            "labels": self.labels,
            "isRead": self.is_read,
            "isStarred": self.is_starred,
            "hasAttachments": self.has_attachments,
            "attachmentCount": len(self.attachments),
            "priority": self.priority.value,
            "category": self.category.value,
            "triageNotes": self.triage_notes
        }

    @classmethod
    def from_gmail_message(cls, data: Dict[str, Any]) -> "EmailMessage":
        """Create from Gmail API message response."""
        headers = {}
        for header in data.get("payload", {}).get("headers", []):
            headers[header["name"].lower()] = header["value"]

        # Parse addresses
        from_addr = EmailAddress.parse(headers.get("from", "unknown@unknown.com"))
        to_addrs = [
            EmailAddress.parse(a.strip())
            for a in headers.get("to", "").split(",") if a.strip()
        ]
        cc_addrs = [
            EmailAddress.parse(a.strip())
            for a in headers.get("cc", "").split(",") if a.strip()
        ]

        # Parse date
        date_str = headers.get("date", "")
        parsed_date = None
        if date_str:
            # Gmail uses various date formats
            for fmt in [
                "%a, %d %b %Y %H:%M:%S %z",
                "%d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S"
            ]:
                try:
                    parsed_date = datetime.strptime(date_str[:31], fmt)
                    break
                except ValueError:
                    continue

        # Extract body
        body_plain = ""
        body_html = ""
        attachments = []

        def extract_body(payload: Dict[str, Any]) -> None:
            nonlocal body_plain, body_html, attachments

            mime_type = payload.get("mimeType", "")

            if mime_type == "text/plain":
                body_data = payload.get("body", {}).get("data", "")
                if body_data:
                    body_plain = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
            elif mime_type == "text/html":
                body_data = payload.get("body", {}).get("data", "")
                if body_data:
                    body_html = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
            elif payload.get("filename"):
                attachments.append(EmailAttachment.from_gmail_part(payload))

            for part in payload.get("parts", []):
                extract_body(part)

        extract_body(data.get("payload", {}))

        # Determine label states
        label_ids = data.get("labelIds", [])
        is_read = "UNREAD" not in label_ids
        is_starred = "STARRED" in label_ids
        is_important = "IMPORTANT" in label_ids
        is_draft = "DRAFT" in label_ids

        return cls(
            message_id=data.get("id", ""),
            thread_id=data.get("threadId", ""),
            subject=headers.get("subject", "(no subject)"),
            snippet=data.get("snippet", ""),
            from_address=from_addr,
            to_addresses=to_addrs,
            cc_addresses=cc_addrs,
            reply_to=EmailAddress.parse(headers["reply-to"]) if "reply-to" in headers else None,
            body_plain=body_plain,
            body_html=body_html,
            date=parsed_date,
            labels=label_ids,
            is_read=is_read,
            is_starred=is_starred,
            is_important=is_important,
            is_draft=is_draft,
            attachments=attachments,
            has_attachments=len(attachments) > 0,
            raw_headers=headers
        )


@dataclass
class EmailThread:
    """Represents an email thread (conversation)."""
    thread_id: str
    subject: str
    messages: List[EmailMessage] = field(default_factory=list)
    snippet: str = ""
    history_id: Optional[str] = None

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def participants(self) -> Set[str]:
        """All email addresses involved in this thread."""
        participants = set()
        for msg in self.messages:
            participants.add(msg.from_address.email)
            for addr in msg.to_addresses:
                participants.add(addr.email)
        return participants

    @property
    def latest_message(self) -> Optional[EmailMessage]:
        if not self.messages:
            return None
        return max(self.messages, key=lambda m: m.date or datetime.min)

    @property
    def has_unread(self) -> bool:
        return any(not m.is_read for m in self.messages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.thread_id,
            "subject": self.subject,
            "snippet": self.snippet,
            "messageCount": self.message_count,
            "participants": list(self.participants),
            "hasUnread": self.has_unread,
            "latestDate": self.latest_message.date.isoformat()
                if self.latest_message and self.latest_message.date else None
        }


@dataclass
class TriageRule:
    """
    Rule for automatic email triage.

    Rules can match on sender, subject, content, and labels,
    then apply specified actions.
    """
    rule_id: str
    name: str
    enabled: bool = True
    priority: int = 0  # Higher = checked first

    # Match conditions (all must match if specified)
    match_from: Optional[List[str]] = None  # Email addresses or domains
    match_to: Optional[List[str]] = None
    match_subject: Optional[List[str]] = None  # Regex patterns
    match_body: Optional[List[str]] = None  # Regex patterns
    match_labels: Optional[List[str]] = None
    match_has_attachment: Optional[bool] = None

    # Actions to take
    actions: List[Tuple[TriageAction, Any]] = field(default_factory=list)

    # Resulting classification
    set_priority: Optional[EmailPriority] = None
    set_category: Optional[EmailCategory] = None

    def matches(self, message: EmailMessage) -> bool:
        """Check if a message matches this rule."""
        # Check from address
        if self.match_from:
            from_email = message.from_address.email.lower()
            from_domain = from_email.split("@")[1] if "@" in from_email else ""
            if not any(
                pattern.lower() in from_email or pattern.lower() == from_domain
                for pattern in self.match_from
            ):
                return False

        # Check to addresses
        if self.match_to:
            to_emails = [a.email.lower() for a in message.to_addresses]
            if not any(
                any(pattern.lower() in email for email in to_emails)
                for pattern in self.match_to
            ):
                return False

        # Check subject
        if self.match_subject:
            if not any(
                re.search(pattern, message.subject, re.IGNORECASE)
                for pattern in self.match_subject
            ):
                return False

        # Check body
        if self.match_body:
            combined_body = message.body_plain + message.body_html
            if not any(
                re.search(pattern, combined_body, re.IGNORECASE)
                for pattern in self.match_body
            ):
                return False

        # Check labels
        if self.match_labels:
            if not any(label in message.labels for label in self.match_labels):
                return False

        # Check attachments
        if self.match_has_attachment is not None:
            if message.has_attachments != self.match_has_attachment:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.rule_id,
            "name": self.name,
            "enabled": self.enabled,
            "priority": self.priority,
            "conditions": {
                "from": self.match_from,
                "to": self.match_to,
                "subject": self.match_subject,
                "body": self.match_body,
                "labels": self.match_labels,
                "hasAttachment": self.match_has_attachment
            },
            "actions": [(a.value, v) for a, v in self.actions],
            "classification": {
                "priority": self.set_priority.value if self.set_priority else None,
                "category": self.set_category.value if self.set_category else None
            }
        }


# =============================================================================
# GMAIL ADAPTER
# =============================================================================

class GmailAdapter(BaseAdapter):
    """
    Gmail adapter with MCP and direct API support.

    Provides:
    - Message retrieval and search
    - Thread management
    - Sending and composing emails
    - Triage rule application
    - Attachment handling
    """

    # MCP server names
    MCP_GSUITE = "mcp-gsuite"
    MCP_GMAIL = "gmail-mcp"

    # Default labels
    LABEL_INBOX = "INBOX"
    LABEL_SENT = "SENT"
    LABEL_DRAFT = "DRAFT"
    LABEL_TRASH = "TRASH"
    LABEL_SPAM = "SPAM"
    LABEL_STARRED = "STARRED"
    LABEL_IMPORTANT = "IMPORTANT"
    LABEL_UNREAD = "UNREAD"

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        api_credentials: Optional[Dict[str, Any]] = None,
        enable_mock: bool = False
    ):
        """
        Initialize Gmail adapter.

        Args:
            mcp_client: MCP client for server communication
            api_credentials: Google API credentials for direct API access
            enable_mock: Enable mock mode for testing
        """
        # Determine MCP server
        mcp_server = None
        if mcp_client:
            if mcp_client.is_server_available(self.MCP_GSUITE):
                mcp_server = self.MCP_GSUITE
            elif mcp_client.is_server_available(self.MCP_GMAIL):
                mcp_server = self.MCP_GMAIL

        super().__init__(
            adapter_name="GmailAdapter",
            mcp_client=mcp_client,
            mcp_server_name=mcp_server,
            api_credentials=api_credentials,
            enable_mock=enable_mock
        )

        self._api_service = None
        self._triage_rules: List[TriageRule] = []

    async def _connect_api(self) -> None:
        """Establish direct Gmail API connection."""
        if not self.api_credentials:
            raise AuthenticationError(
                self.adapter_name,
                "No API credentials provided"
            )
        # In production: build Gmail API service
        pass

    # -------------------------------------------------------------------------
    # TRIAGE RULES
    # -------------------------------------------------------------------------

    def add_triage_rule(self, rule: TriageRule) -> None:
        """Add a triage rule."""
        self._triage_rules.append(rule)
        self._triage_rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_triage_rule(self, rule_id: str) -> bool:
        """Remove a triage rule by ID."""
        for i, rule in enumerate(self._triage_rules):
            if rule.rule_id == rule_id:
                del self._triage_rules[i]
                return True
        return False

    def apply_triage_rules(self, message: EmailMessage) -> EmailMessage:
        """
        Apply triage rules to a message.

        Args:
            message: Message to triage

        Returns:
            Message with triage fields populated
        """
        for rule in self._triage_rules:
            if not rule.enabled:
                continue

            if rule.matches(message):
                # Apply classification
                if rule.set_priority:
                    message.priority = rule.set_priority
                if rule.set_category:
                    message.category = rule.set_category

                message.triage_notes.append(f"Matched rule: {rule.name}")

                # Note: Action execution would happen separately
                # This just classifies the message
                break  # First matching rule wins

        return message

    # -------------------------------------------------------------------------
    # MESSAGE RETRIEVAL
    # -------------------------------------------------------------------------

    async def get_messages(
        self,
        query: Optional[str] = None,
        label_ids: Optional[List[str]] = None,
        max_results: int = 100,
        include_spam_trash: bool = False,
        page_token: Optional[str] = None
    ) -> Tuple[List[EmailMessage], Optional[str]]:
        """
        Retrieve email messages.

        Args:
            query: Gmail search query (e.g., "from:example@gmail.com is:unread")
            label_ids: Filter by label IDs
            max_results: Maximum number of messages
            include_spam_trash: Include spam and trash
            page_token: Token for pagination

        Returns:
            Tuple of (messages, next_page_token)
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_messages_mcp(
                query, label_ids, max_results, include_spam_trash, page_token
            )
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_messages_api(
                query, label_ids, max_results, include_spam_trash, page_token
            )
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_messages_mock(), None
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _get_messages_mcp(
        self,
        query: Optional[str],
        label_ids: Optional[List[str]],
        max_results: int,
        include_spam_trash: bool,
        page_token: Optional[str]
    ) -> Tuple[List[EmailMessage], Optional[str]]:
        """Get messages via MCP server."""
        args = {
            "maxResults": max_results,
            "includeSpamTrash": include_spam_trash
        }
        if query:
            args["q"] = query
        if label_ids:
            args["labelIds"] = label_ids
        if page_token:
            args["pageToken"] = page_token

        if self.mcp_server_name == self.MCP_GSUITE:
            result = await self._call_mcp("gmail.list_messages", args)
        else:
            result = await self._call_mcp("list_messages", args)

        messages = []
        for msg_data in result.get("messages", []):
            # Get full message
            full_msg = await self.get_message(msg_data["id"])
            if full_msg:
                messages.append(full_msg)

        return messages, result.get("nextPageToken")

    async def _get_messages_api(
        self,
        query: Optional[str],
        label_ids: Optional[List[str]],
        max_results: int,
        include_spam_trash: bool,
        page_token: Optional[str]
    ) -> Tuple[List[EmailMessage], Optional[str]]:
        """Get messages via direct API."""
        # Placeholder for API implementation
        return [], None

    def _get_messages_mock(self) -> List[EmailMessage]:
        """Get mock messages for testing."""
        now = datetime.now()
        return [
            EmailMessage(
                message_id="mock_1",
                thread_id="thread_1",
                subject="Urgent: Patient Follow-up Required",
                snippet="Please review the attached test results...",
                from_address=EmailAddress("nurse@hospital.com", "Head Nurse"),
                to_addresses=[EmailAddress("doctor@hospital.com")],
                date=now - timedelta(hours=1),
                labels=["INBOX", "UNREAD", "IMPORTANT"],
                is_read=False,
                is_important=True,
                priority=EmailPriority.URGENT,
                category=EmailCategory.CLINICAL
            ),
            EmailMessage(
                message_id="mock_2",
                thread_id="thread_2",
                subject="Weekly Newsletter - Health Updates",
                snippet="This week's top health news...",
                from_address=EmailAddress("newsletter@health.com", "Health News"),
                to_addresses=[EmailAddress("subscriber@email.com")],
                date=now - timedelta(hours=3),
                labels=["INBOX"],
                is_read=True,
                priority=EmailPriority.LOW,
                category=EmailCategory.NEWSLETTER
            ),
            EmailMessage(
                message_id="mock_3",
                thread_id="thread_3",
                subject="Meeting Tomorrow at 3 PM",
                snippet="Let's discuss the quarterly review...",
                from_address=EmailAddress("colleague@company.com", "John Smith"),
                to_addresses=[EmailAddress("me@company.com")],
                date=now - timedelta(hours=5),
                labels=["INBOX", "UNREAD"],
                is_read=False,
                priority=EmailPriority.NORMAL,
                category=EmailCategory.BUSINESS
            ),
        ]

    async def get_message(
        self,
        message_id: str,
        format: str = "full"
    ) -> Optional[EmailMessage]:
        """
        Get a single message by ID.

        Args:
            message_id: Message ID
            format: Response format (full, metadata, minimal, raw)

        Returns:
            EmailMessage or None if not found
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_message_mcp(message_id, format)
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_message_api(message_id, format)
        elif self.connection_mode == ConnectionMode.MOCK:
            return None
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _get_message_mcp(
        self,
        message_id: str,
        format: str
    ) -> Optional[EmailMessage]:
        """Get message via MCP server."""
        try:
            if self.mcp_server_name == self.MCP_GSUITE:
                result = await self._call_mcp("gmail.get_message", {
                    "messageId": message_id,
                    "format": format
                })
            else:
                result = await self._call_mcp("get_message", {
                    "id": message_id,
                    "format": format
                })

            message = EmailMessage.from_gmail_message(result)
            return self.apply_triage_rules(message)
        except Exception:
            return None

    async def _get_message_api(
        self,
        message_id: str,
        format: str
    ) -> Optional[EmailMessage]:
        """Get message via direct API."""
        # Placeholder for API implementation
        return None

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int = 50
    ) -> List[EmailMessage]:
        """
        Search for messages using Gmail search syntax.

        Supports operators:
        - from: to: subject: label:
        - is:read is:unread is:starred
        - has:attachment larger: smaller:
        - after: before: older: newer:

        Args:
            query: Gmail search query
            max_results: Maximum results to return

        Returns:
            List of matching messages
        """
        messages, _ = await self.get_messages(query=query, max_results=max_results)
        return messages

    # -------------------------------------------------------------------------
    # SENDING
    # -------------------------------------------------------------------------

    async def send_message(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[Tuple[str, bytes, str]]] = None,
        html_body: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Send an email message.

        Args:
            to: List of recipient email addresses
            subject: Email subject
            body: Plain text body
            cc: CC recipients
            bcc: BCC recipients
            reply_to: Reply-to address
            attachments: List of (filename, content, mime_type) tuples
            html_body: HTML body (alternative to plain text)
            thread_id: Thread ID to reply to

        Returns:
            Message ID of sent message, or None on failure
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            return await self._send_message_mcp(
                to, subject, body, cc, bcc, reply_to, attachments, html_body, thread_id
            )
        elif self.connection_mode == ConnectionMode.API:
            return await self._send_message_api(
                to, subject, body, cc, bcc, reply_to, attachments, html_body, thread_id
            )
        elif self.connection_mode == ConnectionMode.MOCK:
            return f"mock_sent_{datetime.now().timestamp()}"
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _send_message_mcp(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]],
        bcc: Optional[List[str]],
        reply_to: Optional[str],
        attachments: Optional[List[Tuple[str, bytes, str]]],
        html_body: Optional[str],
        thread_id: Optional[str]
    ) -> Optional[str]:
        """Send message via MCP server."""
        # Build message
        message_data = {
            "to": to,
            "subject": subject,
            "body": body
        }
        if cc:
            message_data["cc"] = cc
        if bcc:
            message_data["bcc"] = bcc
        if reply_to:
            message_data["replyTo"] = reply_to
        if html_body:
            message_data["htmlBody"] = html_body
        if thread_id:
            message_data["threadId"] = thread_id

        # Note: Attachments would need to be base64 encoded
        if attachments:
            message_data["attachments"] = [
                {
                    "filename": filename,
                    "data": base64.b64encode(content).decode("utf-8"),
                    "mimeType": mime_type
                }
                for filename, content, mime_type in attachments
            ]

        try:
            if self.mcp_server_name == self.MCP_GSUITE:
                result = await self._call_mcp("gmail.send_message", message_data)
            else:
                result = await self._call_mcp("send_message", message_data)

            return result.get("id")
        except Exception as e:
            raise AdapterError(
                f"Failed to send message: {str(e)}",
                self.adapter_name,
                "SEND_FAILED",
                original_error=e
            )

    async def _send_message_api(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]],
        bcc: Optional[List[str]],
        reply_to: Optional[str],
        attachments: Optional[List[Tuple[str, bytes, str]]],
        html_body: Optional[str],
        thread_id: Optional[str]
    ) -> Optional[str]:
        """Send message via direct API."""
        # Placeholder for API implementation
        return None

    # -------------------------------------------------------------------------
    # ATTACHMENT HANDLING
    # -------------------------------------------------------------------------

    async def get_attachment(
        self,
        message_id: str,
        attachment_id: str
    ) -> Optional[bytes]:
        """
        Download an attachment.

        Args:
            message_id: Parent message ID
            attachment_id: Attachment ID

        Returns:
            Attachment content as bytes
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_attachment_mcp(message_id, attachment_id)
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_attachment_api(message_id, attachment_id)
        else:
            return None

    async def _get_attachment_mcp(
        self,
        message_id: str,
        attachment_id: str
    ) -> Optional[bytes]:
        """Get attachment via MCP server."""
        try:
            if self.mcp_server_name == self.MCP_GSUITE:
                result = await self._call_mcp("gmail.get_attachment", {
                    "messageId": message_id,
                    "attachmentId": attachment_id
                })
            else:
                result = await self._call_mcp("get_attachment", {
                    "messageId": message_id,
                    "id": attachment_id
                })

            data = result.get("data", "")
            return base64.urlsafe_b64decode(data)
        except Exception:
            return None

    async def _get_attachment_api(
        self,
        message_id: str,
        attachment_id: str
    ) -> Optional[bytes]:
        """Get attachment via direct API."""
        # Placeholder for API implementation
        return None

    # -------------------------------------------------------------------------
    # MESSAGE ACTIONS
    # -------------------------------------------------------------------------

    async def modify_labels(
        self,
        message_id: str,
        add_labels: Optional[List[str]] = None,
        remove_labels: Optional[List[str]] = None
    ) -> bool:
        """
        Modify message labels.

        Args:
            message_id: Message ID
            add_labels: Labels to add
            remove_labels: Labels to remove

        Returns:
            True if successful
        """
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            try:
                args = {"messageId": message_id}
                if add_labels:
                    args["addLabelIds"] = add_labels
                if remove_labels:
                    args["removeLabelIds"] = remove_labels

                if self.mcp_server_name == self.MCP_GSUITE:
                    await self._call_mcp("gmail.modify_message", args)
                else:
                    await self._call_mcp("modify_labels", args)
                return True
            except Exception:
                return False
        elif self.connection_mode == ConnectionMode.MOCK:
            return True
        else:
            return False

    async def mark_read(self, message_id: str) -> bool:
        """Mark a message as read."""
        return await self.modify_labels(message_id, remove_labels=[self.LABEL_UNREAD])

    async def mark_unread(self, message_id: str) -> bool:
        """Mark a message as unread."""
        return await self.modify_labels(message_id, add_labels=[self.LABEL_UNREAD])

    async def star(self, message_id: str) -> bool:
        """Star a message."""
        return await self.modify_labels(message_id, add_labels=[self.LABEL_STARRED])

    async def unstar(self, message_id: str) -> bool:
        """Remove star from a message."""
        return await self.modify_labels(message_id, remove_labels=[self.LABEL_STARRED])

    async def archive(self, message_id: str) -> bool:
        """Archive a message (remove from inbox)."""
        return await self.modify_labels(message_id, remove_labels=[self.LABEL_INBOX])

    async def trash(self, message_id: str) -> bool:
        """Move a message to trash."""
        self._check_rate_limit()

        if self.connection_mode == ConnectionMode.MCP:
            try:
                if self.mcp_server_name == self.MCP_GSUITE:
                    await self._call_mcp("gmail.trash_message", {"messageId": message_id})
                else:
                    await self._call_mcp("trash_message", {"id": message_id})
                return True
            except Exception:
                return False
        elif self.connection_mode == ConnectionMode.MOCK:
            return True
        else:
            return False
