"""
Intake Agent - Unified Ingestion System for Alfred

Role: Unified ingestion system for all inbound communication channels.
Normalizes, timestamps, and queues all incoming information for Alfred's review.
Never interprets or prioritizes - only collects and structures.

GitHub Tools to integrate (interfaces prepared for):
- whatsapp-mcp for WhatsApp messages
- mcp-gsuite for Gmail integration
- PaddleOCR for document OCR
- MinerU for PDF extraction
"""

from . import OperationsAgent, AgentResponse, AlfredState
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Set
import hashlib
import json
import logging
import re
import uuid

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChannelType(Enum):
    """Supported intake channels."""
    WHATSAPP = "WhatsApp"
    EMAIL = "Email"
    SCAN = "Scan"
    CALENDAR = "Calendar"
    VOICE = "Voice"


class UrgencyMarker(Enum):
    """Urgency classification for inbound items."""
    NONE = "none"
    EXPLICIT_URGENT = "explicit_urgent"
    TIME_SENSITIVE = "time_sensitive"


class IncludeFilter(Enum):
    """Filter types for intake requests."""
    ALL = "all"
    UNREAD = "unread"
    FLAGGED = "flagged"


class TimeWindow(Enum):
    """Predefined time windows for intake requests."""
    SINCE_LAST_CHECK = "since_last_check"
    LAST_HOUR = "last_hour"
    LAST_24_HOURS = "last_24_hours"
    LAST_WEEK = "last_week"
    CUSTOM = "custom"


@dataclass
class Attachment:
    """Represents an attachment in an inbound item."""
    filename: str
    file_type: str
    size_bytes: int
    mime_type: Optional[str] = None
    content_preview: Optional[str] = None  # For OCR'd documents
    raw_reference: Optional[str] = None


@dataclass
class InboundItem:
    """
    Normalized schema for all inbound communication items.

    This is the standard format that all channel adapters must produce,
    regardless of their source format.
    """
    # Required fields
    source: ChannelType
    timestamp: datetime
    sender: str
    raw_reference: str  # ID for full retrieval from original system

    # Optional fields with defaults
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject: Optional[str] = None
    thread_id: Optional[str] = None
    content_preview: str = ""  # First 100 chars
    full_content: Optional[str] = None  # Full content if available
    attachments: List[Attachment] = field(default_factory=list)
    urgency_markers: UrgencyMarker = UrgencyMarker.NONE
    is_read: bool = False
    is_flagged: bool = False

    # Deduplication hash
    _content_hash: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Generate content hash for deduplication."""
        if self._content_hash is None:
            self._content_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate a hash for deduplication based on sender, content, and timestamp."""
        # Use sender, subject, content preview, and approximate timestamp
        # Round timestamp to nearest minute for fuzzy matching
        rounded_time = self.timestamp.replace(second=0, microsecond=0)
        hash_content = f"{self.sender}|{self.subject or ''}|{self.content_preview}|{rounded_time.isoformat()}"
        return hashlib.sha256(hash_content.encode()).hexdigest()

    @property
    def content_hash(self) -> str:
        """Return the content hash for deduplication."""
        return self._content_hash

    @property
    def attachment_count(self) -> int:
        """Return the number of attachments."""
        return len(self.attachments)

    @property
    def attachment_types(self) -> List[str]:
        """Return list of attachment types."""
        return [a.file_type for a in self.attachments]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "subject": self.subject,
            "thread_id": self.thread_id,
            "content_preview": self.content_preview,
            "attachments": {
                "count": self.attachment_count,
                "types": self.attachment_types
            },
            "urgency_markers": self.urgency_markers.value,
            "raw_reference": self.raw_reference,
            "is_read": self.is_read,
            "is_flagged": self.is_flagged
        }


@dataclass
class BatchSummary:
    """Summary statistics for an inbound batch."""
    by_channel: Dict[str, int] = field(default_factory=dict)
    with_attachments: int = 0
    with_urgency_markers: int = 0
    unread_count: int = 0
    flagged_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "by_channel": self.by_channel,
            "with_attachments": self.with_attachments,
            "with_urgency_markers": self.with_urgency_markers,
            "unread_count": self.unread_count,
            "flagged_count": self.flagged_count
        }


@dataclass
class InboundBatch:
    """
    Batch of normalized inbound items.

    This is the output format that the Intake Agent produces for Alfred.
    """
    batch_id: str
    items_count: int
    channels_checked: List[ChannelType]
    items: List[InboundItem]
    summary: BatchSummary
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "complete"
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching the INBOUND_BATCH format."""
        return {
            "batch_id": self.batch_id,
            "items_count": self.items_count,
            "channels_checked": [c.value for c in self.channels_checked],
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
            "batch_summary": self.summary.to_dict(),
            "errors": self.errors
        }

    def to_formatted_output(self) -> str:
        """Format as the specification output format."""
        lines = [
            "INBOUND_BATCH",
            f"- Batch ID: {self.batch_id}",
            f"- Items Count: {self.items_count}",
            f"- Channels Checked: {', '.join(c.value for c in self.channels_checked)}",
            ""
        ]

        if self.items_count == 0:
            lines.append("- Status: No new items since last check")
        else:
            for i, item in enumerate(self.items, 1):
                lines.extend([
                    f"- Item {i}:",
                    f"  - Source: {item.source.value}",
                    f"  - Timestamp: {item.timestamp.isoformat()}",
                    f"  - Sender: {item.sender}",
                    f"  - Subject/Thread: {item.subject or item.thread_id or 'N/A'}",
                    f"  - Content Preview: {item.content_preview[:100]}",
                    f"  - Attachments: {item.attachment_count} ({', '.join(item.attachment_types) or 'none'})",
                    f"  - Urgency Markers: {item.urgency_markers.value}",
                    f"  - Raw Reference: {item.raw_reference}",
                    ""
                ])

            lines.extend([
                "- Batch Summary:",
                f"  - By Channel: {self.summary.by_channel}",
                f"  - With Attachments: {self.summary.with_attachments}",
                f"  - With Urgency Markers: {self.summary.with_urgency_markers}"
            ])

        return "\n".join(lines)


@dataclass
class IntakeRequest:
    """
    Parsed intake request from Alfred.

    Corresponds to the INTAKE_REQUEST format in the specification.
    """
    channels: List[ChannelType]
    time_window: TimeWindow
    include_filter: IncludeFilter
    custom_start: Optional[datetime] = None
    custom_end: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntakeRequest":
        """Parse an intake request from dictionary."""
        channels = []
        channel_input = data.get("channels", ["all"])

        if isinstance(channel_input, str):
            channel_input = [channel_input]

        if "all" in channel_input:
            channels = list(ChannelType)
        else:
            for ch in channel_input:
                try:
                    channels.append(ChannelType(ch))
                except ValueError:
                    # Try case-insensitive match
                    for ct in ChannelType:
                        if ct.value.lower() == ch.lower():
                            channels.append(ct)
                            break

        # Parse time window
        time_window_input = data.get("time_window", "since_last_check")
        try:
            time_window = TimeWindow(time_window_input)
        except ValueError:
            time_window = TimeWindow.SINCE_LAST_CHECK

        # Parse include filter
        include_input = data.get("include", "all")
        try:
            include_filter = IncludeFilter(include_input)
        except ValueError:
            include_filter = IncludeFilter.ALL

        return cls(
            channels=channels,
            time_window=time_window,
            include_filter=include_filter,
            custom_start=data.get("custom_start"),
            custom_end=data.get("custom_end")
        )


class ChannelAdapter(ABC):
    """
    Abstract base class for channel adapters.

    Each channel adapter is responsible for:
    1. Connecting to its data source
    2. Fetching items within a time window
    3. Normalizing items to the InboundItem schema
    4. Extracting sender, subject, content preview, attachments
    5. Detecting urgency markers
    """

    def __init__(self, channel_type: ChannelType):
        self.channel_type = channel_type
        self.last_check: Optional[datetime] = None
        self._is_connected = False
        self._error_count = 0
        self._max_retries = 3

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the channel.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the channel."""
        pass

    @abstractmethod
    async def fetch_items(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_filter: IncludeFilter = IncludeFilter.ALL
    ) -> List[InboundItem]:
        """
        Fetch items from the channel within the specified time window.

        Args:
            since: Start of time window (inclusive)
            until: End of time window (inclusive)
            include_filter: Filter for which items to include

        Returns:
            List of normalized InboundItem objects
        """
        pass

    @abstractmethod
    async def get_full_content(self, raw_reference: str) -> Optional[str]:
        """
        Retrieve full content for an item by its raw reference.

        Args:
            raw_reference: The raw reference ID of the item

        Returns:
            Full content string or None if not found
        """
        pass

    def detect_urgency_markers(self, content: str, subject: Optional[str] = None) -> UrgencyMarker:
        """
        Detect urgency markers in content and subject.

        Looks for explicit urgency patterns without interpreting importance.
        """
        text_to_check = f"{subject or ''} {content}".lower()

        # Explicit urgent patterns
        urgent_patterns = [
            r'\burgent\b',
            r'\basap\b',
            r'\bemergency\b',
            r'\bcritical\b',
            r'\bimmediate(ly)?\b',
            r'!!!',
            r'\[urgent\]',
            r'\*urgent\*',
            r'🚨',
            r'⚠️',
        ]

        for pattern in urgent_patterns:
            if re.search(pattern, text_to_check):
                return UrgencyMarker.EXPLICIT_URGENT

        # Time-sensitive patterns
        time_sensitive_patterns = [
            r'\btoday\b',
            r'\btonight\b',
            r'\bthis morning\b',
            r'\bthis afternoon\b',
            r'\bby \d{1,2}(:\d{2})?\s*(am|pm)?\b',
            r'\bdeadline\b',
            r'\bexpires?\b',
            r'\blast chance\b',
            r'\bending soon\b',
        ]

        for pattern in time_sensitive_patterns:
            if re.search(pattern, text_to_check):
                return UrgencyMarker.TIME_SENSITIVE

        return UrgencyMarker.NONE

    def extract_preview(self, content: str, max_length: int = 100) -> str:
        """Extract a preview from content, truncating if necessary."""
        if not content:
            return ""

        # Clean up whitespace
        content = " ".join(content.split())

        if len(content) <= max_length:
            return content

        # Truncate at word boundary
        truncated = content[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:
            truncated = truncated[:last_space]

        return truncated + "..."

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._is_connected

    def update_last_check(self) -> None:
        """Update the last check timestamp."""
        self.last_check = datetime.now()


class EmailAdapter(ChannelAdapter):
    """
    Email channel adapter.

    Integrates with mcp-gsuite for Gmail integration.
    This is a prepared interface - actual MCP integration to be connected.
    """

    def __init__(self):
        super().__init__(ChannelType.EMAIL)
        self._mcp_client = None  # Will hold mcp-gsuite client
        self._inbox_folder = "INBOX"

    async def connect(self) -> bool:
        """
        Connect to Gmail via mcp-gsuite.

        TODO: Integrate with actual mcp-gsuite MCP server.
        """
        try:
            # Placeholder for mcp-gsuite connection
            # self._mcp_client = await mcp_gsuite.connect()
            logger.info("EmailAdapter: Connection interface ready (awaiting mcp-gsuite integration)")
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"EmailAdapter connection failed: {e}")
            self._is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Gmail."""
        if self._mcp_client:
            # await self._mcp_client.disconnect()
            pass
        self._is_connected = False
        logger.info("EmailAdapter: Disconnected")

    async def fetch_items(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_filter: IncludeFilter = IncludeFilter.ALL
    ) -> List[InboundItem]:
        """
        Fetch emails from Gmail.

        TODO: Implement actual Gmail fetching via mcp-gsuite.
        """
        items = []

        try:
            # Placeholder for mcp-gsuite email fetching
            # emails = await self._mcp_client.list_messages(
            #     folder=self._inbox_folder,
            #     after=since,
            #     before=until,
            #     is_read=None if include_filter == IncludeFilter.ALL else False
            # )

            # Example of how to normalize an email:
            # for email in emails:
            #     item = self._normalize_email(email)
            #     items.append(item)

            logger.debug(f"EmailAdapter: Fetched {len(items)} emails")
            self.update_last_check()

        except Exception as e:
            logger.error(f"EmailAdapter fetch failed: {e}")

        return items

    def _normalize_email(self, email_data: Dict[str, Any]) -> InboundItem:
        """Normalize email data to InboundItem schema."""
        content = email_data.get("body", "")
        subject = email_data.get("subject", "")

        # Extract attachments
        attachments = []
        for att in email_data.get("attachments", []):
            attachments.append(Attachment(
                filename=att.get("filename", "unknown"),
                file_type=att.get("mime_type", "application/octet-stream").split("/")[-1],
                size_bytes=att.get("size", 0),
                mime_type=att.get("mime_type"),
                raw_reference=att.get("attachment_id")
            ))

        return InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.fromisoformat(email_data.get("date", datetime.now().isoformat())),
            sender=email_data.get("from", "unknown"),
            subject=subject,
            thread_id=email_data.get("thread_id"),
            content_preview=self.extract_preview(content),
            full_content=content,
            attachments=attachments,
            urgency_markers=self.detect_urgency_markers(content, subject),
            is_read=email_data.get("is_read", False),
            is_flagged=email_data.get("is_starred", False),
            raw_reference=email_data.get("message_id", str(uuid.uuid4()))
        )

    async def get_full_content(self, raw_reference: str) -> Optional[str]:
        """Retrieve full email content by message ID."""
        try:
            # email = await self._mcp_client.get_message(raw_reference)
            # return email.get("body")
            return None
        except Exception as e:
            logger.error(f"EmailAdapter get_full_content failed: {e}")
            return None


class WhatsAppAdapter(ChannelAdapter):
    """
    WhatsApp channel adapter.

    Integrates with whatsapp-mcp for WhatsApp Business API.
    This is a prepared interface - actual MCP integration to be connected.
    """

    def __init__(self):
        super().__init__(ChannelType.WHATSAPP)
        self._mcp_client = None  # Will hold whatsapp-mcp client
        self._configured_groups = []  # Personal, work groups, etc.

    async def connect(self) -> bool:
        """
        Connect to WhatsApp via whatsapp-mcp.

        TODO: Integrate with actual whatsapp-mcp MCP server.
        """
        try:
            # self._mcp_client = await whatsapp_mcp.connect()
            logger.info("WhatsAppAdapter: Connection interface ready (awaiting whatsapp-mcp integration)")
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"WhatsAppAdapter connection failed: {e}")
            self._is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from WhatsApp."""
        if self._mcp_client:
            # await self._mcp_client.disconnect()
            pass
        self._is_connected = False
        logger.info("WhatsAppAdapter: Disconnected")

    async def fetch_items(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_filter: IncludeFilter = IncludeFilter.ALL
    ) -> List[InboundItem]:
        """
        Fetch WhatsApp messages.

        TODO: Implement actual WhatsApp fetching via whatsapp-mcp.
        """
        items = []

        try:
            # messages = await self._mcp_client.get_messages(
            #     since=since,
            #     until=until,
            #     include_groups=self._configured_groups
            # )

            # for msg in messages:
            #     item = self._normalize_message(msg)
            #     items.append(item)

            logger.debug(f"WhatsAppAdapter: Fetched {len(items)} messages")
            self.update_last_check()

        except Exception as e:
            logger.error(f"WhatsAppAdapter fetch failed: {e}")

        return items

    def _normalize_message(self, message_data: Dict[str, Any]) -> InboundItem:
        """Normalize WhatsApp message to InboundItem schema."""
        content = message_data.get("text", "")

        # Extract media attachments
        attachments = []
        if message_data.get("has_media"):
            media_type = message_data.get("media_type", "unknown")
            attachments.append(Attachment(
                filename=message_data.get("media_filename", f"media.{media_type}"),
                file_type=media_type,
                size_bytes=message_data.get("media_size", 0),
                raw_reference=message_data.get("media_id")
            ))

        # Determine thread (group name or contact name)
        thread_id = message_data.get("group_name") or message_data.get("chat_id")

        return InboundItem(
            source=ChannelType.WHATSAPP,
            timestamp=datetime.fromtimestamp(message_data.get("timestamp", datetime.now().timestamp())),
            sender=message_data.get("sender_name", message_data.get("sender_number", "unknown")),
            thread_id=thread_id,
            content_preview=self.extract_preview(content),
            full_content=content,
            attachments=attachments,
            urgency_markers=self.detect_urgency_markers(content),
            is_read=message_data.get("is_read", False),
            raw_reference=message_data.get("message_id", str(uuid.uuid4()))
        )

    async def get_full_content(self, raw_reference: str) -> Optional[str]:
        """Retrieve full message content by message ID."""
        try:
            # message = await self._mcp_client.get_message(raw_reference)
            # return message.get("text")
            return None
        except Exception as e:
            logger.error(f"WhatsAppAdapter get_full_content failed: {e}")
            return None


class ScanAdapter(ChannelAdapter):
    """
    Document scan channel adapter.

    Integrates with:
    - PaddleOCR for document OCR
    - MinerU for PDF extraction

    Handles secretary-uploaded documents and scanned files.
    """

    def __init__(self, scan_directory: str = "/var/alfred/scans"):
        super().__init__(ChannelType.SCAN)
        self._scan_directory = scan_directory
        self._ocr_engine = None  # Will hold PaddleOCR instance
        self._pdf_extractor = None  # Will hold MinerU instance
        self._supported_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    async def connect(self) -> bool:
        """
        Initialize OCR and PDF extraction engines.

        TODO: Integrate with PaddleOCR and MinerU.
        """
        try:
            # from paddleocr import PaddleOCR
            # self._ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

            # from mineru import PDFExtractor
            # self._pdf_extractor = PDFExtractor()

            logger.info("ScanAdapter: Connection interface ready (awaiting PaddleOCR/MinerU integration)")
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"ScanAdapter initialization failed: {e}")
            self._is_connected = False
            return False

    async def disconnect(self) -> None:
        """Release OCR resources."""
        self._ocr_engine = None
        self._pdf_extractor = None
        self._is_connected = False
        logger.info("ScanAdapter: Disconnected")

    async def fetch_items(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_filter: IncludeFilter = IncludeFilter.ALL
    ) -> List[InboundItem]:
        """
        Scan directory for new documents and process them.

        TODO: Implement file scanning with OCR/PDF extraction.
        """
        items = []

        try:
            # import os
            # for filename in os.listdir(self._scan_directory):
            #     filepath = os.path.join(self._scan_directory, filename)
            #     if not os.path.isfile(filepath):
            #         continue
            #
            #     # Check file extension
            #     ext = os.path.splitext(filename)[1].lower()
            #     if ext not in self._supported_extensions:
            #         continue
            #
            #     # Check modification time
            #     mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            #     if since and mtime < since:
            #         continue
            #     if until and mtime > until:
            #         continue
            #
            #     item = await self._process_document(filepath, filename, mtime)
            #     items.append(item)

            logger.debug(f"ScanAdapter: Processed {len(items)} documents")
            self.update_last_check()

        except Exception as e:
            logger.error(f"ScanAdapter fetch failed: {e}")

        return items

    async def _process_document(
        self,
        filepath: str,
        filename: str,
        timestamp: datetime
    ) -> InboundItem:
        """Process a document file and extract content."""
        ext = filename.split(".")[-1].lower()
        content = ""

        if ext == "pdf":
            content = await self._extract_pdf(filepath)
        else:
            content = await self._perform_ocr(filepath)

        # Try to extract sender from filename or first line
        sender = self._extract_sender(filename, content)

        return InboundItem(
            source=ChannelType.SCAN,
            timestamp=timestamp,
            sender=sender,
            subject=filename,
            content_preview=self.extract_preview(content),
            full_content=content,
            attachments=[Attachment(
                filename=filename,
                file_type=ext,
                size_bytes=0,  # Would get from os.path.getsize
                raw_reference=filepath
            )],
            urgency_markers=self.detect_urgency_markers(content),
            raw_reference=filepath
        )

    async def _extract_pdf(self, filepath: str) -> str:
        """Extract text from PDF using MinerU."""
        try:
            # result = await self._pdf_extractor.extract(filepath)
            # return result.text
            return ""
        except Exception as e:
            logger.error(f"PDF extraction failed for {filepath}: {e}")
            return ""

    async def _perform_ocr(self, filepath: str) -> str:
        """Perform OCR on image using PaddleOCR."""
        try:
            # result = self._ocr_engine.ocr(filepath)
            # text_lines = []
            # for line in result:
            #     if line:
            #         for word_info in line:
            #             text_lines.append(word_info[1][0])
            # return "\n".join(text_lines)
            return ""
        except Exception as e:
            logger.error(f"OCR failed for {filepath}: {e}")
            return ""

    def _extract_sender(self, filename: str, content: str) -> str:
        """Try to extract sender from filename or content."""
        # Common patterns: "From_DrSmith_20240115.pdf", "Smith_referral.pdf"
        patterns = [
            r"From_([A-Za-z]+)",
            r"^([A-Za-z]+)_",
            r"referral_from_([A-Za-z]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1)

        return "secretary_upload"

    async def get_full_content(self, raw_reference: str) -> Optional[str]:
        """Retrieve full content by re-processing the document."""
        try:
            ext = raw_reference.split(".")[-1].lower()
            if ext == "pdf":
                return await self._extract_pdf(raw_reference)
            else:
                return await self._perform_ocr(raw_reference)
        except Exception as e:
            logger.error(f"ScanAdapter get_full_content failed: {e}")
            return None


class CalendarAdapter(ChannelAdapter):
    """
    Calendar channel adapter.

    Handles calendar notifications, invites, and updates.
    Integrates with mcp-gsuite for Google Calendar.
    """

    def __init__(self):
        super().__init__(ChannelType.CALENDAR)
        self._mcp_client = None  # Will hold mcp-gsuite calendar client

    async def connect(self) -> bool:
        """
        Connect to Google Calendar via mcp-gsuite.

        TODO: Integrate with actual mcp-gsuite calendar functions.
        """
        try:
            # self._mcp_client = await mcp_gsuite.calendar.connect()
            logger.info("CalendarAdapter: Connection interface ready (awaiting mcp-gsuite integration)")
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"CalendarAdapter connection failed: {e}")
            self._is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from calendar service."""
        if self._mcp_client:
            # await self._mcp_client.disconnect()
            pass
        self._is_connected = False
        logger.info("CalendarAdapter: Disconnected")

    async def fetch_items(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_filter: IncludeFilter = IncludeFilter.ALL
    ) -> List[InboundItem]:
        """
        Fetch calendar notifications and invites.

        TODO: Implement actual calendar fetching via mcp-gsuite.
        """
        items = []

        try:
            # notifications = await self._mcp_client.get_notifications(
            #     since=since,
            #     until=until
            # )

            # invites = await self._mcp_client.get_pending_invites()

            # for notification in notifications:
            #     item = self._normalize_notification(notification)
            #     items.append(item)

            # for invite in invites:
            #     item = self._normalize_invite(invite)
            #     items.append(item)

            logger.debug(f"CalendarAdapter: Fetched {len(items)} calendar items")
            self.update_last_check()

        except Exception as e:
            logger.error(f"CalendarAdapter fetch failed: {e}")

        return items

    def _normalize_notification(self, notification: Dict[str, Any]) -> InboundItem:
        """Normalize calendar notification to InboundItem schema."""
        event_time = notification.get("event_time", "")
        event_title = notification.get("event_title", "")
        content = f"Event: {event_title} at {event_time}"

        return InboundItem(
            source=ChannelType.CALENDAR,
            timestamp=datetime.fromisoformat(notification.get("created_at", datetime.now().isoformat())),
            sender="Calendar",
            subject=f"Reminder: {event_title}",
            content_preview=self.extract_preview(content),
            full_content=content,
            urgency_markers=UrgencyMarker.TIME_SENSITIVE,  # Calendar items are inherently time-sensitive
            raw_reference=notification.get("event_id", str(uuid.uuid4()))
        )

    def _normalize_invite(self, invite: Dict[str, Any]) -> InboundItem:
        """Normalize calendar invite to InboundItem schema."""
        organizer = invite.get("organizer", {})
        organizer_name = organizer.get("name", organizer.get("email", "Unknown"))
        event_title = invite.get("summary", "Meeting")
        event_time = invite.get("start", {}).get("dateTime", "")

        content = f"Meeting invite: {event_title}\nFrom: {organizer_name}\nTime: {event_time}"

        return InboundItem(
            source=ChannelType.CALENDAR,
            timestamp=datetime.fromisoformat(invite.get("created", datetime.now().isoformat())),
            sender=organizer_name,
            subject=f"Invite: {event_title}",
            content_preview=self.extract_preview(content),
            full_content=content,
            urgency_markers=self.detect_urgency_markers(content),
            is_flagged=True,  # Invites need response
            raw_reference=invite.get("id", str(uuid.uuid4()))
        )

    async def get_full_content(self, raw_reference: str) -> Optional[str]:
        """Retrieve full event details by event ID."""
        try:
            # event = await self._mcp_client.get_event(raw_reference)
            # return json.dumps(event, indent=2)
            return None
        except Exception as e:
            logger.error(f"CalendarAdapter get_full_content failed: {e}")
            return None


class VoiceAdapter(ChannelAdapter):
    """
    Voice message channel adapter.

    Handles transcribed voice messages.
    Can integrate with speech-to-text services.
    """

    def __init__(self, voice_directory: str = "/var/alfred/voice"):
        super().__init__(ChannelType.VOICE)
        self._voice_directory = voice_directory
        self._transcription_service = None

    async def connect(self) -> bool:
        """Initialize voice transcription service."""
        try:
            logger.info("VoiceAdapter: Connection interface ready")
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"VoiceAdapter connection failed: {e}")
            self._is_connected = False
            return False

    async def disconnect(self) -> None:
        """Release transcription resources."""
        self._transcription_service = None
        self._is_connected = False
        logger.info("VoiceAdapter: Disconnected")

    async def fetch_items(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_filter: IncludeFilter = IncludeFilter.ALL
    ) -> List[InboundItem]:
        """Fetch and transcribe voice messages."""
        items = []

        try:
            # Similar to ScanAdapter - scan directory for audio files
            # Transcribe and normalize
            logger.debug(f"VoiceAdapter: Processed {len(items)} voice messages")
            self.update_last_check()

        except Exception as e:
            logger.error(f"VoiceAdapter fetch failed: {e}")

        return items

    async def get_full_content(self, raw_reference: str) -> Optional[str]:
        """Retrieve full transcription by file reference."""
        return None


class DeduplicationEngine:
    """
    Handles deduplication of inbound items across channels.

    Uses content hashing and fuzzy matching to identify duplicates
    that may appear across different channels (e.g., same message
    forwarded via email and WhatsApp).
    """

    def __init__(self, cache_duration_hours: int = 24):
        self._seen_hashes: Dict[str, datetime] = {}
        self._cache_duration = timedelta(hours=cache_duration_hours)

    def deduplicate(self, items: List[InboundItem]) -> List[InboundItem]:
        """
        Remove duplicate items from a list.

        Args:
            items: List of InboundItem objects to deduplicate

        Returns:
            List of unique InboundItem objects
        """
        self._cleanup_cache()

        unique_items = []
        for item in items:
            if not self._is_duplicate(item):
                unique_items.append(item)
                self._mark_seen(item)

        duplicates_removed = len(items) - len(unique_items)
        if duplicates_removed > 0:
            logger.info(f"Deduplication: Removed {duplicates_removed} duplicate items")

        return unique_items

    def _is_duplicate(self, item: InboundItem) -> bool:
        """Check if item has been seen before."""
        return item.content_hash in self._seen_hashes

    def _mark_seen(self, item: InboundItem) -> None:
        """Mark item as seen."""
        self._seen_hashes[item.content_hash] = datetime.now()

    def _cleanup_cache(self) -> None:
        """Remove expired entries from the cache."""
        now = datetime.now()
        expired = [
            hash_key for hash_key, seen_time in self._seen_hashes.items()
            if now - seen_time > self._cache_duration
        ]
        for hash_key in expired:
            del self._seen_hashes[hash_key]

    def clear_cache(self) -> None:
        """Clear all cached hashes."""
        self._seen_hashes.clear()


class IntakeAgent(OperationsAgent):
    """
    Unified Ingestion System for Alfred.

    The Intake Agent is responsible for:
    - Ingesting from all configured channels (WhatsApp, email, scans, calendar)
    - Normalizing formats to standard schema
    - Timestamping all incoming items
    - Deduplicating across channels
    - Tagging by source channel
    - Extracting sender, subject, attachments
    - Queuing in order received
    - Flagging explicit urgency markers

    The Intake Agent does NOT:
    - Interpret meaning or importance
    - Prioritize or triage
    - Respond to incoming messages
    - Filter based on sender status
    - Make decisions about urgency
    """

    def __init__(self):
        super().__init__(name="IntakeAgent")

        # Initialize channel adapters
        self._adapters: Dict[ChannelType, ChannelAdapter] = {
            ChannelType.EMAIL: EmailAdapter(),
            ChannelType.WHATSAPP: WhatsAppAdapter(),
            ChannelType.SCAN: ScanAdapter(),
            ChannelType.CALENDAR: CalendarAdapter(),
            ChannelType.VOICE: VoiceAdapter(),
        }

        # Deduplication engine
        self._deduplicator = DeduplicationEngine()

        # Track last check times per channel
        self._last_checks: Dict[ChannelType, datetime] = {}

        # Batch counter
        self._batch_counter = 0

        # Error tracking
        self._channel_errors: Dict[ChannelType, List[str]] = {ct: [] for ct in ChannelType}

    async def initialize(self) -> bool:
        """
        Initialize all channel adapters.

        Returns:
            True if at least one channel connected successfully
        """
        connected_count = 0

        for channel_type, adapter in self._adapters.items():
            try:
                success = await adapter.connect()
                if success:
                    connected_count += 1
                    logger.info(f"IntakeAgent: {channel_type.value} adapter connected")
                else:
                    logger.warning(f"IntakeAgent: {channel_type.value} adapter failed to connect")
            except Exception as e:
                logger.error(f"IntakeAgent: Error connecting {channel_type.value}: {e}")
                self._channel_errors[channel_type].append(str(e))

        return connected_count > 0

    async def shutdown(self) -> None:
        """Disconnect all channel adapters."""
        for channel_type, adapter in self._adapters.items():
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.error(f"IntakeAgent: Error disconnecting {channel_type.value}: {e}")

    async def check_channels(
        self,
        request: IntakeRequest
    ) -> InboundBatch:
        """
        Check specified channels and return normalized batch.

        This is the main entry point for Alfred to request intake.

        Args:
            request: IntakeRequest specifying channels, time window, and filters

        Returns:
            InboundBatch containing all normalized items
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self._create_blocked_batch(request.channels, reason)

        all_items: List[InboundItem] = []
        errors: List[str] = []

        # Calculate time window
        since, until = self._calculate_time_window(request)

        # Process each requested channel
        for channel_type in request.channels:
            try:
                channel_items = await self.process_channel(
                    channel_type,
                    since=since,
                    until=until,
                    include_filter=request.include_filter
                )
                all_items.extend(channel_items)

                # Update last check time for this channel
                self._last_checks[channel_type] = datetime.now()

            except Exception as e:
                error_msg = f"{channel_type.value}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"IntakeAgent: Channel error - {error_msg}")

        # Deduplicate across channels
        unique_items = self._deduplicator.deduplicate(all_items)

        # Sort by timestamp (queue in order received)
        unique_items.sort(key=lambda x: x.timestamp)

        # Create and return batch
        batch = self.create_batch(
            items=unique_items,
            channels_checked=request.channels,
            errors=errors
        )

        logger.info(
            f"IntakeAgent: Batch {batch.batch_id} created with "
            f"{batch.items_count} items from {len(request.channels)} channels"
        )

        return batch

    async def process_channel(
        self,
        channel_type: ChannelType,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_filter: IncludeFilter = IncludeFilter.ALL
    ) -> List[InboundItem]:
        """
        Process a single channel and return normalized items.

        Args:
            channel_type: The channel to process
            since: Start of time window
            until: End of time window
            include_filter: Filter for which items to include

        Returns:
            List of normalized InboundItem objects
        """
        adapter = self._adapters.get(channel_type)
        if not adapter:
            logger.warning(f"IntakeAgent: No adapter for channel {channel_type.value}")
            return []

        if not adapter.is_connected:
            logger.warning(f"IntakeAgent: Adapter for {channel_type.value} not connected")
            return []

        try:
            items = await adapter.fetch_items(
                since=since,
                until=until,
                include_filter=include_filter
            )

            # Normalize each item (adapters should already do this, but verify)
            normalized_items = [self.normalize_item(item) for item in items]

            return normalized_items

        except Exception as e:
            logger.error(f"IntakeAgent: Error processing {channel_type.value}: {e}")
            raise

    def normalize_item(self, item: InboundItem) -> InboundItem:
        """
        Ensure item meets normalization requirements.

        This is a validation/cleanup step - adapters should already
        produce normalized items, but this ensures consistency.

        Args:
            item: InboundItem to normalize

        Returns:
            Normalized InboundItem
        """
        # Ensure preview is within limits
        if len(item.content_preview) > 100:
            item.content_preview = item.content_preview[:97] + "..."

        # Ensure timestamp is present
        if not item.timestamp:
            item.timestamp = datetime.now()

        # Ensure sender is present
        if not item.sender:
            item.sender = "unknown"

        return item

    def create_batch(
        self,
        items: List[InboundItem],
        channels_checked: List[ChannelType],
        errors: List[str] = None
    ) -> InboundBatch:
        """
        Create an InboundBatch from a list of items.

        Args:
            items: List of normalized InboundItem objects
            channels_checked: List of channels that were checked
            errors: List of any errors encountered

        Returns:
            InboundBatch ready for Alfred
        """
        self._batch_counter += 1
        batch_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._batch_counter:04d}"

        # Calculate summary statistics
        summary = self._calculate_summary(items)

        return InboundBatch(
            batch_id=batch_id,
            items_count=len(items),
            channels_checked=channels_checked,
            items=items,
            summary=summary,
            errors=errors or []
        )

    def _calculate_summary(self, items: List[InboundItem]) -> BatchSummary:
        """Calculate summary statistics for a batch."""
        summary = BatchSummary()

        for item in items:
            # Count by channel
            channel_name = item.source.value
            summary.by_channel[channel_name] = summary.by_channel.get(channel_name, 0) + 1

            # Count with attachments
            if item.attachment_count > 0:
                summary.with_attachments += 1

            # Count with urgency markers
            if item.urgency_markers != UrgencyMarker.NONE:
                summary.with_urgency_markers += 1

            # Count unread
            if not item.is_read:
                summary.unread_count += 1

            # Count flagged
            if item.is_flagged:
                summary.flagged_count += 1

        return summary

    def _calculate_time_window(
        self,
        request: IntakeRequest
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Calculate the actual time window for a request."""
        until = datetime.now()
        since = None

        if request.time_window == TimeWindow.SINCE_LAST_CHECK:
            # Find the oldest last check time among requested channels
            check_times = [
                self._last_checks.get(ch)
                for ch in request.channels
                if ch in self._last_checks
            ]
            if check_times:
                since = min(t for t in check_times if t is not None)

        elif request.time_window == TimeWindow.LAST_HOUR:
            since = until - timedelta(hours=1)

        elif request.time_window == TimeWindow.LAST_24_HOURS:
            since = until - timedelta(hours=24)

        elif request.time_window == TimeWindow.LAST_WEEK:
            since = until - timedelta(weeks=1)

        elif request.time_window == TimeWindow.CUSTOM:
            since = request.custom_start
            until = request.custom_end or until

        return since, until

    def _create_blocked_batch(
        self,
        channels: List[ChannelType],
        reason: str
    ) -> InboundBatch:
        """Create a batch indicating the agent is blocked."""
        return InboundBatch(
            batch_id=f"BLOCKED_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            items_count=0,
            channels_checked=channels,
            items=[],
            summary=BatchSummary(),
            status=f"BLOCKED: {reason}",
            errors=[reason]
        )

    async def get_full_item_content(
        self,
        item: InboundItem
    ) -> Optional[str]:
        """
        Retrieve full content for an item.

        Args:
            item: The InboundItem to get full content for

        Returns:
            Full content string or None if unavailable
        """
        adapter = self._adapters.get(item.source)
        if adapter:
            return await adapter.get_full_content(item.raw_reference)
        return None

    def execute(
        self,
        request_data: Dict[str, Any]
    ) -> AgentResponse:
        """
        Synchronous execution wrapper for Alfred.

        This method provides a synchronous interface that Alfred can call,
        wrapping the async check_channels method.

        Args:
            request_data: Dictionary matching INTAKE_REQUEST format

        Returns:
            AgentResponse containing the InboundBatch
        """
        import asyncio

        try:
            # Parse request
            request = IntakeRequest.from_dict(request_data)

            # Run async method
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.check_channels(request)
                    )
                    batch = future.result()
            else:
                batch = loop.run_until_complete(self.check_channels(request))

            # Create response
            return self.create_response(
                data={
                    "batch": batch.to_dict(),
                    "formatted_output": batch.to_formatted_output()
                },
                success=batch.status == "complete",
                errors=batch.errors
            )

        except Exception as e:
            logger.error(f"IntakeAgent execution failed: {e}")
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[str(e)]
            )

    def get_channel_status(self) -> Dict[str, Any]:
        """
        Get status of all channel adapters.

        Returns:
            Dictionary with status information for each channel
        """
        status = {}
        for channel_type, adapter in self._adapters.items():
            status[channel_type.value] = {
                "connected": adapter.is_connected,
                "last_check": (
                    self._last_checks.get(channel_type, None)
                    if channel_type in self._last_checks
                    else None
                ),
                "recent_errors": self._channel_errors.get(channel_type, [])[-5:]
            }
        return status

    def clear_deduplication_cache(self) -> None:
        """Clear the deduplication cache."""
        self._deduplicator.clear_cache()
        logger.info("IntakeAgent: Deduplication cache cleared")


# Convenience function for creating and configuring the agent
def create_intake_agent() -> IntakeAgent:
    """
    Factory function to create and configure an IntakeAgent.

    Returns:
        Configured IntakeAgent instance
    """
    agent = IntakeAgent()
    return agent


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_intake_agent():
        """Test the Intake Agent."""
        agent = create_intake_agent()

        # Initialize adapters
        await agent.initialize()

        # Create a test request
        request = IntakeRequest(
            channels=[ChannelType.EMAIL, ChannelType.WHATSAPP],
            time_window=TimeWindow.LAST_24_HOURS,
            include_filter=IncludeFilter.ALL
        )

        # Check channels
        batch = await agent.check_channels(request)

        # Print results
        print(batch.to_formatted_output())

        # Cleanup
        await agent.shutdown()

    # Run test
    asyncio.run(test_intake_agent())
