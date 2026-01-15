"""
Calendar Adapter for Alfred

Google Calendar integration with MCP server support and direct API fallback.
Manages calendar events, block type detection, and conflict detection.

MCP Servers Supported:
- google-calendar-mcp
- mcp-gsuite (calendar module)

Direct API:
- Google Calendar API v3
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, time
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
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

class BlockType(Enum):
    """Time block categories for calendar management."""
    CLINICAL = "CLINICAL"         # Patient care, procedures, rounds
    DEEP_WORK = "DEEP_WORK"       # Focused creation, writing, building
    MEETINGS = "MEETINGS"         # Calls, syncs, external meetings
    BUFFER = "BUFFER"             # Transition time, overflow
    RECOVERY = "RECOVERY"         # Rest, meals, breaks
    PERSONAL = "PERSONAL"         # Family, non-work commitments
    COMMUTE = "COMMUTE"           # Travel time
    ADMIN = "ADMIN"               # Administrative tasks
    LEARNING = "LEARNING"         # Education, courses, reading
    UNKNOWN = "UNKNOWN"           # Unclassified


class EventStatus(Enum):
    """Calendar event status."""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class ConflictType(Enum):
    """Types of calendar conflicts."""
    OVERLAP = "overlap"                        # Events overlap in time
    DOUBLE_BOOKING = "double_booking"          # Same time slot
    BUFFER_VIOLATION = "buffer_violation"      # Insufficient buffer between events
    PROTECTED_BLOCK = "protected_block"        # Event during protected time
    OUTSIDE_HOURS = "outside_hours"            # Event outside work hours


class ResponseStatus(Enum):
    """Attendee response status."""
    ACCEPTED = "accepted"
    DECLINED = "declined"
    TENTATIVE = "tentative"
    NEEDS_ACTION = "needsAction"


# Block type detection keywords
BLOCK_TYPE_KEYWORDS: Dict[BlockType, List[str]] = {
    BlockType.CLINICAL: [
        "patient", "clinic", "surgery", "rounds", "consult", "procedure",
        "medical", "appointment", "exam", "treatment", "hospital"
    ],
    BlockType.DEEP_WORK: [
        "focus", "deep work", "writing", "coding", "research", "creation",
        "analysis", "design", "building", "draft", "compose"
    ],
    BlockType.MEETINGS: [
        "meeting", "call", "sync", "standup", "1:1", "one-on-one",
        "interview", "review", "presentation", "demo", "catch up"
    ],
    BlockType.BUFFER: [
        "buffer", "transition", "break", "prep", "preparation", "setup"
    ],
    BlockType.RECOVERY: [
        "lunch", "dinner", "breakfast", "rest", "meditation", "walk",
        "exercise", "gym", "yoga", "recovery", "nap"
    ],
    BlockType.PERSONAL: [
        "personal", "family", "kids", "school", "doctor", "dentist",
        "birthday", "anniversary", "vacation", "holiday"
    ],
    BlockType.COMMUTE: [
        "commute", "travel", "drive", "flight", "train", "uber", "transit"
    ],
    BlockType.ADMIN: [
        "admin", "email", "inbox", "expenses", "paperwork", "forms"
    ],
    BlockType.LEARNING: [
        "course", "class", "lecture", "webinar", "training", "workshop",
        "learning", "study", "reading"
    ],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RecurrenceRule:
    """Recurrence rule for calendar events."""
    frequency: str  # DAILY, WEEKLY, MONTHLY, YEARLY
    interval: int = 1
    count: Optional[int] = None
    until: Optional[datetime] = None
    by_day: Optional[List[str]] = None  # MO, TU, WE, TH, FR, SA, SU
    by_month_day: Optional[List[int]] = None
    by_month: Optional[List[int]] = None

    def to_rrule_string(self) -> str:
        """Convert to iCalendar RRULE string."""
        parts = [f"FREQ={self.frequency}"]
        if self.interval > 1:
            parts.append(f"INTERVAL={self.interval}")
        if self.count:
            parts.append(f"COUNT={self.count}")
        if self.until:
            parts.append(f"UNTIL={self.until.strftime('%Y%m%dT%H%M%SZ')}")
        if self.by_day:
            parts.append(f"BYDAY={','.join(self.by_day)}")
        if self.by_month_day:
            parts.append(f"BYMONTHDAY={','.join(map(str, self.by_month_day))}")
        if self.by_month:
            parts.append(f"BYMONTH={','.join(map(str, self.by_month))}")
        return ";".join(parts)

    @classmethod
    def from_rrule_string(cls, rrule: str) -> "RecurrenceRule":
        """Parse from iCalendar RRULE string."""
        parts = dict(p.split("=") for p in rrule.replace("RRULE:", "").split(";"))
        return cls(
            frequency=parts.get("FREQ", "WEEKLY"),
            interval=int(parts.get("INTERVAL", 1)),
            count=int(parts["COUNT"]) if "COUNT" in parts else None,
            by_day=parts.get("BYDAY", "").split(",") if "BYDAY" in parts else None
        )


@dataclass
class Attendee:
    """Calendar event attendee."""
    email: str
    display_name: Optional[str] = None
    response_status: ResponseStatus = ResponseStatus.NEEDS_ACTION
    optional: bool = False
    organizer: bool = False
    self_: bool = False  # Whether this is the current user

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "displayName": self.display_name,
            "responseStatus": self.response_status.value,
            "optional": self.optional,
            "organizer": self.organizer,
            "self": self.self_
        }


@dataclass
class CalendarEvent:
    """Represents a calendar event."""
    event_id: str
    title: str
    start_time: datetime
    end_time: datetime

    # Optional fields
    description: str = ""
    location: str = ""
    status: EventStatus = EventStatus.CONFIRMED
    calendar_id: str = "primary"

    # Block classification
    block_type: BlockType = BlockType.UNKNOWN
    is_protected: bool = False
    protection_level: str = "standard"  # standard, sacred, flexible

    # Attendees and conferencing
    attendees: List[Attendee] = field(default_factory=list)
    organizer_email: Optional[str] = None
    conference_link: Optional[str] = None
    conference_type: Optional[str] = None  # zoom, meet, teams

    # Recurrence
    is_recurring: bool = False
    recurrence_rule: Optional[RecurrenceRule] = None
    recurring_event_id: Optional[str] = None  # Parent event ID

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    etag: Optional[str] = None
    color_id: Optional[str] = None

    # Custom Alfred fields
    buffer_before_minutes: int = 0
    buffer_after_minutes: int = 0
    intensity: str = "medium"  # low, medium, high
    source: str = "google"  # google, cal_com, manual
    tags: List[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> int:
        """Return duration in minutes."""
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() / 60)

    @property
    def duration_hours(self) -> float:
        """Return duration in hours."""
        return self.duration_minutes / 60.0

    @property
    def is_all_day(self) -> bool:
        """Check if this is an all-day event."""
        return (
            self.start_time.hour == 0 and
            self.start_time.minute == 0 and
            self.end_time.hour == 0 and
            self.end_time.minute == 0 and
            self.duration_hours >= 24
        )

    def overlaps(self, other: "CalendarEvent") -> bool:
        """Check if this event overlaps with another."""
        return (
            self.start_time < other.end_time and
            self.end_time > other.start_time
        )

    def contains(self, dt: datetime) -> bool:
        """Check if a datetime falls within this event."""
        return self.start_time <= dt < self.end_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "id": self.event_id,
            "summary": self.title,
            "description": self.description,
            "location": self.location,
            "start": {
                "dateTime": self.start_time.isoformat(),
                "timeZone": "UTC"
            },
            "end": {
                "dateTime": self.end_time.isoformat(),
                "timeZone": "UTC"
            },
            "status": self.status.value,
            "attendees": [a.to_dict() for a in self.attendees],
            "conferenceData": {
                "entryPoints": [{"uri": self.conference_link}]
            } if self.conference_link else None,
            "recurrence": [f"RRULE:{self.recurrence_rule.to_rrule_string()}"]
                if self.recurrence_rule else None,
            "colorId": self.color_id
        }

    @classmethod
    def from_google_event(cls, data: Dict[str, Any]) -> "CalendarEvent":
        """Create from Google Calendar API response."""
        # Parse start time
        start_data = data.get("start", {})
        if "dateTime" in start_data:
            start_time = datetime.fromisoformat(
                start_data["dateTime"].replace("Z", "+00:00")
            )
        elif "date" in start_data:
            start_time = datetime.strptime(start_data["date"], "%Y-%m-%d")
        else:
            start_time = datetime.now()

        # Parse end time
        end_data = data.get("end", {})
        if "dateTime" in end_data:
            end_time = datetime.fromisoformat(
                end_data["dateTime"].replace("Z", "+00:00")
            )
        elif "date" in end_data:
            end_time = datetime.strptime(end_data["date"], "%Y-%m-%d")
        else:
            end_time = start_time + timedelta(hours=1)

        # Parse attendees
        attendees = []
        for att in data.get("attendees", []):
            attendees.append(Attendee(
                email=att.get("email", ""),
                display_name=att.get("displayName"),
                response_status=ResponseStatus(att.get("responseStatus", "needsAction")),
                optional=att.get("optional", False),
                organizer=att.get("organizer", False),
                self_=att.get("self", False)
            ))

        # Parse recurrence
        recurrence_rule = None
        recurrence_data = data.get("recurrence", [])
        if recurrence_data:
            for rule in recurrence_data:
                if rule.startswith("RRULE:"):
                    recurrence_rule = RecurrenceRule.from_rrule_string(rule)
                    break

        # Extract conference link
        conference_link = None
        conference_type = None
        conf_data = data.get("conferenceData", {})
        for entry in conf_data.get("entryPoints", []):
            if entry.get("entryPointType") == "video":
                conference_link = entry.get("uri")
                conference_type = conf_data.get("conferenceSolution", {}).get("name", "").lower()
                break

        event = cls(
            event_id=data.get("id", ""),
            title=data.get("summary", "Untitled"),
            start_time=start_time,
            end_time=end_time,
            description=data.get("description", ""),
            location=data.get("location", ""),
            status=EventStatus(data.get("status", "confirmed")),
            attendees=attendees,
            organizer_email=data.get("organizer", {}).get("email"),
            conference_link=conference_link,
            conference_type=conference_type,
            is_recurring=bool(recurrence_data) or "recurringEventId" in data,
            recurrence_rule=recurrence_rule,
            recurring_event_id=data.get("recurringEventId"),
            created_at=datetime.fromisoformat(data["created"].replace("Z", "+00:00"))
                if "created" in data else None,
            updated_at=datetime.fromisoformat(data["updated"].replace("Z", "+00:00"))
                if "updated" in data else None,
            etag=data.get("etag"),
            color_id=data.get("colorId")
        )

        return event


@dataclass
class CalendarConflict:
    """Represents a calendar conflict."""
    conflict_type: ConflictType
    events: List[CalendarEvent]
    overlap_start: datetime
    overlap_end: datetime
    severity: str  # high, medium, low
    description: str
    resolution_suggestions: List[str] = field(default_factory=list)

    @property
    def overlap_duration_minutes(self) -> int:
        """Duration of overlap in minutes."""
        delta = self.overlap_end - self.overlap_start
        return int(delta.total_seconds() / 60)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.conflict_type.value,
            "events": [e.event_id for e in self.events],
            "event_titles": [e.title for e in self.events],
            "overlap": {
                "start": self.overlap_start.isoformat(),
                "end": self.overlap_end.isoformat(),
                "minutes": self.overlap_duration_minutes
            },
            "severity": self.severity,
            "description": self.description,
            "suggestions": self.resolution_suggestions
        }


# =============================================================================
# CALENDAR ADAPTER
# =============================================================================

class CalendarAdapter(BaseAdapter):
    """
    Google Calendar adapter with MCP and direct API support.

    Provides:
    - Event CRUD operations
    - Block type detection and classification
    - Conflict detection and resolution suggestions
    - Recurring event handling
    - Multi-calendar support
    """

    # MCP server names
    MCP_GOOGLE_CALENDAR = "google-calendar-mcp"
    MCP_GSUITE = "mcp-gsuite"

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        api_credentials: Optional[Dict[str, Any]] = None,
        default_calendar_id: str = "primary",
        enable_mock: bool = False
    ):
        """
        Initialize Calendar adapter.

        Args:
            mcp_client: MCP client for server communication
            api_credentials: Google API credentials for direct API access
            default_calendar_id: Default calendar ID (usually "primary")
            enable_mock: Enable mock mode for testing
        """
        # Try google-calendar-mcp first, fall back to mcp-gsuite
        mcp_server = None
        if mcp_client:
            if mcp_client.is_server_available(self.MCP_GOOGLE_CALENDAR):
                mcp_server = self.MCP_GOOGLE_CALENDAR
            elif mcp_client.is_server_available(self.MCP_GSUITE):
                mcp_server = self.MCP_GSUITE

        super().__init__(
            adapter_name="CalendarAdapter",
            mcp_client=mcp_client,
            mcp_server_name=mcp_server,
            api_credentials=api_credentials,
            enable_mock=enable_mock
        )

        self.default_calendar_id = default_calendar_id
        self._api_service = None

    async def _connect_api(self) -> None:
        """Establish direct Google Calendar API connection."""
        if not self.api_credentials:
            raise AuthenticationError(
                self.adapter_name,
                "No API credentials provided"
            )

        # In production, this would use google-api-python-client
        # from googleapiclient.discovery import build
        # from google.oauth2.credentials import Credentials
        #
        # creds = Credentials.from_authorized_user_info(self.api_credentials)
        # self._api_service = build('calendar', 'v3', credentials=creds)
        pass

    # -------------------------------------------------------------------------
    # CORE EVENT METHODS
    # -------------------------------------------------------------------------

    async def get_events(
        self,
        start_time: datetime,
        end_time: datetime,
        calendar_id: Optional[str] = None,
        max_results: int = 250,
        single_events: bool = True,
        include_cancelled: bool = False
    ) -> List[CalendarEvent]:
        """
        Retrieve calendar events within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            calendar_id: Calendar ID (defaults to primary)
            max_results: Maximum number of events to return
            single_events: Expand recurring events into instances
            include_cancelled: Include cancelled events

        Returns:
            List of CalendarEvent objects
        """
        self._check_rate_limit()
        calendar_id = calendar_id or self.default_calendar_id

        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_events_mcp(
                start_time, end_time, calendar_id, max_results, single_events
            )
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_events_api(
                start_time, end_time, calendar_id, max_results,
                single_events, include_cancelled
            )
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_events_mock(start_time, end_time)
        else:
            raise AdapterError(
                "Not connected",
                self.adapter_name,
                "NOT_CONNECTED"
            )

    async def _get_events_mcp(
        self,
        start_time: datetime,
        end_time: datetime,
        calendar_id: str,
        max_results: int,
        single_events: bool
    ) -> List[CalendarEvent]:
        """Get events via MCP server."""
        if self.mcp_server_name == self.MCP_GOOGLE_CALENDAR:
            result = await self._call_mcp("list_events", {
                "calendarId": calendar_id,
                "timeMin": start_time.isoformat() + "Z",
                "timeMax": end_time.isoformat() + "Z",
                "maxResults": max_results,
                "singleEvents": single_events,
                "orderBy": "startTime"
            })
        else:  # mcp-gsuite
            result = await self._call_mcp("calendar.list_events", {
                "calendar_id": calendar_id,
                "time_min": start_time.isoformat(),
                "time_max": end_time.isoformat(),
                "max_results": max_results
            })

        events = []
        for item in result.get("items", []):
            event = CalendarEvent.from_google_event(item)
            event.block_type = self.detect_block_type(event)
            events.append(event)

        return events

    async def _get_events_api(
        self,
        start_time: datetime,
        end_time: datetime,
        calendar_id: str,
        max_results: int,
        single_events: bool,
        include_cancelled: bool
    ) -> List[CalendarEvent]:
        """Get events via direct API."""
        if not self._api_service:
            raise AdapterError(
                "API service not initialized",
                self.adapter_name,
                "API_NOT_INITIALIZED"
            )

        # Placeholder for actual API call
        # events_result = self._api_service.events().list(
        #     calendarId=calendar_id,
        #     timeMin=start_time.isoformat() + 'Z',
        #     timeMax=end_time.isoformat() + 'Z',
        #     maxResults=max_results,
        #     singleEvents=single_events,
        #     orderBy='startTime',
        #     showDeleted=include_cancelled
        # ).execute()
        return []

    def _get_events_mock(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[CalendarEvent]:
        """Get mock events for testing."""
        base_date = start_time.date()
        return [
            CalendarEvent(
                event_id="mock_1",
                title="Morning Meeting",
                start_time=datetime.combine(base_date, time(9, 0)),
                end_time=datetime.combine(base_date, time(10, 0)),
                block_type=BlockType.MEETINGS
            ),
            CalendarEvent(
                event_id="mock_2",
                title="Deep Work: Writing",
                start_time=datetime.combine(base_date, time(10, 30)),
                end_time=datetime.combine(base_date, time(12, 30)),
                block_type=BlockType.DEEP_WORK,
                is_protected=True
            ),
            CalendarEvent(
                event_id="mock_3",
                title="Lunch",
                start_time=datetime.combine(base_date, time(12, 30)),
                end_time=datetime.combine(base_date, time(13, 30)),
                block_type=BlockType.RECOVERY
            ),
        ]

    async def create_event(
        self,
        event: CalendarEvent,
        calendar_id: Optional[str] = None,
        send_notifications: bool = True
    ) -> CalendarEvent:
        """
        Create a new calendar event.

        Args:
            event: CalendarEvent to create
            calendar_id: Calendar ID (defaults to primary)
            send_notifications: Send notifications to attendees

        Returns:
            Created CalendarEvent with server-assigned ID
        """
        self._check_rate_limit()
        calendar_id = calendar_id or self.default_calendar_id

        if self.connection_mode == ConnectionMode.MCP:
            return await self._create_event_mcp(event, calendar_id, send_notifications)
        elif self.connection_mode == ConnectionMode.API:
            return await self._create_event_api(event, calendar_id, send_notifications)
        elif self.connection_mode == ConnectionMode.MOCK:
            event.event_id = f"mock_{datetime.now().timestamp()}"
            return event
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _create_event_mcp(
        self,
        event: CalendarEvent,
        calendar_id: str,
        send_notifications: bool
    ) -> CalendarEvent:
        """Create event via MCP server."""
        event_data = {
            "summary": event.title,
            "description": event.description,
            "location": event.location,
            "start": {"dateTime": event.start_time.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": event.end_time.isoformat(), "timeZone": "UTC"},
        }

        if event.attendees:
            event_data["attendees"] = [a.to_dict() for a in event.attendees]

        if event.recurrence_rule:
            event_data["recurrence"] = [f"RRULE:{event.recurrence_rule.to_rrule_string()}"]

        if self.mcp_server_name == self.MCP_GOOGLE_CALENDAR:
            result = await self._call_mcp("create_event", {
                "calendarId": calendar_id,
                "event": event_data,
                "sendNotifications": send_notifications
            })
        else:
            result = await self._call_mcp("calendar.create_event", {
                "calendar_id": calendar_id,
                "event": event_data
            })

        return CalendarEvent.from_google_event(result)

    async def _create_event_api(
        self,
        event: CalendarEvent,
        calendar_id: str,
        send_notifications: bool
    ) -> CalendarEvent:
        """Create event via direct API."""
        # Placeholder for actual API call
        return event

    async def update_event(
        self,
        event: CalendarEvent,
        calendar_id: Optional[str] = None,
        send_notifications: bool = True
    ) -> CalendarEvent:
        """
        Update an existing calendar event.

        Args:
            event: CalendarEvent with updated data
            calendar_id: Calendar ID (defaults to primary)
            send_notifications: Send notifications to attendees

        Returns:
            Updated CalendarEvent
        """
        self._check_rate_limit()
        calendar_id = calendar_id or self.default_calendar_id

        if self.connection_mode == ConnectionMode.MCP:
            return await self._update_event_mcp(event, calendar_id, send_notifications)
        elif self.connection_mode == ConnectionMode.API:
            return await self._update_event_api(event, calendar_id, send_notifications)
        elif self.connection_mode == ConnectionMode.MOCK:
            return event
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _update_event_mcp(
        self,
        event: CalendarEvent,
        calendar_id: str,
        send_notifications: bool
    ) -> CalendarEvent:
        """Update event via MCP server."""
        event_data = event.to_dict()

        if self.mcp_server_name == self.MCP_GOOGLE_CALENDAR:
            result = await self._call_mcp("update_event", {
                "calendarId": calendar_id,
                "eventId": event.event_id,
                "event": event_data,
                "sendNotifications": send_notifications
            })
        else:
            result = await self._call_mcp("calendar.update_event", {
                "calendar_id": calendar_id,
                "event_id": event.event_id,
                "event": event_data
            })

        return CalendarEvent.from_google_event(result)

    async def _update_event_api(
        self,
        event: CalendarEvent,
        calendar_id: str,
        send_notifications: bool
    ) -> CalendarEvent:
        """Update event via direct API."""
        # Placeholder for actual API call
        return event

    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
        send_notifications: bool = True
    ) -> bool:
        """
        Delete a calendar event.

        Args:
            event_id: Event ID to delete
            calendar_id: Calendar ID (defaults to primary)
            send_notifications: Send cancellation notifications

        Returns:
            True if deletion was successful
        """
        self._check_rate_limit()
        calendar_id = calendar_id or self.default_calendar_id

        if self.connection_mode == ConnectionMode.MCP:
            return await self._delete_event_mcp(event_id, calendar_id, send_notifications)
        elif self.connection_mode == ConnectionMode.API:
            return await self._delete_event_api(event_id, calendar_id, send_notifications)
        elif self.connection_mode == ConnectionMode.MOCK:
            return True
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _delete_event_mcp(
        self,
        event_id: str,
        calendar_id: str,
        send_notifications: bool
    ) -> bool:
        """Delete event via MCP server."""
        try:
            if self.mcp_server_name == self.MCP_GOOGLE_CALENDAR:
                await self._call_mcp("delete_event", {
                    "calendarId": calendar_id,
                    "eventId": event_id,
                    "sendNotifications": send_notifications
                })
            else:
                await self._call_mcp("calendar.delete_event", {
                    "calendar_id": calendar_id,
                    "event_id": event_id
                })
            return True
        except Exception:
            return False

    async def _delete_event_api(
        self,
        event_id: str,
        calendar_id: str,
        send_notifications: bool
    ) -> bool:
        """Delete event via direct API."""
        # Placeholder for actual API call
        return True

    # -------------------------------------------------------------------------
    # BLOCK TYPE DETECTION
    # -------------------------------------------------------------------------

    def detect_block_type(self, event: CalendarEvent) -> BlockType:
        """
        Detect the block type of an event based on its content.

        Uses title, description, and other metadata to classify the event
        into appropriate block types for Alfred's scheduling logic.

        Args:
            event: CalendarEvent to classify

        Returns:
            Detected BlockType
        """
        # Combine searchable text
        search_text = f"{event.title} {event.description}".lower()

        # Score each block type
        scores: Dict[BlockType, int] = {bt: 0 for bt in BlockType}

        for block_type, keywords in BLOCK_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in search_text:
                    scores[block_type] += 1

        # Additional heuristics
        if event.attendees and len(event.attendees) > 1:
            scores[BlockType.MEETINGS] += 2

        if event.conference_link:
            scores[BlockType.MEETINGS] += 1

        if event.is_all_day:
            scores[BlockType.PERSONAL] += 1

        # Check for specific patterns
        if re.search(r'\bfocus\s*time\b', search_text, re.IGNORECASE):
            scores[BlockType.DEEP_WORK] += 3
        if re.search(r'\bno\s*meetings?\b', search_text, re.IGNORECASE):
            scores[BlockType.DEEP_WORK] += 2

        # Find highest scoring type
        max_type = max(scores, key=scores.get)
        if scores[max_type] == 0:
            return BlockType.UNKNOWN

        return max_type

    def classify_events(
        self,
        events: List[CalendarEvent]
    ) -> Dict[BlockType, List[CalendarEvent]]:
        """
        Classify a list of events by block type.

        Args:
            events: List of events to classify

        Returns:
            Dictionary mapping block types to events
        """
        classified: Dict[BlockType, List[CalendarEvent]] = {
            bt: [] for bt in BlockType
        }

        for event in events:
            if event.block_type == BlockType.UNKNOWN:
                event.block_type = self.detect_block_type(event)
            classified[event.block_type].append(event)

        return classified

    # -------------------------------------------------------------------------
    # CONFLICT DETECTION
    # -------------------------------------------------------------------------

    def detect_conflicts(
        self,
        events: List[CalendarEvent],
        protected_blocks: Optional[List[CalendarEvent]] = None,
        buffer_minutes: int = 15
    ) -> List[CalendarConflict]:
        """
        Detect conflicts among a list of events.

        Args:
            events: List of events to check
            protected_blocks: Optional list of protected time blocks
            buffer_minutes: Minimum buffer between events

        Returns:
            List of detected conflicts
        """
        conflicts: List[CalendarConflict] = []

        # Sort events by start time
        sorted_events = sorted(events, key=lambda e: e.start_time)

        # Check for overlaps
        for i, event1 in enumerate(sorted_events):
            for event2 in sorted_events[i + 1:]:
                if event1.overlaps(event2):
                    # Determine severity
                    overlap_start = max(event1.start_time, event2.start_time)
                    overlap_end = min(event1.end_time, event2.end_time)
                    overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60

                    if overlap_minutes >= event1.duration_minutes * 0.5:
                        severity = "high"
                    elif overlap_minutes >= 15:
                        severity = "medium"
                    else:
                        severity = "low"

                    conflicts.append(CalendarConflict(
                        conflict_type=ConflictType.OVERLAP,
                        events=[event1, event2],
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                        severity=severity,
                        description=f"'{event1.title}' overlaps with '{event2.title}'",
                        resolution_suggestions=[
                            f"Reschedule '{event1.title}' to after {event2.end_time.strftime('%H:%M')}",
                            f"Reschedule '{event2.title}' to after {event1.end_time.strftime('%H:%M')}",
                            "Decline one of the conflicting events"
                        ]
                    ))

        # Check buffer violations
        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]

            gap = (next_event.start_time - current.end_time).total_seconds() / 60

            if 0 < gap < buffer_minutes:
                conflicts.append(CalendarConflict(
                    conflict_type=ConflictType.BUFFER_VIOLATION,
                    events=[current, next_event],
                    overlap_start=current.end_time,
                    overlap_end=next_event.start_time,
                    severity="low",
                    description=f"Only {int(gap)} minutes between events (need {buffer_minutes})",
                    resolution_suggestions=[
                        f"Add {buffer_minutes - int(gap)} minutes buffer",
                        f"End '{current.title}' earlier",
                        f"Start '{next_event.title}' later"
                    ]
                ))

        # Check against protected blocks
        if protected_blocks:
            for event in sorted_events:
                for protected in protected_blocks:
                    if event.overlaps(protected) and event.event_id != protected.event_id:
                        conflicts.append(CalendarConflict(
                            conflict_type=ConflictType.PROTECTED_BLOCK,
                            events=[event, protected],
                            overlap_start=max(event.start_time, protected.start_time),
                            overlap_end=min(event.end_time, protected.end_time),
                            severity="high" if protected.protection_level == "sacred" else "medium",
                            description=f"'{event.title}' conflicts with protected {protected.block_type.value} time",
                            resolution_suggestions=[
                                f"Reschedule '{event.title}' outside protected time",
                                "Decline this event to protect focus time"
                            ]
                        ))

        return conflicts

    async def find_free_slots(
        self,
        start_time: datetime,
        end_time: datetime,
        duration_minutes: int,
        calendar_id: Optional[str] = None,
        work_hours: Optional[Tuple[time, time]] = None
    ) -> List[Tuple[datetime, datetime]]:
        """
        Find free time slots within a range.

        Args:
            start_time: Start of search range
            end_time: End of search range
            duration_minutes: Required slot duration
            calendar_id: Calendar ID to check
            work_hours: Optional work hours constraint (start, end)

        Returns:
            List of (start, end) tuples for free slots
        """
        events = await self.get_events(start_time, end_time, calendar_id)
        events = sorted(events, key=lambda e: e.start_time)

        # Default work hours: 9 AM to 6 PM
        work_start = work_hours[0] if work_hours else time(9, 0)
        work_end = work_hours[1] if work_hours else time(18, 0)

        free_slots: List[Tuple[datetime, datetime]] = []
        current = start_time

        for event in events:
            # Skip to work hours start if needed
            if current.time() < work_start:
                current = datetime.combine(current.date(), work_start)
            elif current.time() >= work_end:
                current = datetime.combine(
                    current.date() + timedelta(days=1),
                    work_start
                )

            # Check if there's a gap before this event
            if current < event.start_time:
                gap_end = min(event.start_time, datetime.combine(current.date(), work_end))
                gap_minutes = (gap_end - current).total_seconds() / 60

                if gap_minutes >= duration_minutes:
                    free_slots.append((current, gap_end))

            # Move past this event
            current = max(current, event.end_time)

        # Check remaining time after last event
        while current < end_time:
            if current.time() < work_start:
                current = datetime.combine(current.date(), work_start)
            elif current.time() >= work_end:
                current = datetime.combine(
                    current.date() + timedelta(days=1),
                    work_start
                )
                continue

            day_end = datetime.combine(current.date(), work_end)
            if current < day_end and current < end_time:
                slot_end = min(day_end, end_time)
                gap_minutes = (slot_end - current).total_seconds() / 60

                if gap_minutes >= duration_minutes:
                    free_slots.append((current, slot_end))

            current = datetime.combine(
                current.date() + timedelta(days=1),
                work_start
            )

        return free_slots
