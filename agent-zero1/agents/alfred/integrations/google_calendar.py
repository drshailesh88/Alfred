"""
Google Calendar Integration Adapter for ALFRED

Connects ALFRED's Scheduling Agent to Google Calendar via MCP.
Provides structured access to calendar events with ALFRED's block type
classification for schedule governance.

Block Types:
- CLINICAL: Patient care, procedures, rounds
- DEEP_WORK: Focus time, writing, building
- MEETINGS: Calls, syncs
- BUFFER: Transition time
- RECOVERY: Rest, meals, breaks
- PERSONAL: Family, non-work
- COMMUTE: Travel time
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import asyncio
import hashlib
import json
import logging
import re
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class BlockType(Enum):
    """ALFRED's activity block types for schedule governance."""
    CLINICAL = "CLINICAL"
    DEEP_WORK = "DEEP_WORK"
    MEETINGS = "MEETINGS"
    BUFFER = "BUFFER"
    RECOVERY = "RECOVERY"
    PERSONAL = "PERSONAL"
    COMMUTE = "COMMUTE"
    UNKNOWN = "UNKNOWN"


@dataclass
class CalendarEvent:
    """
    Structured calendar event with ALFRED block type classification.

    Attributes:
        event_id: Unique identifier from Google Calendar
        title: Event summary/title
        start: Event start datetime
        end: Event end datetime
        block_type: ALFRED block classification (CLINICAL, DEEP_WORK, etc.)
        description: Optional event description
        is_protected: Whether this time block is protected from scheduling
        attendees: List of attendee email addresses
        location: Event location
        calendar_id: Source calendar ID
        recurring_event_id: Parent recurring event ID if applicable
        color_id: Google Calendar color ID
        raw_data: Original event data from Google Calendar
    """
    event_id: str
    title: str
    start: datetime
    end: datetime
    block_type: str
    description: Optional[str] = None
    is_protected: bool = False
    attendees: List[str] = field(default_factory=list)
    location: Optional[str] = None
    calendar_id: Optional[str] = None
    recurring_event_id: Optional[str] = None
    color_id: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

    def duration_minutes(self) -> int:
        """Calculate event duration in minutes."""
        return int((self.end - self.start).total_seconds() / 60)

    def overlaps_with(self, other: "CalendarEvent") -> bool:
        """Check if this event overlaps with another."""
        return self.start < other.end and self.end > other.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "block_type": self.block_type,
            "description": self.description,
            "is_protected": self.is_protected,
            "attendees": self.attendees,
            "location": self.location,
            "duration_minutes": self.duration_minutes(),
        }


# =============================================================================
# Exceptions
# =============================================================================

class CalendarError(Exception):
    """Base exception for calendar operations."""
    pass


class AuthenticationError(CalendarError):
    """Raised when OAuth authentication fails or token is invalid."""
    pass


class RateLimitError(CalendarError):
    """Raised when Google Calendar API rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class OfflineError(CalendarError):
    """Raised when operating in offline mode or network is unavailable."""
    pass


class EventNotFoundError(CalendarError):
    """Raised when an event cannot be found."""
    pass


# =============================================================================
# Block Type Classifier
# =============================================================================

class BlockTypeClassifier:
    """
    Classifies calendar events into ALFRED's block types based on
    event metadata (title, description, attendees, etc.).
    """

    # Keywords for each block type (case-insensitive matching)
    CLASSIFICATION_RULES: Dict[BlockType, Dict[str, List[str]]] = {
        BlockType.CLINICAL: {
            "title_keywords": [
                "patient", "clinic", "rounds", "procedure", "surgery",
                "consultation", "consult", "exam", "appointment", "visit",
                "care", "treatment", "diagnosis", "medical", "health",
                "hospital", "ward", "ICU", "ER", "OR", "operative",
                "pre-op", "post-op", "discharge", "admission", "chart",
                "EMR", "EHR", "note", "documentation", "dictation",
            ],
            "description_keywords": [
                "patient", "clinical", "medical", "healthcare",
            ],
        },
        BlockType.DEEP_WORK: {
            "title_keywords": [
                "focus", "deep work", "writing", "build", "code", "coding",
                "research", "study", "draft", "create", "design", "develop",
                "think", "strategic", "plan", "planning", "brainstorm",
                "no meeting", "blocked", "work time", "concentration",
                "maker time", "creative", "review", "analysis",
            ],
            "description_keywords": [
                "focus time", "do not disturb", "DND", "concentration",
            ],
        },
        BlockType.MEETINGS: {
            "title_keywords": [
                "meeting", "call", "sync", "standup", "stand-up", "1:1",
                "one-on-one", "huddle", "check-in", "check in", "catch up",
                "catchup", "zoom", "teams", "webex", "google meet", "interview",
                "discussion", "conference", "presentation", "demo", "review",
                "weekly", "monthly", "quarterly", "all-hands", "townhall",
                "retrospective", "retro", "sprint", "scrum", "agile",
            ],
            "description_keywords": [
                "dial-in", "join link", "meeting link", "zoom.us",
                "teams.microsoft.com", "meet.google.com",
            ],
        },
        BlockType.BUFFER: {
            "title_keywords": [
                "buffer", "transition", "travel", "break between",
                "prep time", "preparation", "setup", "wrap-up", "wrap up",
                "debrief", "follow-up", "follow up",
            ],
            "description_keywords": [],
        },
        BlockType.RECOVERY: {
            "title_keywords": [
                "lunch", "breakfast", "dinner", "meal", "eat", "rest",
                "break", "recovery", "recharge", "meditation", "meditate",
                "nap", "sleep", "relax", "gym", "workout", "exercise",
                "walk", "running", "yoga", "stretch", "wellness",
                "self-care", "mental health", "therapy", "counseling",
            ],
            "description_keywords": [
                "rest", "recovery", "break",
            ],
        },
        BlockType.PERSONAL: {
            "title_keywords": [
                "personal", "family", "kids", "children", "school",
                "pickup", "drop off", "dropoff", "appointment", "doctor",
                "dentist", "vet", "home", "errands", "shopping", "birthday",
                "anniversary", "holiday", "vacation", "PTO", "time off",
                "day off", "sick day", "leave", "wedding", "funeral",
                "graduation", "recital", "game", "event",
            ],
            "description_keywords": [
                "personal", "family",
            ],
        },
        BlockType.COMMUTE: {
            "title_keywords": [
                "commute", "travel", "drive", "driving", "flight",
                "train", "bus", "uber", "lyft", "taxi", "airport",
                "departure", "arrival", "transit", "transportation",
            ],
            "description_keywords": [],
        },
    }

    # Keywords that indicate protected time
    PROTECTED_KEYWORDS: List[str] = [
        "protected", "blocked", "do not schedule", "DND", "focus",
        "no meetings", "deep work", "sacred", "reserved", "hold",
        "unavailable", "busy", "important", "critical", "priority",
    ]

    @classmethod
    def classify(cls, event: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Classify an event into a block type and determine if protected.

        Args:
            event: Raw event data from Google Calendar

        Returns:
            Tuple of (block_type, is_protected)
        """
        title = (event.get("summary") or "").lower()
        description = (event.get("description") or "").lower()
        attendees = event.get("attendees", [])

        # Check if protected
        is_protected = cls._is_protected(title, description)

        # Classify block type
        block_type = cls._classify_block_type(title, description, attendees)

        return block_type.value, is_protected

    @classmethod
    def _is_protected(cls, title: str, description: str) -> bool:
        """Determine if an event should be marked as protected."""
        text = f"{title} {description}"
        for keyword in cls.PROTECTED_KEYWORDS:
            if keyword.lower() in text:
                return True
        return False

    @classmethod
    def _classify_block_type(
        cls,
        title: str,
        description: str,
        attendees: List[Dict[str, Any]]
    ) -> BlockType:
        """Classify event into a block type based on content."""
        # Score each block type
        scores: Dict[BlockType, int] = {bt: 0 for bt in BlockType}

        for block_type, rules in cls.CLASSIFICATION_RULES.items():
            # Check title keywords (higher weight)
            for keyword in rules["title_keywords"]:
                if keyword.lower() in title:
                    scores[block_type] += 3

            # Check description keywords (lower weight)
            for keyword in rules["description_keywords"]:
                if keyword.lower() in description:
                    scores[block_type] += 1

        # Additional heuristics

        # If has multiple attendees, likely a meeting
        if len(attendees) > 1:
            scores[BlockType.MEETINGS] += 2

        # If has no attendees and no classification, might be personal
        if not attendees and max(scores.values()) == 0:
            scores[BlockType.PERSONAL] += 1

        # Find highest scoring block type
        max_score = max(scores.values())
        if max_score > 0:
            for block_type, score in scores.items():
                if score == max_score:
                    return block_type

        return BlockType.UNKNOWN


# =============================================================================
# Response Cache
# =============================================================================

class ResponseCache:
    """
    Simple in-memory cache for calendar responses.
    Reduces API calls for frequently accessed data.
    """

    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 5 minutes)
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL override."""
        expiry = time.time() + (ttl if ttl is not None else self._default_ttl)
        self._cache[key] = (value, expiry)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        self._cache.pop(key, None)

    def invalidate_pattern(self, pattern: str) -> None:
        """Remove all keys matching a pattern."""
        keys_to_remove = [k for k in self._cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def _generate_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_data = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()


# =============================================================================
# Google Calendar Adapter
# =============================================================================

class GoogleCalendarAdapter:
    """
    Google Calendar integration adapter for ALFRED.

    Connects to Google Calendar via MCP (Model Context Protocol) server
    and provides structured access to calendar events with ALFRED's
    block type classification.

    Features:
    - Fetch events by date range
    - Create, update, and delete events
    - Check availability
    - Classify events into ALFRED block types
    - Response caching to reduce API calls
    - Error handling for auth, rate limits, and offline mode

    Example:
        adapter = GoogleCalendarAdapter(mcp_client)
        events = await adapter.get_today_events()
        for event in events:
            print(f"{event.title}: {event.block_type}")
    """

    def __init__(
        self,
        mcp_client: Optional[Any] = None,
        calendar_id: str = "primary",
        cache_ttl: int = 300,
        offline_fallback: bool = True,
    ):
        """
        Initialize the Google Calendar adapter.

        Args:
            mcp_client: MCP client for Google Calendar server communication.
                       If None, adapter operates in offline/mock mode.
            calendar_id: Default calendar ID to use (default: "primary")
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
            offline_fallback: Whether to provide fallback data in offline mode
        """
        self._mcp_client = mcp_client
        self._calendar_id = calendar_id
        self._cache = ResponseCache(default_ttl=cache_ttl)
        self._offline_fallback = offline_fallback
        self._is_online = True
        self._last_sync_time: Optional[datetime] = None

        # Offline data store
        self._offline_events: List[CalendarEvent] = []

    # -------------------------------------------------------------------------
    # Public Methods: Event Retrieval
    # -------------------------------------------------------------------------

    async def get_today_events(self) -> List[CalendarEvent]:
        """
        Retrieve today's calendar events.

        Returns:
            List of CalendarEvent objects for today

        Raises:
            AuthenticationError: If OAuth fails
            RateLimitError: If API rate limit exceeded
            OfflineError: If offline and no fallback available
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        return await self.get_events_in_range(today, tomorrow)

    async def get_week_events(self) -> List[CalendarEvent]:
        """
        Retrieve this week's calendar events (Monday to Sunday).

        Returns:
            List of CalendarEvent objects for the current week
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        # Get start of week (Monday)
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=7)
        return await self.get_events_in_range(start_of_week, end_of_week)

    async def get_events_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        calendar_id: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """
        Retrieve calendar events within a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            calendar_id: Optional calendar ID override

        Returns:
            List of CalendarEvent objects within the range
        """
        cal_id = calendar_id or self._calendar_id
        cache_key = f"events_{cal_id}_{start_date.date()}_{end_date.date()}"

        # Check cache first
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached

        try:
            events = await self._fetch_events(start_date, end_date, cal_id)
            self._cache.set(cache_key, events)
            self._is_online = True
            self._last_sync_time = datetime.now()
            return events
        except (AuthenticationError, RateLimitError):
            raise
        except Exception as e:
            logger.warning(f"Failed to fetch events: {e}")
            self._is_online = False
            if self._offline_fallback:
                return self._get_offline_events(start_date, end_date)
            raise OfflineError(f"Unable to fetch events: {e}")

    async def get_protected_blocks(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[CalendarEvent]:
        """
        Get events marked as protected (focus time, recovery, etc.).

        Args:
            start_date: Start of date range (default: today)
            end_date: End of date range (default: 7 days from start)

        Returns:
            List of protected CalendarEvent objects
        """
        if start_date is None:
            start_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if end_date is None:
            end_date = start_date + timedelta(days=7)

        all_events = await self.get_events_in_range(start_date, end_date)
        return [e for e in all_events if e.is_protected]

    # -------------------------------------------------------------------------
    # Public Methods: Event Management
    # -------------------------------------------------------------------------

    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        is_protected: bool = False,
        calendar_id: Optional[str] = None,
    ) -> CalendarEvent:
        """
        Create a new calendar event.

        Args:
            title: Event title/summary
            start: Event start time
            end: Event end time
            description: Optional event description
            location: Optional event location
            attendees: Optional list of attendee emails
            is_protected: Whether to mark as protected
            calendar_id: Optional calendar ID override

        Returns:
            Created CalendarEvent object

        Raises:
            AuthenticationError: If OAuth fails
            CalendarError: If event creation fails
        """
        cal_id = calendar_id or self._calendar_id

        # Build event data for Google Calendar
        event_data = {
            "summary": title,
            "description": description or "",
            "location": location or "",
            "start": {
                "dateTime": start.isoformat(),
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": end.isoformat(),
                "timeZone": "UTC",
            },
        }

        if attendees:
            event_data["attendees"] = [{"email": email} for email in attendees]

        # Add protected indicator to description if needed
        if is_protected and description:
            event_data["description"] = f"[PROTECTED] {description}"
        elif is_protected:
            event_data["description"] = "[PROTECTED]"

        try:
            result = await self._call_mcp(
                "create_event",
                calendar_id=cal_id,
                event=event_data,
            )

            # Invalidate relevant caches
            self._cache.invalidate_pattern(f"events_{cal_id}")

            # Classify and return the created event
            block_type, _ = BlockTypeClassifier.classify(event_data)

            return CalendarEvent(
                event_id=result.get("id", ""),
                title=title,
                start=start,
                end=end,
                block_type=block_type,
                description=description,
                is_protected=is_protected,
                attendees=attendees or [],
                location=location,
                calendar_id=cal_id,
                raw_data=result,
            )
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            raise CalendarError(f"Failed to create event: {e}")

    async def update_event(
        self,
        event_id: str,
        updates: Dict[str, Any],
        calendar_id: Optional[str] = None,
    ) -> CalendarEvent:
        """
        Update an existing calendar event.

        Args:
            event_id: ID of event to update
            updates: Dictionary of fields to update. Supported fields:
                    - title: New event title
                    - start: New start time (datetime)
                    - end: New end time (datetime)
                    - description: New description
                    - location: New location
                    - attendees: New attendee list
            calendar_id: Optional calendar ID override

        Returns:
            Updated CalendarEvent object
        """
        cal_id = calendar_id or self._calendar_id

        # Build patch data
        patch_data: Dict[str, Any] = {}

        if "title" in updates:
            patch_data["summary"] = updates["title"]
        if "description" in updates:
            patch_data["description"] = updates["description"]
        if "location" in updates:
            patch_data["location"] = updates["location"]
        if "start" in updates:
            patch_data["start"] = {
                "dateTime": updates["start"].isoformat(),
                "timeZone": "UTC",
            }
        if "end" in updates:
            patch_data["end"] = {
                "dateTime": updates["end"].isoformat(),
                "timeZone": "UTC",
            }
        if "attendees" in updates:
            patch_data["attendees"] = [
                {"email": email} for email in updates["attendees"]
            ]

        try:
            result = await self._call_mcp(
                "update_event",
                calendar_id=cal_id,
                event_id=event_id,
                event=patch_data,
            )

            # Invalidate caches
            self._cache.invalidate_pattern(f"events_{cal_id}")

            return self._parse_event(result)
        except Exception as e:
            logger.error(f"Failed to update event {event_id}: {e}")
            raise CalendarError(f"Failed to update event: {e}")

    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a calendar event.

        Args:
            event_id: ID of event to delete
            calendar_id: Optional calendar ID override

        Returns:
            True if deletion was successful
        """
        cal_id = calendar_id or self._calendar_id

        try:
            await self._call_mcp(
                "delete_event",
                calendar_id=cal_id,
                event_id=event_id,
            )

            # Invalidate caches
            self._cache.invalidate_pattern(f"events_{cal_id}")

            return True
        except Exception as e:
            logger.error(f"Failed to delete event {event_id}: {e}")
            raise CalendarError(f"Failed to delete event: {e}")

    # -------------------------------------------------------------------------
    # Public Methods: Availability
    # -------------------------------------------------------------------------

    async def check_availability(
        self,
        start: datetime,
        end: datetime,
        calendar_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Check if a time slot is available (no conflicting events).

        Args:
            start: Start of time slot
            end: End of time slot
            calendar_ids: Optional list of calendar IDs to check

        Returns:
            Dictionary with availability status:
            {
                "is_available": bool,
                "conflicts": List[CalendarEvent],
                "suggested_alternatives": List[Dict]  # if not available
            }
        """
        cal_ids = calendar_ids or [self._calendar_id]
        all_conflicts: List[CalendarEvent] = []

        for cal_id in cal_ids:
            events = await self.get_events_in_range(
                start - timedelta(hours=1),  # Buffer for overlaps
                end + timedelta(hours=1),
                calendar_id=cal_id,
            )

            for event in events:
                # Check for overlap
                if event.start < end and event.end > start:
                    all_conflicts.append(event)

        is_available = len(all_conflicts) == 0

        result = {
            "is_available": is_available,
            "conflicts": all_conflicts,
            "suggested_alternatives": [],
        }

        # If not available, suggest alternatives
        if not is_available:
            result["suggested_alternatives"] = self._find_alternatives(
                start, end, all_conflicts
            )

        return result

    async def find_free_slots(
        self,
        date: datetime,
        duration_minutes: int = 60,
        work_start_hour: int = 9,
        work_end_hour: int = 17,
    ) -> List[Dict[str, datetime]]:
        """
        Find free time slots on a given date.

        Args:
            date: Date to search for free slots
            duration_minutes: Minimum duration of free slot
            work_start_hour: Start of work day (default: 9 AM)
            work_end_hour: End of work day (default: 5 PM)

        Returns:
            List of free slots as {"start": datetime, "end": datetime}
        """
        day_start = date.replace(
            hour=work_start_hour, minute=0, second=0, microsecond=0
        )
        day_end = date.replace(
            hour=work_end_hour, minute=0, second=0, microsecond=0
        )

        events = await self.get_events_in_range(day_start, day_end)
        events.sort(key=lambda e: e.start)

        free_slots: List[Dict[str, datetime]] = []
        current_time = day_start

        for event in events:
            # Check gap before this event
            if event.start > current_time:
                gap_minutes = (event.start - current_time).total_seconds() / 60
                if gap_minutes >= duration_minutes:
                    free_slots.append({
                        "start": current_time,
                        "end": event.start,
                    })

            # Move current time to end of this event
            if event.end > current_time:
                current_time = event.end

        # Check remaining time after last event
        if current_time < day_end:
            gap_minutes = (day_end - current_time).total_seconds() / 60
            if gap_minutes >= duration_minutes:
                free_slots.append({
                    "start": current_time,
                    "end": day_end,
                })

        return free_slots

    # -------------------------------------------------------------------------
    # Public Methods: Block Type Analysis
    # -------------------------------------------------------------------------

    async def get_events_by_block_type(
        self,
        block_type: BlockType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[CalendarEvent]:
        """
        Get all events of a specific block type.

        Args:
            block_type: BlockType to filter by
            start_date: Start of date range (default: today)
            end_date: End of date range (default: 7 days from start)

        Returns:
            List of CalendarEvent objects matching the block type
        """
        if start_date is None:
            start_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if end_date is None:
            end_date = start_date + timedelta(days=7)

        all_events = await self.get_events_in_range(start_date, end_date)
        return [e for e in all_events if e.block_type == block_type.value]

    async def get_block_type_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get summary statistics by block type.

        Returns:
            Dictionary mapping block type to statistics:
            {
                "CLINICAL": {
                    "count": 5,
                    "total_minutes": 240,
                    "events": [...]
                },
                ...
            }
        """
        if start_date is None:
            start_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if end_date is None:
            end_date = start_date + timedelta(days=7)

        all_events = await self.get_events_in_range(start_date, end_date)

        summary: Dict[str, Dict[str, Any]] = {}
        for bt in BlockType:
            summary[bt.value] = {
                "count": 0,
                "total_minutes": 0,
                "events": [],
            }

        for event in all_events:
            bt = event.block_type
            if bt in summary:
                summary[bt]["count"] += 1
                summary[bt]["total_minutes"] += event.duration_minutes()
                summary[bt]["events"].append(event)

        return summary

    # -------------------------------------------------------------------------
    # Public Methods: Status and Configuration
    # -------------------------------------------------------------------------

    @property
    def is_online(self) -> bool:
        """Check if adapter is currently online."""
        return self._is_online

    @property
    def last_sync_time(self) -> Optional[datetime]:
        """Get the last successful sync time."""
        return self._last_sync_time

    def set_offline_events(self, events: List[CalendarEvent]) -> None:
        """
        Set events for offline fallback mode.

        Args:
            events: List of events to use when offline
        """
        self._offline_events = events

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Google Calendar.

        Returns:
            Dictionary with connection status and details
        """
        try:
            # Try to list calendars as a connection test
            result = await self._call_mcp("list_calendars")
            self._is_online = True
            return {
                "connected": True,
                "calendar_count": len(result.get("items", [])),
                "primary_calendar": self._calendar_id,
            }
        except AuthenticationError as e:
            return {
                "connected": False,
                "error": "authentication",
                "message": str(e),
            }
        except Exception as e:
            self._is_online = False
            return {
                "connected": False,
                "error": "connection",
                "message": str(e),
            }

    # -------------------------------------------------------------------------
    # Private Methods: MCP Communication
    # -------------------------------------------------------------------------

    async def _call_mcp(
        self,
        method: str,
        **kwargs,
    ) -> Any:
        """
        Call MCP server method with error handling.

        Args:
            method: MCP method to call
            **kwargs: Method arguments

        Returns:
            MCP method result

        Raises:
            AuthenticationError: If OAuth authentication fails
            RateLimitError: If rate limit is exceeded
            OfflineError: If MCP client is not available
        """
        if self._mcp_client is None:
            raise OfflineError("MCP client not configured")

        try:
            # Call the MCP method
            # The exact API depends on the MCP client implementation
            if hasattr(self._mcp_client, 'call'):
                result = await self._mcp_client.call(
                    f"google-calendar/{method}",
                    **kwargs
                )
            elif hasattr(self._mcp_client, 'invoke'):
                result = await self._mcp_client.invoke(
                    "google-calendar",
                    method,
                    kwargs
                )
            else:
                # Generic fallback
                mcp_method = getattr(self._mcp_client, method, None)
                if mcp_method:
                    result = await mcp_method(**kwargs)
                else:
                    raise CalendarError(f"Unknown MCP method: {method}")

            return result

        except Exception as e:
            error_str = str(e).lower()

            # Check for authentication errors
            if any(term in error_str for term in [
                "unauthorized", "authentication", "oauth", "token",
                "401", "invalid_grant", "access_denied"
            ]):
                raise AuthenticationError(
                    f"OAuth authentication failed: {e}. "
                    "Please re-authenticate with Google Calendar."
                )

            # Check for rate limit errors
            if any(term in error_str for term in [
                "rate limit", "quota", "429", "too many requests"
            ]):
                # Try to extract retry-after
                retry_after = None
                if "retry-after" in error_str:
                    match = re.search(r'retry-after[:\s]+(\d+)', error_str)
                    if match:
                        retry_after = int(match.group(1))
                raise RateLimitError(
                    f"Rate limit exceeded: {e}",
                    retry_after=retry_after or 60
                )

            # Re-raise other errors
            raise

    async def _fetch_events(
        self,
        start_date: datetime,
        end_date: datetime,
        calendar_id: str,
    ) -> List[CalendarEvent]:
        """
        Fetch events from Google Calendar via MCP.

        Args:
            start_date: Start of date range
            end_date: End of date range
            calendar_id: Calendar ID to fetch from

        Returns:
            List of CalendarEvent objects
        """
        result = await self._call_mcp(
            "list_events",
            calendar_id=calendar_id,
            time_min=start_date.isoformat() + "Z",
            time_max=end_date.isoformat() + "Z",
            single_events=True,
            order_by="startTime",
        )

        events = []
        for item in result.get("items", []):
            event = self._parse_event(item)
            if event:
                events.append(event)

        return events

    def _parse_event(self, raw_event: Dict[str, Any]) -> Optional[CalendarEvent]:
        """
        Parse raw Google Calendar event into CalendarEvent.

        Args:
            raw_event: Raw event data from Google Calendar API

        Returns:
            CalendarEvent object or None if parsing fails
        """
        try:
            # Extract start/end times
            start_data = raw_event.get("start", {})
            end_data = raw_event.get("end", {})

            # Handle all-day vs timed events
            if "dateTime" in start_data:
                start = datetime.fromisoformat(
                    start_data["dateTime"].replace("Z", "+00:00")
                )
                end = datetime.fromisoformat(
                    end_data["dateTime"].replace("Z", "+00:00")
                )
            elif "date" in start_data:
                # All-day event
                start = datetime.strptime(start_data["date"], "%Y-%m-%d")
                end = datetime.strptime(end_data["date"], "%Y-%m-%d")
            else:
                logger.warning(f"Event missing start time: {raw_event}")
                return None

            # Remove timezone info for consistency
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)

            # Classify block type and protection status
            block_type, is_protected = BlockTypeClassifier.classify(raw_event)

            # Extract attendees
            attendees = [
                a.get("email", "")
                for a in raw_event.get("attendees", [])
                if a.get("email")
            ]

            return CalendarEvent(
                event_id=raw_event.get("id", ""),
                title=raw_event.get("summary", ""),
                start=start,
                end=end,
                block_type=block_type,
                description=raw_event.get("description"),
                is_protected=is_protected,
                attendees=attendees,
                location=raw_event.get("location"),
                calendar_id=raw_event.get("calendarId"),
                recurring_event_id=raw_event.get("recurringEventId"),
                color_id=raw_event.get("colorId"),
                raw_data=raw_event,
            )
        except Exception as e:
            logger.error(f"Failed to parse event: {e}")
            return None

    # -------------------------------------------------------------------------
    # Private Methods: Helpers
    # -------------------------------------------------------------------------

    def _get_offline_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[CalendarEvent]:
        """Get events from offline store for date range."""
        return [
            e for e in self._offline_events
            if start_date <= e.start < end_date
        ]

    def _find_alternatives(
        self,
        start: datetime,
        end: datetime,
        conflicts: List[CalendarEvent],
    ) -> List[Dict[str, datetime]]:
        """
        Find alternative time slots when requested time is not available.

        Args:
            start: Original requested start time
            end: Original requested end time
            conflicts: Conflicting events

        Returns:
            List of alternative slot suggestions
        """
        duration = end - start
        alternatives = []

        # Try earlier same day
        earliest_conflict = min(conflicts, key=lambda e: e.start)
        if earliest_conflict.start > start.replace(hour=9, minute=0):
            alt_end = earliest_conflict.start - timedelta(minutes=15)
            alt_start = alt_end - duration
            if alt_start >= start.replace(hour=9, minute=0):
                alternatives.append({
                    "start": alt_start,
                    "end": alt_end,
                    "reason": "Before first conflict",
                })

        # Try later same day
        latest_conflict = max(conflicts, key=lambda e: e.end)
        if latest_conflict.end < start.replace(hour=17, minute=0):
            alt_start = latest_conflict.end + timedelta(minutes=15)
            alt_end = alt_start + duration
            if alt_end <= start.replace(hour=17, minute=0):
                alternatives.append({
                    "start": alt_start,
                    "end": alt_end,
                    "reason": "After last conflict",
                })

        # Try next day same time
        next_day_start = start + timedelta(days=1)
        next_day_end = end + timedelta(days=1)
        alternatives.append({
            "start": next_day_start,
            "end": next_day_end,
            "reason": "Same time tomorrow",
        })

        return alternatives[:3]  # Return top 3 alternatives
