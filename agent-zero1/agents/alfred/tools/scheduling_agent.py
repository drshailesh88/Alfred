"""
Scheduling Agent (Calendar Agent) for Alfred

Time block management and calendar optimization system. Protects focus time,
enforces buffer zones, manages meeting requests, and ensures recovery windows
are maintained.

This agent operates under Alfred's coordination and outputs structured packets
for calendar status, scheduling responses, and calendar alerts.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Protocol
from . import OperationsAgent, AgentResponse, AlfredState


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class BlockType(Enum):
    """Time block categories for calendar management."""
    CLINICAL = "CLINICAL"        # Patient care, procedures, rounds
    DEEP_WORK = "DEEP_WORK"      # Focused creation, writing, building
    MEETINGS = "MEETINGS"        # Calls, syncs, external meetings
    BUFFER = "BUFFER"            # Transition time, overflow
    RECOVERY = "RECOVERY"        # Rest, meals, breaks
    PERSONAL = "PERSONAL"        # Family, non-work commitments
    COMMUTE = "COMMUTE"          # Travel time (potential learning time)


class AlertType(Enum):
    """Calendar alert types."""
    OVERLOAD = "overload"
    BUFFER_BREACH = "buffer_breach"
    PERSONAL_ENCROACHMENT = "personal_encroachment"
    RECOVERY_SKIP = "recovery_skip"
    DOUBLE_BOOKING = "double_booking"
    BACK_TO_BACK_INTENSIVE = "back_to_back_intensive"


class RequestType(Enum):
    """Calendar request types."""
    STATUS = "status"
    SCHEDULE = "schedule"
    RESCHEDULE = "reschedule"
    PROTECT = "protect"
    ANALYZE = "analyze"


class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = "critical"      # Cannot be moved or declined
    HIGH = "high"              # Important, requires good reason to move
    MEDIUM = "medium"          # Standard priority
    LOW = "low"                # Can be rescheduled easily
    TENTATIVE = "tentative"    # Not confirmed, may be dropped


class IntensityLevel(Enum):
    """Energy intensity levels for activities."""
    HIGH = "high"              # Demanding, requires recovery after
    MEDIUM = "medium"          # Moderate effort
    LOW = "low"                # Light, minimal energy drain
    RECOVERY = "recovery"      # Actively restorative


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SchedulingConfig:
    """Configuration for the scheduling agent."""

    # Buffer requirements (in minutes)
    min_buffer_between_meetings: int = 15
    buffer_after_clinical: int = 30
    buffer_after_deep_work: int = 15
    buffer_before_deep_work: int = 15
    buffer_after_high_intensity: int = 30

    # Meeting limits
    max_meetings_per_day: int = 6
    max_meeting_hours_per_day: float = 4.0
    max_meeting_hours_per_week: float = 15.0
    max_consecutive_meetings: int = 2

    # Protected time defaults
    min_deep_work_hours_per_day: float = 2.0
    min_recovery_minutes_per_day: int = 60
    lunch_duration_minutes: int = 60

    # Energy patterns (24h format: hour -> intensity tolerance)
    peak_hours_start: int = 9
    peak_hours_end: int = 12
    low_energy_hours_start: int = 14
    low_energy_hours_end: int = 15

    # Work boundaries
    work_day_start: time = field(default_factory=lambda: time(8, 0))
    work_day_end: time = field(default_factory=lambda: time(18, 0))
    personal_time_start: time = field(default_factory=lambda: time(18, 0))
    personal_time_end: time = field(default_factory=lambda: time(8, 0))

    # Back-to-back rules
    high_intensity_blocks: List[BlockType] = field(
        default_factory=lambda: [BlockType.CLINICAL, BlockType.DEEP_WORK]
    )


DEFAULT_CONFIG = SchedulingConfig()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimeBlock:
    """Represents a protected time block."""
    block_type: BlockType
    start_time: datetime
    end_time: datetime
    title: str = ""
    description: str = ""
    is_protected: bool = True
    protection_level: str = "standard"  # standard, sacred, flexible
    recurrence: Optional[str] = None    # daily, weekly, monthly, none

    @property
    def duration_minutes(self) -> int:
        """Return duration in minutes."""
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() / 60)

    @property
    def duration_hours(self) -> float:
        """Return duration in hours."""
        return self.duration_minutes / 60.0

    def overlaps(self, other: 'TimeBlock') -> bool:
        """Check if this block overlaps with another."""
        return (self.start_time < other.end_time and
                self.end_time > other.start_time)

    def contains(self, dt: datetime) -> bool:
        """Check if a datetime falls within this block."""
        return self.start_time <= dt < self.end_time


@dataclass
class CalendarEvent:
    """Represents a calendar event."""
    event_id: str
    title: str
    start_time: datetime
    end_time: datetime
    block_type: BlockType = BlockType.MEETINGS
    priority: EventPriority = EventPriority.MEDIUM
    intensity: IntensityLevel = IntensityLevel.MEDIUM
    location: str = ""
    description: str = ""
    attendees: List[str] = field(default_factory=list)
    is_recurring: bool = False
    recurrence_rule: Optional[str] = None
    source: str = "manual"  # manual, google, cal_com, outlook
    external_id: Optional[str] = None
    buffer_before: int = 0   # minutes
    buffer_after: int = 0    # minutes
    can_reschedule: bool = True
    requires_prep: bool = False
    prep_time_minutes: int = 0

    @property
    def duration_minutes(self) -> int:
        """Return duration in minutes."""
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() / 60)

    @property
    def total_time_commitment(self) -> int:
        """Return total time including buffers and prep."""
        return (self.duration_minutes +
                self.buffer_before +
                self.buffer_after +
                self.prep_time_minutes)

    def overlaps(self, other: 'CalendarEvent') -> bool:
        """Check if this event overlaps with another."""
        return (self.start_time < other.end_time and
                self.end_time > other.start_time)


@dataclass
class CalendarConflict:
    """Represents a calendar conflict."""
    conflict_type: str
    events: List[CalendarEvent]
    time_period: Tuple[datetime, datetime]
    severity: str  # high, medium, low
    description: str
    resolution_suggestions: List[str] = field(default_factory=list)


@dataclass
class CalendarAlert:
    """Represents a calendar alert/warning."""
    alert_type: AlertType
    period: Tuple[datetime, datetime]
    issue: str
    impact: str
    recommended_action: str
    severity: str = "medium"  # high, medium, low
    affected_events: List[CalendarEvent] = field(default_factory=list)


@dataclass
class ScheduleStatus:
    """Status summary for a time period."""
    period_start: datetime
    period_end: datetime
    events: List[CalendarEvent]
    protected_blocks: List[TimeBlock]
    conflicts: List[CalendarConflict]
    alerts: List[CalendarAlert]
    meeting_hours: float
    meeting_target_hours: float
    deep_work_hours: float
    recovery_hours: float
    personal_hours: float
    buffer_adequate: bool
    buffer_gaps: List[Tuple[datetime, datetime]]


# =============================================================================
# CALENDAR PROVIDER INTERFACES
# =============================================================================

class CalendarProvider(Protocol):
    """Protocol for calendar provider integrations."""

    def get_events(
        self,
        start: datetime,
        end: datetime
    ) -> List[CalendarEvent]:
        """Fetch events from the calendar provider."""
        ...

    def create_event(self, event: CalendarEvent) -> str:
        """Create an event and return the event ID."""
        ...

    def update_event(self, event: CalendarEvent) -> bool:
        """Update an existing event."""
        ...

    def delete_event(self, event_id: str) -> bool:
        """Delete an event."""
        ...


class GoogleCalendarProvider:
    """
    Interface for Google Calendar via google-calendar-mcp or mcp-gsuite.

    This is a placeholder that prepares the interface for actual MCP integration.
    When connected, this will use the MCP tools to interact with Google Calendar.
    """

    def __init__(self, mcp_client: Optional[Any] = None):
        self.mcp_client = mcp_client
        self.connected = mcp_client is not None

    def get_events(
        self,
        start: datetime,
        end: datetime
    ) -> List[CalendarEvent]:
        """
        Fetch events from Google Calendar.

        MCP Integration Notes:
        - Use google-calendar-mcp's list_events tool
        - Or use mcp-gsuite's calendar.list tool
        """
        if not self.connected:
            return []

        # Placeholder for MCP integration
        # events = self.mcp_client.call("google-calendar-mcp", "list_events", {
        #     "time_min": start.isoformat(),
        #     "time_max": end.isoformat()
        # })
        return []

    def create_event(self, event: CalendarEvent) -> str:
        """Create an event in Google Calendar."""
        if not self.connected:
            return ""

        # Placeholder for MCP integration
        # result = self.mcp_client.call("google-calendar-mcp", "create_event", {...})
        return ""

    def update_event(self, event: CalendarEvent) -> bool:
        """Update an event in Google Calendar."""
        if not self.connected:
            return False
        return False

    def delete_event(self, event_id: str) -> bool:
        """Delete an event from Google Calendar."""
        if not self.connected:
            return False
        return False


class CalComProvider:
    """
    Interface for Cal.com scheduling.

    This is a placeholder that prepares the interface for Cal.com integration.
    Cal.com is used for external booking and scheduling links.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.connected = api_key is not None

    def get_bookings(
        self,
        start: datetime,
        end: datetime
    ) -> List[CalendarEvent]:
        """Fetch bookings from Cal.com."""
        if not self.connected:
            return []

        # Placeholder for Cal.com API integration
        return []

    def get_availability(
        self,
        start: datetime,
        end: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Get available time slots from Cal.com."""
        if not self.connected:
            return []
        return []

    def block_time(
        self,
        start: datetime,
        end: datetime,
        reason: str
    ) -> bool:
        """Block time in Cal.com to prevent bookings."""
        if not self.connected:
            return False
        return False


# =============================================================================
# SCHEDULING AGENT
# =============================================================================

class SchedulingAgent(OperationsAgent):
    """
    Scheduling/Calendar Agent for Alfred.

    Time block management and calendar optimization system. Protects focus time,
    enforces buffer zones, manages meeting requests, and ensures recovery windows
    are maintained.

    Does NOT:
    - Accept meetings without Alfred's criteria
    - Overbook or double-schedule
    - Ignore buffer time requirements
    - Schedule during protected blocks
    - Accept back-to-back high-intensity sessions
    - Allow meeting creep into personal time
    - Schedule without considering energy patterns

    Does:
    - Manage all calendar operations
    - Protect focus blocks as sacred
    - Insert buffer time between meetings
    - Enforce recovery windows
    - Suggest optimal times for different work types
    - Track meeting patterns and time allocation
    - Flag calendar conflicts
    - Recommend rescheduling when overloaded
    """

    def __init__(
        self,
        config: Optional[SchedulingConfig] = None,
        google_provider: Optional[GoogleCalendarProvider] = None,
        cal_com_provider: Optional[CalComProvider] = None
    ):
        super().__init__("SchedulingAgent")
        self.config = config or DEFAULT_CONFIG

        # Calendar providers
        self.google_provider = google_provider or GoogleCalendarProvider()
        self.cal_com_provider = cal_com_provider or CalComProvider()

        # Internal state
        self._events: Dict[str, CalendarEvent] = {}
        self._protected_blocks: Dict[str, TimeBlock] = {}
        self._next_event_id = 1
        self._next_block_id = 1

    # -------------------------------------------------------------------------
    # MAIN INTERFACE METHODS
    # -------------------------------------------------------------------------

    def process_request(
        self,
        request_type: RequestType,
        time_window: Optional[Tuple[datetime, datetime]] = None,
        event_details: Optional[Dict[str, Any]] = None,
        priority: Optional[EventPriority] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a calendar request and return appropriate response.

        Input Format:
        CALENDAR_REQUEST
        - Request Type: status | schedule | reschedule | protect | analyze
        - Time Window: [specific period]
        - Event Details: [if scheduling]
        - Priority: [if scheduling]
        - Constraints: [any specific requirements]
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Default time window to today
        if time_window is None:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            time_window = (today, today + timedelta(days=1))

        if request_type == RequestType.STATUS:
            return self.get_status(time_window)
        elif request_type == RequestType.SCHEDULE:
            if event_details is None:
                return self.create_response(
                    {"error": "Event details required for scheduling"},
                    success=False,
                    errors=["Missing event_details parameter"]
                )
            return self.schedule_event(event_details, priority, constraints)
        elif request_type == RequestType.RESCHEDULE:
            if event_details is None or "event_id" not in event_details:
                return self.create_response(
                    {"error": "Event ID required for rescheduling"},
                    success=False,
                    errors=["Missing event_id in event_details"]
                )
            return self.reschedule_event(
                event_details["event_id"],
                event_details.get("new_time"),
                constraints
            )
        elif request_type == RequestType.PROTECT:
            if event_details is None:
                return self.create_response(
                    {"error": "Block details required for protection"},
                    success=False,
                    errors=["Missing event_details parameter"]
                )
            return self.protect_block(event_details)
        elif request_type == RequestType.ANALYZE:
            return self.analyze_week(time_window)
        else:
            return self.create_response(
                {"error": f"Unknown request type: {request_type}"},
                success=False,
                errors=[f"Invalid request type: {request_type}"]
            )

    def get_status(
        self,
        time_window: Tuple[datetime, datetime]
    ) -> AgentResponse:
        """
        Get calendar status for a time period.

        Output Format:
        CALENDAR_REPORT
        - Report Date, Period Covered
        - Today's Schedule: [time-blocked view]
        - Protected Blocks: [focus, recovery, personal hours]
        - Conflicts/Issues
        - Meeting Load: [hours vs target]
        - Buffer Status: [adequate, gaps]
        - Recommendations
        """
        start, end = time_window

        # Gather events and blocks
        events = self._get_events_in_range(start, end)
        protected = self._get_protected_blocks_in_range(start, end)

        # Calculate metrics
        meeting_hours = self._calculate_meeting_hours(events)
        deep_work_hours = self._calculate_block_type_hours(events, protected, BlockType.DEEP_WORK)
        recovery_hours = self._calculate_block_type_hours(events, protected, BlockType.RECOVERY)
        personal_hours = self._calculate_block_type_hours(events, protected, BlockType.PERSONAL)

        # Check for conflicts and issues
        conflicts = self.check_conflicts(start, end)
        alerts = self._generate_alerts(start, end, events, protected)

        # Check buffer status
        buffer_adequate, buffer_gaps = self._check_buffer_status(events)

        # Generate schedule view
        schedule_view = self._generate_schedule_view(start, end, events, protected)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            events, protected, conflicts, alerts,
            meeting_hours, deep_work_hours, recovery_hours
        )

        # Determine meeting target for period
        days_in_period = (end - start).days or 1
        meeting_target = self.config.max_meeting_hours_per_day * days_in_period

        report_data = {
            "report_type": "CALENDAR_REPORT",
            "report_date": datetime.now().isoformat(),
            "period_covered": {
                "start": start.isoformat(),
                "end": end.isoformat()
            },
            "schedule": schedule_view,
            "protected_blocks": {
                "focus_time_hours": deep_work_hours,
                "recovery_hours": recovery_hours,
                "personal_hours": personal_hours,
                "blocks": [self._block_to_dict(b) for b in protected]
            },
            "conflicts_issues": [self._conflict_to_dict(c) for c in conflicts],
            "meeting_load": {
                "hours_scheduled": meeting_hours,
                "target_hours": meeting_target,
                "status": "over" if meeting_hours > meeting_target else "on_track"
            },
            "buffer_status": {
                "adequate": buffer_adequate,
                "gaps": [
                    {"start": g[0].isoformat(), "end": g[1].isoformat()}
                    for g in buffer_gaps
                ]
            },
            "recommendations": recommendations,
            "alerts": [self._alert_to_dict(a) for a in alerts]
        }

        # Add state-specific notes
        if self.alfred_state == AlfredState.RED:
            report_data["state_note"] = "RED state active - consider clearing calendar"

        return self.create_response(
            report_data,
            warnings=[a.issue for a in alerts if a.severity == "high"]
        )

    def schedule_event(
        self,
        event_details: Dict[str, Any],
        priority: Optional[EventPriority] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Schedule a new event with all protections enforced.

        Output Format:
        SCHEDULE_RESPONSE
        - Request: [what was asked]
        - Status: scheduled | declined | needs_input
        - Details: Event, Time, Duration, Buffer Added, Conflicts Resolved
        - Notes: [any caveats or concerns]
        """
        constraints = constraints or {}
        priority = priority or EventPriority.MEDIUM

        # Parse event details
        title = event_details.get("title", "Untitled Event")
        start_time = self._parse_datetime(event_details.get("start_time"))
        end_time = self._parse_datetime(event_details.get("end_time"))
        duration_minutes = event_details.get("duration_minutes")

        if start_time is None:
            return self.create_response(
                {
                    "report_type": "SCHEDULE_RESPONSE",
                    "request": title,
                    "status": "needs_input",
                    "reason": "Start time is required"
                },
                success=False,
                errors=["Missing or invalid start_time"]
            )

        # Calculate end time if not provided
        if end_time is None and duration_minutes:
            end_time = start_time + timedelta(minutes=duration_minutes)
        elif end_time is None:
            end_time = start_time + timedelta(hours=1)  # Default 1 hour

        # Determine block type and intensity
        block_type = BlockType(
            event_details.get("block_type", BlockType.MEETINGS.value)
        )
        intensity = IntensityLevel(
            event_details.get("intensity", IntensityLevel.MEDIUM.value)
        )

        # Create event object
        event = CalendarEvent(
            event_id=f"evt_{self._next_event_id}",
            title=title,
            start_time=start_time,
            end_time=end_time,
            block_type=block_type,
            priority=priority,
            intensity=intensity,
            location=event_details.get("location", ""),
            description=event_details.get("description", ""),
            attendees=event_details.get("attendees", []),
            can_reschedule=event_details.get("can_reschedule", True),
            requires_prep=event_details.get("requires_prep", False),
            prep_time_minutes=event_details.get("prep_time_minutes", 0)
        )

        # Run scheduling checks
        validation = self._validate_scheduling(event, constraints)

        if not validation["can_schedule"]:
            # Check if we can find alternative time
            if constraints.get("find_alternative", True):
                alternatives = self._find_alternative_times(event, constraints)
                return self.create_response(
                    {
                        "report_type": "SCHEDULE_RESPONSE",
                        "request": title,
                        "status": "declined",
                        "reason": validation["reason"],
                        "violations": validation["violations"],
                        "alternative_times": [
                            {"start": a[0].isoformat(), "end": a[1].isoformat()}
                            for a in alternatives[:5]
                        ]
                    },
                    success=False,
                    errors=[validation["reason"]]
                )
            else:
                return self.create_response(
                    {
                        "report_type": "SCHEDULE_RESPONSE",
                        "request": title,
                        "status": "declined",
                        "reason": validation["reason"],
                        "violations": validation["violations"]
                    },
                    success=False,
                    errors=[validation["reason"]]
                )

        # Calculate and add buffers
        buffer_before, buffer_after = self._calculate_required_buffers(event)
        event.buffer_before = buffer_before
        event.buffer_after = buffer_after

        # Store the event
        self._events[event.event_id] = event
        self._next_event_id += 1

        # Sync to external providers if connected
        external_ids = self._sync_to_providers(event)

        return self.create_response(
            {
                "report_type": "SCHEDULE_RESPONSE",
                "request": title,
                "status": "scheduled",
                "details": {
                    "event_id": event.event_id,
                    "title": event.title,
                    "time": {
                        "start": event.start_time.isoformat(),
                        "end": event.end_time.isoformat()
                    },
                    "duration_minutes": event.duration_minutes,
                    "buffer_added": {
                        "before_minutes": buffer_before,
                        "after_minutes": buffer_after
                    },
                    "conflicts_resolved": validation.get("conflicts_resolved", []),
                    "external_ids": external_ids
                },
                "notes": validation.get("warnings", [])
            }
        )

    def reschedule_event(
        self,
        event_id: str,
        new_time: Optional[datetime] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Reschedule an existing event."""
        constraints = constraints or {}

        if event_id not in self._events:
            return self.create_response(
                {"error": f"Event not found: {event_id}"},
                success=False,
                errors=[f"Event {event_id} not found"]
            )

        event = self._events[event_id]

        if not event.can_reschedule:
            return self.create_response(
                {
                    "report_type": "SCHEDULE_RESPONSE",
                    "request": f"Reschedule {event.title}",
                    "status": "declined",
                    "reason": "Event cannot be rescheduled"
                },
                success=False,
                errors=["Event is marked as non-reschedulable"]
            )

        if new_time is None:
            # Find next available slot
            alternatives = self._find_alternative_times(event, constraints)
            if not alternatives:
                return self.create_response(
                    {
                        "report_type": "SCHEDULE_RESPONSE",
                        "request": f"Reschedule {event.title}",
                        "status": "needs_input",
                        "reason": "No available time slots found"
                    },
                    success=False
                )
            new_time = alternatives[0][0]

        # Store original for response
        original_time = event.start_time
        duration = event.end_time - event.start_time

        # Update event times
        event.start_time = new_time
        event.end_time = new_time + duration

        # Validate new position
        validation = self._validate_scheduling(event, constraints)

        if not validation["can_schedule"]:
            # Revert
            event.start_time = original_time
            event.end_time = original_time + duration

            return self.create_response(
                {
                    "report_type": "SCHEDULE_RESPONSE",
                    "request": f"Reschedule {event.title}",
                    "status": "declined",
                    "reason": validation["reason"]
                },
                success=False,
                errors=[validation["reason"]]
            )

        return self.create_response(
            {
                "report_type": "SCHEDULE_RESPONSE",
                "request": f"Reschedule {event.title}",
                "status": "scheduled",
                "details": {
                    "event_id": event.event_id,
                    "original_time": original_time.isoformat(),
                    "new_time": event.start_time.isoformat(),
                    "duration_minutes": event.duration_minutes
                }
            }
        )

    def check_conflicts(
        self,
        start: datetime,
        end: datetime
    ) -> List[CalendarConflict]:
        """Check for conflicts in a time period."""
        conflicts = []
        events = self._get_events_in_range(start, end)
        protected = self._get_protected_blocks_in_range(start, end)

        # Check event-to-event conflicts
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                if event1.overlaps(event2):
                    conflicts.append(CalendarConflict(
                        conflict_type="double_booking",
                        events=[event1, event2],
                        time_period=(
                            max(event1.start_time, event2.start_time),
                            min(event1.end_time, event2.end_time)
                        ),
                        severity="high",
                        description=f"Overlap: {event1.title} and {event2.title}",
                        resolution_suggestions=[
                            f"Reschedule {event1.title if event1.priority.value > event2.priority.value else event2.title}",
                            "Decline one of the events"
                        ]
                    ))

        # Check event vs protected block conflicts
        for event in events:
            for block in protected:
                if block.is_protected and event.start_time < block.end_time and event.end_time > block.start_time:
                    conflicts.append(CalendarConflict(
                        conflict_type="protected_block_violation",
                        events=[event],
                        time_period=(
                            max(event.start_time, block.start_time),
                            min(event.end_time, block.end_time)
                        ),
                        severity="high" if block.protection_level == "sacred" else "medium",
                        description=f"{event.title} conflicts with protected {block.block_type.value} time",
                        resolution_suggestions=[
                            f"Reschedule {event.title}",
                            f"Reduce protection on {block.title}" if block.protection_level != "sacred" else ""
                        ]
                    ))

        # Check for back-to-back high intensity
        sorted_events = sorted(events, key=lambda e: e.start_time)
        for i in range(len(sorted_events) - 1):
            curr = sorted_events[i]
            next_evt = sorted_events[i + 1]

            if (curr.intensity == IntensityLevel.HIGH and
                next_evt.intensity == IntensityLevel.HIGH):
                gap = (next_evt.start_time - curr.end_time).total_seconds() / 60
                if gap < self.config.buffer_after_high_intensity:
                    conflicts.append(CalendarConflict(
                        conflict_type="back_to_back_intensive",
                        events=[curr, next_evt],
                        time_period=(curr.end_time, next_evt.start_time),
                        severity="medium",
                        description="High-intensity sessions need recovery buffer",
                        resolution_suggestions=[
                            f"Add {self.config.buffer_after_high_intensity}min buffer",
                            "Reschedule one session"
                        ]
                    ))

        return conflicts

    def protect_block(
        self,
        block_details: Dict[str, Any]
    ) -> AgentResponse:
        """Create or update a protected time block."""
        block_type = BlockType(
            block_details.get("block_type", BlockType.DEEP_WORK.value)
        )
        start_time = self._parse_datetime(block_details.get("start_time"))
        end_time = self._parse_datetime(block_details.get("end_time"))

        if start_time is None or end_time is None:
            return self.create_response(
                {"error": "Start and end times are required"},
                success=False,
                errors=["Missing time parameters"]
            )

        block = TimeBlock(
            block_type=block_type,
            start_time=start_time,
            end_time=end_time,
            title=block_details.get("title", f"{block_type.value} Block"),
            description=block_details.get("description", ""),
            is_protected=True,
            protection_level=block_details.get("protection_level", "standard"),
            recurrence=block_details.get("recurrence")
        )

        block_id = f"blk_{self._next_block_id}"
        self._protected_blocks[block_id] = block
        self._next_block_id += 1

        # Check for conflicts with existing events
        conflicts = []
        for event in self._get_events_in_range(start_time, end_time):
            if block.start_time < event.end_time and block.end_time > event.start_time:
                conflicts.append(event.title)

        return self.create_response(
            {
                "report_type": "PROTECTION_RESPONSE",
                "status": "protected",
                "block_id": block_id,
                "details": self._block_to_dict(block),
                "conflicts_detected": conflicts,
                "note": "Existing events conflict with this block" if conflicts else None
            },
            warnings=[f"Conflicts with: {', '.join(conflicts)}"] if conflicts else []
        )

    def analyze_week(
        self,
        time_window: Optional[Tuple[datetime, datetime]] = None
    ) -> AgentResponse:
        """
        Analyze a week's calendar for patterns and recommendations.

        Returns comprehensive analysis of meeting patterns, time allocation,
        buffer adequacy, and specific recommendations.
        """
        if time_window is None:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            # Find start of week (Monday)
            days_since_monday = today.weekday()
            week_start = today - timedelta(days=days_since_monday)
            week_end = week_start + timedelta(days=7)
            time_window = (week_start, week_end)

        start, end = time_window
        events = self._get_events_in_range(start, end)
        protected = self._get_protected_blocks_in_range(start, end)

        # Daily breakdown
        daily_stats = {}
        current_day = start
        while current_day < end:
            next_day = current_day + timedelta(days=1)
            day_events = [
                e for e in events
                if e.start_time >= current_day and e.start_time < next_day
            ]

            daily_stats[current_day.strftime("%A")] = {
                "date": current_day.strftime("%Y-%m-%d"),
                "meeting_count": len([e for e in day_events if e.block_type == BlockType.MEETINGS]),
                "meeting_hours": sum(
                    e.duration_minutes / 60
                    for e in day_events
                    if e.block_type == BlockType.MEETINGS
                ),
                "deep_work_hours": sum(
                    e.duration_minutes / 60
                    for e in day_events
                    if e.block_type == BlockType.DEEP_WORK
                ),
                "first_meeting": min((e.start_time for e in day_events), default=None),
                "last_meeting_end": max((e.end_time for e in day_events), default=None),
                "intensity_score": self._calculate_day_intensity(day_events)
            }
            current_day = next_day

        # Week totals
        total_meeting_hours = sum(
            e.duration_minutes / 60
            for e in events
            if e.block_type == BlockType.MEETINGS
        )
        total_deep_work = sum(
            e.duration_minutes / 60
            for e in events
            if e.block_type == BlockType.DEEP_WORK
        )

        # Meeting patterns
        meeting_patterns = self._analyze_meeting_patterns(events)

        # Buffer analysis
        buffer_adequate, buffer_gaps = self._check_buffer_status(events)

        # Generate alerts
        alerts = self._generate_alerts(start, end, events, protected)

        # Overall assessment
        health_score = self._calculate_schedule_health(
            total_meeting_hours, total_deep_work, buffer_adequate, len(alerts)
        )

        return self.create_response(
            {
                "report_type": "WEEKLY_ANALYSIS",
                "period": {
                    "start": start.isoformat(),
                    "end": end.isoformat()
                },
                "summary": {
                    "health_score": health_score,
                    "total_meeting_hours": round(total_meeting_hours, 1),
                    "meeting_target": self.config.max_meeting_hours_per_week,
                    "total_deep_work_hours": round(total_deep_work, 1),
                    "buffer_status": "adequate" if buffer_adequate else "inadequate",
                    "alert_count": len(alerts)
                },
                "daily_breakdown": daily_stats,
                "meeting_patterns": meeting_patterns,
                "buffer_analysis": {
                    "adequate": buffer_adequate,
                    "gap_count": len(buffer_gaps),
                    "gaps": [
                        {"start": g[0].isoformat(), "end": g[1].isoformat()}
                        for g in buffer_gaps
                    ]
                },
                "alerts": [self._alert_to_dict(a) for a in alerts],
                "recommendations": self._generate_weekly_recommendations(
                    daily_stats, total_meeting_hours, total_deep_work,
                    buffer_adequate, alerts
                )
            }
        )

    def generate_report(
        self,
        time_window: Tuple[datetime, datetime],
        report_type: str = "full"
    ) -> AgentResponse:
        """Generate a comprehensive calendar report."""
        if report_type == "status":
            return self.get_status(time_window)
        elif report_type == "analysis":
            return self.analyze_week(time_window)
        else:
            # Full report combines status and analysis
            status = self.get_status(time_window)
            analysis = self.analyze_week(time_window)

            if not status.success or not analysis.success:
                return self.create_response(
                    {"error": "Failed to generate full report"},
                    success=False,
                    errors=status.errors + analysis.errors
                )

            return self.create_response(
                {
                    "report_type": "FULL_CALENDAR_REPORT",
                    "status": status.data,
                    "analysis": analysis.data,
                    "generated_at": datetime.now().isoformat()
                },
                warnings=status.warnings + analysis.warnings
            )

    # -------------------------------------------------------------------------
    # ALERT GENERATION
    # -------------------------------------------------------------------------

    def generate_alert(
        self,
        alert_type: AlertType,
        period: Tuple[datetime, datetime],
        issue: str,
        impact: str,
        recommended_action: str,
        affected_events: Optional[List[CalendarEvent]] = None
    ) -> CalendarAlert:
        """
        Generate a calendar alert.

        Alert Format:
        CALENDAR_ALERT
        - Alert Type: overload | buffer_breach | personal_encroachment | recovery_skip
        - Period, Issue, Impact
        - Recommended Action
        """
        severity = "high"
        if alert_type in [AlertType.BUFFER_BREACH, AlertType.RECOVERY_SKIP]:
            severity = "medium"

        return CalendarAlert(
            alert_type=alert_type,
            period=period,
            issue=issue,
            impact=impact,
            recommended_action=recommended_action,
            severity=severity,
            affected_events=affected_events or []
        )

    def _generate_alerts(
        self,
        start: datetime,
        end: datetime,
        events: List[CalendarEvent],
        protected: List[TimeBlock]
    ) -> List[CalendarAlert]:
        """Generate all relevant alerts for a time period."""
        alerts = []

        # Check for meeting overload
        meeting_hours = self._calculate_meeting_hours(events)
        days = max(1, (end - start).days)
        daily_avg = meeting_hours / days

        if daily_avg > self.config.max_meeting_hours_per_day:
            alerts.append(self.generate_alert(
                AlertType.OVERLOAD,
                (start, end),
                f"Meeting load ({daily_avg:.1f}h/day) exceeds target ({self.config.max_meeting_hours_per_day}h/day)",
                "Reduced focus time, increased stress, less deep work",
                "Decline or reschedule low-priority meetings"
            ))

        # Check for buffer breaches
        buffer_adequate, buffer_gaps = self._check_buffer_status(events)
        if not buffer_adequate:
            for gap in buffer_gaps[:3]:  # Limit to first 3
                alerts.append(self.generate_alert(
                    AlertType.BUFFER_BREACH,
                    gap,
                    "Insufficient buffer between meetings",
                    "No transition time, potential for late arrivals, mental fatigue",
                    "Add buffer time or reschedule one meeting"
                ))

        # Check for personal time encroachment
        for event in events:
            event_time = event.start_time.time()
            if (event_time >= self.config.personal_time_start or
                event_time < self.config.personal_time_end):
                if event.block_type == BlockType.MEETINGS:
                    alerts.append(self.generate_alert(
                        AlertType.PERSONAL_ENCROACHMENT,
                        (event.start_time, event.end_time),
                        f"Meeting '{event.title}' scheduled during personal time",
                        "Work-life boundary violation, family time affected",
                        "Reschedule to work hours or decline",
                        [event]
                    ))

        # Check for recovery skips
        recovery_hours = self._calculate_block_type_hours(events, protected, BlockType.RECOVERY)
        min_recovery_hours = (self.config.min_recovery_minutes_per_day / 60) * days

        if recovery_hours < min_recovery_hours:
            alerts.append(self.generate_alert(
                AlertType.RECOVERY_SKIP,
                (start, end),
                f"Insufficient recovery time ({recovery_hours:.1f}h vs {min_recovery_hours:.1f}h needed)",
                "Burnout risk, decreased productivity, health impact",
                "Block recovery time as sacred"
            ))

        return alerts

    # -------------------------------------------------------------------------
    # VALIDATION AND PROTECTION
    # -------------------------------------------------------------------------

    def _validate_scheduling(
        self,
        event: CalendarEvent,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate if an event can be scheduled.

        Checks:
        - No double booking
        - Respects protected blocks
        - Maintains buffer requirements
        - No back-to-back high intensity
        - Within work hours (unless forced)
        - Doesn't exceed meeting limits
        """
        violations = []
        warnings = []
        conflicts_resolved = []

        # Check for overlapping events
        for existing in self._events.values():
            if event.event_id != existing.event_id and event.overlaps(existing):
                if constraints.get("force", False):
                    conflicts_resolved.append(existing.title)
                else:
                    violations.append(f"Overlaps with: {existing.title}")

        # Check protected blocks
        for block in self._protected_blocks.values():
            if block.is_protected:
                if (event.start_time < block.end_time and
                    event.end_time > block.start_time):
                    if block.protection_level == "sacred":
                        violations.append(
                            f"Conflicts with sacred {block.block_type.value} time"
                        )
                    elif block.protection_level == "standard":
                        if not constraints.get("override_protection", False):
                            violations.append(
                                f"Conflicts with protected {block.block_type.value} time"
                            )
                        else:
                            warnings.append(
                                f"Overriding {block.block_type.value} protection"
                            )

        # Check work hours
        event_time = event.start_time.time()
        if not constraints.get("allow_outside_hours", False):
            if (event_time < self.config.work_day_start or
                event_time >= self.config.work_day_end):
                if event.block_type == BlockType.MEETINGS:
                    violations.append("Meeting scheduled outside work hours")

        # Check back-to-back high intensity
        if event.intensity == IntensityLevel.HIGH:
            for existing in self._events.values():
                if existing.intensity == IntensityLevel.HIGH:
                    gap_before = (event.start_time - existing.end_time).total_seconds() / 60
                    gap_after = (existing.start_time - event.end_time).total_seconds() / 60

                    if 0 < gap_before < self.config.buffer_after_high_intensity:
                        warnings.append(
                            f"Only {int(gap_before)}min after high-intensity {existing.title}"
                        )
                    if 0 < gap_after < self.config.buffer_after_high_intensity:
                        warnings.append(
                            f"Only {int(gap_after)}min before high-intensity {existing.title}"
                        )

        # Check daily meeting limits
        day_start = event.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        day_events = self._get_events_in_range(day_start, day_end)

        if event.block_type == BlockType.MEETINGS:
            day_meeting_count = sum(
                1 for e in day_events if e.block_type == BlockType.MEETINGS
            )
            if day_meeting_count >= self.config.max_meetings_per_day:
                warnings.append(
                    f"Day already has {day_meeting_count} meetings (limit: {self.config.max_meetings_per_day})"
                )

            day_meeting_hours = sum(
                e.duration_minutes / 60 for e in day_events
                if e.block_type == BlockType.MEETINGS
            ) + (event.duration_minutes / 60)

            if day_meeting_hours > self.config.max_meeting_hours_per_day:
                violations.append(
                    f"Would exceed daily meeting limit ({day_meeting_hours:.1f}h > {self.config.max_meeting_hours_per_day}h)"
                )

        can_schedule = len(violations) == 0
        reason = violations[0] if violations else None

        return {
            "can_schedule": can_schedule,
            "reason": reason,
            "violations": violations,
            "warnings": warnings,
            "conflicts_resolved": conflicts_resolved
        }

    def _calculate_required_buffers(
        self,
        event: CalendarEvent
    ) -> Tuple[int, int]:
        """Calculate required buffer times before and after an event."""
        buffer_before = 0
        buffer_after = 0

        # Default minimum buffer for meetings
        if event.block_type == BlockType.MEETINGS:
            buffer_before = self.config.min_buffer_between_meetings
            buffer_after = self.config.min_buffer_between_meetings

        # Extra buffer for clinical
        if event.block_type == BlockType.CLINICAL:
            buffer_after = max(buffer_after, self.config.buffer_after_clinical)

        # Deep work needs buffer before for mental preparation
        if event.block_type == BlockType.DEEP_WORK:
            buffer_before = max(buffer_before, self.config.buffer_before_deep_work)
            buffer_after = max(buffer_after, self.config.buffer_after_deep_work)

        # High intensity needs recovery buffer
        if event.intensity == IntensityLevel.HIGH:
            buffer_after = max(buffer_after, self.config.buffer_after_high_intensity)

        # Prep time if required
        if event.requires_prep:
            buffer_before = max(buffer_before, event.prep_time_minutes)

        return buffer_before, buffer_after

    def _check_buffer_status(
        self,
        events: List[CalendarEvent]
    ) -> Tuple[bool, List[Tuple[datetime, datetime]]]:
        """Check if buffer requirements are met between events."""
        if len(events) < 2:
            return True, []

        gaps = []
        sorted_events = sorted(events, key=lambda e: e.start_time)

        for i in range(len(sorted_events) - 1):
            curr = sorted_events[i]
            next_evt = sorted_events[i + 1]

            gap_minutes = (next_evt.start_time - curr.end_time).total_seconds() / 60
            required_buffer = max(
                curr.buffer_after,
                next_evt.buffer_before,
                self.config.min_buffer_between_meetings
            )

            if gap_minutes < required_buffer:
                gaps.append((curr.end_time, next_evt.start_time))

        return len(gaps) == 0, gaps

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _get_events_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[CalendarEvent]:
        """Get all events within a time range."""
        events = []

        # Local events
        for event in self._events.values():
            if event.start_time < end and event.end_time > start:
                events.append(event)

        # External provider events
        if self.google_provider.connected:
            events.extend(self.google_provider.get_events(start, end))

        if self.cal_com_provider.connected:
            events.extend(self.cal_com_provider.get_bookings(start, end))

        return sorted(events, key=lambda e: e.start_time)

    def _get_protected_blocks_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[TimeBlock]:
        """Get all protected blocks within a time range."""
        blocks = []
        for block in self._protected_blocks.values():
            if block.start_time < end and block.end_time > start:
                blocks.append(block)
        return sorted(blocks, key=lambda b: b.start_time)

    def _calculate_meeting_hours(
        self,
        events: List[CalendarEvent]
    ) -> float:
        """Calculate total meeting hours."""
        return sum(
            e.duration_minutes / 60
            for e in events
            if e.block_type == BlockType.MEETINGS
        )

    def _calculate_block_type_hours(
        self,
        events: List[CalendarEvent],
        blocks: List[TimeBlock],
        block_type: BlockType
    ) -> float:
        """Calculate total hours for a specific block type."""
        hours = sum(
            e.duration_minutes / 60
            for e in events
            if e.block_type == block_type
        )
        hours += sum(
            b.duration_hours
            for b in blocks
            if b.block_type == block_type
        )
        return hours

    def _calculate_day_intensity(
        self,
        events: List[CalendarEvent]
    ) -> str:
        """Calculate overall intensity score for a day."""
        if not events:
            return "light"

        high_count = sum(1 for e in events if e.intensity == IntensityLevel.HIGH)
        meeting_count = sum(1 for e in events if e.block_type == BlockType.MEETINGS)

        if high_count >= 3 or meeting_count >= 5:
            return "intense"
        elif high_count >= 2 or meeting_count >= 3:
            return "moderate"
        else:
            return "light"

    def _calculate_schedule_health(
        self,
        meeting_hours: float,
        deep_work_hours: float,
        buffer_adequate: bool,
        alert_count: int
    ) -> int:
        """Calculate overall schedule health score (0-100)."""
        score = 100

        # Penalize for exceeding meeting target
        if meeting_hours > self.config.max_meeting_hours_per_week:
            excess = meeting_hours - self.config.max_meeting_hours_per_week
            score -= min(20, excess * 4)

        # Penalize for insufficient deep work
        min_deep_work = self.config.min_deep_work_hours_per_day * 5
        if deep_work_hours < min_deep_work:
            deficit = min_deep_work - deep_work_hours
            score -= min(20, deficit * 4)

        # Penalize for buffer issues
        if not buffer_adequate:
            score -= 15

        # Penalize for alerts
        score -= min(25, alert_count * 5)

        return max(0, score)

    def _find_alternative_times(
        self,
        event: CalendarEvent,
        constraints: Dict[str, Any]
    ) -> List[Tuple[datetime, datetime]]:
        """Find alternative time slots for an event."""
        alternatives = []
        duration = event.end_time - event.start_time

        # Search window: next 5 working days
        search_start = datetime.now().replace(
            hour=self.config.work_day_start.hour,
            minute=self.config.work_day_start.minute,
            second=0, microsecond=0
        )
        if search_start < datetime.now():
            search_start += timedelta(days=1)

        search_end = search_start + timedelta(days=5)

        current = search_start
        while current < search_end and len(alternatives) < 10:
            # Skip non-work hours
            if current.time() < self.config.work_day_start:
                current = current.replace(
                    hour=self.config.work_day_start.hour,
                    minute=self.config.work_day_start.minute
                )
            elif current.time() >= self.config.work_day_end:
                current = (current + timedelta(days=1)).replace(
                    hour=self.config.work_day_start.hour,
                    minute=self.config.work_day_start.minute
                )
                continue

            # Check if slot is available
            test_event = CalendarEvent(
                event_id="test",
                title=event.title,
                start_time=current,
                end_time=current + duration,
                block_type=event.block_type,
                intensity=event.intensity
            )

            validation = self._validate_scheduling(test_event, {"force": False})
            if validation["can_schedule"]:
                alternatives.append((current, current + duration))

            current += timedelta(minutes=30)

        return alternatives

    def _analyze_meeting_patterns(
        self,
        events: List[CalendarEvent]
    ) -> Dict[str, Any]:
        """Analyze meeting patterns."""
        meetings = [e for e in events if e.block_type == BlockType.MEETINGS]

        if not meetings:
            return {"pattern": "no_meetings", "details": {}}

        # Time of day distribution
        morning = sum(1 for m in meetings if m.start_time.hour < 12)
        afternoon = sum(1 for m in meetings if 12 <= m.start_time.hour < 17)
        evening = sum(1 for m in meetings if m.start_time.hour >= 17)

        # Average duration
        avg_duration = sum(m.duration_minutes for m in meetings) / len(meetings)

        # Day distribution
        day_counts = {}
        for m in meetings:
            day = m.start_time.strftime("%A")
            day_counts[day] = day_counts.get(day, 0) + 1

        return {
            "total_count": len(meetings),
            "time_distribution": {
                "morning": morning,
                "afternoon": afternoon,
                "evening": evening
            },
            "average_duration_minutes": round(avg_duration, 0),
            "day_distribution": day_counts,
            "busiest_day": max(day_counts.keys(), key=lambda d: day_counts[d]) if day_counts else None
        }

    def _generate_schedule_view(
        self,
        start: datetime,
        end: datetime,
        events: List[CalendarEvent],
        protected: List[TimeBlock]
    ) -> List[Dict[str, Any]]:
        """Generate a time-blocked view of the schedule."""
        items = []

        for event in events:
            items.append({
                "type": "event",
                "category": event.block_type.value,
                "start": event.start_time.isoformat(),
                "end": event.end_time.isoformat(),
                "title": event.title,
                "priority": event.priority.value,
                "intensity": event.intensity.value
            })

        for block in protected:
            items.append({
                "type": "protected_block",
                "category": block.block_type.value,
                "start": block.start_time.isoformat(),
                "end": block.end_time.isoformat(),
                "title": block.title,
                "protection_level": block.protection_level
            })

        return sorted(items, key=lambda x: x["start"])

    def _generate_recommendations(
        self,
        events: List[CalendarEvent],
        protected: List[TimeBlock],
        conflicts: List[CalendarConflict],
        alerts: List[CalendarAlert],
        meeting_hours: float,
        deep_work_hours: float,
        recovery_hours: float
    ) -> List[str]:
        """Generate specific recommendations based on calendar analysis."""
        recommendations = []

        # Address conflicts
        if conflicts:
            recommendations.append(
                f"Resolve {len(conflicts)} calendar conflict(s) - see conflicts list"
            )

        # Meeting load
        if meeting_hours > self.config.max_meeting_hours_per_day:
            recommendations.append(
                f"Reduce meeting load by {meeting_hours - self.config.max_meeting_hours_per_day:.1f}h"
            )

        # Deep work
        if deep_work_hours < self.config.min_deep_work_hours_per_day:
            recommendations.append(
                f"Add {self.config.min_deep_work_hours_per_day - deep_work_hours:.1f}h of protected deep work time"
            )

        # Recovery
        min_recovery = self.config.min_recovery_minutes_per_day / 60
        if recovery_hours < min_recovery:
            recommendations.append(
                f"Schedule {min_recovery - recovery_hours:.1f}h of recovery time"
            )

        # Alert-specific recommendations
        for alert in alerts:
            if alert.alert_type == AlertType.PERSONAL_ENCROACHMENT:
                recommendations.append(alert.recommended_action)

        return recommendations

    def _generate_weekly_recommendations(
        self,
        daily_stats: Dict[str, Dict],
        meeting_hours: float,
        deep_work_hours: float,
        buffer_adequate: bool,
        alerts: List[CalendarAlert]
    ) -> List[str]:
        """Generate weekly-specific recommendations."""
        recommendations = []

        # Check for unbalanced days
        busiest_day = max(daily_stats.items(), key=lambda x: x[1]["meeting_hours"])
        lightest_day = min(daily_stats.items(), key=lambda x: x[1]["meeting_hours"])

        if busiest_day[1]["meeting_hours"] > lightest_day[1]["meeting_hours"] * 2:
            recommendations.append(
                f"Balance meeting load: {busiest_day[0]} is overloaded, consider moving to {lightest_day[0]}"
            )

        # Weekly meeting target
        if meeting_hours > self.config.max_meeting_hours_per_week:
            recommendations.append(
                f"Weekly meetings ({meeting_hours:.1f}h) exceed target ({self.config.max_meeting_hours_per_week}h) - audit for necessary attendance"
            )

        # Deep work target
        min_weekly_deep_work = self.config.min_deep_work_hours_per_day * 5
        if deep_work_hours < min_weekly_deep_work:
            recommendations.append(
                f"Protect more deep work time - currently {deep_work_hours:.1f}h vs {min_weekly_deep_work:.1f}h target"
            )

        # Buffer recommendations
        if not buffer_adequate:
            recommendations.append(
                "Add buffer time between meetings to allow for transitions"
            )

        # Intensity distribution
        intense_days = sum(1 for d in daily_stats.values() if d["intensity_score"] == "intense")
        if intense_days >= 3:
            recommendations.append(
                f"High intensity on {intense_days} days - spread demanding activities more evenly"
            )

        return recommendations

    def _sync_to_providers(
        self,
        event: CalendarEvent
    ) -> Dict[str, str]:
        """Sync event to external calendar providers."""
        external_ids = {}

        if self.google_provider.connected:
            google_id = self.google_provider.create_event(event)
            if google_id:
                external_ids["google"] = google_id

        return external_ids

    def _parse_datetime(
        self,
        value: Any
    ) -> Optional[datetime]:
        """Parse various datetime formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M")
            except ValueError:
                pass
            try:
                return datetime.strptime(value, "%Y-%m-%dT%H:%M")
            except ValueError:
                pass
        return None

    def _event_to_dict(self, event: CalendarEvent) -> Dict[str, Any]:
        """Convert CalendarEvent to dictionary."""
        return {
            "event_id": event.event_id,
            "title": event.title,
            "start": event.start_time.isoformat(),
            "end": event.end_time.isoformat(),
            "block_type": event.block_type.value,
            "priority": event.priority.value,
            "intensity": event.intensity.value,
            "duration_minutes": event.duration_minutes,
            "location": event.location,
            "attendees": event.attendees,
            "can_reschedule": event.can_reschedule
        }

    def _block_to_dict(self, block: TimeBlock) -> Dict[str, Any]:
        """Convert TimeBlock to dictionary."""
        return {
            "block_type": block.block_type.value,
            "start": block.start_time.isoformat(),
            "end": block.end_time.isoformat(),
            "title": block.title,
            "duration_hours": block.duration_hours,
            "is_protected": block.is_protected,
            "protection_level": block.protection_level,
            "recurrence": block.recurrence
        }

    def _conflict_to_dict(self, conflict: CalendarConflict) -> Dict[str, Any]:
        """Convert CalendarConflict to dictionary."""
        return {
            "type": conflict.conflict_type,
            "events": [e.title for e in conflict.events],
            "period": {
                "start": conflict.time_period[0].isoformat(),
                "end": conflict.time_period[1].isoformat()
            },
            "severity": conflict.severity,
            "description": conflict.description,
            "suggestions": conflict.resolution_suggestions
        }

    def _alert_to_dict(self, alert: CalendarAlert) -> Dict[str, Any]:
        """Convert CalendarAlert to dictionary."""
        return {
            "alert_type": alert.alert_type.value,
            "period": {
                "start": alert.period[0].isoformat(),
                "end": alert.period[1].isoformat()
            },
            "issue": alert.issue,
            "impact": alert.impact,
            "recommended_action": alert.recommended_action,
            "severity": alert.severity
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_scheduling_agent(
    config: Optional[SchedulingConfig] = None,
    google_mcp_client: Optional[Any] = None,
    cal_com_api_key: Optional[str] = None
) -> SchedulingAgent:
    """
    Factory function to create a configured SchedulingAgent.

    Args:
        config: Optional SchedulingConfig for custom settings
        google_mcp_client: MCP client for Google Calendar integration
        cal_com_api_key: API key for Cal.com integration

    Returns:
        Configured SchedulingAgent instance
    """
    google_provider = GoogleCalendarProvider(google_mcp_client) if google_mcp_client else None
    cal_com_provider = CalComProvider(cal_com_api_key) if cal_com_api_key else None

    return SchedulingAgent(
        config=config,
        google_provider=google_provider,
        cal_com_provider=cal_com_provider
    )
