"""
Daily Brief - Morning and Evening Automation System for Alfred

Generates structured daily briefs that aggregate data from all relevant agents
to provide Alfred with actionable morning and evening summaries.

Morning Brief (5-7 minutes):
- Clinical Reputation Status: GREEN/YELLOW/RED
- Today's Constraint: one sentence
- Top 3 Priorities: chosen by Alfred, not user
- One Blocked Thing: what Alfred explicitly disallowed
- Calendar overview for the day

Evening Shutdown (5 minutes):
- What shipped / what didn't (facts only)
- One note captured for tomorrow
- Role switch reminder (home mode)

Notification delivery: email, console
"""

from __future__ import annotations

import asyncio
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Protocol

from . import OperationsAgent, AgentResponse, AlfredState


# =============================================================================
# ENUMS
# =============================================================================

class BriefType(Enum):
    """Type of daily brief."""
    MORNING = "morning"
    EVENING = "evening"


class ReputationStatus(Enum):
    """Clinical reputation status."""
    GREEN = "GREEN"    # All clear, no reputation concerns
    YELLOW = "YELLOW"  # Elevated monitoring, some concerns detected
    RED = "RED"        # Active threat, requires immediate attention


class PrioritySource(Enum):
    """Source of priority determination."""
    ALFRED = "alfred"          # Alfred-chosen based on analysis
    SHIPPING = "shipping"      # Shipping Governor input
    CALENDAR = "calendar"      # Calendar-driven deadline
    REPUTATION = "reputation"  # Reputation-related urgency


class NotificationChannel(Enum):
    """Notification delivery channels."""
    CONSOLE = "console"
    EMAIL = "email"
    SLACK = "slack"
    PUSHOVER = "pushover"


class ConstraintType(Enum):
    """Types of constraints that affect the day."""
    TIME = "time"              # Limited time available
    ENERGY = "energy"          # Energy/recovery constraints
    CONTEXT = "context"        # Context switching limits
    CLINICAL = "clinical"      # Clinical obligations
    SHIPPING = "shipping"      # Shipping deadlines
    REPUTATION = "reputation"  # Reputation management needs


# =============================================================================
# DATA CLASSES - Core Brief Components
# =============================================================================

@dataclass
class PriorityItem:
    """A single priority item for the day."""
    rank: int
    description: str
    source: PrioritySource
    project: Optional[str] = None
    deadline: Optional[datetime] = None
    estimated_minutes: Optional[int] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "description": self.description,
            "source": self.source.value,
            "project": self.project,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "estimated_minutes": self.estimated_minutes,
            "rationale": self.rationale
        }


@dataclass
class BlockedItem:
    """Something Alfred has explicitly disallowed for today."""
    item: str
    reason: str
    blocked_by: str  # Which agent or rule blocked it
    alternative: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item,
            "reason": self.reason,
            "blocked_by": self.blocked_by,
            "alternative": self.alternative
        }


@dataclass
class CalendarBlock:
    """A time block from the calendar."""
    start: datetime
    end: datetime
    title: str
    block_type: str  # CLINICAL, DEEP_WORK, MEETINGS, etc.
    is_protected: bool = False
    notes: str = ""

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "title": self.title,
            "block_type": self.block_type,
            "duration_minutes": self.duration_minutes,
            "is_protected": self.is_protected,
            "notes": self.notes
        }


@dataclass
class DailyConstraint:
    """The single most important constraint for today."""
    constraint_type: ConstraintType
    description: str  # One sentence
    impact: str  # How it affects the day
    source: str  # Which agent identified it

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.constraint_type.value,
            "description": self.description,
            "impact": self.impact,
            "source": self.source
        }


@dataclass
class ShippedItem:
    """Something that was shipped today."""
    name: str
    project: str
    shipped_at: datetime
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "project": self.project,
            "shipped_at": self.shipped_at.isoformat(),
            "description": self.description
        }


@dataclass
class UnshippedItem:
    """Something that did not ship today (facts only, no judgment)."""
    name: str
    project: str
    reason: str  # Factual reason, not excuse
    will_carry_over: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "project": self.project,
            "reason": self.reason,
            "will_carry_over": self.will_carry_over
        }


@dataclass
class TomorrowNote:
    """Single note captured for tomorrow."""
    content: str
    category: str  # context, reminder, task, insight
    captured_at: datetime = field(default_factory=datetime.now)
    priority: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "category": self.category,
            "captured_at": self.captured_at.isoformat(),
            "priority": self.priority
        }


# =============================================================================
# DATA CLASSES - Brief Structures
# =============================================================================

@dataclass
class MorningBrief:
    """
    Morning brief structure (5-7 minutes).

    Components:
    - Clinical Reputation Status: GREEN/YELLOW/RED
    - Today's Constraint: one sentence
    - Top 3 Priorities: chosen by Alfred, not user
    - One Blocked Thing: what Alfred explicitly disallowed
    - Calendar overview for the day
    """
    # Metadata
    generated_at: datetime
    brief_date: date

    # Core components
    reputation_status: ReputationStatus
    reputation_summary: str
    todays_constraint: DailyConstraint
    priorities: List[PriorityItem]  # Top 3
    blocked_item: Optional[BlockedItem]
    calendar_blocks: List[CalendarBlock]

    # Summary metrics
    total_meeting_hours: float = 0.0
    deep_work_hours_available: float = 0.0
    protected_blocks_count: int = 0

    # Additional context
    active_projects_count: int = 0
    days_since_last_ship: int = 0
    pending_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brief_type": "MORNING_BRIEF",
            "generated_at": self.generated_at.isoformat(),
            "brief_date": self.brief_date.isoformat(),
            "reputation": {
                "status": self.reputation_status.value,
                "summary": self.reputation_summary
            },
            "todays_constraint": self.todays_constraint.to_dict(),
            "priorities": [p.to_dict() for p in self.priorities[:3]],
            "blocked_item": self.blocked_item.to_dict() if self.blocked_item else None,
            "calendar": {
                "blocks": [b.to_dict() for b in self.calendar_blocks],
                "total_meeting_hours": round(self.total_meeting_hours, 1),
                "deep_work_hours_available": round(self.deep_work_hours_available, 1),
                "protected_blocks_count": self.protected_blocks_count
            },
            "context": {
                "active_projects": self.active_projects_count,
                "days_since_last_ship": self.days_since_last_ship,
                "pending_alerts": self.pending_alerts
            }
        }

    def to_formatted_output(self) -> str:
        """Generate formatted output for console/notification."""
        lines = [
            "=" * 60,
            f"MORNING BRIEF - {self.brief_date.strftime('%A, %B %d, %Y')}",
            "=" * 60,
            "",
            f"REPUTATION STATUS: {self.reputation_status.value}",
            f"  {self.reputation_summary}",
            "",
            "TODAY'S CONSTRAINT:",
            f"  {self.todays_constraint.description}",
            "",
            "TOP 3 PRIORITIES (Alfred-chosen):",
        ]

        for p in self.priorities[:3]:
            time_note = f" [{p.estimated_minutes}min]" if p.estimated_minutes else ""
            lines.append(f"  {p.rank}. {p.description}{time_note}")
            if p.rationale:
                lines.append(f"     -> {p.rationale}")

        if self.blocked_item:
            lines.extend([
                "",
                "BLOCKED TODAY:",
                f"  {self.blocked_item.item}",
                f"  Reason: {self.blocked_item.reason}"
            ])
            if self.blocked_item.alternative:
                lines.append(f"  Alternative: {self.blocked_item.alternative}")

        lines.extend([
            "",
            "CALENDAR OVERVIEW:",
            f"  Meetings: {self.total_meeting_hours:.1f}h | Deep Work: {self.deep_work_hours_available:.1f}h | Protected: {self.protected_blocks_count} blocks",
            ""
        ])

        for block in self.calendar_blocks[:8]:  # Show first 8 blocks
            time_str = f"{block.start.strftime('%H:%M')}-{block.end.strftime('%H:%M')}"
            protected_marker = " [PROTECTED]" if block.is_protected else ""
            lines.append(f"  {time_str} | {block.title}{protected_marker}")

        if len(self.calendar_blocks) > 8:
            lines.append(f"  ... and {len(self.calendar_blocks) - 8} more blocks")

        if self.pending_alerts:
            lines.extend([
                "",
                "PENDING ALERTS:",
            ])
            for alert in self.pending_alerts[:3]:
                lines.append(f"  - {alert}")

        lines.extend([
            "",
            "=" * 60,
            f"Generated at {self.generated_at.strftime('%H:%M')} | Active Projects: {self.active_projects_count}",
            "=" * 60
        ])

        return "\n".join(lines)


@dataclass
class EveningShutdown:
    """
    Evening shutdown structure (5 minutes).

    Components:
    - What shipped / what didn't (facts only)
    - One note captured for tomorrow
    - Role switch reminder (home mode)
    """
    # Metadata
    generated_at: datetime
    shutdown_date: date

    # Core components
    shipped_items: List[ShippedItem]
    unshipped_items: List[UnshippedItem]
    tomorrow_note: Optional[TomorrowNote]

    # Role switch
    role_switch_reminder: str
    home_mode_active: bool = False

    # Summary metrics
    total_shipped: int = 0
    total_unshipped: int = 0
    shipping_health: str = "HEALTHY"  # HEALTHY, WARNING, CRITICAL

    # Tomorrow preview
    tomorrow_first_commitment: Optional[str] = None
    tomorrow_first_commitment_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brief_type": "EVENING_SHUTDOWN",
            "generated_at": self.generated_at.isoformat(),
            "shutdown_date": self.shutdown_date.isoformat(),
            "shipped": {
                "items": [s.to_dict() for s in self.shipped_items],
                "count": self.total_shipped
            },
            "unshipped": {
                "items": [u.to_dict() for u in self.unshipped_items],
                "count": self.total_unshipped
            },
            "tomorrow_note": self.tomorrow_note.to_dict() if self.tomorrow_note else None,
            "role_switch": {
                "reminder": self.role_switch_reminder,
                "home_mode_active": self.home_mode_active
            },
            "shipping_health": self.shipping_health,
            "tomorrow_preview": {
                "first_commitment": self.tomorrow_first_commitment,
                "first_commitment_time": self.tomorrow_first_commitment_time.isoformat() if self.tomorrow_first_commitment_time else None
            }
        }

    def to_formatted_output(self) -> str:
        """Generate formatted output for console/notification."""
        lines = [
            "=" * 60,
            f"EVENING SHUTDOWN - {self.shutdown_date.strftime('%A, %B %d, %Y')}",
            "=" * 60,
            "",
        ]

        # Shipped items
        if self.shipped_items:
            lines.append(f"SHIPPED TODAY ({self.total_shipped}):")
            for item in self.shipped_items:
                lines.append(f"  + {item.name} ({item.project})")
        else:
            lines.append("SHIPPED TODAY: None")

        lines.append("")

        # Unshipped items (facts only)
        if self.unshipped_items:
            lines.append(f"DID NOT SHIP ({self.total_unshipped}):")
            for item in self.unshipped_items:
                carry_note = " -> carries over" if item.will_carry_over else " -> dropped"
                lines.append(f"  - {item.name} ({item.project})")
                lines.append(f"    Reason: {item.reason}{carry_note}")

        lines.append("")

        # Tomorrow note
        if self.tomorrow_note:
            lines.extend([
                "NOTE FOR TOMORROW:",
                f"  [{self.tomorrow_note.category.upper()}] {self.tomorrow_note.content}",
                ""
            ])

        # Tomorrow preview
        if self.tomorrow_first_commitment:
            time_str = self.tomorrow_first_commitment_time.strftime('%H:%M') if self.tomorrow_first_commitment_time else "TBD"
            lines.extend([
                "TOMORROW STARTS WITH:",
                f"  {time_str} - {self.tomorrow_first_commitment}",
                ""
            ])

        # Role switch reminder
        lines.extend([
            "-" * 60,
            "ROLE SWITCH REMINDER:",
            f"  {self.role_switch_reminder}",
            ""
        ])

        if self.home_mode_active:
            lines.append("  [HOME MODE NOW ACTIVE]")

        lines.extend([
            "",
            "=" * 60,
            f"Shutdown at {self.generated_at.strftime('%H:%M')} | Shipping Health: {self.shipping_health}",
            "=" * 60
        ])

        return "\n".join(lines)


@dataclass
class DailyBrief:
    """
    Container for both morning and evening briefs.
    Represents a complete daily brief cycle.
    """
    brief_date: date
    morning_brief: Optional[MorningBrief] = None
    evening_shutdown: Optional[EveningShutdown] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brief_date": self.brief_date.isoformat(),
            "morning_brief": self.morning_brief.to_dict() if self.morning_brief else None,
            "evening_shutdown": self.evening_shutdown.to_dict() if self.evening_shutdown else None
        }


# =============================================================================
# NOTIFICATION INTERFACES
# =============================================================================

class NotificationDelivery(ABC):
    """Abstract base for notification delivery channels."""

    @abstractmethod
    async def deliver(self, brief: MorningBrief | EveningShutdown,
                     recipient: str) -> bool:
        """Deliver brief to recipient."""
        pass

    @abstractmethod
    def get_channel_name(self) -> str:
        """Return channel name."""
        pass


class ConsoleNotification(NotificationDelivery):
    """Console/stdout notification delivery."""

    async def deliver(self, brief: MorningBrief | EveningShutdown,
                     recipient: str = "") -> bool:
        """Print brief to console."""
        print(brief.to_formatted_output())
        return True

    def get_channel_name(self) -> str:
        return "console"


class EmailNotification(NotificationDelivery):
    """Email notification delivery."""

    def __init__(self, smtp_host: str = "localhost",
                 smtp_port: int = 587,
                 smtp_user: Optional[str] = None,
                 smtp_password: Optional[str] = None,
                 from_address: str = "alfred@localhost",
                 use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_address = from_address
        self.use_tls = use_tls

    async def deliver(self, brief: MorningBrief | EveningShutdown,
                     recipient: str) -> bool:
        """Send brief via email."""
        try:
            # Determine brief type for subject
            if isinstance(brief, MorningBrief):
                subject = f"Alfred Morning Brief - {brief.brief_date.strftime('%B %d, %Y')}"
            else:
                subject = f"Alfred Evening Shutdown - {brief.shutdown_date.strftime('%B %d, %Y')}"

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_address
            msg["To"] = recipient

            # Plain text version
            text_content = brief.to_formatted_output()
            text_part = MIMEText(text_content, "plain")
            msg.attach(text_part)

            # HTML version (enhanced formatting)
            html_content = self._to_html(brief)
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            return True

        except Exception as e:
            # Log error but don't fail
            print(f"Email delivery failed: {e}")
            return False

    def _to_html(self, brief: MorningBrief | EveningShutdown) -> str:
        """Convert brief to HTML format."""
        # Simple HTML conversion
        text = brief.to_formatted_output()
        html_lines = ["<html><body><pre style='font-family: monospace;'>"]
        html_lines.append(text.replace("\n", "<br>"))
        html_lines.append("</pre></body></html>")
        return "\n".join(html_lines)

    def get_channel_name(self) -> str:
        return "email"


# =============================================================================
# AGENT DATA COLLECTORS
# =============================================================================

class AgentDataCollector:
    """
    Collects data from various Alfred sub-agents for brief generation.

    Interfaces with:
    - Reputation Sentinel: reputation status
    - Scheduling Agent: calendar data
    - Shipping Governor: shipping status
    - World Radar: constraint signals
    - Intake Agent: pending items
    """

    def __init__(self):
        # Agent references (will be injected or fetched)
        self._reputation_sentinel = None
        self._scheduling_agent = None
        self._shipping_governor = None
        self._world_radar = None
        self._intake_agent = None

    def set_agents(self, agents: Dict[str, Any]) -> None:
        """Set agent references for data collection."""
        self._reputation_sentinel = agents.get("reputation_sentinel")
        self._scheduling_agent = agents.get("scheduling_agent")
        self._shipping_governor = agents.get("shipping_governor")
        self._world_radar = agents.get("world_radar")
        self._intake_agent = agents.get("intake_agent")

    async def get_reputation_status(self) -> Tuple[ReputationStatus, str]:
        """Get current reputation status from Reputation Sentinel."""
        if self._reputation_sentinel is None:
            return ReputationStatus.GREEN, "No reputation data available (agent not connected)"

        try:
            # Call reputation sentinel for current status
            if hasattr(self._reputation_sentinel, 'get_status'):
                status_data = self._reputation_sentinel.get_status()
                state = status_data.get("alfred_state", "GREEN")

                if state == "RED":
                    return ReputationStatus.RED, "Active reputation concern detected"
                elif state == "YELLOW":
                    return ReputationStatus.YELLOW, "Elevated monitoring in effect"
                else:
                    return ReputationStatus.GREEN, "All clear - no reputation concerns"

            return ReputationStatus.GREEN, "Reputation sentinel connected, status nominal"

        except Exception as e:
            return ReputationStatus.GREEN, f"Unable to fetch reputation status: {str(e)}"

    async def get_calendar_data(self, target_date: date) -> Tuple[List[CalendarBlock], Dict[str, float]]:
        """Get calendar data for the target date."""
        blocks = []
        metrics = {
            "meeting_hours": 0.0,
            "deep_work_hours": 0.0,
            "protected_count": 0
        }

        if self._scheduling_agent is None:
            return blocks, metrics

        try:
            # Create time window for the day
            day_start = datetime.combine(target_date, time(0, 0))
            day_end = datetime.combine(target_date, time(23, 59))

            if hasattr(self._scheduling_agent, 'get_status'):
                # Get status response which includes calendar data
                response = self._scheduling_agent.get_status((day_start, day_end))

                if response.success and "schedule" in response.data:
                    for item in response.data["schedule"]:
                        block = CalendarBlock(
                            start=datetime.fromisoformat(item["start"]),
                            end=datetime.fromisoformat(item["end"]),
                            title=item.get("title", "Untitled"),
                            block_type=item.get("category", item.get("block_type", "MEETINGS")),
                            is_protected=item.get("is_protected", item.get("protection_level") == "sacred")
                        )
                        blocks.append(block)

                        # Calculate metrics
                        hours = block.duration_minutes / 60.0
                        if block.block_type == "MEETINGS":
                            metrics["meeting_hours"] += hours
                        elif block.block_type == "DEEP_WORK":
                            metrics["deep_work_hours"] += hours
                        if block.is_protected:
                            metrics["protected_count"] += 1

            return blocks, metrics

        except Exception as e:
            return blocks, metrics

    async def get_shipping_data(self) -> Dict[str, Any]:
        """Get shipping status from Shipping Governor."""
        data = {
            "health": "HEALTHY",
            "days_since_ship": 0,
            "active_projects": 0,
            "shipped_today": [],
            "unshipped_today": [],
            "blocked_items": []
        }

        if self._shipping_governor is None:
            return data

        try:
            if hasattr(self._shipping_governor, 'generate_report'):
                response = self._shipping_governor.generate_report()

                if response.success:
                    report = response.data
                    data["health"] = report.get("overall_shipping_health", "HEALTHY")
                    data["active_projects"] = report.get("summary", {}).get("active_projects", 0)
                    data["days_since_ship"] = int(report.get("summary", {}).get("average_days_without_output", 0))

                    # Extract recent outputs
                    for output in report.get("recent_outputs", []):
                        if output.get("shipped_date") == date.today().isoformat():
                            data["shipped_today"].append(output)

                    # Check for blocked items from assessments
                    for assessment in report.get("project_assessments", []):
                        if assessment.get("recommended_action") == "KILL":
                            data["blocked_items"].append({
                                "item": f"Continue {assessment.get('project')}",
                                "reason": assessment.get("rationale", "Zombie work"),
                                "blocked_by": "Shipping Governor"
                            })

            return data

        except Exception as e:
            return data

    async def get_constraint_signal(self) -> Optional[DailyConstraint]:
        """Get the primary constraint for today from World Radar and other sources."""
        # Default constraint based on general assessment
        constraint = DailyConstraint(
            constraint_type=ConstraintType.TIME,
            description="Protect focus time - context switches are expensive",
            impact="Limit meetings and reactive work",
            source="Default"
        )

        if self._world_radar is not None:
            try:
                if hasattr(self._world_radar, '_active_signals'):
                    signals = self._world_radar._active_signals
                    if signals:
                        # Use highest priority signal as constraint
                        top_signal = signals[0]
                        constraint = DailyConstraint(
                            constraint_type=ConstraintType.REPUTATION if top_signal.domain.value == "reputation" else ConstraintType.CLINICAL,
                            description=top_signal.event[:100] if len(top_signal.event) > 100 else top_signal.event,
                            impact=top_signal.constraint_impact.projected_state,
                            source="World Radar"
                        )
            except Exception:
                pass

        return constraint

    async def get_priorities(self, calendar_blocks: List[CalendarBlock],
                            shipping_data: Dict[str, Any]) -> List[PriorityItem]:
        """Generate top 3 priorities based on aggregated data."""
        priorities = []

        # Priority 1: Shipping-based (if there are stalled projects)
        if shipping_data.get("health") in ["WARNING", "CRITICAL"]:
            priorities.append(PriorityItem(
                rank=1,
                description="Ship something today - output cadence is stalling",
                source=PrioritySource.SHIPPING,
                rationale=f"Shipping health is {shipping_data.get('health')}, {shipping_data.get('days_since_ship', 0)} days since last output"
            ))

        # Priority from calendar: first protected deep work or clinical block
        for block in calendar_blocks:
            if block.is_protected and block.block_type in ["DEEP_WORK", "CLINICAL"]:
                priorities.append(PriorityItem(
                    rank=len(priorities) + 1,
                    description=f"Honor protected block: {block.title}",
                    source=PrioritySource.CALENDAR,
                    deadline=block.start,
                    estimated_minutes=block.duration_minutes,
                    rationale="Protected time is sacred - do not interrupt"
                ))
                break

        # Default priorities if we don't have enough
        default_priorities = [
            PriorityItem(
                rank=len(priorities) + 1,
                description="Complete one meaningful task before checking messages",
                source=PrioritySource.ALFRED,
                rationale="Deep work before reactive work"
            ),
            PriorityItem(
                rank=len(priorities) + 2,
                description="Review and process any urgent intake items",
                source=PrioritySource.ALFRED,
                estimated_minutes=15,
                rationale="Stay responsive without being reactive"
            ),
            PriorityItem(
                rank=len(priorities) + 3,
                description="Make progress on the current primary project",
                source=PrioritySource.ALFRED,
                rationale="Consistent progress over time"
            )
        ]

        # Fill to get 3 priorities
        while len(priorities) < 3 and default_priorities:
            priorities.append(default_priorities.pop(0))

        # Re-rank
        for i, p in enumerate(priorities[:3]):
            p.rank = i + 1

        return priorities[:3]


# =============================================================================
# DAILY BRIEF AGENT
# =============================================================================

class DailyBriefAgent(OperationsAgent):
    """
    Daily Brief Agent - Morning and Evening Automation System

    Generates Alfred's morning and evening briefs by aggregating data from
    all relevant agents and producing structured, actionable summaries.

    DOES:
    - Generate morning briefs with reputation status, constraints, priorities
    - Generate evening shutdowns with shipping status and tomorrow notes
    - Aggregate data from Reputation Sentinel, Scheduling, Shipping Governor
    - Deliver notifications via console, email, or other channels
    - Respect role boundaries (home mode in evenings)
    - Provide facts, not opinions or judgments

    DOES NOT:
    - Make decisions for the user
    - Override user priorities with its own
    - Generate briefs during inappropriate hours
    - Share sensitive data in notifications
    - Skip the role switch reminder
    """

    # Default schedule times
    DEFAULT_MORNING_TIME = time(6, 30)
    DEFAULT_EVENING_TIME = time(18, 0)

    def __init__(self,
                 morning_time: Optional[time] = None,
                 evening_time: Optional[time] = None,
                 notification_channels: Optional[List[NotificationDelivery]] = None):
        """
        Initialize Daily Brief Agent.

        Args:
            morning_time: Time for morning brief generation
            evening_time: Time for evening shutdown generation
            notification_channels: List of notification delivery channels
        """
        super().__init__("DailyBriefAgent")

        self.morning_time = morning_time or self.DEFAULT_MORNING_TIME
        self.evening_time = evening_time or self.DEFAULT_EVENING_TIME

        # Notification channels
        self._notification_channels: List[NotificationDelivery] = notification_channels or [
            ConsoleNotification()
        ]

        # Data collector
        self._collector = AgentDataCollector()

        # Brief history
        self._briefs: Dict[date, DailyBrief] = {}

        # Configuration
        self._role_switch_messages = [
            "Work day complete. Transition to home mode - family first, work can wait.",
            "Time to switch roles. Close the laptop, be present with family.",
            "Evening shutdown complete. Tomorrow's challenges will wait for tomorrow.",
            "Role switch: Professional mode off, personal mode on. Be present.",
            "Work is paused, not stopped. Your family deserves your full attention now."
        ]

        # Current note for tomorrow (captured during the day)
        self._tomorrow_note: Optional[TomorrowNote] = None

    def set_agents(self, agents: Dict[str, Any]) -> None:
        """Configure agent references for data collection."""
        self._collector.set_agents(agents)

    def add_notification_channel(self, channel: NotificationDelivery) -> None:
        """Add a notification delivery channel."""
        self._notification_channels.append(channel)

    def configure_email(self, smtp_host: str, smtp_port: int,
                       smtp_user: str, smtp_password: str,
                       from_address: str) -> None:
        """Configure email notification channel."""
        email_channel = EmailNotification(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            from_address=from_address
        )
        self.add_notification_channel(email_channel)

    def capture_tomorrow_note(self, content: str,
                             category: str = "reminder",
                             priority: bool = False) -> None:
        """Capture a note for tomorrow's brief."""
        self._tomorrow_note = TomorrowNote(
            content=content,
            category=category,
            priority=priority
        )

    # =========================================================================
    # MORNING BRIEF GENERATION
    # =========================================================================

    async def generate_morning_brief(self,
                                     target_date: Optional[date] = None) -> MorningBrief:
        """
        Generate the morning brief for the specified date.

        Args:
            target_date: Date for the brief (defaults to today)

        Returns:
            MorningBrief with all components populated
        """
        target_date = target_date or date.today()

        # Collect data from all agents
        reputation_status, reputation_summary = await self._collector.get_reputation_status()
        calendar_blocks, calendar_metrics = await self._collector.get_calendar_data(target_date)
        shipping_data = await self._collector.get_shipping_data()
        constraint = await self._collector.get_constraint_signal()
        priorities = await self._collector.get_priorities(calendar_blocks, shipping_data)

        # Determine blocked item (if any)
        blocked_item = None
        if shipping_data.get("blocked_items"):
            blocked = shipping_data["blocked_items"][0]
            blocked_item = BlockedItem(
                item=blocked["item"],
                reason=blocked["reason"],
                blocked_by=blocked["blocked_by"]
            )

        # Generate pending alerts
        pending_alerts = []
        if reputation_status == ReputationStatus.YELLOW:
            pending_alerts.append("Reputation monitoring elevated")
        if reputation_status == ReputationStatus.RED:
            pending_alerts.append("CRITICAL: Active reputation threat")
        if shipping_data.get("health") == "CRITICAL":
            pending_alerts.append("Shipping health critical - output stalled")

        # Create morning brief
        brief = MorningBrief(
            generated_at=datetime.now(),
            brief_date=target_date,
            reputation_status=reputation_status,
            reputation_summary=reputation_summary,
            todays_constraint=constraint,
            priorities=priorities,
            blocked_item=blocked_item,
            calendar_blocks=sorted(calendar_blocks, key=lambda b: b.start),
            total_meeting_hours=calendar_metrics.get("meeting_hours", 0.0),
            deep_work_hours_available=calendar_metrics.get("deep_work_hours", 0.0),
            protected_blocks_count=calendar_metrics.get("protected_count", 0),
            active_projects_count=shipping_data.get("active_projects", 0),
            days_since_last_ship=shipping_data.get("days_since_ship", 0),
            pending_alerts=pending_alerts
        )

        # Store in history
        if target_date not in self._briefs:
            self._briefs[target_date] = DailyBrief(brief_date=target_date)
        self._briefs[target_date].morning_brief = brief

        return brief

    # =========================================================================
    # EVENING SHUTDOWN GENERATION
    # =========================================================================

    async def generate_evening_shutdown(self,
                                        target_date: Optional[date] = None) -> EveningShutdown:
        """
        Generate the evening shutdown for the specified date.

        Args:
            target_date: Date for the shutdown (defaults to today)

        Returns:
            EveningShutdown with all components populated
        """
        target_date = target_date or date.today()

        # Get shipping data
        shipping_data = await self._collector.get_shipping_data()

        # Convert shipped items
        shipped_items = []
        for item in shipping_data.get("shipped_today", []):
            shipped_items.append(ShippedItem(
                name=item.get("name", "Unknown"),
                project=item.get("project", "Unknown"),
                shipped_at=datetime.fromisoformat(item.get("shipped_date", datetime.now().isoformat())),
                description=item.get("description", "")
            ))

        # Unshipped items (need to track these separately - placeholder)
        unshipped_items = []

        # Tomorrow preview
        tomorrow = target_date + timedelta(days=1)
        calendar_blocks, _ = await self._collector.get_calendar_data(tomorrow)

        tomorrow_first = None
        tomorrow_first_time = None
        if calendar_blocks:
            first_block = sorted(calendar_blocks, key=lambda b: b.start)[0]
            tomorrow_first = first_block.title
            tomorrow_first_time = first_block.start

        # Role switch message
        import random
        role_switch = random.choice(self._role_switch_messages)

        # Create shutdown
        shutdown = EveningShutdown(
            generated_at=datetime.now(),
            shutdown_date=target_date,
            shipped_items=shipped_items,
            unshipped_items=unshipped_items,
            tomorrow_note=self._tomorrow_note,
            role_switch_reminder=role_switch,
            home_mode_active=True,
            total_shipped=len(shipped_items),
            total_unshipped=len(unshipped_items),
            shipping_health=shipping_data.get("health", "HEALTHY"),
            tomorrow_first_commitment=tomorrow_first,
            tomorrow_first_commitment_time=tomorrow_first_time
        )

        # Store in history
        if target_date not in self._briefs:
            self._briefs[target_date] = DailyBrief(brief_date=target_date)
        self._briefs[target_date].evening_shutdown = shutdown

        # Clear tomorrow note after use
        self._tomorrow_note = None

        return shutdown

    # =========================================================================
    # NOTIFICATION DELIVERY
    # =========================================================================

    async def deliver_brief(self, brief: MorningBrief | EveningShutdown,
                           recipient: Optional[str] = None,
                           channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Deliver brief via configured notification channels.

        Args:
            brief: The brief to deliver
            recipient: Recipient for email notifications
            channels: Specific channels to use (None = all)

        Returns:
            Dictionary of channel -> success status
        """
        results = {}

        for channel in self._notification_channels:
            channel_name = channel.get_channel_name()

            # Skip if specific channels requested and this isn't one
            if channels and channel_name not in channels:
                continue

            try:
                success = await channel.deliver(brief, recipient or "")
                results[channel_name] = success
            except Exception as e:
                results[channel_name] = False

        return results

    # =========================================================================
    # AGENT RESPONSE INTERFACE
    # =========================================================================

    async def process_brief_request(self,
                                   brief_type: BriefType,
                                   target_date: Optional[date] = None,
                                   deliver: bool = True,
                                   recipient: Optional[str] = None) -> AgentResponse:
        """
        Process a brief generation request.

        Args:
            brief_type: Type of brief to generate
            target_date: Target date for the brief
            deliver: Whether to deliver via notification channels
            recipient: Email recipient (if applicable)

        Returns:
            AgentResponse with brief data
        """
        target_date = target_date or date.today()

        try:
            if brief_type == BriefType.MORNING:
                brief = await self.generate_morning_brief(target_date)
            else:
                brief = await self.generate_evening_shutdown(target_date)

            # Deliver if requested
            delivery_results = {}
            if deliver:
                delivery_results = await self.deliver_brief(brief, recipient)

            return self.create_response(
                data={
                    "brief": brief.to_dict(),
                    "formatted_output": brief.to_formatted_output(),
                    "delivery_results": delivery_results
                },
                success=True
            )

        except Exception as e:
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[str(e)]
            )

    def process_request(self, request_data: Dict[str, Any]) -> AgentResponse:
        """
        Synchronous interface for processing brief requests.

        Args:
            request_data: Dictionary with request parameters
                - type: "morning" or "evening"
                - date: ISO date string (optional)
                - deliver: bool (optional)
                - recipient: email address (optional)

        Returns:
            AgentResponse with brief data
        """
        brief_type_str = request_data.get("type", "morning")
        brief_type = BriefType.MORNING if brief_type_str == "morning" else BriefType.EVENING

        target_date = None
        if "date" in request_data:
            target_date = date.fromisoformat(request_data["date"])

        deliver = request_data.get("deliver", True)
        recipient = request_data.get("recipient")

        # Run async method
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.process_brief_request(brief_type, target_date, deliver, recipient)
            )
            return result
        finally:
            loop.close()

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def generate_morning_brief_sync(self,
                                    target_date: Optional[date] = None) -> MorningBrief:
        """Synchronous wrapper for morning brief generation."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.generate_morning_brief(target_date)
            )
        finally:
            loop.close()

    def generate_evening_shutdown_sync(self,
                                       target_date: Optional[date] = None) -> EveningShutdown:
        """Synchronous wrapper for evening shutdown generation."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.generate_evening_shutdown(target_date)
            )
        finally:
            loop.close()

    def get_brief_history(self,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None) -> List[DailyBrief]:
        """Get brief history for a date range."""
        if start_date is None:
            start_date = date.today() - timedelta(days=7)
        if end_date is None:
            end_date = date.today()

        return [
            brief for brief_date, brief in self._briefs.items()
            if start_date <= brief_date <= end_date
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent": self.name,
            "alfred_state": self.alfred_state.value,
            "morning_time": self.morning_time.isoformat(),
            "evening_time": self.evening_time.isoformat(),
            "notification_channels": [c.get_channel_name() for c in self._notification_channels],
            "briefs_stored": len(self._briefs),
            "has_tomorrow_note": self._tomorrow_note is not None
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_daily_brief_agent(
    morning_time: Optional[time] = None,
    evening_time: Optional[time] = None,
    include_email: bool = False,
    email_config: Optional[Dict[str, str]] = None
) -> DailyBriefAgent:
    """
    Factory function to create a configured DailyBriefAgent.

    Args:
        morning_time: Time for morning brief
        evening_time: Time for evening shutdown
        include_email: Whether to include email notifications
        email_config: Email configuration dict with smtp_host, smtp_port, etc.

    Returns:
        Configured DailyBriefAgent instance
    """
    channels = [ConsoleNotification()]

    if include_email and email_config:
        channels.append(EmailNotification(
            smtp_host=email_config.get("smtp_host", "localhost"),
            smtp_port=email_config.get("smtp_port", 587),
            smtp_user=email_config.get("smtp_user"),
            smtp_password=email_config.get("smtp_password"),
            from_address=email_config.get("from_address", "alfred@localhost")
        ))

    return DailyBriefAgent(
        morning_time=morning_time,
        evening_time=evening_time,
        notification_channels=channels
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main Agent
    "DailyBriefAgent",

    # Brief Types
    "DailyBrief",
    "MorningBrief",
    "EveningShutdown",

    # Component Data Classes
    "PriorityItem",
    "BlockedItem",
    "CalendarBlock",
    "DailyConstraint",
    "ShippedItem",
    "UnshippedItem",
    "TomorrowNote",

    # Enums
    "BriefType",
    "ReputationStatus",
    "PrioritySource",
    "NotificationChannel",
    "ConstraintType",

    # Notification
    "NotificationDelivery",
    "ConsoleNotification",
    "EmailNotification",

    # Utilities
    "AgentDataCollector",
    "create_daily_brief_agent",
]
