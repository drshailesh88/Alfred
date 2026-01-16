"""
Daily Brief System for ALFRED

Generates Alfred's morning and evening briefs as specified in PLAN.md.

Morning Brief (5-7 minutes):
1. Clinical Reputation Status: GREEN/YELLOW/RED
2. Today's Constraint: one sentence
3. Top 3 Priorities: chosen by Alfred based on context
4. One Blocked Thing: what Alfred explicitly disallowed

Evening Shutdown (5 minutes):
1. What shipped / what didn't (facts only)
2. One note captured for tomorrow
3. Role switch reminder (home mode)

Storage: data/alfred/briefs/ with date stamps
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os


class ReputationState(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class PriorityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Priority:
    """A single priority item for the daily brief."""
    title: str
    linked_output: str
    priority_level: str = PriorityLevel.HIGH.value
    source: str = ""  # e.g., "shipping_governor", "calendar", "intake"
    deadline: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "linked_output": self.linked_output,
            "priority_level": self.priority_level,
            "source": self.source,
            "deadline": self.deadline
        }

    def format_display(self) -> str:
        deadline_str = f" (due: {self.deadline})" if self.deadline else ""
        return f"{self.title} -> {self.linked_output}{deadline_str}"


@dataclass
class BlockedItem:
    """An item that Alfred has explicitly blocked."""
    item: str
    reason: str
    blocked_at: str
    blocker_source: str  # Which system blocked it

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item,
            "reason": self.reason,
            "blocked_at": self.blocked_at,
            "blocker_source": self.blocker_source
        }


@dataclass
class ShippedItem:
    """An item that was completed/shipped today."""
    description: str
    shipped_at: str
    project: Optional[str] = None
    output_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "shipped_at": self.shipped_at,
            "project": self.project,
            "output_type": self.output_type
        }


@dataclass
class CalendarSnapshot:
    """Today's calendar summary."""
    total_events: int
    protected_blocks: int
    meetings: int
    meeting_minutes: int
    focus_blocks: int
    focus_minutes: int
    conflicts: int
    schedule: List[Dict[str, str]] = field(default_factory=list)

    def format_display(self) -> str:
        lines = []
        if not self.schedule:
            lines.append("  No events scheduled")
        else:
            for event in self.schedule[:8]:  # Limit to 8 events for brevity
                time_str = event.get("time", "")
                title = event.get("title", "")
                block_type = event.get("block_type", "")
                protected = " [PROTECTED]" if event.get("protected") else ""
                lines.append(f"  {time_str} | {title} ({block_type}){protected}")
            if len(self.schedule) > 8:
                lines.append(f"  ... and {len(self.schedule) - 8} more events")

        return "\n".join(lines)


@dataclass
class MorningBrief:
    """Complete morning brief structure."""
    date: str
    reputation_status: str
    reputation_reason: Optional[str]
    todays_constraint: str
    top_priorities: List[Priority]
    blocked_today: List[BlockedItem]
    calendar_snapshot: CalendarSnapshot
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "reputation_status": self.reputation_status,
            "reputation_reason": self.reputation_reason,
            "todays_constraint": self.todays_constraint,
            "top_priorities": [p.to_dict() for p in self.top_priorities],
            "blocked_today": [b.to_dict() for b in self.blocked_today],
            "calendar_snapshot": {
                "total_events": self.calendar_snapshot.total_events,
                "protected_blocks": self.calendar_snapshot.protected_blocks,
                "meetings": self.calendar_snapshot.meetings,
                "focus_blocks": self.calendar_snapshot.focus_blocks,
                "schedule": self.calendar_snapshot.schedule
            },
            "generated_at": self.generated_at
        }

    def format_display(self) -> str:
        """Generate the formatted morning brief output."""
        lines = [
            f"ALFRED MORNING BRIEF - {self.date}",
            "=" * 40,
            "",
            f"REPUTATION STATUS: {self.reputation_status}"
        ]

        if self.reputation_status != "GREEN" and self.reputation_reason:
            lines.append(f"  {self.reputation_reason}")

        lines.extend([
            "",
            "TODAY'S CONSTRAINT:",
            f"  {self.todays_constraint}",
            "",
            "TOP 3 PRIORITIES:"
        ])

        for i, priority in enumerate(self.top_priorities[:3], 1):
            lines.append(f"  {i}. {priority.format_display()}")

        if not self.top_priorities:
            lines.append("  (No priorities identified)")

        lines.extend([
            "",
            "BLOCKED TODAY:"
        ])

        for blocked in self.blocked_today:
            lines.append(f"  - {blocked.item}: {blocked.reason}")

        if not self.blocked_today:
            lines.append("  - (Nothing explicitly blocked)")

        lines.extend([
            "",
            "CALENDAR SNAPSHOT:",
            self.calendar_snapshot.format_display()
        ])

        return "\n".join(lines)


@dataclass
class EveningBrief:
    """Complete evening shutdown brief structure."""
    date: str
    shipped_items: List[ShippedItem]
    not_shipped_items: List[str]
    note_for_tomorrow: str
    role_switch_message: str
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "shipped_items": [s.to_dict() for s in self.shipped_items],
            "not_shipped_items": self.not_shipped_items,
            "note_for_tomorrow": self.note_for_tomorrow,
            "generated_at": self.generated_at
        }

    def format_display(self) -> str:
        """Generate the formatted evening brief output."""
        lines = [
            f"ALFRED EVENING SHUTDOWN - {self.date}",
            "=" * 40,
            "",
            "SHIPPED TODAY:"
        ]

        for item in self.shipped_items:
            project_str = f" [{item.project}]" if item.project else ""
            lines.append(f"  - {item.description}{project_str}")

        if not self.shipped_items:
            lines.append("  - (Nothing shipped today)")

        lines.extend([
            "",
            "DID NOT SHIP:"
        ])

        for item in self.not_shipped_items:
            lines.append(f"  - {item}")

        if not self.not_shipped_items:
            lines.append("  - (Nothing pending)")

        lines.extend([
            "",
            "CAPTURED FOR TOMORROW:",
            f"  {self.note_for_tomorrow}" if self.note_for_tomorrow else "  (No note captured)",
            "",
            "ROLE SWITCH:",
            f"  {self.role_switch_message}"
        ])

        return "\n".join(lines)


class DailyBrief:
    """
    Daily Brief System for ALFRED

    Aggregates data from multiple tools and generates structured
    morning and evening briefs. Integrates with:
    - ReputationSentinel: Get reputation state
    - ShippingGovernor: Get shipping status and blocked items
    - SchedulingAgent: Get calendar snapshot
    - ContentManager: Get pending content tasks

    Storage: data/alfred/briefs/ with date stamps
    """

    DEFAULT_ROLE_SWITCH_MESSAGE = (
        "It's time to be home. Clinical and builder roles are now inactive."
    )

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the Daily Brief system.

        Args:
            storage_dir: Optional custom storage directory.
                        Defaults to data/alfred/briefs/
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            # Default: relative to agent-zero1 root
            base_path = Path(__file__).parent.parent.parent.parent
            self.storage_dir = base_path / "data" / "alfred" / "briefs"

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state for today's blocked items and notes
        self._today_blocked: List[BlockedItem] = []
        self._today_notes: List[str] = []
        self._tomorrow_note: str = ""

        # External tool references (set via set_tools method)
        self._reputation_sentinel = None
        self._shipping_governor = None
        self._scheduling_agent = None
        self._content_manager = None

    def set_tools(
        self,
        reputation_sentinel=None,
        shipping_governor=None,
        scheduling_agent=None,
        content_manager=None
    ):
        """
        Set references to external tools for data aggregation.

        Args:
            reputation_sentinel: ReputationSentinel tool instance
            shipping_governor: ShippingGovernor tool instance
            scheduling_agent: SchedulingAgent tool instance
            content_manager: ContentManager tool instance
        """
        self._reputation_sentinel = reputation_sentinel
        self._shipping_governor = shipping_governor
        self._scheduling_agent = scheduling_agent
        self._content_manager = content_manager

    # =========================================================================
    # Data Retrieval Methods
    # =========================================================================

    def get_reputation_status(self) -> Tuple[str, Optional[str]]:
        """
        Get current reputation status from ReputationSentinel.

        Returns:
            Tuple of (state: GREEN/YELLOW/RED, reason: Optional[str])
        """
        if self._reputation_sentinel is None:
            # Fallback: read from reputation data file if available
            return self._get_reputation_from_storage()

        # If we have a tool instance, query it directly
        try:
            # Access internal state from ReputationSentinel
            state = getattr(self._reputation_sentinel, 'current_state', 'green')
            risk_score = getattr(self._reputation_sentinel, 'current_risk_score', 0)

            state_upper = state.upper()
            reason = None

            if state_upper == "YELLOW":
                reason = f"Elevated monitoring active (risk score: {risk_score})"
            elif state_upper == "RED":
                reason = f"Active threat detected (risk score: {risk_score})"

            return state_upper, reason
        except Exception:
            return self._get_reputation_from_storage()

    def _get_reputation_from_storage(self) -> Tuple[str, Optional[str]]:
        """Fallback: read reputation state from storage files."""
        try:
            reputation_file = (
                Path(__file__).parent.parent.parent.parent /
                "data" / "alfred" / "reputation" / "current_state.json"
            )
            if reputation_file.exists():
                with open(reputation_file, "r") as f:
                    data = json.load(f)
                    state = data.get("current_state", "green").upper()
                    reason = data.get("reason")
                    return state, reason
        except Exception:
            pass
        return "GREEN", None

    def get_todays_priorities(self) -> List[Priority]:
        """
        Analyze pending work and return top priorities.

        Pulls from:
        - ShippingGovernor: Stalled projects needing attention
        - SchedulingAgent: Calendar deadlines
        - ContentManager: Pending content items

        Returns:
            List of Priority objects, sorted by importance
        """
        priorities = []

        # Get shipping priorities
        shipping_priorities = self._get_shipping_priorities()
        priorities.extend(shipping_priorities)

        # Get calendar-based priorities
        calendar_priorities = self._get_calendar_priorities()
        priorities.extend(calendar_priorities)

        # Get content priorities
        content_priorities = self._get_content_priorities()
        priorities.extend(content_priorities)

        # Sort by priority level and take top 3
        priority_order = {
            PriorityLevel.CRITICAL.value: 0,
            PriorityLevel.HIGH.value: 1,
            PriorityLevel.MEDIUM.value: 2,
            PriorityLevel.LOW.value: 3
        }

        priorities.sort(key=lambda p: priority_order.get(p.priority_level, 4))
        return priorities[:3]

    def _get_shipping_priorities(self) -> List[Priority]:
        """Get priorities from ShippingGovernor."""
        priorities = []

        if self._shipping_governor is None:
            # Fallback: read from shipping data file
            try:
                shipping_file = (
                    Path(__file__).parent.parent.parent.parent /
                    "data" / "alfred" / "shipping_governor.json"
                )
                if shipping_file.exists():
                    with open(shipping_file, "r") as f:
                        data = json.load(f)
                        for name, project in data.get("projects", {}).items():
                            if project.get("status") in ["SHIPPED", "KILLED", "PAUSED"]:
                                continue

                            # Calculate days without output
                            last_milestone = project.get("last_milestone_date")
                            if last_milestone:
                                try:
                                    last_date = datetime.fromisoformat(last_milestone)
                                    days_without = (datetime.now() - last_date).days
                                except ValueError:
                                    days_without = 999
                            else:
                                days_without = 999

                            # High priority if stalled
                            if days_without >= 7:
                                level = PriorityLevel.CRITICAL.value if days_without >= 14 else PriorityLevel.HIGH.value
                                priorities.append(Priority(
                                    title=f"Ship {name}",
                                    linked_output=project.get("target_output", "deliverable"),
                                    priority_level=level,
                                    source="shipping_governor"
                                ))
            except Exception:
                pass
        else:
            # Query tool directly
            try:
                data = self._shipping_governor._load_data()
                for name, project in data.get("projects", {}).items():
                    if project.get("status") in ["SHIPPED", "KILLED", "PAUSED"]:
                        continue
                    last_milestone = project.get("last_milestone_date")
                    if last_milestone:
                        try:
                            last_date = datetime.fromisoformat(last_milestone)
                            days_without = (datetime.now() - last_date).days
                        except ValueError:
                            days_without = 999
                    else:
                        days_without = 999

                    if days_without >= 7:
                        level = PriorityLevel.CRITICAL.value if days_without >= 14 else PriorityLevel.HIGH.value
                        priorities.append(Priority(
                            title=f"Ship {name}",
                            linked_output=project.get("target_output", "deliverable"),
                            priority_level=level,
                            source="shipping_governor"
                        ))
            except Exception:
                pass

        return priorities

    def _get_calendar_priorities(self) -> List[Priority]:
        """Get priorities from SchedulingAgent."""
        priorities = []

        if self._scheduling_agent:
            try:
                state = self._scheduling_agent.get_state()
                today = datetime.now().date()

                for event in state.events:
                    if event.start_time.date() == today:
                        if event.priority in ["critical", "high"] and event.block_type == "DEEP_WORK":
                            priorities.append(Priority(
                                title=event.title,
                                linked_output=event.description or "scheduled output",
                                priority_level=event.priority,
                                source="scheduling_agent",
                                deadline=event.start_time.strftime("%H:%M")
                            ))
            except Exception:
                pass

        return priorities

    def _get_content_priorities(self) -> List[Priority]:
        """Get priorities from ContentManager."""
        # ContentManager integration - would need specific implementation
        # For now, return empty list
        return []

    def get_blocked_items(self) -> List[BlockedItem]:
        """
        Get items that Alfred has explicitly blocked today.

        Returns:
            List of BlockedItem objects
        """
        # Return items blocked today (in-memory + stored)
        blocked = self._today_blocked.copy()

        # Also load from today's brief file if exists
        today_str = date.today().strftime("%Y-%m-%d")
        today_file = self.storage_dir / f"morning_{today_str}.json"

        if today_file.exists():
            try:
                with open(today_file, "r") as f:
                    data = json.load(f)
                    for item in data.get("blocked_today", []):
                        blocked_item = BlockedItem(
                            item=item.get("item", ""),
                            reason=item.get("reason", ""),
                            blocked_at=item.get("blocked_at", ""),
                            blocker_source=item.get("blocker_source", "")
                        )
                        # Avoid duplicates
                        if not any(b.item == blocked_item.item for b in blocked):
                            blocked.append(blocked_item)
            except Exception:
                pass

        return blocked

    def get_shipped_items(self) -> List[ShippedItem]:
        """
        Get items completed today.

        Returns:
            List of ShippedItem objects
        """
        shipped = []
        today = date.today()

        # Get from ShippingGovernor
        try:
            shipping_file = (
                Path(__file__).parent.parent.parent.parent /
                "data" / "alfred" / "shipping_governor.json"
            )
            if shipping_file.exists():
                with open(shipping_file, "r") as f:
                    data = json.load(f)
                    for output in data.get("outputs", []):
                        shipped_date = output.get("shipped_date", "")
                        if shipped_date:
                            try:
                                output_date = datetime.fromisoformat(shipped_date).date()
                                if output_date == today:
                                    shipped.append(ShippedItem(
                                        description=output.get("description", ""),
                                        shipped_at=shipped_date,
                                        project=output.get("project"),
                                        output_type=output.get("output_type")
                                    ))
                            except ValueError:
                                pass
        except Exception:
            pass

        return shipped

    def get_calendar_snapshot(self) -> CalendarSnapshot:
        """
        Get today's calendar summary.

        Returns:
            CalendarSnapshot object
        """
        if self._scheduling_agent:
            try:
                state = self._scheduling_agent.get_state()
                today = datetime.now()
                events = state.get_events_for_date(today)
                events.sort(key=lambda e: e.start_time)

                protected = [e for e in events if e.is_protected]
                meetings = [e for e in events if e.block_type == "MEETINGS"]
                focus = [e for e in events if e.block_type == "DEEP_WORK"]

                # Find conflicts
                conflicts = []
                for i, e1 in enumerate(events):
                    for e2 in events[i + 1:]:
                        if e1.start_time < e2.end_time and e1.end_time > e2.start_time:
                            conflicts.append((e1, e2))

                schedule = []
                for e in events:
                    schedule.append({
                        "time": f"{e.start_time.strftime('%H:%M')}-{e.end_time.strftime('%H:%M')}",
                        "title": e.title,
                        "block_type": e.block_type,
                        "protected": e.is_protected
                    })

                return CalendarSnapshot(
                    total_events=len(events),
                    protected_blocks=len(protected),
                    meetings=len(meetings),
                    meeting_minutes=sum(e.duration_minutes() for e in meetings),
                    focus_blocks=len(focus),
                    focus_minutes=sum(e.duration_minutes() for e in focus),
                    conflicts=len(conflicts),
                    schedule=schedule
                )
            except Exception:
                pass

        # Return empty snapshot if no tool available
        return CalendarSnapshot(
            total_events=0,
            protected_blocks=0,
            meetings=0,
            meeting_minutes=0,
            focus_blocks=0,
            focus_minutes=0,
            conflicts=0,
            schedule=[]
        )

    def get_todays_constraint(self) -> str:
        """
        Get today's primary constraint.

        Pulls from ShippingGovernor and SchedulingAgent to determine
        what is protected or limited today.

        Returns:
            One sentence describing today's constraint
        """
        constraints = []

        # Check calendar for protected blocks
        if self._scheduling_agent:
            try:
                state = self._scheduling_agent.get_state()
                today = datetime.now()
                events = state.get_events_for_date(today)
                protected = [e for e in events if e.is_protected]

                if protected:
                    times = [e.title for e in protected[:2]]
                    constraints.append(f"Protected time: {', '.join(times)}")
            except Exception:
                pass

        # Check shipping health
        try:
            shipping_file = (
                Path(__file__).parent.parent.parent.parent /
                "data" / "alfred" / "shipping_governor.json"
            )
            if shipping_file.exists():
                with open(shipping_file, "r") as f:
                    data = json.load(f)
                    last_output = data.get("last_output_date")
                    if last_output:
                        try:
                            days_since = (datetime.now() - datetime.fromisoformat(last_output)).days
                            if days_since >= 7:
                                constraints.append(f"Shipping stalled ({days_since} days) - no new building until output")
                        except ValueError:
                            pass
        except Exception:
            pass

        # Check reputation state
        state, _ = self.get_reputation_status()
        if state == "YELLOW":
            constraints.append("Elevated reputation monitoring - reactive content restricted")
        elif state == "RED":
            constraints.append("Reputation crisis - all public-facing content paused")

        if constraints:
            return constraints[0]  # Return most important constraint

        return "No specific constraints - normal operations"

    # =========================================================================
    # Brief Generation Methods
    # =========================================================================

    def generate_morning_brief(self) -> MorningBrief:
        """
        Generate the complete morning brief.

        Returns:
            MorningBrief object with all components
        """
        today_str = date.today().strftime("%Y-%m-%d")

        # Get reputation status
        reputation_status, reputation_reason = self.get_reputation_status()

        # Get today's constraint
        todays_constraint = self.get_todays_constraint()

        # Get priorities
        priorities = self.get_todays_priorities()

        # Get blocked items
        blocked_items = self.get_blocked_items()

        # Get calendar snapshot
        calendar = self.get_calendar_snapshot()

        brief = MorningBrief(
            date=today_str,
            reputation_status=reputation_status,
            reputation_reason=reputation_reason,
            todays_constraint=todays_constraint,
            top_priorities=priorities,
            blocked_today=blocked_items,
            calendar_snapshot=calendar,
            generated_at=datetime.now().isoformat()
        )

        # Save brief to storage
        self._save_morning_brief(brief)

        return brief

    def generate_evening_brief(
        self,
        note_for_tomorrow: Optional[str] = None,
        custom_role_switch_message: Optional[str] = None
    ) -> EveningBrief:
        """
        Generate the complete evening shutdown brief.

        Args:
            note_for_tomorrow: Optional note to capture for tomorrow
            custom_role_switch_message: Optional custom role switch message

        Returns:
            EveningBrief object with all components
        """
        today_str = date.today().strftime("%Y-%m-%d")

        # Get shipped items
        shipped = self.get_shipped_items()

        # Get what didn't ship (active projects that stalled)
        not_shipped = self._get_unshipped_items()

        # Note for tomorrow
        if note_for_tomorrow:
            self._tomorrow_note = note_for_tomorrow
        note = self._tomorrow_note or self._generate_auto_note()

        # Role switch message
        role_switch = custom_role_switch_message or self.DEFAULT_ROLE_SWITCH_MESSAGE

        brief = EveningBrief(
            date=today_str,
            shipped_items=shipped,
            not_shipped_items=not_shipped,
            note_for_tomorrow=note,
            role_switch_message=role_switch,
            generated_at=datetime.now().isoformat()
        )

        # Save brief to storage
        self._save_evening_brief(brief)

        return brief

    def _get_unshipped_items(self) -> List[str]:
        """Get items that were expected but didn't ship today."""
        unshipped = []

        # Load today's morning brief to see what was prioritized
        today_str = date.today().strftime("%Y-%m-%d")
        morning_file = self.storage_dir / f"morning_{today_str}.json"

        if morning_file.exists():
            try:
                with open(morning_file, "r") as f:
                    data = json.load(f)
                    shipped_today = self.get_shipped_items()
                    shipped_titles = {s.description.lower() for s in shipped_today}

                    for priority in data.get("top_priorities", []):
                        title = priority.get("title", "")
                        # If priority wasn't shipped, add to unshipped
                        if title.lower() not in shipped_titles:
                            linked = priority.get("linked_output", "")
                            unshipped.append(f"{title}" + (f" ({linked})" if linked else ""))
            except Exception:
                pass

        return unshipped[:5]  # Limit to 5 items

    def _generate_auto_note(self) -> str:
        """Generate an automatic note based on today's events."""
        # Simple heuristic: note the first unshipped item
        unshipped = self._get_unshipped_items()
        if unshipped:
            return f"Continue: {unshipped[0]}"

        shipped = self.get_shipped_items()
        if shipped:
            return f"Follow up on: {shipped[-1].description}"

        return "Review priorities and set clear outputs for the day."

    # =========================================================================
    # Blocking and Note Management
    # =========================================================================

    def add_blocked_item(
        self,
        item: str,
        reason: str,
        blocker_source: str = "alfred"
    ):
        """
        Add an item to today's blocked list.

        Args:
            item: Description of what was blocked
            reason: Why it was blocked
            blocker_source: Which system blocked it (e.g., reputation_sentinel)
        """
        blocked = BlockedItem(
            item=item,
            reason=reason,
            blocked_at=datetime.now().isoformat(),
            blocker_source=blocker_source
        )
        self._today_blocked.append(blocked)

    def set_tomorrow_note(self, note: str):
        """
        Set the note for tomorrow's brief.

        Args:
            note: Note to capture for tomorrow
        """
        self._tomorrow_note = note

    # =========================================================================
    # Storage Methods
    # =========================================================================

    def _save_morning_brief(self, brief: MorningBrief):
        """Save morning brief to storage."""
        filepath = self.storage_dir / f"morning_{brief.date}.json"
        with open(filepath, "w") as f:
            json.dump(brief.to_dict(), f, indent=2)

    def _save_evening_brief(self, brief: EveningBrief):
        """Save evening brief to storage."""
        filepath = self.storage_dir / f"evening_{brief.date}.json"
        with open(filepath, "w") as f:
            json.dump(brief.to_dict(), f, indent=2)

    def load_brief(self, brief_type: str, date_str: str) -> Optional[Dict]:
        """
        Load a previously generated brief.

        Args:
            brief_type: "morning" or "evening"
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Brief data as dictionary, or None if not found
        """
        filepath = self.storage_dir / f"{brief_type}_{date_str}.json"
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return None

    def list_briefs(self, days: int = 7) -> Dict[str, List[str]]:
        """
        List available briefs for the past N days.

        Args:
            days: Number of days to look back (default 7)

        Returns:
            Dictionary with 'morning' and 'evening' keys, each containing list of dates
        """
        result = {"morning": [], "evening": []}

        for i in range(days):
            target_date = date.today() - timedelta(days=i)
            date_str = target_date.strftime("%Y-%m-%d")

            morning_file = self.storage_dir / f"morning_{date_str}.json"
            evening_file = self.storage_dir / f"evening_{date_str}.json"

            if morning_file.exists():
                result["morning"].append(date_str)
            if evening_file.exists():
                result["evening"].append(date_str)

        return result

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def morning(self) -> str:
        """
        Generate and return formatted morning brief.

        Convenience method for quick access.

        Returns:
            Formatted morning brief string
        """
        brief = self.generate_morning_brief()
        return brief.format_display()

    def evening(self, note: Optional[str] = None) -> str:
        """
        Generate and return formatted evening brief.

        Convenience method for quick access.

        Args:
            note: Optional note for tomorrow

        Returns:
            Formatted evening brief string
        """
        brief = self.generate_evening_brief(note_for_tomorrow=note)
        return brief.format_display()

    def __repr__(self) -> str:
        return f"DailyBrief(storage_dir={self.storage_dir})"


# =========================================================================
# Tool Interface for Agent Zero
# =========================================================================

class DailyBriefTool:
    """
    Tool wrapper for Daily Brief system.

    Provides Agent Zero compatible interface for brief generation.
    """

    def __init__(self):
        self.brief_system = DailyBrief()

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute daily brief tool methods."""
        method = kwargs.get("method", "morning")

        if method == "morning":
            brief = self.brief_system.generate_morning_brief()
            return {
                "status": "success",
                "brief_type": "morning",
                "formatted": brief.format_display(),
                "data": brief.to_dict()
            }

        elif method == "evening":
            note = kwargs.get("note_for_tomorrow")
            brief = self.brief_system.generate_evening_brief(note_for_tomorrow=note)
            return {
                "status": "success",
                "brief_type": "evening",
                "formatted": brief.format_display(),
                "data": brief.to_dict()
            }

        elif method == "block_item":
            item = kwargs.get("item", "")
            reason = kwargs.get("reason", "")
            source = kwargs.get("source", "alfred")
            self.brief_system.add_blocked_item(item, reason, source)
            return {
                "status": "success",
                "message": f"Blocked: {item}"
            }

        elif method == "set_note":
            note = kwargs.get("note", "")
            self.brief_system.set_tomorrow_note(note)
            return {
                "status": "success",
                "message": f"Note set for tomorrow"
            }

        elif method == "list_briefs":
            days = kwargs.get("days", 7)
            briefs = self.brief_system.list_briefs(days)
            return {
                "status": "success",
                "briefs": briefs
            }

        elif method == "load_brief":
            brief_type = kwargs.get("brief_type", "morning")
            date_str = kwargs.get("date", date.today().strftime("%Y-%m-%d"))
            data = self.brief_system.load_brief(brief_type, date_str)
            if data:
                return {"status": "success", "data": data}
            return {"status": "not_found", "message": f"No {brief_type} brief for {date_str}"}

        elif method == "status":
            state, reason = self.brief_system.get_reputation_status()
            return {
                "status": "success",
                "reputation_state": state,
                "reason": reason,
                "constraint": self.brief_system.get_todays_constraint()
            }

        return {"status": "error", "message": f"Unknown method: {method}"}


# Standalone usage example
if __name__ == "__main__":
    # Initialize the daily brief system
    brief = DailyBrief()

    # Generate morning brief
    print("=" * 50)
    print("MORNING BRIEF")
    print("=" * 50)
    print(brief.morning())

    print("\n" + "=" * 50)
    print("EVENING BRIEF")
    print("=" * 50)
    print(brief.evening(note="Review ALFRED implementation progress"))
