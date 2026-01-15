# Relationship Nudge Agent
# Preserves social fabric through memory and timing.
# Tracks important relationships, surfaces fading connections, reminds of important dates.

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json

from . import OperationsAgent, AgentResponse, AlfredState


class RelationshipCategory(Enum):
    """
    Categories of relationships with expected contact frequency.

    Each category has an associated expected contact interval in days.
    """
    INNER_CIRCLE = "inner_circle"  # Family, closest friends - weekly contact expected
    CLOSE = "close"                 # Good friends, key colleagues - monthly contact
    IMPORTANT = "important"         # Valued connections - quarterly contact
    PROFESSIONAL = "professional"   # Mentors, collaborators, peers - as needed
    EXTENDED = "extended"           # Wider network - annual minimum

    @property
    def expected_contact_days(self) -> int:
        """Return the expected number of days between contacts for this category."""
        expectations = {
            RelationshipCategory.INNER_CIRCLE: 7,    # Weekly
            RelationshipCategory.CLOSE: 30,          # Monthly
            RelationshipCategory.IMPORTANT: 90,      # Quarterly
            RelationshipCategory.PROFESSIONAL: 90,   # As needed, check quarterly
            RelationshipCategory.EXTENDED: 365,      # Annual
        }
        return expectations.get(self, 90)

    @property
    def grace_period_days(self) -> int:
        """Return grace period before marking as overdue."""
        grace = {
            RelationshipCategory.INNER_CIRCLE: 3,
            RelationshipCategory.CLOSE: 7,
            RelationshipCategory.IMPORTANT: 14,
            RelationshipCategory.PROFESSIONAL: 30,
            RelationshipCategory.EXTENDED: 30,
        }
        return grace.get(self, 14)


class ContactIntensity(Enum):
    """Suggested intensity level for contact."""
    LIGHT_TOUCH = "light_touch"  # Quick text, emoji reaction, brief acknowledgment
    MEANINGFUL = "meaningful"     # Longer conversation, genuine catch-up
    PRIORITY = "priority"         # Needs immediate attention, significant effort


class SuggestedAction(Enum):
    """Types of contact actions to suggest."""
    TEXT = "text"   # Quick message, SMS, WhatsApp
    CALL = "call"   # Phone or video call
    MEET = "meet"   # In-person meeting
    NOTE = "note"   # Handwritten note, card, thoughtful gesture


class OccasionType(Enum):
    """Types of important occasions to track."""
    BIRTHDAY = "birthday"
    ANNIVERSARY = "anniversary"
    MEMORIAL = "memorial"
    PROFESSIONAL = "professional"  # Work anniversary, promotion date
    CUSTOM = "custom"


class TimeHorizon(Enum):
    """Scope of relationship check request."""
    OVERDUE_ONLY = "overdue_only"
    UPCOMING = "upcoming"
    COMPREHENSIVE = "comprehensive"


class RequestScope(Enum):
    """Scope of people to include in check."""
    ALL = "all"
    CATEGORY = "category"
    SPECIFIC_PEOPLE = "specific_people"


@dataclass
class Person:
    """Represents a person in the relationship network."""
    id: str
    name: str
    category: RelationshipCategory
    notes: str = ""
    preferred_contact_method: Optional[SuggestedAction] = None
    timezone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "notes": self.notes,
            "preferred_contact_method": self.preferred_contact_method.value if self.preferred_contact_method else None,
            "timezone": self.timezone,
        }


@dataclass
class ContactRecord:
    """Records a single contact interaction."""
    date: date
    contact_type: SuggestedAction
    notes: str = ""
    was_meaningful: bool = True
    initiated_by_user: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "contact_type": self.contact_type.value,
            "notes": self.notes,
            "was_meaningful": self.was_meaningful,
            "initiated_by_user": self.initiated_by_user,
        }


@dataclass
class Relationship:
    """Represents a relationship with contact history."""
    person: Person
    contacts: List[ContactRecord] = field(default_factory=list)
    relationship_notes: str = ""
    paused_until: Optional[date] = None  # Temporarily pause nudges

    @property
    def last_contact(self) -> Optional[date]:
        """Return the date of most recent contact."""
        if not self.contacts:
            return None
        return max(contact.date for contact in self.contacts)

    @property
    def last_meaningful_contact(self) -> Optional[date]:
        """Return the date of most recent meaningful contact."""
        meaningful = [c for c in self.contacts if c.was_meaningful]
        if not meaningful:
            return None
        return max(contact.date for contact in meaningful)

    def days_since_contact(self, reference_date: Optional[date] = None) -> Optional[int]:
        """Calculate days since last contact."""
        if not self.last_contact:
            return None
        ref = reference_date or date.today()
        return (ref - self.last_contact).days

    def days_since_meaningful_contact(self, reference_date: Optional[date] = None) -> Optional[int]:
        """Calculate days since last meaningful contact."""
        if not self.last_meaningful_contact:
            return None
        ref = reference_date or date.today()
        return (ref - self.last_meaningful_contact).days

    def is_overdue(self, reference_date: Optional[date] = None) -> bool:
        """Check if contact is overdue for this relationship."""
        days = self.days_since_meaningful_contact(reference_date)
        if days is None:
            return True  # Never contacted = overdue
        expected = self.person.category.expected_contact_days
        grace = self.person.category.grace_period_days
        return days > (expected + grace)

    def days_overdue(self, reference_date: Optional[date] = None) -> int:
        """Calculate how many days overdue the contact is."""
        days = self.days_since_meaningful_contact(reference_date)
        if days is None:
            return 999  # Unknown = very overdue
        expected = self.person.category.expected_contact_days
        grace = self.person.category.grace_period_days
        overdue = days - (expected + grace)
        return max(0, overdue)

    def is_fading(self, reference_date: Optional[date] = None) -> bool:
        """Check if this connection is at risk of fading (approaching overdue)."""
        days = self.days_since_meaningful_contact(reference_date)
        if days is None:
            return True
        expected = self.person.category.expected_contact_days
        # Fading if within 25% of expected time remaining
        threshold = expected * 0.75
        return days > threshold and not self.is_overdue(reference_date)

    def is_healthy(self, reference_date: Optional[date] = None) -> bool:
        """Check if relationship contact is healthy (within expected range)."""
        days = self.days_since_meaningful_contact(reference_date)
        if days is None:
            return False
        expected = self.person.category.expected_contact_days
        return days <= expected * 0.5  # Healthy if within half the expected time

    def is_paused(self, reference_date: Optional[date] = None) -> bool:
        """Check if nudges are paused for this relationship."""
        if not self.paused_until:
            return False
        ref = reference_date or date.today()
        return ref < self.paused_until

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person": self.person.to_dict(),
            "contacts": [c.to_dict() for c in self.contacts],
            "relationship_notes": self.relationship_notes,
            "paused_until": self.paused_until.isoformat() if self.paused_until else None,
            "last_contact": self.last_contact.isoformat() if self.last_contact else None,
            "days_since_contact": self.days_since_contact(),
            "is_overdue": self.is_overdue(),
            "days_overdue": self.days_overdue(),
        }


@dataclass
class ImportantDate:
    """An important date to remember for a person."""
    person_id: str
    person_name: str
    occasion_type: OccasionType
    date: date  # Month and day matter; year used for age calculation if applicable
    description: str = ""
    year_known: bool = True
    reminder_days_before: int = 7

    def days_until(self, reference_date: Optional[date] = None) -> int:
        """Calculate days until this occasion occurs next."""
        ref = reference_date or date.today()
        # Create this year's occurrence
        this_year_date = date(ref.year, self.date.month, self.date.day)

        if this_year_date < ref:
            # Already passed this year, calculate for next year
            next_year_date = date(ref.year + 1, self.date.month, self.date.day)
            return (next_year_date - ref).days
        else:
            return (this_year_date - ref).days

    def is_upcoming(self, days_ahead: int = 30, reference_date: Optional[date] = None) -> bool:
        """Check if this date is coming up within specified days."""
        return 0 <= self.days_until(reference_date) <= days_ahead

    def needs_reminder(self, reference_date: Optional[date] = None) -> bool:
        """Check if a reminder should be triggered."""
        days = self.days_until(reference_date)
        return 0 <= days <= self.reminder_days_before

    def get_age(self, reference_date: Optional[date] = None) -> Optional[int]:
        """Calculate age/years if year is known (e.g., birthday age)."""
        if not self.year_known:
            return None
        ref = reference_date or date.today()
        age = ref.year - self.date.year
        # Adjust if birthday hasn't occurred yet this year
        if (ref.month, ref.day) < (self.date.month, self.date.day):
            age -= 1
        return age

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "person_name": self.person_name,
            "occasion_type": self.occasion_type.value,
            "date": self.date.isoformat(),
            "description": self.description,
            "year_known": self.year_known,
            "days_until": self.days_until(),
            "needs_reminder": self.needs_reminder(),
        }


@dataclass
class OverdueContact:
    """Represents an overdue contact for reporting."""
    name: str
    category: RelationshipCategory
    last_contact: Optional[date]
    days_overdue: int
    context: str
    suggested_action: SuggestedAction
    intensity: ContactIntensity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "last_contact": self.last_contact.isoformat() if self.last_contact else "Never",
            "days_overdue": self.days_overdue,
            "context": self.context,
            "suggested_action": self.suggested_action.value,
            "intensity": self.intensity.value,
        }


@dataclass
class UpcomingDate:
    """Represents an upcoming important date for reporting."""
    date: date
    name: str
    occasion: str
    suggested_acknowledgment: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "name": self.name,
            "occasion": self.occasion,
            "suggested_acknowledgment": self.suggested_acknowledgment,
        }


@dataclass
class RelationshipReminder:
    """A gentle reminder about a relationship."""
    who: str
    last_contact: Optional[date]
    relationship: RelationshipCategory
    why_now: str
    suggested_action: str
    optional_defer_to: Optional[date] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "who": self.who,
            "last_contact": self.last_contact.isoformat() if self.last_contact else "Unknown",
            "relationship": self.relationship.value,
            "why_now": self.why_now,
            "suggested_action": self.suggested_action,
            "optional_defer_to": self.optional_defer_to.isoformat() if self.optional_defer_to else None,
        }


@dataclass
class EnergyLevel:
    """User's current social energy capacity."""
    level: int  # 1-10 scale
    max_actions_suggested: int = 3
    prefer_low_effort: bool = False

    @classmethod
    def from_level(cls, level: int) -> "EnergyLevel":
        """Create EnergyLevel from a 1-10 scale."""
        level = max(1, min(10, level))
        if level <= 3:
            return cls(level=level, max_actions_suggested=1, prefer_low_effort=True)
        elif level <= 6:
            return cls(level=level, max_actions_suggested=3, prefer_low_effort=True)
        else:
            return cls(level=level, max_actions_suggested=5, prefer_low_effort=False)


@dataclass
class RelationshipCheckRequest:
    """Request for relationship check."""
    scope: RequestScope = RequestScope.ALL
    category_filter: Optional[RelationshipCategory] = None
    specific_people: Optional[List[str]] = None
    time_horizon: TimeHorizon = TimeHorizon.COMPREHENSIVE
    include_birthdays: bool = True
    include_anniversaries: bool = True
    include_all_occasions: bool = False
    energy_level: Optional[EnergyLevel] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.value,
            "category_filter": self.category_filter.value if self.category_filter else None,
            "specific_people": self.specific_people,
            "time_horizon": self.time_horizon.value,
            "include_birthdays": self.include_birthdays,
            "include_anniversaries": self.include_anniversaries,
            "include_all_occasions": self.include_all_occasions,
            "energy_level": self.energy_level.level if self.energy_level else None,
        }


class RelationshipNudgeAgent(OperationsAgent):
    """
    Relationship Nudge Agent - Preserves social fabric through memory and timing.

    Tracks important relationships, surfaces when connections are fading,
    reminds of important dates, and suggests light-touch maintenance
    without creating additional burden.

    Key Principles:
    - Does NOT manage relationships (user does that)
    - Does NOT send messages automatically
    - Does NOT manipulate or engineer interactions
    - Does NOT track casual acquaintances obsessively
    - Does NOT create obligation or guilt
    - Does NOT prioritize transactional relationships
    - Does NOT ignore user's social energy limits

    - DOES track time since meaningful contact
    - DOES remember important dates
    - DOES surface fading connections before lost
    - DOES suggest appropriate check-in intensity
    - DOES note relationship context and history
    - DOES respect user's capacity constraints
    - DOES prioritize based on relationship importance
    """

    def __init__(self):
        super().__init__("RelationshipNudgeAgent")
        self._relationships: Dict[str, Relationship] = {}
        self._important_dates: List[ImportantDate] = []
        self._default_energy = EnergyLevel.from_level(7)

    # =========================================================================
    # Core Data Management
    # =========================================================================

    def add_person(self, person: Person) -> None:
        """Add a person to track."""
        if person.id not in self._relationships:
            self._relationships[person.id] = Relationship(person=person)

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship with existing history."""
        self._relationships[relationship.person.id] = relationship

    def get_relationship(self, person_id: str) -> Optional[Relationship]:
        """Get a relationship by person ID."""
        return self._relationships.get(person_id)

    def record_contact(self, person_id: str, contact: ContactRecord) -> bool:
        """Record a contact with a person."""
        relationship = self._relationships.get(person_id)
        if not relationship:
            return False
        relationship.contacts.append(contact)
        return True

    def add_important_date(self, important_date: ImportantDate) -> None:
        """Add an important date to track."""
        self._important_dates.append(important_date)

    def pause_nudges(self, person_id: str, until: date) -> bool:
        """Pause nudges for a person until a specific date."""
        relationship = self._relationships.get(person_id)
        if not relationship:
            return False
        relationship.paused_until = until
        return True

    def set_energy_level(self, level: int) -> None:
        """Set the default energy level for suggestions."""
        self._default_energy = EnergyLevel.from_level(level)

    # =========================================================================
    # Core Analysis Methods
    # =========================================================================

    def find_overdue(
        self,
        category_filter: Optional[RelationshipCategory] = None,
        reference_date: Optional[date] = None
    ) -> List[OverdueContact]:
        """
        Find all overdue contacts.

        Args:
            category_filter: Optional filter to specific category
            reference_date: Date to use for calculations (default: today)

        Returns:
            List of OverdueContact objects, sorted by severity
        """
        overdue_list = []

        for relationship in self._relationships.values():
            # Skip paused relationships
            if relationship.is_paused(reference_date):
                continue

            # Apply category filter if specified
            if category_filter and relationship.person.category != category_filter:
                continue

            # Check if overdue
            if not relationship.is_overdue(reference_date):
                continue

            # Build overdue contact record
            days_over = relationship.days_overdue(reference_date)
            action, intensity = self._suggest_action_for_relationship(relationship, days_over)

            # Get context from last contact
            context = ""
            if relationship.contacts:
                last = max(relationship.contacts, key=lambda c: c.date)
                context = last.notes or f"Last contact via {last.contact_type.value}"

            overdue = OverdueContact(
                name=relationship.person.name,
                category=relationship.person.category,
                last_contact=relationship.last_contact,
                days_overdue=days_over,
                context=context,
                suggested_action=action,
                intensity=intensity
            )
            overdue_list.append(overdue)

        # Sort by priority: category weight * days overdue
        def priority_score(o: OverdueContact) -> float:
            category_weight = {
                RelationshipCategory.INNER_CIRCLE: 5,
                RelationshipCategory.CLOSE: 4,
                RelationshipCategory.IMPORTANT: 3,
                RelationshipCategory.PROFESSIONAL: 2,
                RelationshipCategory.EXTENDED: 1,
            }
            return category_weight.get(o.category, 1) * (o.days_overdue + 1)

        return sorted(overdue_list, key=priority_score, reverse=True)

    def find_fading(
        self,
        category_filter: Optional[RelationshipCategory] = None,
        reference_date: Optional[date] = None
    ) -> List[Relationship]:
        """
        Find connections that are fading but not yet overdue.

        Args:
            category_filter: Optional filter to specific category
            reference_date: Date to use for calculations

        Returns:
            List of fading relationships
        """
        fading = []

        for relationship in self._relationships.values():
            if relationship.is_paused(reference_date):
                continue

            if category_filter and relationship.person.category != category_filter:
                continue

            if relationship.is_fading(reference_date):
                fading.append(relationship)

        return fading

    def find_healthy(
        self,
        category_filter: Optional[RelationshipCategory] = None,
        reference_date: Optional[date] = None
    ) -> List[Relationship]:
        """
        Find relationships with healthy recent contact.

        Args:
            category_filter: Optional filter to specific category
            reference_date: Date to use for calculations

        Returns:
            List of healthy relationships
        """
        healthy = []

        for relationship in self._relationships.values():
            if category_filter and relationship.person.category != category_filter:
                continue

            if relationship.is_healthy(reference_date):
                healthy.append(relationship)

        return healthy

    def get_upcoming_dates(
        self,
        days_ahead: int = 30,
        include_birthdays: bool = True,
        include_anniversaries: bool = True,
        include_all: bool = False,
        reference_date: Optional[date] = None
    ) -> List[UpcomingDate]:
        """
        Get upcoming important dates.

        Args:
            days_ahead: How many days ahead to look
            include_birthdays: Include birthday occasions
            include_anniversaries: Include anniversary occasions
            include_all: Include all occasion types
            reference_date: Date to use for calculations

        Returns:
            List of UpcomingDate objects, sorted by date
        """
        ref = reference_date or date.today()
        upcoming = []

        for imp_date in self._important_dates:
            # Filter by occasion type
            if not include_all:
                if imp_date.occasion_type == OccasionType.BIRTHDAY and not include_birthdays:
                    continue
                if imp_date.occasion_type == OccasionType.ANNIVERSARY and not include_anniversaries:
                    continue

            # Check if upcoming
            if not imp_date.is_upcoming(days_ahead, reference_date):
                continue

            # Calculate the actual date this year or next
            days_until = imp_date.days_until(reference_date)
            actual_date = ref + timedelta(days=days_until)

            # Generate suggested acknowledgment
            acknowledgment = self._suggest_acknowledgment(imp_date)

            upcoming.append(UpcomingDate(
                date=actual_date,
                name=imp_date.person_name,
                occasion=f"{imp_date.occasion_type.value}: {imp_date.description}" if imp_date.description else imp_date.occasion_type.value,
                suggested_acknowledgment=acknowledgment
            ))

        return sorted(upcoming, key=lambda u: u.date)

    def suggest_actions(
        self,
        energy_level: Optional[EnergyLevel] = None,
        max_suggestions: Optional[int] = None,
        reference_date: Optional[date] = None
    ) -> List[RelationshipReminder]:
        """
        Generate energy-aware action suggestions.

        Args:
            energy_level: User's current energy level
            max_suggestions: Maximum number of suggestions
            reference_date: Date to use for calculations

        Returns:
            List of RelationshipReminder objects
        """
        energy = energy_level or self._default_energy
        max_actions = max_suggestions or energy.max_actions_suggested

        reminders = []

        # Get overdue contacts
        overdue = self.find_overdue(reference_date=reference_date)

        for contact in overdue[:max_actions]:
            # Adjust action based on energy
            action = contact.suggested_action
            if energy.prefer_low_effort and action in [SuggestedAction.MEET, SuggestedAction.CALL]:
                action = SuggestedAction.TEXT

            intensity_desc = {
                ContactIntensity.LIGHT_TOUCH: "Quick check-in is fine",
                ContactIntensity.MEANINGFUL: "Worth a real conversation",
                ContactIntensity.PRIORITY: "Needs genuine attention soon",
            }

            why_now = f"{contact.days_overdue} days overdue for {contact.category.value} relationship"
            if contact.context:
                why_now += f". {contact.context}"

            # Calculate deferral option
            defer_days = min(7, contact.category.expected_contact_days // 4)
            defer_date = (reference_date or date.today()) + timedelta(days=defer_days)

            reminders.append(RelationshipReminder(
                who=contact.name,
                last_contact=contact.last_contact,
                relationship=contact.category,
                why_now=why_now,
                suggested_action=f"{action.value.title()}: {intensity_desc.get(contact.intensity, '')}",
                optional_defer_to=defer_date
            ))

        return reminders

    def calculate_health_summary(
        self,
        reference_date: Optional[date] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate relationship health summary by category.

        Returns:
            Dict with category health statistics
        """
        summary = {}

        for category in RelationshipCategory:
            relationships = [
                r for r in self._relationships.values()
                if r.person.category == category
            ]

            if not relationships:
                continue

            total = len(relationships)
            healthy = sum(1 for r in relationships if r.is_healthy(reference_date))
            fading = sum(1 for r in relationships if r.is_fading(reference_date))
            overdue = sum(1 for r in relationships if r.is_overdue(reference_date))

            summary[category.value] = {
                "total": total,
                "healthy_count": healthy,
                "healthy_percent": round((healthy / total) * 100, 1),
                "fading_count": fading,
                "fading_percent": round((fading / total) * 100, 1),
                "overdue_count": overdue,
                "overdue_percent": round((overdue / total) * 100, 1),
            }

        return summary

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def check_relationships(
        self,
        request: RelationshipCheckRequest
    ) -> AgentResponse:
        """
        Perform a comprehensive relationship check.

        This is the main entry point for the agent, handling the full
        RELATIONSHIP_CHECK_REQUEST flow and returning a RELATIONSHIP_NUDGE response.

        Args:
            request: The check request parameters

        Returns:
            AgentResponse with RELATIONSHIP_NUDGE data
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Handle RED state - defer non-urgent nudges
        if self.alfred_state == AlfredState.RED:
            return self.create_response(
                data={
                    "status": "DEFERRED",
                    "reason": "Non-urgent relationship nudges deferred during RED state",
                    "urgent_only": True,
                },
                warnings=["Operating in RED state - only urgent nudges processed"]
            )

        reference_date = date.today()
        energy = request.energy_level or self._default_energy

        # Apply scope filters
        category_filter = None
        if request.scope == RequestScope.CATEGORY and request.category_filter:
            category_filter = request.category_filter

        # Gather data based on time horizon
        overdue_contacts = []
        upcoming_dates = []
        fading_connections = []
        healthy_relationships = []

        if request.time_horizon in [TimeHorizon.OVERDUE_ONLY, TimeHorizon.COMPREHENSIVE]:
            overdue_contacts = self.find_overdue(
                category_filter=category_filter,
                reference_date=reference_date
            )

        if request.time_horizon in [TimeHorizon.UPCOMING, TimeHorizon.COMPREHENSIVE]:
            upcoming_dates = self.get_upcoming_dates(
                days_ahead=30,
                include_birthdays=request.include_birthdays,
                include_anniversaries=request.include_anniversaries,
                include_all=request.include_all_occasions,
                reference_date=reference_date
            )

        if request.time_horizon == TimeHorizon.COMPREHENSIVE:
            fading_connections = self.find_fading(
                category_filter=category_filter,
                reference_date=reference_date
            )
            healthy_relationships = self.find_healthy(
                category_filter=category_filter,
                reference_date=reference_date
            )

        # Calculate health summary
        health_summary = self.calculate_health_summary(reference_date)

        # Generate energy-adjusted suggestions
        suggestions = self.suggest_actions(
            energy_level=energy,
            reference_date=reference_date
        )

        # Identify top 3 priorities for limited capacity
        priorities = suggestions[:3] if suggestions else []

        # Build response
        return self.generate_nudge(
            report_date=datetime.now(),
            overdue_contacts=overdue_contacts,
            upcoming_dates=upcoming_dates,
            fading_connections=fading_connections,
            recent_positive=healthy_relationships,
            health_summary=health_summary,
            energy_suggestions=suggestions,
            priorities=priorities,
            request=request
        )

    def generate_nudge(
        self,
        report_date: datetime,
        overdue_contacts: List[OverdueContact],
        upcoming_dates: List[UpcomingDate],
        fading_connections: List[Relationship],
        recent_positive: List[Relationship],
        health_summary: Dict[str, Dict[str, Any]],
        energy_suggestions: List[RelationshipReminder],
        priorities: List[RelationshipReminder],
        request: RelationshipCheckRequest
    ) -> AgentResponse:
        """
        Generate the RELATIONSHIP_NUDGE response packet.

        Args:
            report_date: When this report was generated
            overdue_contacts: List of overdue contacts
            upcoming_dates: List of upcoming important dates
            fading_connections: List of fading relationships
            recent_positive: List of healthy relationships
            health_summary: Category health statistics
            energy_suggestions: Energy-adjusted suggestions
            priorities: Top priority actions
            request: Original request parameters

        Returns:
            AgentResponse with full RELATIONSHIP_NUDGE data
        """
        data = {
            "report_type": "RELATIONSHIP_NUDGE",
            "report_date": report_date.isoformat(),
            "request_params": request.to_dict(),

            "overdue_contacts": [c.to_dict() for c in overdue_contacts],
            "overdue_count": len(overdue_contacts),

            "upcoming_dates": [d.to_dict() for d in upcoming_dates],
            "upcoming_count": len(upcoming_dates),

            "fading_connections": [
                {
                    "name": r.person.name,
                    "category": r.person.category.value,
                    "days_since_contact": r.days_since_meaningful_contact(),
                    "reason": "Approaching expected contact threshold",
                }
                for r in fading_connections
            ],
            "fading_count": len(fading_connections),

            "recent_positive_interactions": [
                {
                    "name": r.person.name,
                    "category": r.person.category.value,
                    "last_contact": r.last_contact.isoformat() if r.last_contact else None,
                }
                for r in recent_positive[:5]  # Limit to top 5
            ],

            "relationship_health_summary": health_summary,

            "energy_adjusted_suggestions": [s.to_dict() for s in energy_suggestions],

            "this_weeks_priority": [p.to_dict() for p in priorities],
        }

        # Add warnings if needed
        warnings = []
        if len(overdue_contacts) > 10:
            warnings.append(f"High number of overdue contacts ({len(overdue_contacts)}). Consider focusing on highest priority.")

        inner_circle_overdue = [
            c for c in overdue_contacts
            if c.category == RelationshipCategory.INNER_CIRCLE
        ]
        if inner_circle_overdue:
            warnings.append(f"{len(inner_circle_overdue)} inner circle relationship(s) overdue - these should be addressed first.")

        return self.create_response(
            data=data,
            success=True,
            warnings=warnings
        )

    def generate_reminder(
        self,
        person_id: str,
        reference_date: Optional[date] = None
    ) -> Optional[AgentResponse]:
        """
        Generate a single RELATIONSHIP_REMINDER for a specific person.

        Args:
            person_id: ID of the person to generate reminder for
            reference_date: Date to use for calculations

        Returns:
            AgentResponse with RELATIONSHIP_REMINDER data, or None if no reminder needed
        """
        relationship = self._relationships.get(person_id)
        if not relationship:
            return None

        if relationship.is_paused(reference_date):
            return None

        if not relationship.is_overdue(reference_date) and not relationship.is_fading(reference_date):
            return None

        days_over = relationship.days_overdue(reference_date)
        action, intensity = self._suggest_action_for_relationship(relationship, days_over)

        # Determine why now
        if relationship.is_overdue(reference_date):
            why_now = f"Contact overdue by {days_over} days"
        else:
            why_now = "Connection at risk of fading"

        # Check for upcoming important dates
        person_dates = [d for d in self._important_dates if d.person_id == person_id]
        upcoming = [d for d in person_dates if d.needs_reminder(reference_date)]
        if upcoming:
            why_now += f". Upcoming: {upcoming[0].occasion_type.value} in {upcoming[0].days_until(reference_date)} days"

        # Calculate defer option
        defer_days = min(7, relationship.person.category.expected_contact_days // 4)
        defer_date = (reference_date or date.today()) + timedelta(days=defer_days)

        reminder = RelationshipReminder(
            who=relationship.person.name,
            last_contact=relationship.last_contact,
            relationship=relationship.person.category,
            why_now=why_now,
            suggested_action=f"{action.value}: {intensity.value} intensity",
            optional_defer_to=defer_date
        )

        return self.create_response(
            data={
                "report_type": "RELATIONSHIP_REMINDER",
                "reminder": reminder.to_dict(),
            }
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _suggest_action_for_relationship(
        self,
        relationship: Relationship,
        days_overdue: int
    ) -> Tuple[SuggestedAction, ContactIntensity]:
        """
        Suggest action and intensity based on relationship and overdue status.

        Args:
            relationship: The relationship to analyze
            days_overdue: Number of days overdue

        Returns:
            Tuple of (SuggestedAction, ContactIntensity)
        """
        category = relationship.person.category
        preferred = relationship.person.preferred_contact_method

        # Determine intensity based on category and how overdue
        if category == RelationshipCategory.INNER_CIRCLE:
            if days_overdue > 14:
                intensity = ContactIntensity.PRIORITY
            elif days_overdue > 7:
                intensity = ContactIntensity.MEANINGFUL
            else:
                intensity = ContactIntensity.LIGHT_TOUCH
        elif category == RelationshipCategory.CLOSE:
            if days_overdue > 30:
                intensity = ContactIntensity.PRIORITY
            elif days_overdue > 14:
                intensity = ContactIntensity.MEANINGFUL
            else:
                intensity = ContactIntensity.LIGHT_TOUCH
        else:
            if days_overdue > 60:
                intensity = ContactIntensity.MEANINGFUL
            else:
                intensity = ContactIntensity.LIGHT_TOUCH

        # Determine action
        if preferred:
            action = preferred
        elif intensity == ContactIntensity.PRIORITY:
            action = SuggestedAction.CALL
        elif intensity == ContactIntensity.MEANINGFUL:
            if category in [RelationshipCategory.INNER_CIRCLE, RelationshipCategory.CLOSE]:
                action = SuggestedAction.CALL
            else:
                action = SuggestedAction.TEXT
        else:
            action = SuggestedAction.TEXT

        return action, intensity

    def _suggest_acknowledgment(self, imp_date: ImportantDate) -> str:
        """
        Suggest an acknowledgment for an important date.

        Args:
            imp_date: The important date

        Returns:
            Suggested acknowledgment string
        """
        occasion = imp_date.occasion_type

        if occasion == OccasionType.BIRTHDAY:
            age = imp_date.get_age()
            if age:
                return f"Birthday message - turning {age}. A call or thoughtful note would be meaningful."
            return "Birthday message - a call or thoughtful note would be meaningful."

        elif occasion == OccasionType.ANNIVERSARY:
            years = imp_date.get_age()
            if years:
                return f"Anniversary message - {years} years. Consider a heartfelt message or card."
            return "Anniversary message - consider a heartfelt message."

        elif occasion == OccasionType.MEMORIAL:
            return "Reach out with care and support. A simple message acknowledging the day."

        elif occasion == OccasionType.PROFESSIONAL:
            return "Professional milestone acknowledgment. A brief congratulatory message."

        else:
            return f"Acknowledge: {imp_date.description or 'Special occasion'}"

    def check_state_permission(self) -> Tuple[bool, str]:
        """
        Check if agent can operate in current Alfred state.

        Operations agents mostly continue in all states, but relationship
        nudges defer non-urgent items in RED state.
        """
        if self.alfred_state == AlfredState.RED:
            # We still permit operation but with reduced scope
            return True, "Operating with reduced scope - non-urgent nudges deferred"
        return True, "Operation permitted"

    # =========================================================================
    # Serialization
    # =========================================================================

    def export_data(self) -> Dict[str, Any]:
        """Export all relationship data for persistence."""
        return {
            "relationships": {
                pid: rel.to_dict()
                for pid, rel in self._relationships.items()
            },
            "important_dates": [d.to_dict() for d in self._important_dates],
            "default_energy_level": self._default_energy.level,
        }

    def import_relationships(self, relationships: List[Relationship]) -> None:
        """Import relationships from external source."""
        for rel in relationships:
            self._relationships[rel.person.id] = rel

    def import_dates(self, dates: List[ImportantDate]) -> None:
        """Import important dates from external source."""
        self._important_dates.extend(dates)

    def clear_data(self) -> None:
        """Clear all stored data."""
        self._relationships.clear()
        self._important_dates.clear()


# =============================================================================
# Convenience Functions for External Use
# =============================================================================

def create_relationship_check_request(
    scope: str = "all",
    time_horizon: str = "comprehensive",
    category: Optional[str] = None,
    include_birthdays: bool = True,
    include_anniversaries: bool = True,
    energy_level: int = 7
) -> RelationshipCheckRequest:
    """
    Create a RelationshipCheckRequest from simple parameters.

    Args:
        scope: "all", "category", or "specific_people"
        time_horizon: "overdue_only", "upcoming", or "comprehensive"
        category: Category name if scope is "category"
        include_birthdays: Include birthday occasions
        include_anniversaries: Include anniversary occasions
        energy_level: User energy level 1-10

    Returns:
        Configured RelationshipCheckRequest
    """
    scope_enum = RequestScope(scope) if scope in [e.value for e in RequestScope] else RequestScope.ALL
    horizon_enum = TimeHorizon(time_horizon) if time_horizon in [e.value for e in TimeHorizon] else TimeHorizon.COMPREHENSIVE

    cat_enum = None
    if category:
        try:
            cat_enum = RelationshipCategory(category)
        except ValueError:
            pass

    return RelationshipCheckRequest(
        scope=scope_enum,
        category_filter=cat_enum,
        time_horizon=horizon_enum,
        include_birthdays=include_birthdays,
        include_anniversaries=include_anniversaries,
        energy_level=EnergyLevel.from_level(energy_level)
    )


def create_person(
    id: str,
    name: str,
    category: str,
    notes: str = "",
    preferred_contact: Optional[str] = None
) -> Person:
    """
    Create a Person from simple parameters.

    Args:
        id: Unique identifier
        name: Person's name
        category: Relationship category name
        notes: Optional notes
        preferred_contact: Preferred contact method ("text", "call", "meet", "note")

    Returns:
        Configured Person object
    """
    cat_enum = RelationshipCategory(category) if category in [e.value for e in RelationshipCategory] else RelationshipCategory.IMPORTANT

    contact_enum = None
    if preferred_contact:
        try:
            contact_enum = SuggestedAction(preferred_contact)
        except ValueError:
            pass

    return Person(
        id=id,
        name=name,
        category=cat_enum,
        notes=notes,
        preferred_contact_method=contact_enum
    )


def create_important_date(
    person_id: str,
    person_name: str,
    occasion: str,
    month: int,
    day: int,
    year: Optional[int] = None,
    description: str = ""
) -> ImportantDate:
    """
    Create an ImportantDate from simple parameters.

    Args:
        person_id: Associated person ID
        person_name: Person's name
        occasion: Type of occasion ("birthday", "anniversary", etc.)
        month: Month (1-12)
        day: Day of month
        year: Optional year (for age calculation)
        description: Optional description

    Returns:
        Configured ImportantDate object
    """
    try:
        occasion_enum = OccasionType(occasion)
    except ValueError:
        occasion_enum = OccasionType.CUSTOM

    if year:
        date_obj = date(year, month, day)
        year_known = True
    else:
        date_obj = date(2000, month, day)  # Placeholder year
        year_known = False

    return ImportantDate(
        person_id=person_id,
        person_name=person_name,
        occasion_type=occasion_enum,
        date=date_obj,
        description=description,
        year_known=year_known
    )
