"""
Content Manager - Overall content orchestration system across all platforms.

Tracks content pipeline from idea to publication, coordinates between content agents,
maintains publishing calendar, ensures consistency across platforms.

DOES:
- Track all content from idea to shipped
- Coordinate between Research, Substack, Twitter, YouTube agents
- Maintain publishing calendar
- Ensure cross-platform consistency
- Flag stalled content
- Track content performance linkage
- Manage content dependencies
- Ensure evidence requirements met before drafting

DOES NOT:
- Create content directly (delegates to content agents)
- Make editorial decisions without Alfred's input
- Publish without approval
- Override Alfred's state constraints
- Prioritize virality over positioning
- Allow scope creep in content projects
- Ignore the Positioning Charter
"""

from . import OperationsAgent, AgentResponse, AlfredState
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json
import uuid


class ContentStage(Enum):
    """Content pipeline stages from idea to repurposed."""
    IDEA = "IDEA"               # Captured, not yet researched
    RESEARCH = "RESEARCH"       # Evidence being gathered
    OUTLINE = "OUTLINE"         # Structure defined
    DRAFT = "DRAFT"             # Being written/scripted
    REVIEW = "REVIEW"           # Awaiting Alfred approval
    SCHEDULED = "SCHEDULED"     # Approved, publication time set
    PUBLISHED = "PUBLISHED"     # Live on platform
    REPURPOSED = "REPURPOSED"   # Adapted for other platforms
    KILLED = "KILLED"           # Abandoned/cancelled


class Platform(Enum):
    """Target platforms for content."""
    SUBSTACK = "Substack"
    TWITTER = "Twitter"
    YOUTUBE = "YouTube"
    ALL = "all"


class Priority(Enum):
    """Content priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertType(Enum):
    """Types of content alerts."""
    STALLED = "stalled"
    DEADLINE_RISK = "deadline_risk"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    EVIDENCE_GAP = "evidence_gap"


class RequestType(Enum):
    """Types of content requests."""
    STATUS = "status"
    CREATE = "create"
    UPDATE = "update"
    SCHEDULE = "schedule"
    KILL = "kill"


@dataclass
class ContentItem:
    """Represents a single piece of content in the pipeline."""
    id: str
    title: str
    topic: str
    stage: ContentStage
    platform: Platform
    priority: Priority
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    stage_entered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    deadline: Optional[str] = None
    owner: str = "unassigned"  # Which agent owns current stage
    progress_pct: int = 0
    blockers: List[str] = field(default_factory=list)
    next_action: str = ""
    evidence_refs: List[str] = field(default_factory=list)  # Research citations
    dependencies: List[str] = field(default_factory=list)  # Content IDs this depends on
    repurposed_from: Optional[str] = None  # Original content ID if repurposed
    repurposed_to: List[str] = field(default_factory=list)  # Content IDs created from this
    scheduled_time: Optional[str] = None
    published_url: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    positioning_alignment: str = ""  # How it aligns with Positioning Charter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "topic": self.topic,
            "stage": self.stage.value,
            "platform": self.platform.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stage_entered_at": self.stage_entered_at,
            "deadline": self.deadline,
            "owner": self.owner,
            "progress_pct": self.progress_pct,
            "blockers": self.blockers,
            "next_action": self.next_action,
            "evidence_refs": self.evidence_refs,
            "dependencies": self.dependencies,
            "repurposed_from": self.repurposed_from,
            "repurposed_to": self.repurposed_to,
            "scheduled_time": self.scheduled_time,
            "published_url": self.published_url,
            "performance_metrics": self.performance_metrics,
            "notes": self.notes,
            "positioning_alignment": self.positioning_alignment
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentItem":
        return cls(
            id=data["id"],
            title=data["title"],
            topic=data["topic"],
            stage=ContentStage(data["stage"]),
            platform=Platform(data["platform"]),
            priority=Priority(data["priority"]),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            stage_entered_at=data.get("stage_entered_at", datetime.now().isoformat()),
            deadline=data.get("deadline"),
            owner=data.get("owner", "unassigned"),
            progress_pct=data.get("progress_pct", 0),
            blockers=data.get("blockers", []),
            next_action=data.get("next_action", ""),
            evidence_refs=data.get("evidence_refs", []),
            dependencies=data.get("dependencies", []),
            repurposed_from=data.get("repurposed_from"),
            repurposed_to=data.get("repurposed_to", []),
            scheduled_time=data.get("scheduled_time"),
            published_url=data.get("published_url"),
            performance_metrics=data.get("performance_metrics", {}),
            notes=data.get("notes", []),
            positioning_alignment=data.get("positioning_alignment", "")
        )


@dataclass
class PublishingCalendarEntry:
    """Entry in the publishing calendar."""
    id: str
    content_id: str
    title: str
    platform: Platform
    scheduled_date: str
    scheduled_time: str
    status: str = "scheduled"  # scheduled, published, cancelled
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content_id": self.content_id,
            "title": self.title,
            "platform": self.platform.value,
            "scheduled_date": self.scheduled_date,
            "scheduled_time": self.scheduled_time,
            "status": self.status,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublishingCalendarEntry":
        return cls(
            id=data["id"],
            content_id=data["content_id"],
            title=data["title"],
            platform=Platform(data["platform"]),
            scheduled_date=data["scheduled_date"],
            scheduled_time=data["scheduled_time"],
            status=data.get("status", "scheduled"),
            notes=data.get("notes", "")
        )


@dataclass
class ContentAlert:
    """Alert for content issues."""
    id: str
    alert_type: AlertType
    content_id: str
    content_title: str
    issue: str
    days_stalled: int = 0
    impact: str = ""
    recommended_action: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "alert_type": self.alert_type.value,
            "content_id": self.content_id,
            "content_title": self.content_title,
            "issue": self.issue,
            "days_stalled": self.days_stalled,
            "impact": self.impact,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at,
            "resolved": self.resolved
        }

    def format_alert(self) -> str:
        """Format alert for display."""
        return f"""CONTENT_ALERT
- Alert Type: {self.alert_type.value}
- Content: {self.content_title} ({self.content_id})
- Issue: {self.issue}
- Days Stalled: {self.days_stalled}
- Impact: {self.impact}
- Recommended Action: {self.recommended_action}"""


@dataclass
class ContentRequest:
    """Request for content operations."""
    request_type: RequestType
    content_id: Optional[str] = None
    topic: Optional[str] = None
    title: Optional[str] = None
    platform_target: Platform = Platform.ALL
    priority: Priority = Priority.MEDIUM
    deadline: Optional[str] = None
    new_stage: Optional[ContentStage] = None
    scheduled_time: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def parse(cls, request_text: str) -> "ContentRequest":
        """Parse a content request from text format."""
        lines = request_text.strip().split("\n")
        params = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_").replace("-", "_")
                value = value.strip()
                params[key] = value

        request_type = RequestType(params.get("request_type", "status").lower())

        platform_str = params.get("platform_target", "all").lower()
        try:
            platform = Platform(platform_str.capitalize() if platform_str != "all" else "all")
        except ValueError:
            platform = Platform.ALL

        priority_str = params.get("priority", "medium").lower()
        try:
            priority = Priority(priority_str)
        except ValueError:
            priority = Priority.MEDIUM

        return cls(
            request_type=request_type,
            content_id=params.get("content_id"),
            topic=params.get("topic"),
            title=params.get("title"),
            platform_target=platform,
            priority=priority,
            deadline=params.get("deadline"),
            scheduled_time=params.get("scheduled_time"),
            notes=params.get("notes")
        )


class ContentManager(OperationsAgent):
    """
    Content Manager - Overall content orchestration system.

    Coordinates the content pipeline across all platforms, tracking content
    from idea to publication and beyond. Does not create content directly
    but ensures smooth coordination between content agents.
    """

    # Stage ownership mapping - which agent owns each stage
    STAGE_OWNERS = {
        ContentStage.IDEA: "ContentManager",
        ContentStage.RESEARCH: "ResearchAgent",
        ContentStage.OUTLINE: "ContentManager",
        ContentStage.DRAFT: None,  # Depends on platform
        ContentStage.REVIEW: "Alfred",
        ContentStage.SCHEDULED: "ContentManager",
        ContentStage.PUBLISHED: None,  # Depends on platform
        ContentStage.REPURPOSED: "ContentManager",
        ContentStage.KILLED: "ContentManager"
    }

    # Platform-specific draft owners
    PLATFORM_DRAFT_OWNERS = {
        Platform.SUBSTACK: "SubstackAgent",
        Platform.TWITTER: "TwitterAgent",
        Platform.YOUTUBE: "YouTubeAgent",
        Platform.ALL: "ContentManager"
    }

    # Valid stage transitions
    VALID_TRANSITIONS = {
        ContentStage.IDEA: {ContentStage.RESEARCH, ContentStage.KILLED},
        ContentStage.RESEARCH: {ContentStage.OUTLINE, ContentStage.IDEA, ContentStage.KILLED},
        ContentStage.OUTLINE: {ContentStage.DRAFT, ContentStage.RESEARCH, ContentStage.KILLED},
        ContentStage.DRAFT: {ContentStage.REVIEW, ContentStage.OUTLINE, ContentStage.KILLED},
        ContentStage.REVIEW: {ContentStage.SCHEDULED, ContentStage.DRAFT, ContentStage.KILLED},
        ContentStage.SCHEDULED: {ContentStage.PUBLISHED, ContentStage.REVIEW, ContentStage.KILLED},
        ContentStage.PUBLISHED: {ContentStage.REPURPOSED},
        ContentStage.REPURPOSED: set(),
        ContentStage.KILLED: set()
    }

    # Stall thresholds in days per stage
    STALL_THRESHOLDS = {
        ContentStage.IDEA: 14,       # Ideas should move to research within 2 weeks
        ContentStage.RESEARCH: 7,    # Research should complete within a week
        ContentStage.OUTLINE: 3,     # Outlines should be quick
        ContentStage.DRAFT: 7,       # Drafts within a week
        ContentStage.REVIEW: 2,      # Reviews should be fast
        ContentStage.SCHEDULED: 30,  # Scheduled items have their own timeline
    }

    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__("ContentManager")
        self.storage_path = storage_path or Path("~/.alfred/content/pipeline.json").expanduser()
        self._content_items: Dict[str, ContentItem] = {}
        self._calendar: Dict[str, PublishingCalendarEntry] = {}
        self._alerts: Dict[str, ContentAlert] = {}
        self._load()

    def _load(self):
        """Load content data from persistent storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                    for item_data in data.get("content_items", []):
                        item = ContentItem.from_dict(item_data)
                        self._content_items[item.id] = item

                    for entry_data in data.get("calendar", []):
                        entry = PublishingCalendarEntry.from_dict(entry_data)
                        self._calendar[entry.id] = entry

                    for alert_data in data.get("alerts", []):
                        alert = ContentAlert(
                            id=alert_data["id"],
                            alert_type=AlertType(alert_data["alert_type"]),
                            content_id=alert_data["content_id"],
                            content_title=alert_data["content_title"],
                            issue=alert_data["issue"],
                            days_stalled=alert_data.get("days_stalled", 0),
                            impact=alert_data.get("impact", ""),
                            recommended_action=alert_data.get("recommended_action", ""),
                            created_at=alert_data.get("created_at", datetime.now().isoformat()),
                            resolved=alert_data.get("resolved", False)
                        )
                        self._alerts[alert.id] = alert
            except Exception as e:
                print(f"Warning: Could not load content data from {self.storage_path}: {e}")

    def _save(self):
        """Save content data to persistent storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            data = {
                "last_updated": datetime.now().isoformat(),
                "content_items": [item.to_dict() for item in self._content_items.values()],
                "calendar": [entry.to_dict() for entry in self._calendar.values()],
                "alerts": [alert.to_dict() for alert in self._alerts.values()]
            }
            json.dump(data, f, indent=2, default=str)

    def _generate_id(self, prefix: str = "CNT") -> str:
        """Generate a unique content ID."""
        return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

    def _get_stage_owner(self, stage: ContentStage, platform: Platform) -> str:
        """Get the agent that owns a particular stage."""
        owner = self.STAGE_OWNERS.get(stage)
        if owner is None:
            # Platform-specific stages
            if stage in (ContentStage.DRAFT, ContentStage.PUBLISHED):
                owner = self.PLATFORM_DRAFT_OWNERS.get(platform, "ContentManager")
        return owner or "ContentManager"

    def _calculate_days_in_stage(self, item: ContentItem) -> int:
        """Calculate how many days content has been in current stage."""
        stage_entered = datetime.fromisoformat(item.stage_entered_at)
        return (datetime.now() - stage_entered).days

    def _check_evidence_requirements(self, item: ContentItem) -> tuple[bool, str]:
        """Check if evidence requirements are met for drafting."""
        if item.stage == ContentStage.OUTLINE and not item.evidence_refs:
            return False, "No evidence references. Research required before drafting."
        if len(item.evidence_refs) < 2 and item.priority == Priority.HIGH:
            return False, "High-priority content requires at least 2 evidence references."
        return True, "Evidence requirements met."

    def _check_dependencies(self, item: ContentItem) -> tuple[bool, List[str]]:
        """Check if all dependencies are satisfied."""
        blocked_by = []
        for dep_id in item.dependencies:
            dep_item = self._content_items.get(dep_id)
            if dep_item and dep_item.stage not in (ContentStage.PUBLISHED, ContentStage.REPURPOSED):
                blocked_by.append(f"{dep_item.title} ({dep_id}) - currently {dep_item.stage.value}")
        return len(blocked_by) == 0, blocked_by

    # ========== Core Operations ==========

    def create_content(
        self,
        topic: str,
        title: Optional[str] = None,
        platform: Platform = Platform.ALL,
        priority: Priority = Priority.MEDIUM,
        deadline: Optional[str] = None,
        positioning_alignment: str = "",
        notes: Optional[str] = None
    ) -> AgentResponse:
        """
        Create a new content item in the pipeline.

        Does not create content directly - just registers the idea in the pipeline
        for tracking and coordination.
        """
        # Check state permissions
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # In YELLOW state, only allow high-priority content
        if self.alfred_state == AlfredState.YELLOW and priority != Priority.HIGH:
            return self.create_response(
                data={"status": "DEFERRED", "reason": "Only high-priority content in YELLOW state"},
                success=False,
                warnings=["Content creation deferred due to YELLOW state"]
            )

        # In RED state, no new content
        if self.alfred_state == AlfredState.RED:
            return self.blocked_response("No new content in RED state")

        content_id = self._generate_id()
        item = ContentItem(
            id=content_id,
            title=title or f"[Draft] {topic}",
            topic=topic,
            stage=ContentStage.IDEA,
            platform=platform,
            priority=priority,
            deadline=deadline,
            owner=self._get_stage_owner(ContentStage.IDEA, platform),
            next_action="Move to RESEARCH stage when ready",
            positioning_alignment=positioning_alignment,
            notes=[notes] if notes else []
        )

        self._content_items[content_id] = item
        self._save()

        return self.create_response(
            data={
                "status": "CREATED",
                "content_id": content_id,
                "item": item.to_dict(),
                "next_steps": [
                    "Assign to ResearchAgent for evidence gathering",
                    "Define positioning alignment if not specified",
                    "Set deadline if time-sensitive"
                ]
            }
        )

    def update_stage(
        self,
        content_id: str,
        new_stage: ContentStage,
        progress_pct: Optional[int] = None,
        blockers: Optional[List[str]] = None,
        next_action: Optional[str] = None,
        evidence_refs: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> AgentResponse:
        """
        Update a content item's stage in the pipeline.

        Enforces valid transitions and evidence requirements.
        """
        item = self._content_items.get(content_id)
        if not item:
            return self.create_response(
                data={"status": "NOT_FOUND", "content_id": content_id},
                success=False,
                errors=[f"Content item {content_id} not found"]
            )

        # Check if transition is valid
        valid_next_stages = self.VALID_TRANSITIONS.get(item.stage, set())
        if new_stage not in valid_next_stages:
            return self.create_response(
                data={
                    "status": "INVALID_TRANSITION",
                    "current_stage": item.stage.value,
                    "requested_stage": new_stage.value,
                    "valid_transitions": [s.value for s in valid_next_stages]
                },
                success=False,
                errors=[f"Cannot transition from {item.stage.value} to {new_stage.value}"]
            )

        # Check evidence requirements before drafting
        if new_stage == ContentStage.DRAFT:
            evidence_ok, evidence_msg = self._check_evidence_requirements(item)
            if not evidence_ok:
                return self.create_response(
                    data={
                        "status": "EVIDENCE_REQUIRED",
                        "message": evidence_msg,
                        "current_evidence": item.evidence_refs
                    },
                    success=False,
                    errors=[evidence_msg]
                )

        # Check dependencies
        deps_ok, blocked_by = self._check_dependencies(item)
        if not deps_ok and new_stage not in (ContentStage.KILLED, ContentStage.IDEA):
            return self.create_response(
                data={
                    "status": "DEPENDENCY_BLOCKED",
                    "blocked_by": blocked_by
                },
                success=False,
                errors=["Blocked by unfinished dependencies"],
                warnings=blocked_by
            )

        # Check state constraints
        if self.alfred_state == AlfredState.RED and new_stage in (
            ContentStage.DRAFT, ContentStage.REVIEW, ContentStage.SCHEDULED, ContentStage.PUBLISHED
        ):
            return self.blocked_response("Content progression blocked in RED state")

        if self.alfred_state == AlfredState.YELLOW and new_stage == ContentStage.PUBLISHED:
            return self.blocked_response("Publishing blocked in YELLOW state")

        # Update the item
        old_stage = item.stage
        item.stage = new_stage
        item.stage_entered_at = datetime.now().isoformat()
        item.updated_at = datetime.now().isoformat()
        item.owner = self._get_stage_owner(new_stage, item.platform)

        if progress_pct is not None:
            item.progress_pct = progress_pct
        if blockers is not None:
            item.blockers = blockers
        if next_action is not None:
            item.next_action = next_action
        if evidence_refs:
            item.evidence_refs.extend(evidence_refs)
        if notes:
            item.notes.append(f"[{datetime.now().strftime('%Y-%m-%d')}] {notes}")

        # Set default next actions based on stage
        if next_action is None:
            default_actions = {
                ContentStage.RESEARCH: "Gather evidence and citations",
                ContentStage.OUTLINE: "Define structure and key points",
                ContentStage.DRAFT: "Create content draft",
                ContentStage.REVIEW: "Awaiting Alfred approval",
                ContentStage.SCHEDULED: "Confirm publication time",
                ContentStage.PUBLISHED: "Monitor performance",
                ContentStage.REPURPOSED: "Track repurposed versions",
                ContentStage.KILLED: "Archive and document learnings"
            }
            item.next_action = default_actions.get(new_stage, "")

        # Clear any resolved alerts for this content
        self._resolve_alerts_for_content(content_id)

        self._save()

        return self.create_response(
            data={
                "status": "UPDATED",
                "content_id": content_id,
                "previous_stage": old_stage.value,
                "new_stage": new_stage.value,
                "owner": item.owner,
                "next_action": item.next_action,
                "item": item.to_dict()
            }
        )

    def check_pipeline(self) -> AgentResponse:
        """
        Check overall pipeline status and generate report.

        Returns counts and status for each stage, plus any concerns.
        """
        # Count items by stage
        stage_counts = {stage: 0 for stage in ContentStage}
        for item in self._content_items.values():
            stage_counts[item.stage] += 1

        # Count by platform
        platform_counts = {platform: 0 for platform in Platform}
        for item in self._content_items.values():
            if item.stage not in (ContentStage.KILLED, ContentStage.PUBLISHED, ContentStage.REPURPOSED):
                platform_counts[item.platform] += 1

        # Count by priority
        priority_counts = {priority: 0 for priority in Priority}
        active_stages = {ContentStage.IDEA, ContentStage.RESEARCH, ContentStage.OUTLINE,
                        ContentStage.DRAFT, ContentStage.REVIEW, ContentStage.SCHEDULED}
        for item in self._content_items.values():
            if item.stage in active_stages:
                priority_counts[item.priority] += 1

        # Identify bottlenecks
        bottlenecks = []
        if stage_counts[ContentStage.REVIEW] > 3:
            bottlenecks.append(f"Review queue backed up: {stage_counts[ContentStage.REVIEW]} items awaiting approval")
        if stage_counts[ContentStage.IDEA] > 10:
            bottlenecks.append(f"Idea backlog growing: {stage_counts[ContentStage.IDEA]} unprocessed ideas")
        if stage_counts[ContentStage.DRAFT] > 5:
            bottlenecks.append(f"Draft stage congested: {stage_counts[ContentStage.DRAFT]} items in draft")

        return self.create_response(
            data={
                "status": "PIPELINE_CHECK_COMPLETE",
                "pipeline_overview": {
                    "ideas": stage_counts[ContentStage.IDEA],
                    "in_research": stage_counts[ContentStage.RESEARCH],
                    "in_outline": stage_counts[ContentStage.OUTLINE],
                    "in_draft": stage_counts[ContentStage.DRAFT],
                    "in_review": stage_counts[ContentStage.REVIEW],
                    "scheduled": stage_counts[ContentStage.SCHEDULED],
                    "published": stage_counts[ContentStage.PUBLISHED],
                    "repurposed": stage_counts[ContentStage.REPURPOSED],
                    "killed": stage_counts[ContentStage.KILLED]
                },
                "by_platform": {p.value: c for p, c in platform_counts.items()},
                "by_priority": {p.value: c for p, c in priority_counts.items()},
                "total_active": sum(stage_counts[s] for s in active_stages),
                "bottlenecks": bottlenecks,
                "health": "HEALTHY" if not bottlenecks else "ATTENTION_NEEDED"
            }
        )

    def detect_stalls(self) -> AgentResponse:
        """
        Detect stalled content items and generate alerts.

        Checks each item against stage-specific thresholds.
        """
        stalled_items = []
        new_alerts = []

        for item in self._content_items.values():
            # Skip terminal stages
            if item.stage in (ContentStage.PUBLISHED, ContentStage.REPURPOSED, ContentStage.KILLED):
                continue

            days_in_stage = self._calculate_days_in_stage(item)
            threshold = self.STALL_THRESHOLDS.get(item.stage, 7)

            if days_in_stage >= threshold:
                # Create alert if not already exists
                alert_id = f"STALL-{item.id}"
                if alert_id not in self._alerts or self._alerts[alert_id].resolved:
                    alert = ContentAlert(
                        id=alert_id,
                        alert_type=AlertType.STALLED,
                        content_id=item.id,
                        content_title=item.title,
                        issue=f"Stalled in {item.stage.value} stage for {days_in_stage} days (threshold: {threshold})",
                        days_stalled=days_in_stage,
                        impact=self._assess_stall_impact(item),
                        recommended_action=self._recommend_stall_action(item)
                    )
                    self._alerts[alert_id] = alert
                    new_alerts.append(alert.to_dict())

                stalled_items.append({
                    "content_id": item.id,
                    "title": item.title,
                    "stage": item.stage.value,
                    "days_stalled": days_in_stage,
                    "threshold": threshold,
                    "blockers": item.blockers,
                    "owner": item.owner
                })

        # Check for deadline risks
        deadline_risks = self._check_deadline_risks()

        # Check for dependency blocks
        dependency_blocks = self._check_dependency_blocks()

        self._save()

        return self.create_response(
            data={
                "status": "STALL_CHECK_COMPLETE",
                "stalled_count": len(stalled_items),
                "stalled_items": stalled_items,
                "new_alerts": new_alerts,
                "deadline_risks": deadline_risks,
                "dependency_blocks": dependency_blocks,
                "total_active_alerts": len([a for a in self._alerts.values() if not a.resolved])
            },
            warnings=[f"Stalled: {item['title']}" for item in stalled_items[:5]]
        )

    def _assess_stall_impact(self, item: ContentItem) -> str:
        """Assess the impact of a stalled item."""
        impacts = []

        if item.priority == Priority.HIGH:
            impacts.append("High-priority content delayed")

        if item.deadline:
            deadline_dt = datetime.fromisoformat(item.deadline)
            if deadline_dt < datetime.now() + timedelta(days=7):
                impacts.append("Deadline at risk")

        # Check if other content depends on this
        dependent_content = [
            c for c in self._content_items.values()
            if item.id in c.dependencies and c.stage not in (ContentStage.KILLED, ContentStage.PUBLISHED)
        ]
        if dependent_content:
            impacts.append(f"Blocking {len(dependent_content)} other content item(s)")

        return "; ".join(impacts) if impacts else "Minimal immediate impact"

    def _recommend_stall_action(self, item: ContentItem) -> str:
        """Recommend action for stalled content."""
        if item.blockers:
            return f"Address blockers: {', '.join(item.blockers[:3])}"

        actions = {
            ContentStage.IDEA: "Move to RESEARCH or KILL if no longer relevant",
            ContentStage.RESEARCH: "Complete research or document why blocked",
            ContentStage.OUTLINE: "Finalize outline or return to RESEARCH if evidence insufficient",
            ContentStage.DRAFT: "Complete draft or escalate resource constraints",
            ContentStage.REVIEW: "Request Alfred review urgently",
            ContentStage.SCHEDULED: "Confirm publication readiness"
        }

        return actions.get(item.stage, "Review and update status")

    def _check_deadline_risks(self) -> List[Dict[str, Any]]:
        """Check for items at risk of missing deadlines."""
        risks = []
        now = datetime.now()

        for item in self._content_items.values():
            if not item.deadline or item.stage in (ContentStage.PUBLISHED, ContentStage.KILLED):
                continue

            deadline_dt = datetime.fromisoformat(item.deadline)
            days_until_deadline = (deadline_dt - now).days

            # Estimate days needed based on current stage
            stages_remaining = self._stages_until_published(item.stage)
            estimated_days_needed = stages_remaining * 3  # Rough estimate

            if days_until_deadline < estimated_days_needed:
                risk = {
                    "content_id": item.id,
                    "title": item.title,
                    "deadline": item.deadline,
                    "days_remaining": days_until_deadline,
                    "estimated_days_needed": estimated_days_needed,
                    "current_stage": item.stage.value,
                    "risk_level": "HIGH" if days_until_deadline < 3 else "MEDIUM"
                }
                risks.append(risk)

                # Create alert
                alert_id = f"DEADLINE-{item.id}"
                if alert_id not in self._alerts or self._alerts[alert_id].resolved:
                    self._alerts[alert_id] = ContentAlert(
                        id=alert_id,
                        alert_type=AlertType.DEADLINE_RISK,
                        content_id=item.id,
                        content_title=item.title,
                        issue=f"Deadline in {days_until_deadline} days, estimated {estimated_days_needed} days needed",
                        impact=f"May miss {item.deadline} deadline",
                        recommended_action="Expedite or negotiate deadline extension"
                    )

        return risks

    def _check_dependency_blocks(self) -> List[Dict[str, Any]]:
        """Check for items blocked by dependencies."""
        blocks = []

        for item in self._content_items.values():
            if item.stage in (ContentStage.PUBLISHED, ContentStage.KILLED, ContentStage.REPURPOSED):
                continue

            deps_ok, blocked_by = self._check_dependencies(item)
            if not deps_ok:
                block = {
                    "content_id": item.id,
                    "title": item.title,
                    "stage": item.stage.value,
                    "blocked_by": blocked_by
                }
                blocks.append(block)

                # Create alert
                alert_id = f"DEPBLOCK-{item.id}"
                if alert_id not in self._alerts or self._alerts[alert_id].resolved:
                    self._alerts[alert_id] = ContentAlert(
                        id=alert_id,
                        alert_type=AlertType.DEPENDENCY_BLOCKED,
                        content_id=item.id,
                        content_title=item.title,
                        issue=f"Blocked by {len(blocked_by)} dependency(ies)",
                        impact="Cannot progress until dependencies complete",
                        recommended_action=f"Prioritize: {blocked_by[0] if blocked_by else 'unknown'}"
                    )

        return blocks

    def _stages_until_published(self, current_stage: ContentStage) -> int:
        """Count stages remaining until PUBLISHED."""
        stage_order = [
            ContentStage.IDEA, ContentStage.RESEARCH, ContentStage.OUTLINE,
            ContentStage.DRAFT, ContentStage.REVIEW, ContentStage.SCHEDULED,
            ContentStage.PUBLISHED
        ]
        try:
            current_idx = stage_order.index(current_stage)
            published_idx = stage_order.index(ContentStage.PUBLISHED)
            return published_idx - current_idx
        except ValueError:
            return 0

    def _resolve_alerts_for_content(self, content_id: str):
        """Mark alerts as resolved when content progresses."""
        for alert in self._alerts.values():
            if alert.content_id == content_id and not alert.resolved:
                alert.resolved = True

    def schedule_publication(
        self,
        content_id: str,
        scheduled_datetime: str,
        notes: str = ""
    ) -> AgentResponse:
        """
        Schedule a content item for publication.

        Requires content to be in REVIEW stage with Alfred approval.
        """
        item = self._content_items.get(content_id)
        if not item:
            return self.create_response(
                data={"status": "NOT_FOUND"},
                success=False,
                errors=[f"Content item {content_id} not found"]
            )

        # Must be in REVIEW stage
        if item.stage != ContentStage.REVIEW:
            return self.create_response(
                data={
                    "status": "INVALID_STATE",
                    "current_stage": item.stage.value,
                    "required_stage": ContentStage.REVIEW.value
                },
                success=False,
                errors=["Content must be in REVIEW stage to schedule"]
            )

        # Check state constraints
        if self.alfred_state != AlfredState.GREEN:
            return self.blocked_response(f"Scheduling blocked in {self.alfred_state.value} state")

        # Parse scheduled datetime
        try:
            scheduled_dt = datetime.fromisoformat(scheduled_datetime)
            scheduled_date = scheduled_dt.strftime("%Y-%m-%d")
            scheduled_time = scheduled_dt.strftime("%H:%M")
        except ValueError as e:
            return self.create_response(
                data={"status": "INVALID_DATETIME"},
                success=False,
                errors=[f"Invalid datetime format: {e}"]
            )

        # Create calendar entry
        entry_id = self._generate_id("CAL")
        calendar_entry = PublishingCalendarEntry(
            id=entry_id,
            content_id=content_id,
            title=item.title,
            platform=item.platform,
            scheduled_date=scheduled_date,
            scheduled_time=scheduled_time,
            notes=notes
        )
        self._calendar[entry_id] = calendar_entry

        # Update content item
        item.stage = ContentStage.SCHEDULED
        item.stage_entered_at = datetime.now().isoformat()
        item.updated_at = datetime.now().isoformat()
        item.scheduled_time = scheduled_datetime
        item.owner = self._get_stage_owner(ContentStage.SCHEDULED, item.platform)
        item.next_action = f"Publish on {scheduled_date} at {scheduled_time}"
        item.notes.append(f"[{datetime.now().strftime('%Y-%m-%d')}] Scheduled for {scheduled_datetime}")

        self._save()

        return self.create_response(
            data={
                "status": "SCHEDULED",
                "content_id": content_id,
                "calendar_entry_id": entry_id,
                "scheduled_date": scheduled_date,
                "scheduled_time": scheduled_time,
                "platform": item.platform.value,
                "calendar_entry": calendar_entry.to_dict(),
                "item": item.to_dict()
            }
        )

    def generate_status(
        self,
        include_calendar: bool = True,
        include_alerts: bool = True,
        days_ahead: int = 14
    ) -> AgentResponse:
        """
        Generate comprehensive content status report.

        Follows the CONTENT_STATUS output format specification.
        """
        now = datetime.now()

        # Run stall detection first
        self.detect_stalls()

        # Pipeline overview
        stage_counts = {stage: 0 for stage in ContentStage}
        for item in self._content_items.values():
            stage_counts[item.stage] += 1

        pipeline_overview = {
            "ideas": stage_counts[ContentStage.IDEA],
            "in_research": stage_counts[ContentStage.RESEARCH],
            "in_outline": stage_counts[ContentStage.OUTLINE],
            "in_draft": stage_counts[ContentStage.DRAFT],
            "in_review": stage_counts[ContentStage.REVIEW],
            "scheduled": stage_counts[ContentStage.SCHEDULED],
            "published_total": stage_counts[ContentStage.PUBLISHED],
            "repurposed_total": stage_counts[ContentStage.REPURPOSED]
        }

        # Active content items
        active_stages = {
            ContentStage.IDEA, ContentStage.RESEARCH, ContentStage.OUTLINE,
            ContentStage.DRAFT, ContentStage.REVIEW, ContentStage.SCHEDULED
        }
        active_items = []
        for item in self._content_items.values():
            if item.stage in active_stages:
                active_items.append({
                    "title": item.title,
                    "stage": item.stage.value,
                    "platform": item.platform.value,
                    "owner": item.owner,
                    "progress": f"{item.progress_pct}%",
                    "blockers": item.blockers,
                    "next_action": item.next_action,
                    "days_in_stage": self._calculate_days_in_stage(item),
                    "priority": item.priority.value,
                    "deadline": item.deadline
                })

        # Sort by priority and days in stage
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        active_items.sort(key=lambda x: (priority_order.get(Priority(x["priority"]), 1), -x["days_in_stage"]))

        # Publishing calendar
        calendar_entries = []
        if include_calendar:
            cutoff_date = now + timedelta(days=days_ahead)
            for entry in self._calendar.values():
                if entry.status != "cancelled":
                    entry_date = datetime.strptime(entry.scheduled_date, "%Y-%m-%d")
                    if entry_date <= cutoff_date:
                        calendar_entries.append({
                            "date": entry.scheduled_date,
                            "time": entry.scheduled_time,
                            "title": entry.title,
                            "platform": entry.platform.value,
                            "status": entry.status
                        })
            calendar_entries.sort(key=lambda x: (x["date"], x["time"]))

        # Stalled items
        stalled_items = []
        for item in self._content_items.values():
            if item.stage in active_stages:
                days = self._calculate_days_in_stage(item)
                threshold = self.STALL_THRESHOLDS.get(item.stage, 7)
                if days >= threshold:
                    stalled_items.append({
                        "title": item.title,
                        "stage": item.stage.value,
                        "days_stalled": days,
                        "threshold": threshold
                    })

        # Active alerts
        active_alerts = []
        if include_alerts:
            for alert in self._alerts.values():
                if not alert.resolved:
                    active_alerts.append(alert.to_dict())

        # Cross-platform coordination
        cross_platform = self._analyze_cross_platform_coordination()

        # Recommendations
        recommendations = self._generate_recommendations(
            pipeline_overview, stalled_items, active_alerts
        )

        # Format the status report
        status_report = {
            "report_date": now.isoformat(),
            "alfred_state": self.alfred_state.value,
            "pipeline_overview": pipeline_overview,
            "active_content_items": active_items,
            "publishing_calendar": calendar_entries,
            "stalled_items": stalled_items,
            "active_alerts": active_alerts,
            "cross_platform_coordination": cross_platform,
            "recommendations": recommendations
        }

        return self.create_response(
            data={
                "status": "STATUS_GENERATED",
                "report": status_report,
                "summary": {
                    "total_active": len(active_items),
                    "stalled_count": len(stalled_items),
                    "alert_count": len(active_alerts),
                    "upcoming_publications": len(calendar_entries)
                }
            },
            warnings=[f"Stalled: {s['title']}" for s in stalled_items]
        )

    def _analyze_cross_platform_coordination(self) -> Dict[str, Any]:
        """Analyze cross-platform content coordination."""
        # Find content that should be repurposed
        repurpose_candidates = []
        for item in self._content_items.values():
            if item.stage == ContentStage.PUBLISHED and not item.repurposed_to:
                if item.platform != Platform.ALL:
                    repurpose_candidates.append({
                        "content_id": item.id,
                        "title": item.title,
                        "source_platform": item.platform.value,
                        "suggested_platforms": self._suggest_repurpose_platforms(item)
                    })

        # Find related content across platforms
        topic_clusters = {}
        for item in self._content_items.values():
            if item.stage not in (ContentStage.KILLED,):
                topic_key = item.topic.lower()[:30]  # Simple topic clustering
                if topic_key not in topic_clusters:
                    topic_clusters[topic_key] = []
                topic_clusters[topic_key].append({
                    "id": item.id,
                    "title": item.title,
                    "platform": item.platform.value,
                    "stage": item.stage.value
                })

        # Find multi-platform clusters
        multi_platform_topics = {
            k: v for k, v in topic_clusters.items()
            if len(set(item["platform"] for item in v)) > 1 or len(v) > 1
        }

        return {
            "repurpose_candidates": repurpose_candidates[:5],
            "multi_platform_topics": list(multi_platform_topics.keys())[:5],
            "platform_balance": self._assess_platform_balance()
        }

    def _suggest_repurpose_platforms(self, item: ContentItem) -> List[str]:
        """Suggest platforms for repurposing content."""
        all_platforms = {Platform.SUBSTACK, Platform.TWITTER, Platform.YOUTUBE}
        current = {item.platform}
        if item.platform == Platform.ALL:
            return []
        return [p.value for p in (all_platforms - current)]

    def _assess_platform_balance(self) -> Dict[str, Any]:
        """Assess content balance across platforms."""
        active_stages = {
            ContentStage.IDEA, ContentStage.RESEARCH, ContentStage.OUTLINE,
            ContentStage.DRAFT, ContentStage.REVIEW, ContentStage.SCHEDULED
        }

        platform_counts = {p: 0 for p in Platform}
        for item in self._content_items.values():
            if item.stage in active_stages:
                platform_counts[item.platform] += 1

        total = sum(platform_counts.values())
        if total == 0:
            return {"balanced": True, "distribution": {}}

        distribution = {p.value: count / total * 100 for p, count in platform_counts.items()}

        # Check for imbalances
        imbalances = []
        for platform, pct in distribution.items():
            if platform != "all" and pct > 60:
                imbalances.append(f"{platform} over-weighted at {pct:.0f}%")
            elif platform != "all" and pct < 10 and platform_counts[Platform(platform)] == 0:
                imbalances.append(f"{platform} has no active content")

        return {
            "balanced": len(imbalances) == 0,
            "distribution": distribution,
            "imbalances": imbalances
        }

    def _generate_recommendations(
        self,
        pipeline_overview: Dict[str, int],
        stalled_items: List[Dict],
        active_alerts: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations based on pipeline state."""
        recommendations = []

        # Review queue recommendations
        if pipeline_overview["in_review"] > 3:
            recommendations.append(
                f"URGENT: {pipeline_overview['in_review']} items awaiting review. "
                "Schedule review session to prevent bottleneck."
            )

        # Idea backlog
        if pipeline_overview["ideas"] > 10:
            recommendations.append(
                f"Idea backlog at {pipeline_overview['ideas']}. "
                "Consider pruning low-value ideas or moving promising ones to research."
            )

        # Stalled content
        if stalled_items:
            high_priority_stalled = [s for s in stalled_items if s.get("days_stalled", 0) > 14]
            if high_priority_stalled:
                recommendations.append(
                    f"{len(high_priority_stalled)} item(s) stalled over 2 weeks. "
                    "Review for blockers or consider killing."
                )

        # Pipeline balance
        if pipeline_overview["in_draft"] > pipeline_overview["in_review"] * 2:
            recommendations.append(
                "Draft queue building up. Ensure drafts are review-ready before creating more."
            )

        # Empty stages
        if pipeline_overview["in_research"] == 0 and pipeline_overview["ideas"] > 0:
            recommendations.append(
                "No content in research. Move an idea to research to maintain pipeline flow."
            )

        # Deadline alerts
        deadline_alerts = [a for a in active_alerts if a.get("alert_type") == "deadline_risk"]
        if deadline_alerts:
            recommendations.append(
                f"{len(deadline_alerts)} item(s) at risk of missing deadlines. "
                "Review and reprioritize or negotiate extensions."
            )

        # State-specific recommendations
        if self.alfred_state == AlfredState.YELLOW:
            recommendations.append(
                "YELLOW state active. Focus on completing in-progress content rather than starting new."
            )
        elif self.alfred_state == AlfredState.RED:
            recommendations.append(
                "RED state active. All content progression paused. Focus on threat resolution."
            )

        return recommendations if recommendations else ["Pipeline healthy. Continue normal operations."]

    # ========== Additional Utility Methods ==========

    def get_content(self, content_id: str) -> AgentResponse:
        """Get details of a specific content item."""
        item = self._content_items.get(content_id)
        if not item:
            return self.create_response(
                data={"status": "NOT_FOUND"},
                success=False,
                errors=[f"Content item {content_id} not found"]
            )

        return self.create_response(
            data={
                "status": "FOUND",
                "item": item.to_dict(),
                "days_in_stage": self._calculate_days_in_stage(item),
                "evidence_status": self._check_evidence_requirements(item),
                "dependency_status": self._check_dependencies(item)
            }
        )

    def list_content(
        self,
        stage: Optional[ContentStage] = None,
        platform: Optional[Platform] = None,
        priority: Optional[Priority] = None,
        include_killed: bool = False
    ) -> AgentResponse:
        """List content items with optional filters."""
        items = []

        for item in self._content_items.values():
            # Apply filters
            if stage and item.stage != stage:
                continue
            if platform and item.platform != platform:
                continue
            if priority and item.priority != priority:
                continue
            if not include_killed and item.stage == ContentStage.KILLED:
                continue

            items.append({
                "id": item.id,
                "title": item.title,
                "stage": item.stage.value,
                "platform": item.platform.value,
                "priority": item.priority.value,
                "days_in_stage": self._calculate_days_in_stage(item),
                "owner": item.owner
            })

        # Sort by priority and days in stage
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        items.sort(key=lambda x: (priority_order.get(Priority(x["priority"]), 1), -x["days_in_stage"]))

        return self.create_response(
            data={
                "status": "LIST_COMPLETE",
                "count": len(items),
                "items": items,
                "filters_applied": {
                    "stage": stage.value if stage else None,
                    "platform": platform.value if platform else None,
                    "priority": priority.value if priority else None
                }
            }
        )

    def add_dependency(self, content_id: str, depends_on: str) -> AgentResponse:
        """Add a dependency between content items."""
        item = self._content_items.get(content_id)
        dependency = self._content_items.get(depends_on)

        if not item:
            return self.create_response(
                data={"status": "NOT_FOUND"},
                success=False,
                errors=[f"Content item {content_id} not found"]
            )

        if not dependency:
            return self.create_response(
                data={"status": "DEPENDENCY_NOT_FOUND"},
                success=False,
                errors=[f"Dependency {depends_on} not found"]
            )

        if depends_on in item.dependencies:
            return self.create_response(
                data={"status": "ALREADY_EXISTS"},
                success=False,
                warnings=["Dependency already exists"]
            )

        # Check for circular dependency
        if content_id in dependency.dependencies:
            return self.create_response(
                data={"status": "CIRCULAR_DEPENDENCY"},
                success=False,
                errors=["Cannot create circular dependency"]
            )

        item.dependencies.append(depends_on)
        item.updated_at = datetime.now().isoformat()
        self._save()

        return self.create_response(
            data={
                "status": "DEPENDENCY_ADDED",
                "content_id": content_id,
                "depends_on": depends_on,
                "all_dependencies": item.dependencies
            }
        )

    def add_evidence(self, content_id: str, evidence_refs: List[str]) -> AgentResponse:
        """Add evidence references to content item."""
        item = self._content_items.get(content_id)
        if not item:
            return self.create_response(
                data={"status": "NOT_FOUND"},
                success=False,
                errors=[f"Content item {content_id} not found"]
            )

        item.evidence_refs.extend(evidence_refs)
        item.updated_at = datetime.now().isoformat()
        self._save()

        evidence_status = self._check_evidence_requirements(item)

        return self.create_response(
            data={
                "status": "EVIDENCE_ADDED",
                "content_id": content_id,
                "total_evidence": len(item.evidence_refs),
                "evidence_refs": item.evidence_refs,
                "evidence_requirements_met": evidence_status[0],
                "evidence_message": evidence_status[1]
            }
        )

    def kill_content(self, content_id: str, reason: str) -> AgentResponse:
        """Kill/cancel a content item."""
        item = self._content_items.get(content_id)
        if not item:
            return self.create_response(
                data={"status": "NOT_FOUND"},
                success=False,
                errors=[f"Content item {content_id} not found"]
            )

        if item.stage == ContentStage.PUBLISHED:
            return self.create_response(
                data={"status": "CANNOT_KILL_PUBLISHED"},
                success=False,
                errors=["Cannot kill published content"]
            )

        old_stage = item.stage
        item.stage = ContentStage.KILLED
        item.stage_entered_at = datetime.now().isoformat()
        item.updated_at = datetime.now().isoformat()
        item.notes.append(f"[{datetime.now().strftime('%Y-%m-%d')}] KILLED: {reason}")

        # Resolve any alerts
        self._resolve_alerts_for_content(content_id)

        # Remove from calendar if scheduled
        calendar_to_remove = [
            entry_id for entry_id, entry in self._calendar.items()
            if entry.content_id == content_id
        ]
        for entry_id in calendar_to_remove:
            self._calendar[entry_id].status = "cancelled"

        self._save()

        return self.create_response(
            data={
                "status": "KILLED",
                "content_id": content_id,
                "previous_stage": old_stage.value,
                "reason": reason,
                "calendar_entries_cancelled": len(calendar_to_remove)
            }
        )

    def repurpose_content(
        self,
        source_content_id: str,
        target_platform: Platform,
        new_title: Optional[str] = None
    ) -> AgentResponse:
        """Create a new content item by repurposing existing published content."""
        source = self._content_items.get(source_content_id)
        if not source:
            return self.create_response(
                data={"status": "NOT_FOUND"},
                success=False,
                errors=[f"Source content {source_content_id} not found"]
            )

        if source.stage != ContentStage.PUBLISHED:
            return self.create_response(
                data={"status": "NOT_PUBLISHED"},
                success=False,
                errors=["Can only repurpose published content"]
            )

        if target_platform == source.platform:
            return self.create_response(
                data={"status": "SAME_PLATFORM"},
                success=False,
                errors=["Target platform must be different from source"]
            )

        # Create new content item
        new_id = self._generate_id()
        new_item = ContentItem(
            id=new_id,
            title=new_title or f"[Repurposed] {source.title}",
            topic=source.topic,
            stage=ContentStage.OUTLINE,  # Start at outline since research is done
            platform=target_platform,
            priority=source.priority,
            owner=self._get_stage_owner(ContentStage.OUTLINE, target_platform),
            evidence_refs=source.evidence_refs.copy(),  # Inherit evidence
            repurposed_from=source_content_id,
            positioning_alignment=source.positioning_alignment,
            next_action="Adapt outline for target platform",
            notes=[f"[{datetime.now().strftime('%Y-%m-%d')}] Repurposed from {source.title} ({source_content_id})"]
        )

        # Update source to track repurposing
        source.repurposed_to.append(new_id)
        source.stage = ContentStage.REPURPOSED
        source.updated_at = datetime.now().isoformat()

        self._content_items[new_id] = new_item
        self._save()

        return self.create_response(
            data={
                "status": "REPURPOSED",
                "new_content_id": new_id,
                "source_content_id": source_content_id,
                "target_platform": target_platform.value,
                "new_item": new_item.to_dict()
            }
        )

    def process_request(self, request: ContentRequest) -> AgentResponse:
        """Process a content request based on its type."""
        if request.request_type == RequestType.STATUS:
            if request.content_id:
                return self.get_content(request.content_id)
            return self.generate_status()

        elif request.request_type == RequestType.CREATE:
            return self.create_content(
                topic=request.topic or "Untitled",
                title=request.title,
                platform=request.platform_target,
                priority=request.priority,
                deadline=request.deadline,
                notes=request.notes
            )

        elif request.request_type == RequestType.UPDATE:
            if not request.content_id:
                return self.create_response(
                    data={"status": "MISSING_CONTENT_ID"},
                    success=False,
                    errors=["Content ID required for update"]
                )
            if request.new_stage:
                return self.update_stage(
                    content_id=request.content_id,
                    new_stage=request.new_stage,
                    notes=request.notes
                )
            return self.get_content(request.content_id)

        elif request.request_type == RequestType.SCHEDULE:
            if not request.content_id or not request.scheduled_time:
                return self.create_response(
                    data={"status": "MISSING_PARAMS"},
                    success=False,
                    errors=["Content ID and scheduled time required"]
                )
            return self.schedule_publication(
                content_id=request.content_id,
                scheduled_datetime=request.scheduled_time,
                notes=request.notes or ""
            )

        elif request.request_type == RequestType.KILL:
            if not request.content_id:
                return self.create_response(
                    data={"status": "MISSING_CONTENT_ID"},
                    success=False,
                    errors=["Content ID required to kill"]
                )
            return self.kill_content(
                content_id=request.content_id,
                reason=request.notes or "No reason provided"
            )

        return self.create_response(
            data={"status": "UNKNOWN_REQUEST_TYPE"},
            success=False,
            errors=[f"Unknown request type: {request.request_type}"]
        )

    def get_calendar(self, days_ahead: int = 30) -> AgentResponse:
        """Get the publishing calendar for upcoming days."""
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)

        entries = []
        for entry in self._calendar.values():
            if entry.status != "cancelled":
                try:
                    entry_date = datetime.strptime(entry.scheduled_date, "%Y-%m-%d")
                    if now.date() <= entry_date.date() <= cutoff.date():
                        entries.append(entry.to_dict())
                except ValueError:
                    continue

        entries.sort(key=lambda x: (x["scheduled_date"], x["scheduled_time"]))

        return self.create_response(
            data={
                "status": "CALENDAR_RETRIEVED",
                "period": f"Next {days_ahead} days",
                "entry_count": len(entries),
                "entries": entries
            }
        )

    def format_status_report(self) -> str:
        """Format a human-readable status report following the specification."""
        response = self.generate_status()
        if not response.success:
            return f"Error generating status: {response.errors}"

        report = response.data.get("report", {})

        lines = [
            "CONTENT_STATUS",
            f"- Report Date: {report.get('report_date', 'Unknown')}",
            f"- Alfred State: {report.get('alfred_state', 'Unknown')}",
            "",
            "Pipeline Overview:",
        ]

        overview = report.get("pipeline_overview", {})
        lines.append(f"  Ideas: {overview.get('ideas', 0)}")
        lines.append(f"  In Research: {overview.get('in_research', 0)}")
        lines.append(f"  In Outline: {overview.get('in_outline', 0)}")
        lines.append(f"  In Draft: {overview.get('in_draft', 0)}")
        lines.append(f"  In Review: {overview.get('in_review', 0)}")
        lines.append(f"  Scheduled: {overview.get('scheduled', 0)}")

        lines.append("")
        lines.append("Active Content Items:")
        for item in report.get("active_content_items", [])[:10]:
            lines.append(f"  - {item['title']}")
            lines.append(f"    Stage: {item['stage']} | Platform: {item['platform']} | Owner: {item['owner']}")
            lines.append(f"    Progress: {item['progress']} | Next: {item['next_action']}")
            if item['blockers']:
                lines.append(f"    Blockers: {', '.join(item['blockers'])}")

        lines.append("")
        lines.append("Publishing Calendar:")
        for entry in report.get("publishing_calendar", [])[:10]:
            lines.append(f"  - {entry['date']} {entry['time']}: {entry['title']} ({entry['platform']})")

        if report.get("stalled_items"):
            lines.append("")
            lines.append("Stalled Items:")
            for item in report["stalled_items"]:
                lines.append(f"  - {item['title']}: {item['days_stalled']} days in {item['stage']}")

        lines.append("")
        lines.append("Cross-Platform Coordination:")
        coord = report.get("cross_platform_coordination", {})
        if coord.get("repurpose_candidates"):
            lines.append("  Repurpose candidates:")
            for candidate in coord["repurpose_candidates"][:3]:
                lines.append(f"    - {candidate['title']} -> {', '.join(candidate['suggested_platforms'])}")

        lines.append("")
        lines.append("Recommendations:")
        for rec in report.get("recommendations", []):
            lines.append(f"  - {rec}")

        return "\n".join(lines)


# Convenience function for creating a ContentManager instance
def create_content_manager(storage_path: Optional[Path] = None) -> ContentManager:
    """Create and return a ContentManager instance."""
    return ContentManager(storage_path=storage_path)
