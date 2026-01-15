"""
Weekly Brief Automation System for Alfred

Generates Alfred's weekly strategic brief (20 minutes).
Aggregates data from multiple sub-agents and memory systems to produce
a comprehensive weekly overview with strategic recommendations.

Weekly Strategic Brief Contents:
- Reputation trajectory: Stable / Improving / At Risk
- Drift check: one sentence about positioning drift
- What worked / didn't (max 3 bullets each)
- Next week's constraints (non-negotiable)
- One question Alfred needs answered (directional)
- Content plan for next week
- Learning queue for next week
- Builder work: explicitly scheduled or explicitly banned

Data Sources:
- Content Strategy Analyst
- Social Metrics Harvester
- Audience Signals Extractor
- Shipping Governor
- Financial Sentinel
- Pattern Registry (memory)

Does NOT:
- Generate content directly
- Make final decisions for Alfred
- Override other agent recommendations
- Ignore data from constituent agents
- Cherry-pick only positive signals
- Dismiss concerning patterns

Does:
- Aggregate weekly data from all relevant agents
- Synthesize cross-agent insights
- Generate strategic recommendations
- Track week-over-week trends
- Surface the most critical information
- Provide directional questions for Alfred
- Explicitly schedule or ban builder work
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json

from . import StrategyAgent, AgentResponse, AlfredState


# =============================================================================
# ENUMS
# =============================================================================

class ReputationTrajectory(Enum):
    """Overall reputation trajectory assessment."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    AT_RISK = "AT_RISK"


class DriftSeverity(Enum):
    """Severity of positioning drift."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class BuilderWorkStatus(Enum):
    """Status of builder work for the week."""
    SCHEDULED = "SCHEDULED"     # Explicitly scheduled with deliverables
    BANNED = "BANNED"           # Explicitly banned due to shipping issues
    CONDITIONAL = "CONDITIONAL"  # Allowed only if conditions are met


class ConstraintType(Enum):
    """Types of constraints for the week."""
    SHIPPING = "shipping"       # Must ship something
    RECOVERY = "recovery"       # Must address reputation issue
    FINANCIAL = "financial"     # Budget constraints
    HEALTH = "health"           # Energy/capacity constraints
    DEADLINE = "deadline"       # External deadline
    STRATEGIC = "strategic"     # Strategic decision required


class TrendDirection(Enum):
    """Direction of week-over-week trends."""
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    NEW = "new"  # No previous data to compare


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WeeklyMetrics:
    """Aggregated metrics for the week."""
    total_content_published: int = 0
    total_reach: int = 0
    total_engagement: int = 0
    engagement_rate: float = 0.0
    follower_change: int = 0

    # Deltas from previous week
    reach_delta: Optional[float] = None
    engagement_delta: Optional[float] = None
    follower_delta: Optional[float] = None

    # Platform breakdown
    platform_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_content_published": self.total_content_published,
            "total_reach": self.total_reach,
            "total_engagement": self.total_engagement,
            "engagement_rate": round(self.engagement_rate, 2),
            "follower_change": self.follower_change,
            "reach_delta_percent": round(self.reach_delta, 1) if self.reach_delta else None,
            "engagement_delta_percent": round(self.engagement_delta, 1) if self.engagement_delta else None,
            "follower_delta": self.follower_delta,
            "platform_metrics": self.platform_metrics
        }


@dataclass
class ContentItem:
    """Represents a content item for the brief."""
    title: str
    platform: str
    performance: str  # "top", "average", "underperformed"
    key_insight: str
    engagement_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "platform": self.platform,
            "performance": self.performance,
            "key_insight": self.key_insight,
            "engagement_rate": self.engagement_rate
        }


@dataclass
class WeeklyConstraint:
    """A non-negotiable constraint for the week."""
    constraint_type: ConstraintType
    description: str
    deadline: Optional[date] = None
    source: Optional[str] = None  # Which agent/system identified this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.constraint_type.value,
            "description": self.description,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "source": self.source
        }


@dataclass
class LearningItem:
    """An item for the learning queue."""
    topic: str
    priority: str  # "high", "medium", "low"
    linked_output: Optional[str] = None  # What content this learning supports
    estimated_time_hours: Optional[float] = None
    source: Optional[str] = None  # Why this was added to queue

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "priority": self.priority,
            "linked_output": self.linked_output,
            "estimated_time_hours": self.estimated_time_hours,
            "source": self.source
        }


@dataclass
class ContentPlanItem:
    """A planned content item for next week."""
    title: str
    platform: str
    content_type: str  # "article", "video", "thread", etc.
    pillar: Optional[str] = None
    target_publish_date: Optional[date] = None
    status: str = "planned"  # "planned", "in_progress", "ready"
    priority: str = "medium"
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "platform": self.platform,
            "content_type": self.content_type,
            "pillar": self.pillar,
            "target_publish_date": self.target_publish_date.isoformat() if self.target_publish_date else None,
            "status": self.status,
            "priority": self.priority,
            "rationale": self.rationale
        }


@dataclass
class ContentPlan:
    """Content plan for the upcoming week."""
    week_start: date
    week_end: date
    items: List[ContentPlanItem] = field(default_factory=list)
    total_planned: int = 0
    pillar_distribution: Dict[str, int] = field(default_factory=dict)
    platform_distribution: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def add_item(self, item: ContentPlanItem):
        """Add an item to the content plan."""
        self.items.append(item)
        self.total_planned = len(self.items)

        # Update distributions
        if item.pillar:
            self.pillar_distribution[item.pillar] = self.pillar_distribution.get(item.pillar, 0) + 1
        self.platform_distribution[item.platform] = self.platform_distribution.get(item.platform, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week_start": self.week_start.isoformat(),
            "week_end": self.week_end.isoformat(),
            "items": [item.to_dict() for item in self.items],
            "total_planned": self.total_planned,
            "pillar_distribution": self.pillar_distribution,
            "platform_distribution": self.platform_distribution,
            "notes": self.notes
        }


@dataclass
class BuilderWorkAssignment:
    """Builder work assignment for the week."""
    status: BuilderWorkStatus
    rationale: str
    scheduled_items: List[Dict[str, str]] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)  # For CONDITIONAL status
    shipping_health: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "rationale": self.rationale,
            "scheduled_items": self.scheduled_items,
            "conditions": self.conditions,
            "shipping_health": self.shipping_health
        }


@dataclass
class StrategicSummary:
    """Strategic summary section of the weekly brief."""
    reputation_trajectory: ReputationTrajectory
    trajectory_rationale: str
    drift_severity: DriftSeverity
    drift_description: str  # One sentence about positioning drift
    alignment_score: Optional[float] = None
    week_over_week_trend: TrendDirection = TrendDirection.FLAT
    key_risk: Optional[str] = None
    key_opportunity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reputation_trajectory": self.reputation_trajectory.value,
            "trajectory_rationale": self.trajectory_rationale,
            "drift_severity": self.drift_severity.value,
            "drift_description": self.drift_description,
            "alignment_score": round(self.alignment_score, 1) if self.alignment_score else None,
            "week_over_week_trend": self.week_over_week_trend.value,
            "key_risk": self.key_risk,
            "key_opportunity": self.key_opportunity
        }


@dataclass
class PatternAlert:
    """Alert about a behavioral pattern from Pattern Registry."""
    pattern_type: str
    description: str
    trajectory: str
    severity: int
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "trajectory": self.trajectory,
            "severity": self.severity,
            "recommended_action": self.recommended_action
        }


@dataclass
class WeeklyBrief:
    """The complete weekly strategic brief."""
    brief_id: str
    generated_at: datetime
    week_start: date
    week_end: date

    # Strategic Summary
    strategic_summary: StrategicSummary

    # What worked / didn't (max 3 each)
    what_worked: List[str] = field(default_factory=list)
    what_didnt_work: List[str] = field(default_factory=list)

    # Constraints
    next_week_constraints: List[WeeklyConstraint] = field(default_factory=list)

    # The question
    directional_question: str = ""
    question_context: str = ""

    # Plans
    content_plan: Optional[ContentPlan] = None
    learning_queue: List[LearningItem] = field(default_factory=list)
    builder_work: Optional[BuilderWorkAssignment] = None

    # Supporting data
    weekly_metrics: Optional[WeeklyMetrics] = None
    pattern_alerts: List[PatternAlert] = field(default_factory=list)
    financial_summary: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    data_sources_used: List[str] = field(default_factory=list)
    processing_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brief_id": self.brief_id,
            "generated_at": self.generated_at.isoformat(),
            "week": {
                "start": self.week_start.isoformat(),
                "end": self.week_end.isoformat()
            },
            "strategic_summary": self.strategic_summary.to_dict(),
            "what_worked": self.what_worked[:3],
            "what_didnt_work": self.what_didnt_work[:3],
            "next_week_constraints": [c.to_dict() for c in self.next_week_constraints],
            "directional_question": {
                "question": self.directional_question,
                "context": self.question_context
            },
            "content_plan": self.content_plan.to_dict() if self.content_plan else None,
            "learning_queue": [item.to_dict() for item in self.learning_queue],
            "builder_work": self.builder_work.to_dict() if self.builder_work else None,
            "weekly_metrics": self.weekly_metrics.to_dict() if self.weekly_metrics else None,
            "pattern_alerts": [alert.to_dict() for alert in self.pattern_alerts],
            "financial_summary": self.financial_summary,
            "data_sources_used": self.data_sources_used,
            "processing_notes": self.processing_notes
        }


# =============================================================================
# WEEKLY BRIEF GENERATOR
# =============================================================================

class WeeklyBriefGenerator(StrategyAgent):
    """
    Weekly Brief Generator - Produces Alfred's weekly strategic brief.

    Aggregates data from:
    - Content Strategy Analyst
    - Social Metrics Harvester
    - Audience Signals Extractor
    - Shipping Governor
    - Financial Sentinel
    - Pattern Registry (memory)

    Generates a 20-minute strategic brief with:
    - Reputation trajectory assessment
    - Positioning drift check
    - What worked/didn't analysis
    - Non-negotiable constraints
    - Directional question for Alfred
    - Content plan for next week
    - Learning queue
    - Builder work status
    """

    # Thresholds for assessments
    ALIGNMENT_HEALTHY_THRESHOLD = 70
    ALIGNMENT_WARNING_THRESHOLD = 50
    DRIFT_SEVERE_THRESHOLD = 40
    DRIFT_MODERATE_THRESHOLD = 25
    SHIPPING_CRITICAL_DAYS = 14

    def __init__(self):
        super().__init__("Weekly Brief Generator")
        self._previous_briefs: List[WeeklyBrief] = []
        self._brief_counter = 0

    def generate_brief(
        self,
        strategy_data: Optional[Dict[str, Any]] = None,
        metrics_data: Optional[Dict[str, Any]] = None,
        audience_data: Optional[Dict[str, Any]] = None,
        shipping_data: Optional[Dict[str, Any]] = None,
        financial_data: Optional[Dict[str, Any]] = None,
        pattern_data: Optional[Dict[str, Any]] = None,
        week_start: Optional[date] = None,
        week_end: Optional[date] = None
    ) -> AgentResponse:
        """
        Generate the weekly strategic brief.

        Args:
            strategy_data: Output from Content Strategy Analyst
            metrics_data: Output from Social Metrics Harvester
            audience_data: Output from Audience Signals Extractor
            shipping_data: Output from Shipping Governor
            financial_data: Output from Financial Sentinel
            pattern_data: Output from Pattern Registry
            week_start: Start of the week being reported
            week_end: End of the week being reported

        Returns:
            AgentResponse containing the WeeklyBrief
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Default to current week if not specified
        if not week_start:
            today = date.today()
            week_start = today - timedelta(days=today.weekday())  # Monday
        if not week_end:
            week_end = week_start + timedelta(days=6)  # Sunday

        # Generate brief ID
        self._brief_counter += 1
        brief_id = f"brief_{week_start.strftime('%Y%W')}_{self._brief_counter:03d}"

        data_sources = []
        processing_notes = []

        try:
            # 1. Generate Strategic Summary
            strategic_summary = self._generate_strategic_summary(
                strategy_data, metrics_data, audience_data
            )
            if strategy_data:
                data_sources.append("Content Strategy Analyst")

            # 2. Extract what worked / didn't
            what_worked, what_didnt = self._extract_performance_insights(
                strategy_data, metrics_data, audience_data
            )
            if metrics_data:
                data_sources.append("Social Metrics Harvester")
            if audience_data:
                data_sources.append("Audience Signals Extractor")

            # 3. Determine constraints
            constraints = self._determine_constraints(
                shipping_data, financial_data, strategic_summary
            )
            if shipping_data:
                data_sources.append("Shipping Governor")
            if financial_data:
                data_sources.append("Financial Sentinel")

            # 4. Generate directional question
            question, context = self._generate_directional_question(
                strategic_summary, what_didnt, constraints, pattern_data
            )
            if pattern_data:
                data_sources.append("Pattern Registry")

            # 5. Create content plan
            content_plan = self._create_content_plan(
                strategy_data, what_worked, week_end + timedelta(days=1)
            )

            # 6. Build learning queue
            learning_queue = self._build_learning_queue(
                audience_data, strategy_data, what_didnt
            )

            # 7. Determine builder work status
            builder_work = self._determine_builder_work(
                shipping_data, constraints
            )

            # 8. Extract metrics summary
            weekly_metrics = self._extract_weekly_metrics(metrics_data)

            # 9. Extract pattern alerts
            pattern_alerts = self._extract_pattern_alerts(pattern_data)

            # 10. Extract financial summary
            financial_summary = self._extract_financial_summary(financial_data)

            # Build the brief
            brief = WeeklyBrief(
                brief_id=brief_id,
                generated_at=datetime.now(),
                week_start=week_start,
                week_end=week_end,
                strategic_summary=strategic_summary,
                what_worked=what_worked[:3],
                what_didnt_work=what_didnt[:3],
                next_week_constraints=constraints,
                directional_question=question,
                question_context=context,
                content_plan=content_plan,
                learning_queue=learning_queue,
                builder_work=builder_work,
                weekly_metrics=weekly_metrics,
                pattern_alerts=pattern_alerts,
                financial_summary=financial_summary,
                data_sources_used=data_sources,
                processing_notes=processing_notes
            )

            # Store for historical comparison
            self._previous_briefs.append(brief)
            if len(self._previous_briefs) > 12:  # Keep ~3 months
                self._previous_briefs = self._previous_briefs[-12:]

            # Collect warnings
            warnings = self._collect_warnings(brief)

            return self.create_response(
                data={
                    "brief": brief.to_dict(),
                    "formatted_brief": self.format_brief(brief)
                },
                success=True,
                warnings=warnings
            )

        except Exception as e:
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Brief generation failed: {str(e)}"]
            )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _generate_strategic_summary(
        self,
        strategy_data: Optional[Dict[str, Any]],
        metrics_data: Optional[Dict[str, Any]],
        audience_data: Optional[Dict[str, Any]]
    ) -> StrategicSummary:
        """Generate the strategic summary section."""

        # Extract alignment score and drift from strategy data
        alignment_score = None
        drift_severity = DriftSeverity.NONE
        drift_description = "No significant positioning drift detected."

        if strategy_data:
            memo = strategy_data.get("memo", {})
            alignment_score = memo.get("alignment_score")

            drift_analysis = memo.get("drift_analysis", {})
            if drift_analysis.get("drift_detected"):
                drift_sev = drift_analysis.get("drift_severity", "none")
                drift_severity = DriftSeverity(drift_sev)
                drift_direction = drift_analysis.get("drift_direction", "")
                if drift_direction:
                    drift_description = f"Positioning drift: {drift_direction}"

        # Determine reputation trajectory
        trajectory = ReputationTrajectory.STABLE
        rationale = "Metrics and positioning remain consistent with previous period."

        # Check for risks
        key_risk = None
        key_opportunity = None

        if alignment_score is not None:
            if alignment_score < self.ALIGNMENT_WARNING_THRESHOLD:
                trajectory = ReputationTrajectory.AT_RISK
                rationale = f"Alignment score ({alignment_score:.1f}) below healthy threshold."
                key_risk = "Positioning alignment critically low"
            elif alignment_score < self.ALIGNMENT_HEALTHY_THRESHOLD:
                trajectory = ReputationTrajectory.AT_RISK
                rationale = f"Alignment score ({alignment_score:.1f}) needs attention."
                key_risk = "Positioning drift accumulating"
            elif drift_severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]:
                trajectory = ReputationTrajectory.AT_RISK
                rationale = f"{drift_severity.value.title()} positioning drift detected."
                key_risk = drift_description

        # Check audience signals for trust indicators
        if audience_data:
            report = audience_data.get("report", {})
            trust_killers = report.get("top_3_trust_killers", [])
            trust_builders = report.get("top_3_trust_builders", [])

            if len(trust_killers) > len(trust_builders):
                if trajectory != ReputationTrajectory.AT_RISK:
                    trajectory = ReputationTrajectory.AT_RISK
                    rationale = "Trust signals indicate more negatives than positives."
                key_risk = key_risk or "Trust erosion detected in audience feedback"
            elif len(trust_builders) > len(trust_killers) and trajectory == ReputationTrajectory.STABLE:
                trajectory = ReputationTrajectory.IMPROVING
                rationale = "Trust signals indicate positive momentum."
                key_opportunity = "Strong trust-building signals from audience"

        # Check metrics trends
        week_trend = TrendDirection.FLAT
        if metrics_data:
            report = metrics_data.get("report", {})
            cross_platform = report.get("cross_platform_summary", {})

            # Check engagement trends (simplified - would need historical data)
            total_engagement = cross_platform.get("total_engagement", 0)
            if total_engagement > 0:
                if trajectory == ReputationTrajectory.STABLE:
                    key_opportunity = key_opportunity or "Engagement metrics showing positive signals"

        return StrategicSummary(
            reputation_trajectory=trajectory,
            trajectory_rationale=rationale,
            drift_severity=drift_severity,
            drift_description=drift_description,
            alignment_score=alignment_score,
            week_over_week_trend=week_trend,
            key_risk=key_risk,
            key_opportunity=key_opportunity
        )

    def _extract_performance_insights(
        self,
        strategy_data: Optional[Dict[str, Any]],
        metrics_data: Optional[Dict[str, Any]],
        audience_data: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Extract what worked and what didn't from the data."""
        what_worked = []
        what_didnt = []

        # From strategy data
        if strategy_data:
            memo = strategy_data.get("memo", {})

            # Successes
            successes = memo.get("successes", [])
            for success in successes[:3]:
                if isinstance(success, dict):
                    content = success.get("content", "")[:50]
                    why = success.get("why", "performed well")
                    what_worked.append(f"{content}: {why[:60]}")

            # Failures
            failures = memo.get("failures", [])
            for failure in failures[:3]:
                if isinstance(failure, dict):
                    content = failure.get("content", "")[:50]
                    diagnosis = failure.get("diagnosis", "underperformed")
                    what_didnt.append(f"{content}: {diagnosis[:60]}")

            # Double down recommendations (positive signal)
            double_down = memo.get("double_down", [])
            for rec in double_down[:2]:
                if isinstance(rec, dict):
                    target = rec.get("target", "")
                    if target and len(what_worked) < 3:
                        what_worked.append(f"Double down on: {target[:50]}")

        # From audience data
        if audience_data:
            report = audience_data.get("report", {})

            # Trust builders
            trust_builders = report.get("top_3_trust_builders", [])
            for builder in trust_builders[:2]:
                if isinstance(builder, dict) and len(what_worked) < 3:
                    theme = builder.get("theme", "")
                    if theme:
                        what_worked.append(f"Trust built through: {theme[:50]}")

            # Confusions (things that didn't work)
            confusions = report.get("top_5_confusions", [])
            for confusion in confusions[:2]:
                if isinstance(confusion, dict) and len(what_didnt) < 3:
                    theme = confusion.get("theme", "")
                    if theme:
                        what_didnt.append(f"Audience confused about: {theme[:50]}")

        # Ensure we have at least some content
        if not what_worked:
            what_worked.append("No standout successes identified this week")
        if not what_didnt:
            what_didnt.append("No significant failures identified this week")

        return what_worked, what_didnt

    def _determine_constraints(
        self,
        shipping_data: Optional[Dict[str, Any]],
        financial_data: Optional[Dict[str, Any]],
        strategic_summary: StrategicSummary
    ) -> List[WeeklyConstraint]:
        """Determine non-negotiable constraints for next week."""
        constraints = []

        # Shipping constraints
        if shipping_data:
            health = shipping_data.get("overall_shipping_health", "")

            if health == "CRITICAL":
                constraints.append(WeeklyConstraint(
                    constraint_type=ConstraintType.SHIPPING,
                    description="MUST ship something before any builder work",
                    source="Shipping Governor"
                ))
            elif health == "WARNING":
                constraints.append(WeeklyConstraint(
                    constraint_type=ConstraintType.SHIPPING,
                    description="Address shipping backlog - at least one output required",
                    source="Shipping Governor"
                ))

            # Check for specific project deadlines
            alerts = shipping_data.get("shipping_alerts", [])
            for alert in alerts[:2]:
                if isinstance(alert, dict):
                    project = alert.get("project", "")
                    days = alert.get("days_without_output", 0)
                    if days >= self.SHIPPING_CRITICAL_DAYS:
                        constraints.append(WeeklyConstraint(
                            constraint_type=ConstraintType.DEADLINE,
                            description=f"Ship or kill '{project}' - {days} days stalled",
                            source="Shipping Governor"
                        ))

        # Financial constraints
        if financial_data:
            budget_status = financial_data.get("budget_status", {})
            budgets = budget_status.get("budgets", {})

            overall = budgets.get("overall", {})
            if overall.get("status") == "over":
                constraints.append(WeeklyConstraint(
                    constraint_type=ConstraintType.FINANCIAL,
                    description="Budget exceeded - no new tool purchases",
                    source="Financial Sentinel"
                ))

            # Check for upcoming renewals that need decisions
            renewals = financial_data.get("upcoming_renewals", [])
            urgent_renewals = [r for r in renewals if r.get("days_until_renewal", 999) <= 7]
            if urgent_renewals:
                names = [r.get("name", "unknown") for r in urgent_renewals[:2]]
                constraints.append(WeeklyConstraint(
                    constraint_type=ConstraintType.DEADLINE,
                    description=f"Renewal decisions needed: {', '.join(names)}",
                    deadline=date.today() + timedelta(days=7),
                    source="Financial Sentinel"
                ))

        # Strategic constraints from reputation status
        if strategic_summary.reputation_trajectory == ReputationTrajectory.AT_RISK:
            constraints.append(WeeklyConstraint(
                constraint_type=ConstraintType.RECOVERY,
                description=strategic_summary.key_risk or "Address reputation concerns",
                source="Strategic Summary"
            ))

        if strategic_summary.drift_severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]:
            constraints.append(WeeklyConstraint(
                constraint_type=ConstraintType.STRATEGIC,
                description=f"Correct positioning drift: {strategic_summary.drift_description}",
                source="Content Strategy Analyst"
            ))

        return constraints

    def _generate_directional_question(
        self,
        strategic_summary: StrategicSummary,
        what_didnt: List[str],
        constraints: List[WeeklyConstraint],
        pattern_data: Optional[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Generate the one directional question Alfred needs answered."""

        # Priority 1: Critical strategic issues
        if strategic_summary.reputation_trajectory == ReputationTrajectory.AT_RISK:
            if strategic_summary.drift_severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]:
                return (
                    "Is the current positioning drift intentional, or should we course-correct?",
                    f"Context: {strategic_summary.drift_description}"
                )
            if strategic_summary.key_risk:
                return (
                    f"How do we address: {strategic_summary.key_risk}?",
                    "Context: Reputation trajectory showing risk signals"
                )

        # Priority 2: Pattern-based questions
        if pattern_data:
            worsening = pattern_data.get("worsening_patterns", [])
            if worsening:
                pattern = worsening[0]
                return (
                    f"Pattern alert: {pattern.get('description', 'behavioral pattern')} - intervention needed?",
                    f"Context: Pattern trajectory is worsening, severity {pattern.get('average_severity', 'unknown')}"
                )

        # Priority 3: Constraint-driven questions
        shipping_constraints = [c for c in constraints if c.constraint_type == ConstraintType.SHIPPING]
        if shipping_constraints:
            return (
                "What's blocking shipping, and what's the minimum viable output we can ship this week?",
                "Context: Shipping health requires attention before other work"
            )

        # Priority 4: From failures
        if what_didnt and "confused" in what_didnt[0].lower():
            confusion_topic = what_didnt[0].split(":")[-1].strip()
            return (
                f"How should we clarify our messaging about {confusion_topic[:40]}?",
                "Context: Audience feedback indicates confusion"
            )

        # Priority 5: Opportunity-based
        if strategic_summary.key_opportunity:
            return (
                f"How do we capitalize on: {strategic_summary.key_opportunity}?",
                "Context: Positive signals detected - opportunity to accelerate"
            )

        # Default question
        return (
            "What is the single most important thing to accomplish this week?",
            "Context: No critical issues detected - focus on highest-impact work"
        )

    def _create_content_plan(
        self,
        strategy_data: Optional[Dict[str, Any]],
        what_worked: List[str],
        next_week_start: date
    ) -> ContentPlan:
        """Create the content plan for next week."""
        plan = ContentPlan(
            week_start=next_week_start,
            week_end=next_week_start + timedelta(days=6)
        )

        # Extract recommendations from strategy data
        if strategy_data:
            memo = strategy_data.get("memo", {})

            # From double-down recommendations
            double_down = memo.get("double_down", [])
            for rec in double_down[:2]:
                if isinstance(rec, dict):
                    target = rec.get("target", "")
                    rationale = rec.get("rationale", "")
                    if target:
                        plan.add_item(ContentPlanItem(
                            title=f"Follow-up: {target[:40]}",
                            platform="tbd",
                            content_type="continuation",
                            priority="high",
                            rationale=rationale[:100] if rationale else "Based on success this week"
                        ))

            # From recommended experiments
            experiments = memo.get("recommended_experiments", [])
            for exp in experiments[:1]:
                if isinstance(exp, dict):
                    name = exp.get("name", "")
                    if name:
                        plan.add_item(ContentPlanItem(
                            title=name[:50],
                            platform="tbd",
                            content_type="experiment",
                            priority="medium",
                            rationale="Testing hypothesis from strategy analysis"
                        ))

        # From what worked insights
        for insight in what_worked[:2]:
            if "Double down on" in insight:
                topic = insight.replace("Double down on:", "").strip()
                plan.add_item(ContentPlanItem(
                    title=f"Expand: {topic[:40]}",
                    platform="tbd",
                    content_type="expansion",
                    priority="high",
                    rationale="Scaling what worked this week"
                ))

        # Add notes
        if not plan.items:
            plan.notes.append("No specific content recommendations - review strategy analysis")
        else:
            plan.notes.append(f"{len(plan.items)} items recommended based on this week's performance")

        return plan

    def _build_learning_queue(
        self,
        audience_data: Optional[Dict[str, Any]],
        strategy_data: Optional[Dict[str, Any]],
        what_didnt: List[str]
    ) -> List[LearningItem]:
        """Build the learning queue for next week."""
        queue = []

        # From audience questions (high priority)
        if audience_data:
            report = audience_data.get("report", {})
            questions = report.get("top_5_questions", [])

            for q in questions[:2]:
                if isinstance(q, dict):
                    theme = q.get("theme", "")
                    opportunity = q.get("content_opportunity", "")
                    if theme:
                        queue.append(LearningItem(
                            topic=f"Research for audience question: {theme[:40]}",
                            priority="high",
                            linked_output=opportunity[:50] if opportunity else None,
                            source="Audience Signals Extractor"
                        ))

        # From strategy gaps
        if strategy_data:
            memo = strategy_data.get("memo", {})
            strategic_questions = memo.get("strategic_questions", [])

            for sq in strategic_questions[:1]:
                if sq:
                    queue.append(LearningItem(
                        topic=f"Strategic research: {sq[:50]}",
                        priority="medium",
                        source="Content Strategy Analyst"
                    ))

        # From failures
        for failure in what_didnt[:1]:
            if "confused" in failure.lower():
                topic = failure.split(":")[-1].strip()
                queue.append(LearningItem(
                    topic=f"Clarify messaging: {topic[:40]}",
                    priority="high",
                    linked_output="Clarification content",
                    source="Performance Analysis"
                ))

        return queue[:5]  # Limit to 5 items

    def _determine_builder_work(
        self,
        shipping_data: Optional[Dict[str, Any]],
        constraints: List[WeeklyConstraint]
    ) -> BuilderWorkAssignment:
        """Determine builder work status for next week."""

        shipping_health = "UNKNOWN"
        if shipping_data:
            shipping_health = shipping_data.get("overall_shipping_health", "UNKNOWN")

        # Check for shipping constraints
        has_shipping_constraint = any(
            c.constraint_type == ConstraintType.SHIPPING
            for c in constraints
        )

        if shipping_health == "CRITICAL" or has_shipping_constraint:
            return BuilderWorkAssignment(
                status=BuilderWorkStatus.BANNED,
                rationale="Building without shipping is procrastination. Ship first.",
                shipping_health=shipping_health,
                conditions=["Ship at least one output before any builder work is permitted"]
            )

        if shipping_health == "WARNING":
            return BuilderWorkAssignment(
                status=BuilderWorkStatus.CONDITIONAL,
                rationale="Builder work allowed only after shipping commitment is met.",
                shipping_health=shipping_health,
                conditions=[
                    "Complete one shipping task first",
                    "Builder work must have linked output",
                    "Maximum 2 hours builder time"
                ]
            )

        # Healthy shipping - allow scheduled builder work
        return BuilderWorkAssignment(
            status=BuilderWorkStatus.SCHEDULED,
            rationale="Shipping health is good. Builder work permitted with linked outputs.",
            shipping_health=shipping_health,
            scheduled_items=[
                {"description": "Review and plan builder priorities", "time": "30 min"}
            ],
            conditions=["All builder work must have linked output defined"]
        )

    def _extract_weekly_metrics(
        self,
        metrics_data: Optional[Dict[str, Any]]
    ) -> Optional[WeeklyMetrics]:
        """Extract weekly metrics summary."""
        if not metrics_data:
            return None

        report = metrics_data.get("report", {})
        cross_platform = report.get("cross_platform_summary", {})

        metrics = WeeklyMetrics(
            total_content_published=cross_platform.get("total_output", 0),
            total_reach=cross_platform.get("total_reach", 0),
            total_engagement=cross_platform.get("total_engagement", 0),
            follower_change=cross_platform.get("net_follower_growth", 0)
        )

        # Calculate engagement rate
        if metrics.total_reach > 0:
            metrics.engagement_rate = (metrics.total_engagement / metrics.total_reach) * 100

        # Extract platform breakdown
        platform_metrics = report.get("platform_metrics", {})
        for platform, data in platform_metrics.items():
            if isinstance(data, dict):
                metrics.platform_metrics[platform] = {
                    "output": data.get("output", {}).get("total_output", 0),
                    "engagement_rate": data.get("engagement_rate", 0)
                }

        return metrics

    def _extract_pattern_alerts(
        self,
        pattern_data: Optional[Dict[str, Any]]
    ) -> List[PatternAlert]:
        """Extract pattern alerts from Pattern Registry data."""
        alerts = []

        if not pattern_data:
            return alerts

        # Worsening patterns
        worsening = pattern_data.get("worsening_patterns", [])
        for pattern in worsening[:3]:
            if isinstance(pattern, dict):
                alerts.append(PatternAlert(
                    pattern_type=pattern.get("pattern_type", "unknown"),
                    description=pattern.get("description", ""),
                    trajectory="worsening",
                    severity=int(pattern.get("average_severity", 5)),
                    recommended_action="Review and implement intervention"
                ))

        # Recent occurrences of concerning patterns
        recent = pattern_data.get("recent_occurrences", [])
        for occ in recent[:2]:
            if isinstance(occ, dict) and occ.get("severity", 0) >= 7:
                alerts.append(PatternAlert(
                    pattern_type=occ.get("pattern_type", "unknown"),
                    description=occ.get("context", ""),
                    trajectory="active",
                    severity=occ.get("severity", 5),
                    recommended_action="Monitor closely this week"
                ))

        return alerts

    def _extract_financial_summary(
        self,
        financial_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract financial summary."""
        if not financial_data:
            return {}

        summary = {}

        # Monthly recurring
        monthly = financial_data.get("monthly_recurring", {})
        summary["monthly_recurring"] = monthly.get("total", 0)
        summary["monthly_recurring_delta"] = monthly.get("delta_from_last")

        # Budget status
        budget_status = financial_data.get("budget_status", {})
        budgets = budget_status.get("budgets", {})
        if "overall" in budgets:
            summary["budget_status"] = budgets["overall"].get("status", "unknown")
            summary["budget_remaining"] = budgets["overall"].get("remaining", 0)

        # Waste
        unused = financial_data.get("unused_underused", {})
        summary["monthly_waste"] = unused.get("monthly_waste", 0)

        # Recommendations
        recommendations = financial_data.get("recommendations", [])
        summary["top_recommendations"] = recommendations[:2]

        return summary

    def _collect_warnings(self, brief: WeeklyBrief) -> List[str]:
        """Collect warnings to include in the response."""
        warnings = []

        if brief.strategic_summary.reputation_trajectory == ReputationTrajectory.AT_RISK:
            warnings.append(f"REPUTATION AT RISK: {brief.strategic_summary.trajectory_rationale}")

        if brief.strategic_summary.drift_severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]:
            warnings.append(f"POSITIONING DRIFT: {brief.strategic_summary.drift_description}")

        if brief.builder_work and brief.builder_work.status == BuilderWorkStatus.BANNED:
            warnings.append("BUILDER WORK BANNED: Ship first before any building")

        for constraint in brief.next_week_constraints:
            if constraint.constraint_type in [ConstraintType.SHIPPING, ConstraintType.RECOVERY]:
                warnings.append(f"CONSTRAINT: {constraint.description}")

        return warnings

    # =========================================================================
    # Output Formatting
    # =========================================================================

    def format_brief(self, brief: WeeklyBrief) -> str:
        """Format the weekly brief as a readable string."""
        lines = [
            "=" * 70,
            "WEEKLY STRATEGIC BRIEF",
            "=" * 70,
            f"Week: {brief.week_start.isoformat()} to {brief.week_end.isoformat()}",
            f"Generated: {brief.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"Brief ID: {brief.brief_id}",
            "",
            "-" * 70,
            "STRATEGIC SUMMARY",
            "-" * 70,
            f"Reputation Trajectory: {brief.strategic_summary.reputation_trajectory.value}",
            f"  Rationale: {brief.strategic_summary.trajectory_rationale}",
            "",
            f"Drift Check: {brief.strategic_summary.drift_description}",
        ]

        if brief.strategic_summary.alignment_score:
            lines.append(f"Alignment Score: {brief.strategic_summary.alignment_score:.1f}/100")

        if brief.strategic_summary.key_risk:
            lines.append(f"Key Risk: {brief.strategic_summary.key_risk}")

        if brief.strategic_summary.key_opportunity:
            lines.append(f"Key Opportunity: {brief.strategic_summary.key_opportunity}")

        # What worked / didn't
        lines.extend([
            "",
            "-" * 70,
            "WHAT WORKED (max 3)",
            "-" * 70,
        ])
        for i, item in enumerate(brief.what_worked[:3], 1):
            lines.append(f"  {i}. {item}")

        lines.extend([
            "",
            "-" * 70,
            "WHAT DIDN'T WORK (max 3)",
            "-" * 70,
        ])
        for i, item in enumerate(brief.what_didnt_work[:3], 1):
            lines.append(f"  {i}. {item}")

        # Constraints
        lines.extend([
            "",
            "-" * 70,
            "NEXT WEEK'S CONSTRAINTS (non-negotiable)",
            "-" * 70,
        ])
        if brief.next_week_constraints:
            for c in brief.next_week_constraints:
                deadline_str = f" [Deadline: {c.deadline.isoformat()}]" if c.deadline else ""
                lines.append(f"  [{c.constraint_type.value.upper()}] {c.description}{deadline_str}")
        else:
            lines.append("  No critical constraints identified")

        # Directional question
        lines.extend([
            "",
            "-" * 70,
            "THE QUESTION ALFRED NEEDS ANSWERED",
            "-" * 70,
            f"  {brief.directional_question}",
            f"  Context: {brief.question_context}",
        ])

        # Content plan
        lines.extend([
            "",
            "-" * 70,
            "CONTENT PLAN FOR NEXT WEEK",
            "-" * 70,
        ])
        if brief.content_plan and brief.content_plan.items:
            for i, item in enumerate(brief.content_plan.items[:5], 1):
                lines.append(f"  {i}. [{item.priority.upper()}] {item.title}")
                if item.rationale:
                    lines.append(f"     Rationale: {item.rationale[:60]}")
        else:
            lines.append("  No specific content items planned")

        # Learning queue
        lines.extend([
            "",
            "-" * 70,
            "LEARNING QUEUE FOR NEXT WEEK",
            "-" * 70,
        ])
        if brief.learning_queue:
            for i, item in enumerate(brief.learning_queue[:5], 1):
                linked = f" -> {item.linked_output}" if item.linked_output else ""
                lines.append(f"  {i}. [{item.priority.upper()}] {item.topic}{linked}")
        else:
            lines.append("  No learning items queued")

        # Builder work
        lines.extend([
            "",
            "-" * 70,
            "BUILDER WORK STATUS",
            "-" * 70,
        ])
        if brief.builder_work:
            lines.append(f"  Status: {brief.builder_work.status.value}")
            lines.append(f"  Rationale: {brief.builder_work.rationale}")
            if brief.builder_work.conditions:
                lines.append("  Conditions:")
                for cond in brief.builder_work.conditions:
                    lines.append(f"    - {cond}")
            if brief.builder_work.scheduled_items:
                lines.append("  Scheduled Items:")
                for item in brief.builder_work.scheduled_items:
                    lines.append(f"    - {item.get('description', '')} ({item.get('time', '')})")

        # Pattern alerts
        if brief.pattern_alerts:
            lines.extend([
                "",
                "-" * 70,
                "PATTERN ALERTS",
                "-" * 70,
            ])
            for alert in brief.pattern_alerts[:3]:
                lines.append(f"  [{alert.pattern_type.upper()}] {alert.description}")
                lines.append(f"    Severity: {alert.severity}/10, Trajectory: {alert.trajectory}")
                lines.append(f"    Action: {alert.recommended_action}")

        # Financial summary
        if brief.financial_summary:
            lines.extend([
                "",
                "-" * 70,
                "FINANCIAL SNAPSHOT",
                "-" * 70,
            ])
            if "monthly_recurring" in brief.financial_summary:
                delta = brief.financial_summary.get("monthly_recurring_delta")
                delta_str = f" ({'+' if delta >= 0 else ''}{delta:.2f})" if delta else ""
                lines.append(f"  Monthly Recurring: ${brief.financial_summary['monthly_recurring']:.2f}{delta_str}")
            if "budget_status" in brief.financial_summary:
                lines.append(f"  Budget Status: {brief.financial_summary['budget_status'].upper()}")
            if brief.financial_summary.get("monthly_waste", 0) > 0:
                lines.append(f"  Monthly Waste: ${brief.financial_summary['monthly_waste']:.2f}")

        # Footer
        lines.extend([
            "",
            "-" * 70,
            "DATA SOURCES",
            "-" * 70,
            f"  {', '.join(brief.data_sources_used) if brief.data_sources_used else 'None specified'}",
            "",
            "=" * 70,
            "END OF WEEKLY BRIEF",
            "=" * 70,
        ])

        return "\n".join(lines)

    # =========================================================================
    # Request Processing
    # =========================================================================

    def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process a WEEKLY_BRIEF_REQUEST.

        Expected request format:
        {
            "strategy_data": {...},  # From Content Strategy Analyst
            "metrics_data": {...},   # From Social Metrics Harvester
            "audience_data": {...},  # From Audience Signals Extractor
            "shipping_data": {...},  # From Shipping Governor
            "financial_data": {...}, # From Financial Sentinel
            "pattern_data": {...},   # From Pattern Registry
            "week_start": "YYYY-MM-DD",  # Optional
            "week_end": "YYYY-MM-DD"     # Optional
        }
        """
        week_start = None
        week_end = None

        if request.get("week_start"):
            week_start = date.fromisoformat(request["week_start"])
        if request.get("week_end"):
            week_end = date.fromisoformat(request["week_end"])

        return self.generate_brief(
            strategy_data=request.get("strategy_data"),
            metrics_data=request.get("metrics_data"),
            audience_data=request.get("audience_data"),
            shipping_data=request.get("shipping_data"),
            financial_data=request.get("financial_data"),
            pattern_data=request.get("pattern_data"),
            week_start=week_start,
            week_end=week_end
        )

    def get_previous_briefs(self, count: int = 4) -> List[Dict[str, Any]]:
        """Get previous briefs for comparison."""
        return [b.to_dict() for b in self._previous_briefs[-count:]]

    def get_trend_comparison(self) -> Dict[str, Any]:
        """Compare current brief with previous for trends."""
        if len(self._previous_briefs) < 2:
            return {"status": "insufficient_data", "briefs_available": len(self._previous_briefs)}

        current = self._previous_briefs[-1]
        previous = self._previous_briefs[-2]

        comparison = {
            "current_week": current.week_start.isoformat(),
            "previous_week": previous.week_start.isoformat(),
            "trajectory_change": {
                "from": previous.strategic_summary.reputation_trajectory.value,
                "to": current.strategic_summary.reputation_trajectory.value
            },
            "drift_change": {
                "from": previous.strategic_summary.drift_severity.value,
                "to": current.strategic_summary.drift_severity.value
            }
        }

        # Compare alignment scores
        if current.strategic_summary.alignment_score and previous.strategic_summary.alignment_score:
            comparison["alignment_change"] = (
                current.strategic_summary.alignment_score -
                previous.strategic_summary.alignment_score
            )

        return comparison


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_weekly_brief_generator() -> WeeklyBriefGenerator:
    """Factory function to create a Weekly Brief Generator instance."""
    return WeeklyBriefGenerator()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create generator
    generator = WeeklyBriefGenerator()

    # Example data (would come from other agents in practice)
    example_strategy_data = {
        "memo": {
            "alignment_score": 72.5,
            "drift_analysis": {
                "drift_detected": True,
                "drift_severity": "minor",
                "drift_direction": "over-indexing on technical content"
            },
            "successes": [
                {"content": "Deep dive on API design", "why": "High engagement, strong saves"}
            ],
            "failures": [
                {"content": "Quick tip thread", "diagnosis": "Low reach, unclear hook"}
            ],
            "double_down": [
                {"target": "technical deep dives", "rationale": "Strong audience response"}
            ]
        }
    }

    example_shipping_data = {
        "overall_shipping_health": "WARNING",
        "shipping_alerts": [
            {"project": "Course Module 3", "days_without_output": 10}
        ]
    }

    # Generate brief
    response = generator.generate_brief(
        strategy_data=example_strategy_data,
        shipping_data=example_shipping_data
    )

    if response.success:
        print(response.data["formatted_brief"])
    else:
        print(f"Error: {response.errors}")
