"""
Weekly Brief System for ALFRED

Role: Generate Alfred's weekly strategic brief - 20 minutes of essential strategic analysis.

Integration Points:
- Content Strategy Analyst: Strategy memo
- Social Metrics Harvester: Performance data
- Audience Signals Extractor: Audience insights
- Shipping Governor: Builder status
- Learning Curator: Learning queue

Output: Comprehensive weekly strategic brief saved to data/alfred/briefs/weekly/
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path


class ReputationTrajectory(Enum):
    STABLE = "Stable"
    IMPROVING = "Improving"
    AT_RISK = "At Risk"


class BuilderStatus(Enum):
    SCHEDULED = "SCHEDULED"
    BANNED = "BANNED"


@dataclass
class ReputationAnalysis:
    trajectory: str
    explanation: str
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    positive_signals: List[Dict[str, Any]] = field(default_factory=list)
    trend_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAnalysis:
    summary: str
    drift_detected: bool = False
    drift_direction: str = ""
    severity: str = ""
    correction_needed: str = ""


@dataclass
class PerformanceItem:
    item: str
    evidence: str
    category: str = ""


@dataclass
class ContentPlanItem:
    day: str
    platform: str
    topic: str
    status: str
    notes: str = ""


@dataclass
class LearningQueueItem:
    question: str
    resource: str
    duration: str
    linked_output: str


@dataclass
class BuilderConstraint:
    status: str
    details: str
    projects: List[Dict[str, Any]] = field(default_factory=list)
    hours_allocated: int = 0


@dataclass
class WeeklyBriefData:
    brief_id: str
    generated_at: str
    week_start: str
    week_end: str
    reputation_trajectory: ReputationAnalysis
    drift_analysis: DriftAnalysis
    what_worked: List[PerformanceItem]
    what_didnt_work: List[PerformanceItem]
    non_negotiables: List[str]
    directional_question: str
    content_plan: List[ContentPlanItem]
    learning_queue: List[LearningQueueItem]
    builder_constraint: BuilderConstraint
    strategy_memo_attached: bool
    raw_data: Dict[str, Any] = field(default_factory=dict)


class WeeklyBrief:
    """
    Weekly Brief Generator for ALFRED

    Aggregates weekly data from all agents, analyzes patterns,
    and generates strategic recommendations.

    Methods:
    - generate_weekly_brief(): Returns full strategic brief
    - get_reputation_trajectory(): Analyzes week's reputation data
    - get_drift_analysis(): Checks positioning drift
    - get_performance_summary(): What worked/didn't
    - get_content_plan(): Next week's content schedule
    - get_learning_queue(): Next week's learning items
    - get_builder_constraints(): Building allowed/banned
    """

    BRIEF_STORAGE_PATH = "data/alfred/briefs/weekly"

    def __init__(self, base_path: str = None):
        """Initialize WeeklyBrief with optional base path."""
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Default to agent-zero1 directory
            self.base_path = Path(__file__).parent.parent.parent.parent

        self.briefs_path = self.base_path / self.BRIEF_STORAGE_PATH
        self.briefs_path.mkdir(parents=True, exist_ok=True)

        # Integration data from agents
        self._metrics_data: Optional[Dict] = None
        self._audience_signals: Optional[Dict] = None
        self._strategy_memo: Optional[Dict] = None
        self._shipping_report: Optional[Dict] = None
        self._learning_data: Optional[Dict] = None
        self._reputation_data: Optional[Dict] = None

    def set_metrics_data(self, data: Dict) -> None:
        """Set metrics data from Social Metrics Harvester."""
        self._metrics_data = data

    def set_audience_signals(self, data: Dict) -> None:
        """Set audience signals from Audience Signals Extractor."""
        self._audience_signals = data

    def set_strategy_memo(self, data: Dict) -> None:
        """Set strategy memo from Content Strategy Analyst."""
        self._strategy_memo = data

    def set_shipping_report(self, data: Dict) -> None:
        """Set shipping report from Shipping Governor."""
        self._shipping_report = data

    def set_learning_data(self, data: Dict) -> None:
        """Set learning data from Learning Curator."""
        self._learning_data = data

    def set_reputation_data(self, data: Dict) -> None:
        """Set reputation data from Reputation Sentinel."""
        self._reputation_data = data

    def generate_weekly_brief(
        self,
        week_start: str = None,
        week_end: str = None,
        metrics_data: Dict = None,
        audience_signals: Dict = None,
        strategy_memo: Dict = None,
        shipping_report: Dict = None,
        learning_data: Dict = None,
        reputation_data: Dict = None,
        positioning_charter: Dict = None
    ) -> Dict[str, Any]:
        """
        Generate the full weekly strategic brief.

        This is the main method that aggregates all agent data and produces
        the complete brief as specified in PLAN.md.

        Args:
            week_start: Start date of the week (YYYY-MM-DD)
            week_end: End date of the week (YYYY-MM-DD)
            metrics_data: From Social Metrics Harvester
            audience_signals: From Audience Signals Extractor
            strategy_memo: From Content Strategy Analyst
            shipping_report: From Shipping Governor
            learning_data: From Learning Curator
            reputation_data: From Reputation Sentinel
            positioning_charter: Reference positioning document

        Returns:
            Dict containing the complete weekly brief
        """
        # Use provided data or fall back to stored data
        metrics_data = metrics_data or self._metrics_data or {}
        audience_signals = audience_signals or self._audience_signals or {}
        strategy_memo = strategy_memo or self._strategy_memo or {}
        shipping_report = shipping_report or self._shipping_report or {}
        learning_data = learning_data or self._learning_data or {}
        reputation_data = reputation_data or self._reputation_data or {}

        # Calculate week dates
        if not week_end:
            week_end = datetime.now().strftime("%Y-%m-%d")
        if not week_start:
            week_start = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")

        # Generate each section
        reputation_analysis = self.get_reputation_trajectory(
            reputation_data=reputation_data,
            metrics_data=metrics_data
        )

        drift_analysis = self.get_drift_analysis(
            strategy_memo=strategy_memo,
            positioning_charter=positioning_charter
        )

        what_worked, what_didnt = self.get_performance_summary(
            metrics_data=metrics_data,
            strategy_memo=strategy_memo,
            audience_signals=audience_signals
        )

        non_negotiables = self._generate_non_negotiables(
            reputation_analysis=reputation_analysis,
            shipping_report=shipping_report,
            strategy_memo=strategy_memo
        )

        directional_question = self._generate_directional_question(
            drift_analysis=drift_analysis,
            reputation_analysis=reputation_analysis,
            audience_signals=audience_signals
        )

        content_plan = self.get_content_plan(
            strategy_memo=strategy_memo,
            audience_signals=audience_signals,
            what_worked=what_worked
        )

        learning_queue = self.get_learning_queue(
            learning_data=learning_data,
            content_plan=content_plan
        )

        builder_constraint = self.get_builder_constraints(
            shipping_report=shipping_report
        )

        # Build brief data
        brief_id = f"WEEKLY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        brief_data = WeeklyBriefData(
            brief_id=brief_id,
            generated_at=datetime.now().isoformat(),
            week_start=week_start,
            week_end=week_end,
            reputation_trajectory=reputation_analysis,
            drift_analysis=drift_analysis,
            what_worked=what_worked,
            what_didnt_work=what_didnt,
            non_negotiables=non_negotiables,
            directional_question=directional_question,
            content_plan=content_plan,
            learning_queue=learning_queue,
            builder_constraint=builder_constraint,
            strategy_memo_attached=bool(strategy_memo),
            raw_data={
                "metrics_provided": bool(metrics_data),
                "audience_signals_provided": bool(audience_signals),
                "strategy_memo_provided": bool(strategy_memo),
                "shipping_report_provided": bool(shipping_report),
                "learning_data_provided": bool(learning_data),
                "reputation_data_provided": bool(reputation_data)
            }
        )

        # Generate formatted output
        formatted_brief = self._format_brief(brief_data, strategy_memo)

        # Save brief
        self._save_brief(brief_data, formatted_brief)

        return {
            "WEEKLY_BRIEF": {
                "brief_id": brief_id,
                "generated_at": brief_data.generated_at,
                "formatted_output": formatted_brief,
                "structured_data": self._brief_to_dict(brief_data),
                "saved_to": str(self.briefs_path / f"{brief_id}.json")
            }
        }

    def get_reputation_trajectory(
        self,
        reputation_data: Dict = None,
        metrics_data: Dict = None
    ) -> ReputationAnalysis:
        """
        Analyze the week's reputation trajectory.

        Categories:
        - Stable: No significant negative events, consistent engagement
        - Improving: Positive sentiment growth, trust signals increasing
        - At Risk: Negative events detected, trust killers identified

        Args:
            reputation_data: From Reputation Sentinel
            metrics_data: From Social Metrics Harvester

        Returns:
            ReputationAnalysis with trajectory, explanation, and supporting data
        """
        reputation_data = reputation_data or {}
        metrics_data = metrics_data or {}

        risk_events = []
        positive_signals = []
        trajectory = ReputationTrajectory.STABLE

        # Analyze reputation packets if available
        packets = reputation_data.get("packets", [])
        for packet in packets:
            risk_score = packet.get("risk_score", 0)
            if risk_score > 70:
                risk_events.append({
                    "event": packet.get("event", "Unknown event"),
                    "platform": packet.get("platform", "Unknown"),
                    "classification": packet.get("classification", "unknown"),
                    "risk_score": risk_score
                })
            elif packet.get("recommended_state") == "GREEN" and risk_score < 30:
                positive_signals.append({
                    "event": packet.get("event", "Positive signal"),
                    "platform": packet.get("platform", "Unknown")
                })

        # Analyze metrics for engagement trends
        metrics_report = metrics_data.get("METRICS_REPORT", {})
        cross_platform = metrics_report.get("cross_platform_summary", {})
        follower_delta = cross_platform.get("total_follower_delta", 0)
        engagement_rate = cross_platform.get("avg_engagement_rate", 0)

        if follower_delta > 0 and engagement_rate > 0.02:
            positive_signals.append({
                "event": f"Positive growth: +{follower_delta} followers, {engagement_rate:.1%} engagement",
                "platform": "Cross-platform"
            })
        elif follower_delta < -50:
            risk_events.append({
                "event": f"Follower decline: {follower_delta} followers lost",
                "platform": "Cross-platform",
                "classification": "engagement_decline",
                "risk_score": 50
            })

        # Determine trajectory
        if len(risk_events) >= 2 or any(e.get("risk_score", 0) > 80 for e in risk_events):
            trajectory = ReputationTrajectory.AT_RISK
            explanation = f"At risk due to {len(risk_events)} flagged event(s) this week requiring attention."
        elif len(positive_signals) >= 3 and len(risk_events) == 0:
            trajectory = ReputationTrajectory.IMPROVING
            explanation = "Positive trajectory with strong engagement signals and no risk events."
        else:
            trajectory = ReputationTrajectory.STABLE
            if len(risk_events) == 0 and len(positive_signals) == 0:
                explanation = "No significant reputation signals this week. Maintain current approach."
            elif len(risk_events) > 0:
                explanation = f"Generally stable with {len(risk_events)} minor concern(s) to monitor."
            else:
                explanation = f"Stable with {len(positive_signals)} positive signal(s) detected."

        return ReputationAnalysis(
            trajectory=trajectory.value,
            explanation=explanation,
            risk_events=risk_events,
            positive_signals=positive_signals,
            trend_data={
                "follower_delta": follower_delta,
                "engagement_rate": engagement_rate,
                "risk_event_count": len(risk_events),
                "positive_signal_count": len(positive_signals)
            }
        )

    def get_drift_analysis(
        self,
        strategy_memo: Dict = None,
        positioning_charter: Dict = None
    ) -> DriftAnalysis:
        """
        Check for positioning drift from the Positioning Charter.

        Args:
            strategy_memo: From Content Strategy Analyst
            positioning_charter: Reference positioning document

        Returns:
            DriftAnalysis with summary and drift details
        """
        strategy_memo = strategy_memo or {}

        # Extract drift info from strategy memo
        memo_data = strategy_memo.get("STRATEGY_MEMO", {})
        drift_alert = memo_data.get("drift_alert", {})
        perf_vs_pos = memo_data.get("performance_vs_positioning", {})

        drift_detected = drift_alert.get("detected", False)
        direction = drift_alert.get("direction", "")
        severity = drift_alert.get("severity", "")
        correction = drift_alert.get("correction_needed", "")
        alignment_score = perf_vs_pos.get("alignment_score", 0)

        # Generate one-sentence summary
        if not drift_detected and alignment_score >= 0.7:
            summary = "No drift detected. Content remains aligned with positioning charter."
        elif not drift_detected:
            summary = f"Alignment at {alignment_score:.0%}. Minor adjustments may improve positioning consistency."
        elif severity == "high":
            summary = f"HIGH PRIORITY: {direction.replace('_', ' ').title()} detected. {correction}"
        elif severity == "medium":
            summary = f"Moderate drift toward {direction.replace('_', ' ')}. Correction recommended before it compounds."
        else:
            summary = f"Minor drift detected ({direction.replace('_', ' ')}). Monitor next week."

        return DriftAnalysis(
            summary=summary,
            drift_detected=drift_detected,
            drift_direction=direction,
            severity=severity,
            correction_needed=correction
        )

    def get_performance_summary(
        self,
        metrics_data: Dict = None,
        strategy_memo: Dict = None,
        audience_signals: Dict = None,
        max_items: int = 3
    ) -> Tuple[List[PerformanceItem], List[PerformanceItem]]:
        """
        Generate what worked and what didn't work summaries.

        Args:
            metrics_data: From Social Metrics Harvester
            strategy_memo: From Content Strategy Analyst
            audience_signals: From Audience Signals Extractor
            max_items: Maximum items per category (default 3)

        Returns:
            Tuple of (what_worked, what_didnt_work) lists
        """
        what_worked = []
        what_didnt = []

        # Extract from strategy memo
        memo_data = (strategy_memo or {}).get("STRATEGY_MEMO", {})

        for item in memo_data.get("what_worked", [])[:max_items]:
            what_worked.append(PerformanceItem(
                item=item.get("content", "Content performed well"),
                evidence=item.get("evidence", "See metrics"),
                category=item.get("lever", "content_resonance")
            ))

        for item in memo_data.get("what_didnt_work", [])[:max_items]:
            what_didnt.append(PerformanceItem(
                item=item.get("content", "Content underperformed"),
                evidence=item.get("evidence", "See metrics"),
                category=item.get("diagnosis", "needs_analysis")
            ))

        # Supplement from audience signals if we don't have enough items
        signals_data = (audience_signals or {}).get("AUDIENCE_SIGNALS", {})

        if len(what_worked) < max_items:
            trust_builders = signals_data.get("top_3_trust_builders", [])
            for builder in trust_builders[:max_items - len(what_worked)]:
                what_worked.append(PerformanceItem(
                    item=f"Trust signal: {builder.get('pattern', 'Positive engagement')}",
                    evidence=f"Frequency: {builder.get('frequency', 0)} occurrences",
                    category="trust_builder"
                ))

        if len(what_didnt) < max_items:
            trust_killers = signals_data.get("top_3_trust_killers", [])
            for killer in trust_killers[:max_items - len(what_didnt)]:
                what_didnt.append(PerformanceItem(
                    item=f"Trust concern: {killer.get('pattern', 'Audience friction')}",
                    evidence=f"Frequency: {killer.get('frequency', 0)} occurrences",
                    category="trust_killer"
                ))

        # Ensure at least one item in each category
        if not what_worked:
            what_worked.append(PerformanceItem(
                item="No standout successes identified",
                evidence="Insufficient data for analysis",
                category="no_data"
            ))

        if not what_didnt:
            what_didnt.append(PerformanceItem(
                item="No significant failures identified",
                evidence="Week performed within expected parameters",
                category="no_issues"
            ))

        return what_worked, what_didnt

    def get_content_plan(
        self,
        strategy_memo: Dict = None,
        audience_signals: Dict = None,
        what_worked: List[PerformanceItem] = None
    ) -> List[ContentPlanItem]:
        """
        Generate next week's content plan.

        Args:
            strategy_memo: From Content Strategy Analyst
            audience_signals: From Audience Signals Extractor
            what_worked: Performance items that worked

        Returns:
            List of ContentPlanItem for the week
        """
        content_plan = []

        # Days of the week
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        platforms = ["Substack", "Twitter", "YouTube", "Twitter", "Instagram", "Rest", "Rest"]

        # Get experiments and recommendations from strategy memo
        memo_data = (strategy_memo or {}).get("STRATEGY_MEMO", {})
        experiments = memo_data.get("experiments_next_week", [])
        double_down = memo_data.get("double_down", [])

        # Get content opportunities from audience signals
        signals_data = (audience_signals or {}).get("AUDIENCE_SIGNALS", {})
        opportunities = signals_data.get("content_opportunities", [])

        # Build topics list
        topics = []

        # Add from double down recommendations
        for item in double_down[:2]:
            if isinstance(item, str):
                topics.append(item.replace("Continue: ", ""))
            elif isinstance(item, dict):
                topics.append(item.get("content", "Topic from last week"))

        # Add from content opportunities
        for opp in opportunities[:3]:
            if isinstance(opp, dict):
                topics.append(opp.get("topic", "Audience-requested topic"))
            else:
                topics.append(str(opp))

        # Add experiment topics
        for exp in experiments[:2]:
            if isinstance(exp, dict):
                topics.append(f"Experiment: {exp.get('name', 'Content experiment')}")

        # Fill remaining with placeholders
        while len(topics) < 5:
            topics.append("TBD - Review queue")

        # Create plan
        topic_idx = 0
        for i, (day, platform) in enumerate(zip(days, platforms)):
            if platform == "Rest":
                content_plan.append(ContentPlanItem(
                    day=day,
                    platform="-",
                    topic="Rest / Review",
                    status="N/A",
                    notes="No publishing scheduled"
                ))
            else:
                topic = topics[topic_idx] if topic_idx < len(topics) else "TBD"
                topic_idx += 1

                status = "Scheduled" if i < 3 else "Planned"

                content_plan.append(ContentPlanItem(
                    day=day,
                    platform=platform,
                    topic=topic[:50] if isinstance(topic, str) else str(topic)[:50],
                    status=status,
                    notes=""
                ))

        return content_plan

    def get_learning_queue(
        self,
        learning_data: Dict = None,
        content_plan: List[ContentPlanItem] = None
    ) -> List[LearningQueueItem]:
        """
        Generate next week's learning items.

        Each item must have a linked output (no learning without execution link).

        Args:
            learning_data: From Learning Curator
            content_plan: Next week's content plan for linking

        Returns:
            List of LearningQueueItem for the week
        """
        learning_queue = []
        learning_data = learning_data or {}

        # Extract queued items from learning data
        items = learning_data.get("items", [])
        if not items and "learning_plan" in learning_data:
            items = learning_data.get("learning_plan", [])

        for item in items[:5]:  # Max 5 items per week
            if isinstance(item, dict):
                question = item.get("question", item.get("item_id", "Learning item"))
                resource = item.get("resource", "TBD")
                duration = item.get("duration_minutes", item.get("duration", 30))
                linked_output = item.get("linked_output", "")

                # Ensure duration is a string with units
                if isinstance(duration, int):
                    duration = f"{duration} min"

                learning_queue.append(LearningQueueItem(
                    question=question[:80] if isinstance(question, str) else str(question)[:80],
                    resource=resource[:50] if isinstance(resource, str) else str(resource)[:50],
                    duration=str(duration),
                    linked_output=linked_output[:50] if isinstance(linked_output, str) else str(linked_output)[:50]
                ))

        # If no learning items, add placeholder with content plan link
        if not learning_queue and content_plan:
            for plan_item in content_plan[:2]:
                if plan_item.platform != "-":
                    learning_queue.append(LearningQueueItem(
                        question=f"Research for: {plan_item.topic[:30]}",
                        resource="TBD - identify resource",
                        duration="30 min",
                        linked_output=f"{plan_item.day} {plan_item.platform} post"
                    ))

        # Always ensure at least one item with warning if queue is empty
        if not learning_queue:
            learning_queue.append(LearningQueueItem(
                question="No learning items queued",
                resource="N/A",
                duration="0 min",
                linked_output="WARNING: Learning queue empty - intentional?"
            ))

        return learning_queue

    def get_builder_constraints(
        self,
        shipping_report: Dict = None
    ) -> BuilderConstraint:
        """
        Determine builder work constraints for next week.

        Building is either:
        - SCHEDULED: Specific projects with time allocation
        - BANNED: No building until shipping resumes

        Args:
            shipping_report: From Shipping Governor

        Returns:
            BuilderConstraint with status and details
        """
        shipping_report = shipping_report or {}

        # Parse shipping report
        if isinstance(shipping_report, str):
            # Handle string format from shipping governor
            is_critical = "CRITICAL" in shipping_report.upper()
            is_warning = "WARNING" in shipping_report.upper()
        else:
            # Handle dict format
            health = shipping_report.get("overall_health", shipping_report.get("health", ""))
            is_critical = health == "CRITICAL" or "CRITICAL" in str(shipping_report)
            is_warning = health == "WARNING" or "WARNING" in str(shipping_report)

        # Extract project info
        projects = []
        if isinstance(shipping_report, dict):
            project_assessments = shipping_report.get("project_assessments", [])
            for proj in project_assessments:
                if isinstance(proj, dict):
                    projects.append({
                        "name": proj.get("name", "Unknown project"),
                        "days_without_output": proj.get("days_without_output", 0),
                        "recommended_action": proj.get("recommended_action", "unknown")
                    })

        if is_critical:
            return BuilderConstraint(
                status=BuilderStatus.BANNED.value,
                details="BANNED: No building until shipping resumes. Shipping is critically stalled.",
                projects=projects,
                hours_allocated=0
            )
        elif is_warning:
            return BuilderConstraint(
                status=BuilderStatus.BANNED.value,
                details="BANNED: No new building. Focus on shipping existing projects.",
                projects=projects,
                hours_allocated=0
            )
        else:
            # Healthy - can schedule building
            active_projects = [p for p in projects if p.get("recommended_action") == "continue"]
            if active_projects:
                project_names = ", ".join([p["name"] for p in active_projects[:2]])
                return BuilderConstraint(
                    status=BuilderStatus.SCHEDULED.value,
                    details=f"SCHEDULED: {project_names} - 4 hours total",
                    projects=active_projects,
                    hours_allocated=4
                )
            else:
                return BuilderConstraint(
                    status=BuilderStatus.SCHEDULED.value,
                    details="SCHEDULED: General building time available - 4 hours max",
                    projects=[],
                    hours_allocated=4
                )

    def _generate_non_negotiables(
        self,
        reputation_analysis: ReputationAnalysis,
        shipping_report: Dict,
        strategy_memo: Dict
    ) -> List[str]:
        """Generate next week's non-negotiable constraints."""
        non_negotiables = []

        # Reputation-based constraints
        if reputation_analysis.trajectory == ReputationTrajectory.AT_RISK.value:
            non_negotiables.append("No real-time replies or quote tweets until risk resolves")

        # Drift-based constraints
        memo_data = (strategy_memo or {}).get("STRATEGY_MEMO", {})
        stop_items = memo_data.get("stop_immediately", [])
        for item in stop_items[:1]:
            if isinstance(item, str):
                non_negotiables.append(item)

        # Shipping-based constraints
        if isinstance(shipping_report, dict):
            if shipping_report.get("overall_health") == "CRITICAL":
                non_negotiables.append("Ship one thing before any new building")

        # Default constraints
        if len(non_negotiables) < 2:
            non_negotiables.append("Maintain clinical reputation as top priority")
            non_negotiables.append("No learning without linked output")

        return non_negotiables[:3]  # Max 3 non-negotiables

    def _generate_directional_question(
        self,
        drift_analysis: DriftAnalysis,
        reputation_analysis: ReputationAnalysis,
        audience_signals: Dict
    ) -> str:
        """Generate the one directional question Alfred needs answered."""

        # Priority 1: Reputation at risk
        if reputation_analysis.trajectory == ReputationTrajectory.AT_RISK.value:
            risk_events = reputation_analysis.risk_events
            if risk_events:
                event = risk_events[0].get("event", "recent event")
                return f"Do you want to address '{event[:50]}' with a clarification, or let it pass?"
            return "What is your current tolerance for reputational risk exposure?"

        # Priority 2: Drift detected
        if drift_analysis.drift_detected:
            if drift_analysis.severity == "high":
                return f"Is the drift toward {drift_analysis.drift_direction.replace('_', ' ')} intentional or should we course-correct?"
            return "Are you comfortable with current positioning, or should we tighten alignment?"

        # Priority 3: Content direction
        signals_data = (audience_signals or {}).get("AUDIENCE_SIGNALS", {})
        questions = signals_data.get("top_5_questions", [])
        if questions:
            top_question = questions[0]
            if isinstance(top_question, dict):
                theme = top_question.get("theme", "topic")
                return f"Should we create dedicated content addressing '{theme}'?"

        # Default question
        return "What is the one thing you most want to accomplish next week?"

    def _format_brief(self, brief_data: WeeklyBriefData, strategy_memo: Dict = None) -> str:
        """Format the brief as the specified output string."""

        # Content plan table
        content_table_rows = []
        for item in brief_data.content_plan:
            content_table_rows.append(
                f"| {item.day:<9} | {item.platform:<8} | {item.topic:<35} | {item.status:<10} |"
            )
        content_table = "\n".join(content_table_rows)

        # Learning queue table
        learning_table_rows = []
        for item in brief_data.learning_queue:
            learning_table_rows.append(
                f"| {item.question:<30} | {item.resource:<20} | {item.duration:<8} | {item.linked_output:<20} |"
            )
        learning_table = "\n".join(learning_table_rows)

        # What worked bullets
        worked_bullets = "\n".join([
            f"- {item.item} ({item.evidence})"
            for item in brief_data.what_worked[:3]
        ])

        # What didn't work bullets
        didnt_bullets = "\n".join([
            f"- {item.item} ({item.evidence})"
            for item in brief_data.what_didnt_work[:3]
        ])

        # Non-negotiables
        non_neg_list = "\n".join([
            f"{i+1}. {item}"
            for i, item in enumerate(brief_data.non_negotiables)
        ])

        # Strategy memo section
        strategy_section = ""
        if strategy_memo:
            memo_data = strategy_memo.get("STRATEGY_MEMO", {})
            exec_summary = memo_data.get("executive_summary", "See attached memo.")
            strategy_section = f"""
---
STRATEGIC ASSESSMENT (Content Strategy Analyst)
{exec_summary}

For full analysis, see attached Strategy Memo ID: {memo_data.get('memo_id', 'N/A')}
"""
        else:
            strategy_section = """
---
Strategic assessment by Content Strategy Analyst not attached (no data provided).
"""

        formatted = f"""ALFRED WEEKLY STRATEGIC BRIEF
Week of {brief_data.week_start} to {brief_data.week_end}
========================================

REPUTATION TRAJECTORY: {brief_data.reputation_trajectory.trajectory}
{brief_data.reputation_trajectory.explanation}

DRIFT CHECK:
{brief_data.drift_analysis.summary}

WHAT WORKED THIS WEEK:
{worked_bullets}

WHAT DIDN'T WORK:
{didnt_bullets}

NEXT WEEK'S NON-NEGOTIABLES:
{non_neg_list}

QUESTION FOR YOU:
{brief_data.directional_question}

CONTENT PLAN (Next Week):
| Day       | Platform | Topic                               | Status     |
|-----------|----------|-------------------------------------|------------|
{content_table}

LEARNING QUEUE:
| Question                       | Resource             | Duration | Linked Output        |
|--------------------------------|----------------------|----------|----------------------|
{learning_table}

BUILDER WORK:
{brief_data.builder_constraint.details}
{strategy_section}

---
Brief ID: {brief_data.brief_id}
Generated: {brief_data.generated_at}
"""
        return formatted

    def _brief_to_dict(self, brief_data: WeeklyBriefData) -> Dict:
        """Convert WeeklyBriefData to dictionary."""
        return {
            "brief_id": brief_data.brief_id,
            "generated_at": brief_data.generated_at,
            "week_start": brief_data.week_start,
            "week_end": brief_data.week_end,
            "reputation_trajectory": {
                "trajectory": brief_data.reputation_trajectory.trajectory,
                "explanation": brief_data.reputation_trajectory.explanation,
                "risk_events": brief_data.reputation_trajectory.risk_events,
                "positive_signals": brief_data.reputation_trajectory.positive_signals,
                "trend_data": brief_data.reputation_trajectory.trend_data
            },
            "drift_analysis": {
                "summary": brief_data.drift_analysis.summary,
                "drift_detected": brief_data.drift_analysis.drift_detected,
                "drift_direction": brief_data.drift_analysis.drift_direction,
                "severity": brief_data.drift_analysis.severity,
                "correction_needed": brief_data.drift_analysis.correction_needed
            },
            "what_worked": [
                {"item": w.item, "evidence": w.evidence, "category": w.category}
                for w in brief_data.what_worked
            ],
            "what_didnt_work": [
                {"item": w.item, "evidence": w.evidence, "category": w.category}
                for w in brief_data.what_didnt_work
            ],
            "non_negotiables": brief_data.non_negotiables,
            "directional_question": brief_data.directional_question,
            "content_plan": [
                {
                    "day": c.day, "platform": c.platform, "topic": c.topic,
                    "status": c.status, "notes": c.notes
                }
                for c in brief_data.content_plan
            ],
            "learning_queue": [
                {
                    "question": l.question, "resource": l.resource,
                    "duration": l.duration, "linked_output": l.linked_output
                }
                for l in brief_data.learning_queue
            ],
            "builder_constraint": {
                "status": brief_data.builder_constraint.status,
                "details": brief_data.builder_constraint.details,
                "projects": brief_data.builder_constraint.projects,
                "hours_allocated": brief_data.builder_constraint.hours_allocated
            },
            "strategy_memo_attached": brief_data.strategy_memo_attached,
            "raw_data": brief_data.raw_data
        }

    def _save_brief(self, brief_data: WeeklyBriefData, formatted_brief: str) -> None:
        """Save the brief to storage."""
        # Save JSON version
        json_path = self.briefs_path / f"{brief_data.brief_id}.json"
        with open(json_path, "w") as f:
            json.dump(self._brief_to_dict(brief_data), f, indent=2)

        # Save formatted text version
        txt_path = self.briefs_path / f"{brief_data.brief_id}.txt"
        with open(txt_path, "w") as f:
            f.write(formatted_brief)

    def load_brief(self, brief_id: str) -> Optional[Dict]:
        """Load a previously saved brief."""
        json_path = self.briefs_path / f"{brief_id}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
        return None

    def list_briefs(self, limit: int = 10) -> List[Dict]:
        """List recent briefs."""
        briefs = []
        json_files = sorted(
            self.briefs_path.glob("WEEKLY_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for json_file in json_files[:limit]:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    briefs.append({
                        "brief_id": data.get("brief_id"),
                        "generated_at": data.get("generated_at"),
                        "week_start": data.get("week_start"),
                        "week_end": data.get("week_end"),
                        "reputation_trajectory": data.get("reputation_trajectory", {}).get("trajectory")
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        return briefs


# Convenience function for direct invocation
def generate_weekly_brief(**kwargs) -> Dict:
    """Generate a weekly brief with provided data."""
    brief_generator = WeeklyBrief()
    return brief_generator.generate_weekly_brief(**kwargs)
