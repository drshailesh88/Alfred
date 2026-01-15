"""
Content Strategy Analyst - The "mentor" agent for content strategy

Role: Compares actual content performance to the Positioning Charter, detects drift,
diagnoses failures, and produces weekly strategic memos. Provides the feedback loop
between output and positioning.

Does NOT:
- Execute or create content
- Make final strategic decisions (Alfred does)
- Optimize purely for metrics
- Recommend "post more" as solution
- Chase trends or virality
- Ignore positioning for performance
- Provide vague advice

Does:
- Compare performance to Positioning Charter
- Detect positioning drift
- Diagnose why content succeeded or failed
- Recommend specific adjustments
- Produce weekly strategic memos
- Track experiments and outcomes
- Identify what to double down on
- Flag what to stop immediately
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json
import statistics

from . import StrategyAgent, AgentResponse, AlfredState


class AlertType(Enum):
    """Types of strategy alerts."""
    DRIFT = "drift"
    PERFORMANCE_CLIFF = "performance_cliff"
    POSITIONING_VIOLATION = "positioning_violation"
    OPPORTUNITY = "opportunity"


class AlertUrgency(Enum):
    """Urgency levels for alerts."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SuccessLever(Enum):
    """Levers that drive content success."""
    FORMAT = "format"
    TOPIC = "topic"
    HOOK = "hook"
    TIMING = "timing"
    VOICE = "voice"
    DEPTH = "depth"


class ExperimentStatus(Enum):
    """Status of content experiments."""
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class VoiceRule:
    """A voice rule defining what to always or never do."""
    rule_type: str  # "always" or "never"
    description: str
    examples: List[str] = field(default_factory=list)


@dataclass
class PlatformGoal:
    """Goals for a specific platform."""
    platform: str
    success_definition: str
    primary_metric: str
    target_value: Optional[float] = None
    secondary_metrics: List[str] = field(default_factory=list)


@dataclass
class ContentPillar:
    """A content pillar defining a topic area."""
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    target_percentage: float = 20.0  # Target % of content in this pillar
    current_percentage: float = 0.0
    performance_trend: str = "stable"  # "improving", "declining", "stable"


@dataclass
class PositioningCharter:
    """
    The Positioning Charter defines who you are and who you are not.
    This is the foundation against which all content strategy is measured.
    """
    # Core identity
    who_you_are: str  # One sentence identity
    who_you_are_not: List[str]  # Hard boundaries

    # Audience focus
    priority_audience: str  # Primary audience this quarter
    secondary_audiences: List[str] = field(default_factory=list)

    # Content structure
    content_pillars: List[ContentPillar] = field(default_factory=list)

    # Voice and style
    voice_rules: List[VoiceRule] = field(default_factory=list)

    # Platform strategy
    platform_goals: List[PlatformGoal] = field(default_factory=list)

    # Metadata
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "who_you_are": self.who_you_are,
            "who_you_are_not": self.who_you_are_not,
            "priority_audience": self.priority_audience,
            "secondary_audiences": self.secondary_audiences,
            "content_pillars": [
                {
                    "name": p.name,
                    "description": p.description,
                    "keywords": p.keywords,
                    "target_percentage": p.target_percentage,
                    "current_percentage": p.current_percentage,
                    "performance_trend": p.performance_trend
                }
                for p in self.content_pillars
            ],
            "voice_rules": [
                {
                    "rule_type": r.rule_type,
                    "description": r.description,
                    "examples": r.examples
                }
                for r in self.voice_rules
            ],
            "platform_goals": [
                {
                    "platform": g.platform,
                    "success_definition": g.success_definition,
                    "primary_metric": g.primary_metric,
                    "target_value": g.target_value,
                    "secondary_metrics": g.secondary_metrics
                }
                for g in self.platform_goals
            ],
            "created_date": self.created_date,
            "last_updated": self.last_updated,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositioningCharter":
        pillars = [
            ContentPillar(
                name=p["name"],
                description=p["description"],
                keywords=p.get("keywords", []),
                target_percentage=p.get("target_percentage", 20.0),
                current_percentage=p.get("current_percentage", 0.0),
                performance_trend=p.get("performance_trend", "stable")
            )
            for p in data.get("content_pillars", [])
        ]

        voice_rules = [
            VoiceRule(
                rule_type=r["rule_type"],
                description=r["description"],
                examples=r.get("examples", [])
            )
            for r in data.get("voice_rules", [])
        ]

        platform_goals = [
            PlatformGoal(
                platform=g["platform"],
                success_definition=g["success_definition"],
                primary_metric=g["primary_metric"],
                target_value=g.get("target_value"),
                secondary_metrics=g.get("secondary_metrics", [])
            )
            for g in data.get("platform_goals", [])
        ]

        return cls(
            who_you_are=data["who_you_are"],
            who_you_are_not=data.get("who_you_are_not", []),
            priority_audience=data["priority_audience"],
            secondary_audiences=data.get("secondary_audiences", []),
            content_pillars=pillars,
            voice_rules=voice_rules,
            platform_goals=platform_goals,
            created_date=data.get("created_date", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            version=data.get("version", "1.0")
        )


@dataclass
class Experiment:
    """A content experiment being tracked."""
    id: str
    name: str
    hypothesis: str
    metric: str  # What we measure
    target_value: Optional[float] = None
    baseline_value: Optional[float] = None
    duration_days: int = 7
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    outcome: Optional[str] = None  # "success", "failure", "inconclusive"
    result_value: Optional[float] = None
    learnings: List[str] = field(default_factory=list)
    content_ids: List[str] = field(default_factory=list)  # Content in the experiment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "metric": self.metric,
            "target_value": self.target_value,
            "baseline_value": self.baseline_value,
            "duration_days": self.duration_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status.value,
            "outcome": self.outcome,
            "result_value": self.result_value,
            "learnings": self.learnings,
            "content_ids": self.content_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        return cls(
            id=data["id"],
            name=data["name"],
            hypothesis=data["hypothesis"],
            metric=data["metric"],
            target_value=data.get("target_value"),
            baseline_value=data.get("baseline_value"),
            duration_days=data.get("duration_days", 7),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            status=ExperimentStatus(data.get("status", "proposed")),
            outcome=data.get("outcome"),
            result_value=data.get("result_value"),
            learnings=data.get("learnings", []),
            content_ids=data.get("content_ids", [])
        )


@dataclass
class ContentPerformance:
    """Performance data for a piece of content."""
    content_id: str
    title: str
    platform: str
    pillar: Optional[str] = None
    published_date: Optional[str] = None

    # Metrics
    impressions: int = 0
    engagement: int = 0
    engagement_rate: float = 0.0
    saves: int = 0
    shares: int = 0
    comments: int = 0
    watch_time_minutes: float = 0.0
    completion_rate: float = 0.0
    click_through_rate: float = 0.0

    # Qualitative signals
    authority_signals: List[str] = field(default_factory=list)  # Peer engagement, etc.
    audience_feedback_themes: List[str] = field(default_factory=list)

    # Analysis
    performance_tier: str = "average"  # "top", "above_average", "average", "below_average", "poor"
    success_factors: List[str] = field(default_factory=list)
    failure_factors: List[str] = field(default_factory=list)


@dataclass
class DriftAnalysis:
    """Analysis of positioning drift."""
    drift_detected: bool = False
    drift_direction: Optional[str] = None  # What direction are we drifting toward
    drift_severity: str = "none"  # "none", "minor", "moderate", "severe"
    drift_source: Optional[str] = None  # What's causing the drift
    pillar_distribution: Dict[str, float] = field(default_factory=dict)
    off_pillar_percentage: float = 0.0
    voice_consistency_score: float = 100.0
    audience_alignment_score: float = 100.0
    correction_needed: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class StrategyRecommendation:
    """A strategic recommendation."""
    action: str  # "double_down", "stop", "adjust", "experiment"
    target: str  # What content/format/topic
    rationale: str
    priority: str = "medium"  # "high", "medium", "low"
    expected_impact: Optional[str] = None


@dataclass
class StrategyMemo:
    """The weekly strategy memo output."""
    memo_date: str
    period_analyzed: str

    # Executive summary
    executive_summary: str

    # Performance vs positioning
    alignment_score: float  # 0-100
    drift_analysis: DriftAnalysis = field(default_factory=DriftAnalysis)

    # What worked
    successes: List[Dict[str, Any]] = field(default_factory=list)

    # What didn't work
    failures: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    double_down: List[StrategyRecommendation] = field(default_factory=list)
    stop_immediately: List[StrategyRecommendation] = field(default_factory=list)

    # Experiments
    completed_experiments: List[Experiment] = field(default_factory=list)
    recommended_experiments: List[Experiment] = field(default_factory=list)

    # Strategic questions
    strategic_questions: List[str] = field(default_factory=list)

    # Scoreboard
    scoreboard: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memo_date": self.memo_date,
            "period_analyzed": self.period_analyzed,
            "executive_summary": self.executive_summary,
            "alignment_score": self.alignment_score,
            "drift_analysis": {
                "drift_detected": self.drift_analysis.drift_detected,
                "drift_direction": self.drift_analysis.drift_direction,
                "drift_severity": self.drift_analysis.drift_severity,
                "drift_source": self.drift_analysis.drift_source,
                "pillar_distribution": self.drift_analysis.pillar_distribution,
                "off_pillar_percentage": self.drift_analysis.off_pillar_percentage,
                "voice_consistency_score": self.drift_analysis.voice_consistency_score,
                "audience_alignment_score": self.drift_analysis.audience_alignment_score,
                "correction_needed": self.drift_analysis.correction_needed,
                "evidence": self.drift_analysis.evidence
            },
            "successes": self.successes,
            "failures": self.failures,
            "double_down": [
                {
                    "action": r.action,
                    "target": r.target,
                    "rationale": r.rationale,
                    "priority": r.priority,
                    "expected_impact": r.expected_impact
                }
                for r in self.double_down
            ],
            "stop_immediately": [
                {
                    "action": r.action,
                    "target": r.target,
                    "rationale": r.rationale,
                    "priority": r.priority,
                    "expected_impact": r.expected_impact
                }
                for r in self.stop_immediately
            ],
            "completed_experiments": [e.to_dict() for e in self.completed_experiments],
            "recommended_experiments": [e.to_dict() for e in self.recommended_experiments],
            "strategic_questions": self.strategic_questions,
            "scoreboard": self.scoreboard
        }


@dataclass
class StrategyAlert:
    """An urgent strategy alert."""
    alert_type: AlertType
    issue: str
    evidence: List[str]
    urgency: AlertUrgency
    recommended_action: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "issue": self.issue,
            "evidence": self.evidence,
            "urgency": self.urgency.value,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp
        }


class ContentStrategyAnalyst(StrategyAgent):
    """
    Content Strategy Analyst - The "mentor" agent for content strategy.

    Compares actual content performance to the Positioning Charter, detects drift,
    diagnoses failures, and produces weekly strategic memos.
    """

    # Performance thresholds
    TOP_PERFORMER_PERCENTILE = 80
    POOR_PERFORMER_PERCENTILE = 20
    DRIFT_THRESHOLD_MINOR = 15  # % deviation from target pillar distribution
    DRIFT_THRESHOLD_MODERATE = 25
    DRIFT_THRESHOLD_SEVERE = 40
    VOICE_CONSISTENCY_THRESHOLD = 70
    ALIGNMENT_WARNING_THRESHOLD = 70
    PERFORMANCE_CLIFF_THRESHOLD = -30  # % decline from previous period

    def __init__(self):
        super().__init__("Content Strategy Analyst")
        self._positioning_charter: Optional[PositioningCharter] = None
        self._experiments: List[Experiment] = []
        self._historical_memos: List[StrategyMemo] = []
        self._pending_alerts: List[StrategyAlert] = []

    @property
    def positioning_charter(self) -> Optional[PositioningCharter]:
        return self._positioning_charter

    @positioning_charter.setter
    def positioning_charter(self, charter: PositioningCharter):
        self._positioning_charter = charter

    def check_state_permission(self) -> Tuple[bool, str]:
        """Check if agent can operate in current state."""
        if self.alfred_state == AlfredState.RED:
            return False, "Strategy work paused in RED state - focus on recovery"
        return True, "Operation permitted"

    def analyze_period(
        self,
        period: str,
        metrics_data: Dict[str, Any],
        audience_signals: Dict[str, Any],
        positioning_charter: Optional[Dict[str, Any]] = None,
        active_experiments: Optional[List[Dict[str, Any]]] = None,
        specific_questions: Optional[List[str]] = None
    ) -> AgentResponse:
        """
        Main analysis method - produces the strategy memo.

        Args:
            period: The period being analyzed (e.g., "2024-W03" or "2024-01")
            metrics_data: Raw metrics from Social Metrics Harvester
            audience_signals: Qualitative data from Audience Signals Extractor
            positioning_charter: The positioning charter (if not already set)
            active_experiments: Currently running experiments
            specific_questions: Specific areas to focus analysis on

        Returns:
            AgentResponse containing the strategy memo
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Set or update positioning charter
        if positioning_charter:
            self._positioning_charter = PositioningCharter.from_dict(positioning_charter)

        if not self._positioning_charter:
            return self.create_response(
                data={"error": "No Positioning Charter defined"},
                success=False,
                errors=["Cannot analyze without a Positioning Charter. Please define one first."]
            )

        # Load active experiments
        if active_experiments:
            self._experiments = [
                Experiment.from_dict(e) for e in active_experiments
                if e.get("status") == "active"
            ]

        # Parse and structure the metrics data
        content_performance = self._parse_metrics_data(metrics_data)

        # Detect drift
        drift_analysis = self.detect_drift(content_performance)

        # Diagnose performance
        successes, failures = self.diagnose_performance(
            content_performance,
            audience_signals
        )

        # Score alignment
        alignment_score = self.score_alignment(
            content_performance,
            drift_analysis,
            audience_signals
        )

        # Evaluate completed experiments
        completed_experiments = self._evaluate_experiments(content_performance)

        # Generate recommendations
        double_down, stop_immediately = self._generate_recommendations(
            successes,
            failures,
            drift_analysis,
            content_performance
        )

        # Recommend new experiments
        recommended_experiments = self.recommend_experiments(
            successes,
            failures,
            specific_questions
        )

        # Generate strategic questions
        strategic_questions = self._generate_strategic_questions(
            drift_analysis,
            successes,
            failures,
            specific_questions
        )

        # Build scoreboard
        scoreboard = self._build_scoreboard(
            content_performance,
            drift_analysis,
            alignment_score
        )

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            alignment_score,
            drift_analysis,
            successes,
            failures,
            content_performance
        )

        # Build the memo
        memo = self.generate_memo(
            period=period,
            executive_summary=executive_summary,
            alignment_score=alignment_score,
            drift_analysis=drift_analysis,
            successes=successes,
            failures=failures,
            double_down=double_down,
            stop_immediately=stop_immediately,
            completed_experiments=completed_experiments,
            recommended_experiments=recommended_experiments,
            strategic_questions=strategic_questions,
            scoreboard=scoreboard
        )

        # Store memo for historical reference
        self._historical_memos.append(memo)

        # Check for alerts
        alerts = self._check_for_alerts(
            drift_analysis,
            content_performance,
            alignment_score
        )

        # Build response
        response_data = {
            "memo": memo.to_dict(),
            "alerts": [a.to_dict() for a in alerts]
        }

        warnings = []
        if drift_analysis.drift_detected:
            warnings.append(f"Positioning drift detected: {drift_analysis.drift_direction}")
        if alignment_score < self.ALIGNMENT_WARNING_THRESHOLD:
            warnings.append(f"Alignment score ({alignment_score}) below threshold")

        return self.create_response(
            data=response_data,
            success=True,
            warnings=warnings
        )

    def detect_drift(
        self,
        content_performance: List[ContentPerformance]
    ) -> DriftAnalysis:
        """
        Detect positioning drift by comparing content distribution to charter.

        Drift detection algorithm:
        1. Calculate actual pillar distribution from content
        2. Compare to target distribution from charter
        3. Identify off-pillar content
        4. Assess voice consistency
        5. Check audience alignment
        """
        if not self._positioning_charter:
            return DriftAnalysis()

        analysis = DriftAnalysis()

        # Calculate pillar distribution
        pillar_counts: Dict[str, int] = {}
        off_pillar_count = 0
        total_content = len(content_performance)

        if total_content == 0:
            return analysis

        # Define pillar names for matching
        pillar_names = {p.name.lower() for p in self._positioning_charter.content_pillars}

        for content in content_performance:
            if content.pillar:
                pillar_lower = content.pillar.lower()
                if pillar_lower in pillar_names:
                    pillar_counts[content.pillar] = pillar_counts.get(content.pillar, 0) + 1
                else:
                    off_pillar_count += 1
            else:
                off_pillar_count += 1

        # Calculate percentages
        for pillar_name in pillar_counts:
            analysis.pillar_distribution[pillar_name] = (
                pillar_counts[pillar_name] / total_content * 100
            )

        analysis.off_pillar_percentage = off_pillar_count / total_content * 100

        # Check for drift
        max_deviation = 0
        drift_direction_pillars = []

        for pillar in self._positioning_charter.content_pillars:
            actual_pct = analysis.pillar_distribution.get(pillar.name, 0)
            target_pct = pillar.target_percentage
            deviation = abs(actual_pct - target_pct)

            if deviation > max_deviation:
                max_deviation = deviation

            if actual_pct < target_pct - self.DRIFT_THRESHOLD_MINOR:
                drift_direction_pillars.append(f"under-indexing on {pillar.name}")
            elif actual_pct > target_pct + self.DRIFT_THRESHOLD_MINOR:
                drift_direction_pillars.append(f"over-indexing on {pillar.name}")

        # Off-pillar content check
        if analysis.off_pillar_percentage > self.DRIFT_THRESHOLD_MINOR:
            drift_direction_pillars.append(
                f"{analysis.off_pillar_percentage:.1f}% content outside defined pillars"
            )

        # Determine drift severity
        if max_deviation >= self.DRIFT_THRESHOLD_SEVERE or analysis.off_pillar_percentage > self.DRIFT_THRESHOLD_MODERATE:
            analysis.drift_severity = "severe"
            analysis.drift_detected = True
        elif max_deviation >= self.DRIFT_THRESHOLD_MODERATE or analysis.off_pillar_percentage > self.DRIFT_THRESHOLD_MINOR:
            analysis.drift_severity = "moderate"
            analysis.drift_detected = True
        elif max_deviation >= self.DRIFT_THRESHOLD_MINOR:
            analysis.drift_severity = "minor"
            analysis.drift_detected = True

        if drift_direction_pillars:
            analysis.drift_direction = "; ".join(drift_direction_pillars)
            analysis.evidence = drift_direction_pillars

        # Generate correction recommendation
        if analysis.drift_detected:
            analysis.correction_needed = self._generate_drift_correction(
                analysis,
                self._positioning_charter
            )

        return analysis

    def diagnose_performance(
        self,
        content_performance: List[ContentPerformance],
        audience_signals: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Diagnose why content succeeded or failed.

        Returns:
            Tuple of (successes, failures) with detailed diagnosis
        """
        if not content_performance:
            return [], []

        # Calculate performance benchmarks
        engagement_rates = [c.engagement_rate for c in content_performance if c.engagement_rate > 0]

        if engagement_rates:
            median_engagement = statistics.median(engagement_rates)
            top_threshold = statistics.quantiles(engagement_rates, n=5)[3] if len(engagement_rates) >= 5 else median_engagement * 1.5
            poor_threshold = statistics.quantiles(engagement_rates, n=5)[0] if len(engagement_rates) >= 5 else median_engagement * 0.5
        else:
            median_engagement = 0
            top_threshold = 0
            poor_threshold = 0

        successes = []
        failures = []

        # Extract audience themes for context
        trust_builders = audience_signals.get("trust_builders", [])
        confusions = audience_signals.get("confusions", [])
        praise_patterns = audience_signals.get("praise_patterns", [])

        for content in content_performance:
            # Determine performance tier
            if content.engagement_rate >= top_threshold and top_threshold > 0:
                content.performance_tier = "top"
            elif content.engagement_rate >= median_engagement:
                content.performance_tier = "above_average"
            elif content.engagement_rate >= poor_threshold and poor_threshold > 0:
                content.performance_tier = "average"
            elif content.engagement_rate > 0:
                content.performance_tier = "below_average"
            else:
                content.performance_tier = "poor"

            # Analyze successes
            if content.performance_tier in ["top", "above_average"]:
                success_factors = self._identify_success_factors(content, audience_signals)
                lever = self._identify_primary_lever(content, success_factors)

                successes.append({
                    "content": content.title,
                    "content_id": content.content_id,
                    "platform": content.platform,
                    "pillar": content.pillar,
                    "evidence": {
                        "engagement_rate": content.engagement_rate,
                        "impressions": content.impressions,
                        "saves": content.saves,
                        "shares": content.shares,
                        "authority_signals": content.authority_signals
                    },
                    "why": self._explain_success(content, success_factors, audience_signals),
                    "lever": lever.value,
                    "success_factors": success_factors
                })

            # Analyze failures
            elif content.performance_tier in ["below_average", "poor"]:
                failure_factors = self._identify_failure_factors(content, audience_signals)
                lesson = self._extract_lesson(content, failure_factors)

                failures.append({
                    "content": content.title,
                    "content_id": content.content_id,
                    "platform": content.platform,
                    "pillar": content.pillar,
                    "evidence": {
                        "engagement_rate": content.engagement_rate,
                        "impressions": content.impressions,
                        "completion_rate": content.completion_rate
                    },
                    "diagnosis": self._diagnose_failure(content, failure_factors, audience_signals),
                    "lesson": lesson,
                    "failure_factors": failure_factors
                })

        # Sort by impact
        successes.sort(key=lambda x: x["evidence"].get("engagement_rate", 0), reverse=True)
        failures.sort(key=lambda x: x["evidence"].get("engagement_rate", 0))

        return successes[:5], failures[:5]  # Top 5 of each

    def score_alignment(
        self,
        content_performance: List[ContentPerformance],
        drift_analysis: DriftAnalysis,
        audience_signals: Dict[str, Any]
    ) -> float:
        """
        Calculate alignment score (0-100) based on:
        - Pillar distribution adherence (40%)
        - Voice consistency (25%)
        - Audience alignment (25%)
        - Goal progress (10%)
        """
        if not self._positioning_charter:
            return 0.0

        score = 0.0

        # 1. Pillar distribution score (40 points max)
        pillar_score = 40.0
        if drift_analysis.pillar_distribution:
            deviations = []
            for pillar in self._positioning_charter.content_pillars:
                actual = drift_analysis.pillar_distribution.get(pillar.name, 0)
                target = pillar.target_percentage
                deviation = abs(actual - target)
                deviations.append(deviation)

            if deviations:
                avg_deviation = sum(deviations) / len(deviations)
                # Subtract points for deviation (max 40 point loss)
                pillar_score = max(0, 40 - avg_deviation)

        # Penalize for off-pillar content
        pillar_score *= (1 - drift_analysis.off_pillar_percentage / 100)
        score += pillar_score

        # 2. Voice consistency score (25 points max)
        voice_score = self._calculate_voice_score(content_performance)
        score += voice_score * 0.25
        drift_analysis.voice_consistency_score = voice_score

        # 3. Audience alignment score (25 points max)
        audience_score = self._calculate_audience_alignment(audience_signals)
        score += audience_score * 0.25
        drift_analysis.audience_alignment_score = audience_score

        # 4. Goal progress score (10 points max)
        goal_score = self._calculate_goal_progress(content_performance)
        score += goal_score * 0.10

        return min(100, max(0, score))

    def recommend_experiments(
        self,
        successes: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        specific_questions: Optional[List[str]] = None
    ) -> List[Experiment]:
        """
        Recommend up to 3 experiments for the next period.

        Experiment sources:
        1. Amplify what's working (test if it scales)
        2. Diagnose failures (test hypothesis about why)
        3. Answer specific questions from Alfred
        """
        experiments = []
        experiment_id = datetime.now().strftime("%Y%m%d")

        # Experiment 1: Scale what's working
        if successes:
            top_success = successes[0]
            lever = top_success.get("lever", "topic")

            experiments.append(Experiment(
                id=f"{experiment_id}-scale-{len(experiments)+1}",
                name=f"Scale {lever} from '{top_success['content'][:30]}...'",
                hypothesis=f"The {lever} that worked in '{top_success['content'][:30]}...' will produce similar results when applied to other content",
                metric="engagement_rate",
                duration_days=7,
                status=ExperimentStatus.PROPOSED,
                learnings=[f"Based on success of: {top_success['content']}"]
            ))

        # Experiment 2: Test failure hypothesis
        if failures:
            top_failure = failures[0]
            diagnosis = top_failure.get("diagnosis", "")

            # Create counter-hypothesis experiment
            experiments.append(Experiment(
                id=f"{experiment_id}-diagnose-{len(experiments)+1}",
                name=f"Test alternative approach for '{top_failure['pillar'] or 'content'}'",
                hypothesis=f"Addressing the issue ({diagnosis[:50]}...) will improve performance",
                metric="engagement_rate",
                duration_days=7,
                status=ExperimentStatus.PROPOSED,
                learnings=[f"Based on failure of: {top_failure['content']}"]
            ))

        # Experiment 3: Answer specific question or explore opportunity
        if specific_questions and len(specific_questions) > 0:
            question = specific_questions[0]
            experiments.append(Experiment(
                id=f"{experiment_id}-explore-{len(experiments)+1}",
                name=f"Answer: {question[:40]}...",
                hypothesis=f"Testing will answer: {question}",
                metric="engagement_rate",  # Default metric
                duration_days=7,
                status=ExperimentStatus.PROPOSED,
                learnings=[f"Exploring question: {question}"]
            ))
        elif len(experiments) < 3 and self._positioning_charter:
            # Find underrepresented pillar
            for pillar in self._positioning_charter.content_pillars:
                if pillar.performance_trend == "declining":
                    experiments.append(Experiment(
                        id=f"{experiment_id}-revive-{len(experiments)+1}",
                        name=f"Revive {pillar.name} pillar performance",
                        hypothesis=f"A new approach to {pillar.name} content will reverse declining engagement",
                        metric="engagement_rate",
                        duration_days=7,
                        status=ExperimentStatus.PROPOSED,
                        learnings=[f"Pillar '{pillar.name}' showing declining performance"]
                    ))
                    break

        return experiments[:3]  # Max 3 experiments

    def generate_memo(
        self,
        period: str,
        executive_summary: str,
        alignment_score: float,
        drift_analysis: DriftAnalysis,
        successes: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        double_down: List[StrategyRecommendation],
        stop_immediately: List[StrategyRecommendation],
        completed_experiments: List[Experiment],
        recommended_experiments: List[Experiment],
        strategic_questions: List[str],
        scoreboard: Dict[str, Any]
    ) -> StrategyMemo:
        """Generate the complete strategy memo."""
        return StrategyMemo(
            memo_date=datetime.now().isoformat(),
            period_analyzed=period,
            executive_summary=executive_summary,
            alignment_score=alignment_score,
            drift_analysis=drift_analysis,
            successes=successes,
            failures=failures,
            double_down=double_down,
            stop_immediately=stop_immediately,
            completed_experiments=completed_experiments,
            recommended_experiments=recommended_experiments,
            strategic_questions=strategic_questions,
            scoreboard=scoreboard
        )

    def create_alert(
        self,
        alert_type: AlertType,
        issue: str,
        evidence: List[str],
        urgency: AlertUrgency,
        recommended_action: str
    ) -> StrategyAlert:
        """Create a strategy alert."""
        alert = StrategyAlert(
            alert_type=alert_type,
            issue=issue,
            evidence=evidence,
            urgency=urgency,
            recommended_action=recommended_action
        )
        self._pending_alerts.append(alert)
        return alert

    def get_pending_alerts(self) -> List[StrategyAlert]:
        """Get and clear pending alerts."""
        alerts = self._pending_alerts.copy()
        self._pending_alerts.clear()
        return alerts

    # ========== Private Helper Methods ==========

    def _parse_metrics_data(self, metrics_data: Dict[str, Any]) -> List[ContentPerformance]:
        """Parse raw metrics data into ContentPerformance objects."""
        content_list = []

        # Handle the content performance matrix from harvester
        content_matrix = metrics_data.get("content_performance_matrix", [])
        if isinstance(content_matrix, list):
            for item in content_matrix:
                content = ContentPerformance(
                    content_id=item.get("content_id", str(hash(item.get("title", "")))),
                    title=item.get("title", item.get("content", "Unknown")),
                    platform=item.get("platform", "unknown"),
                    pillar=item.get("pillar"),
                    published_date=item.get("published_date"),
                    impressions=item.get("impressions", item.get("reach", 0)),
                    engagement=item.get("engagement", 0),
                    engagement_rate=item.get("engagement_rate", item.get("rate", 0)),
                    saves=item.get("saves", item.get("bookmarks", 0)),
                    shares=item.get("shares", item.get("retweets", 0)),
                    comments=item.get("comments", 0),
                    watch_time_minutes=item.get("watch_time", 0),
                    completion_rate=item.get("completion_rate", item.get("avg_completion", 0)),
                    click_through_rate=item.get("click_rate", item.get("ctr", 0)),
                    authority_signals=item.get("authority_signals", [])
                )
                content_list.append(content)

        # Also parse platform-specific data
        for platform in ["twitter", "youtube", "substack", "instagram"]:
            platform_data = metrics_data.get(platform, metrics_data.get(f"platform_{platform}", {}))
            if platform_data:
                top_performing = platform_data.get("top_performing")
                if top_performing and isinstance(top_performing, dict):
                    # Check if already in list
                    if not any(c.title == top_performing.get("title") for c in content_list):
                        content_list.append(ContentPerformance(
                            content_id=top_performing.get("id", str(hash(top_performing.get("title", "")))),
                            title=top_performing.get("title", "Unknown"),
                            platform=platform,
                            impressions=top_performing.get("impressions", top_performing.get("views", 0)),
                            engagement_rate=top_performing.get("engagement_rate", 0)
                        ))

        return content_list

    def _identify_success_factors(
        self,
        content: ContentPerformance,
        audience_signals: Dict[str, Any]
    ) -> List[str]:
        """Identify what factors contributed to success."""
        factors = []

        # High save rate indicates valuable content
        if content.saves > 0 and content.impressions > 0:
            save_rate = content.saves / content.impressions
            if save_rate > 0.01:  # > 1% save rate
                factors.append("High save rate - indicates reference value")

        # High share rate indicates resonance
        if content.shares > 0 and content.impressions > 0:
            share_rate = content.shares / content.impressions
            if share_rate > 0.01:
                factors.append("High share rate - content resonates")

        # Authority signals
        if content.authority_signals:
            factors.append(f"Authority signals: {', '.join(content.authority_signals[:3])}")

        # Completion rate for video
        if content.completion_rate > 50:
            factors.append(f"Strong completion rate ({content.completion_rate}%) - holding attention")

        # Check alignment with trust builders
        trust_builders = audience_signals.get("trust_builders", [])
        for builder in trust_builders:
            if isinstance(builder, dict) and builder.get("content_id") == content.content_id:
                factors.append(f"Trust building: {builder.get('reason', 'builds credibility')}")

        # Pillar alignment
        if content.pillar and self._positioning_charter:
            for pillar in self._positioning_charter.content_pillars:
                if pillar.name.lower() == content.pillar.lower():
                    if pillar.performance_trend == "improving":
                        factors.append(f"In high-performing pillar: {pillar.name}")

        if not factors:
            factors.append("Above-average engagement - investigating specific factors")

        return factors

    def _identify_failure_factors(
        self,
        content: ContentPerformance,
        audience_signals: Dict[str, Any]
    ) -> List[str]:
        """Identify what factors contributed to failure."""
        factors = []

        # Low impressions might indicate distribution issue
        if content.impressions < 100:  # Low threshold
            factors.append("Low distribution - content not reaching audience")

        # Low completion rate for video
        if content.watch_time_minutes > 0 and content.completion_rate < 30:
            factors.append(f"Low completion rate ({content.completion_rate}%) - losing audience early")

        # Check for confusion signals
        confusions = audience_signals.get("confusions", [])
        for confusion in confusions:
            if isinstance(confusion, dict):
                if confusion.get("content_id") == content.content_id:
                    factors.append(f"Audience confusion: {confusion.get('issue', 'unclear messaging')}")

        # Check for trust killers
        trust_killers = audience_signals.get("trust_killers", [])
        for killer in trust_killers:
            if isinstance(killer, dict):
                if killer.get("content_id") == content.content_id:
                    factors.append(f"Trust issue: {killer.get('reason', 'damaged credibility')}")

        # Off-pillar content
        if not content.pillar:
            factors.append("Content not aligned with any pillar")
        elif self._positioning_charter:
            pillar_names = {p.name.lower() for p in self._positioning_charter.content_pillars}
            if content.pillar.lower() not in pillar_names:
                factors.append(f"Off-pillar content: {content.pillar}")

        # Low engagement despite impressions
        if content.impressions > 1000 and content.engagement_rate < 1:
            factors.append("Low engagement despite reach - content not resonating")

        if not factors:
            factors.append("Below-average engagement - requires deeper analysis")

        return factors

    def _identify_primary_lever(
        self,
        content: ContentPerformance,
        success_factors: List[str]
    ) -> SuccessLever:
        """Identify the primary lever driving success."""
        # Analyze success factors to determine primary lever
        factors_text = " ".join(success_factors).lower()

        if "save" in factors_text or "reference" in factors_text:
            return SuccessLever.DEPTH
        elif "share" in factors_text or "resonat" in factors_text:
            return SuccessLever.TOPIC
        elif "completion" in factors_text or "attention" in factors_text:
            return SuccessLever.HOOK
        elif "authority" in factors_text or "credib" in factors_text:
            return SuccessLever.VOICE
        elif "pillar" in factors_text:
            return SuccessLever.TOPIC
        else:
            return SuccessLever.FORMAT

    def _explain_success(
        self,
        content: ContentPerformance,
        success_factors: List[str],
        audience_signals: Dict[str, Any]
    ) -> str:
        """Generate an explanation of why content succeeded."""
        if not success_factors:
            return "Content performed above average without clear differentiating factors."

        # Build explanation
        primary_factor = success_factors[0]

        explanation = f"Primary driver: {primary_factor}"

        if len(success_factors) > 1:
            explanation += f". Additional factors: {'; '.join(success_factors[1:3])}"

        # Add audience context if available
        praise = audience_signals.get("praise_patterns", [])
        if praise:
            if isinstance(praise[0], dict):
                explanation += f". Audience feedback indicates: {praise[0].get('theme', 'positive reception')}"
            elif isinstance(praise[0], str):
                explanation += f". Audience feedback: {praise[0]}"

        return explanation

    def _diagnose_failure(
        self,
        content: ContentPerformance,
        failure_factors: List[str],
        audience_signals: Dict[str, Any]
    ) -> str:
        """Generate a diagnosis of why content failed."""
        if not failure_factors:
            return "Content underperformed without clear cause - may indicate audience mismatch or timing issue."

        primary_factor = failure_factors[0]
        diagnosis = f"Primary issue: {primary_factor}"

        if len(failure_factors) > 1:
            diagnosis += f". Contributing factors: {'; '.join(failure_factors[1:3])}"

        return diagnosis

    def _extract_lesson(
        self,
        content: ContentPerformance,
        failure_factors: List[str]
    ) -> str:
        """Extract a lesson from a content failure."""
        if not failure_factors:
            return "Monitor similar content closely to identify patterns."

        # Map factors to lessons
        factor_text = " ".join(failure_factors).lower()

        if "distribution" in factor_text or "reach" in factor_text:
            return "Review distribution strategy - content may need different timing or platform approach"
        elif "completion" in factor_text or "attention" in factor_text:
            return "Improve hook or restructure content - losing audience too early"
        elif "confusion" in factor_text:
            return "Simplify messaging - audience is getting lost"
        elif "trust" in factor_text or "credib" in factor_text:
            return "Review tone and claims - may be overstepping expertise or positioning"
        elif "pillar" in factor_text or "align" in factor_text:
            return "Stay closer to core pillars - off-topic content underperforms"
        elif "resonat" in factor_text or "engagement" in factor_text:
            return "Topic may not match audience interest - validate before creating similar content"
        else:
            return "Analyze specific failure factors and avoid repeating in future content"

    def _generate_drift_correction(
        self,
        drift_analysis: DriftAnalysis,
        charter: PositioningCharter
    ) -> str:
        """Generate specific correction recommendation for drift."""
        corrections = []

        # Address pillar imbalances
        for pillar in charter.content_pillars:
            actual = drift_analysis.pillar_distribution.get(pillar.name, 0)
            target = pillar.target_percentage

            if actual < target - self.DRIFT_THRESHOLD_MINOR:
                corrections.append(
                    f"Increase {pillar.name} content from {actual:.1f}% to {target:.1f}%"
                )
            elif actual > target + self.DRIFT_THRESHOLD_MINOR:
                corrections.append(
                    f"Reduce {pillar.name} content from {actual:.1f}% to {target:.1f}%"
                )

        # Address off-pillar content
        if drift_analysis.off_pillar_percentage > self.DRIFT_THRESHOLD_MINOR:
            corrections.append(
                f"Reduce off-pillar content from {drift_analysis.off_pillar_percentage:.1f}% - "
                f"classify or retire content outside pillars: {', '.join(p.name for p in charter.content_pillars)}"
            )

        if corrections:
            return " | ".join(corrections)
        return "Minor drift detected - continue monitoring"

    def _calculate_voice_score(self, content_performance: List[ContentPerformance]) -> float:
        """Calculate voice consistency score based on content signals."""
        if not content_performance or not self._positioning_charter:
            return 100.0

        # This would ideally analyze actual content for voice consistency
        # For now, use proxy metrics

        score = 100.0

        # Penalize for content without authority signals (may indicate off-voice)
        content_with_authority = sum(
            1 for c in content_performance if c.authority_signals
        )
        authority_rate = content_with_authority / len(content_performance) if content_performance else 0

        # If less than 50% of content has authority signals, reduce score
        if authority_rate < 0.5:
            score -= (0.5 - authority_rate) * 40

        return max(0, min(100, score))

    def _calculate_audience_alignment(self, audience_signals: Dict[str, Any]) -> float:
        """Calculate audience alignment score."""
        if not audience_signals:
            return 50.0  # Neutral without data

        score = 70.0  # Start at baseline

        # Positive signals increase score
        trust_builders = audience_signals.get("trust_builders", [])
        praise_patterns = audience_signals.get("praise_patterns", [])

        score += min(15, len(trust_builders) * 3)
        score += min(10, len(praise_patterns) * 2)

        # Negative signals decrease score
        trust_killers = audience_signals.get("trust_killers", [])
        confusions = audience_signals.get("confusions", [])

        score -= min(20, len(trust_killers) * 5)
        score -= min(10, len(confusions) * 2)

        return max(0, min(100, score))

    def _calculate_goal_progress(self, content_performance: List[ContentPerformance]) -> float:
        """Calculate progress toward platform goals."""
        if not self._positioning_charter or not self._positioning_charter.platform_goals:
            return 50.0

        if not content_performance:
            return 0.0

        goal_scores = []

        for goal in self._positioning_charter.platform_goals:
            platform_content = [
                c for c in content_performance
                if c.platform.lower() == goal.platform.lower()
            ]

            if not platform_content or not goal.target_value:
                continue

            # Calculate average performance for the primary metric
            metric_values = []
            for content in platform_content:
                if goal.primary_metric == "engagement_rate":
                    metric_values.append(content.engagement_rate)
                elif goal.primary_metric == "impressions":
                    metric_values.append(content.impressions)
                elif goal.primary_metric == "completion_rate":
                    metric_values.append(content.completion_rate)

            if metric_values:
                avg_value = sum(metric_values) / len(metric_values)
                goal_score = min(100, (avg_value / goal.target_value) * 100)
                goal_scores.append(goal_score)

        return sum(goal_scores) / len(goal_scores) if goal_scores else 50.0

    def _generate_recommendations(
        self,
        successes: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        drift_analysis: DriftAnalysis,
        content_performance: List[ContentPerformance]
    ) -> Tuple[List[StrategyRecommendation], List[StrategyRecommendation]]:
        """Generate double-down and stop-immediately recommendations."""
        double_down = []
        stop_immediately = []

        # Double down on successes
        for success in successes[:3]:
            lever = success.get("lever", "topic")
            double_down.append(StrategyRecommendation(
                action="double_down",
                target=f"{lever} from '{success['content'][:40]}...'",
                rationale=success.get("why", "Strong performance indicates audience interest"),
                priority="high" if success == successes[0] else "medium",
                expected_impact="Continued high engagement in this area"
            ))

        # Stop what's failing consistently
        failure_patterns = {}
        for failure in failures:
            pillar = failure.get("pillar", "unknown")
            if pillar not in failure_patterns:
                failure_patterns[pillar] = []
            failure_patterns[pillar].append(failure)

        for pillar, pillar_failures in failure_patterns.items():
            if len(pillar_failures) >= 2:  # Multiple failures in same pillar
                stop_immediately.append(StrategyRecommendation(
                    action="stop",
                    target=f"Current approach to {pillar} content",
                    rationale=f"Multiple underperformers ({len(pillar_failures)}) in this area - needs rethinking",
                    priority="high",
                    expected_impact="Stop resource waste on failing approach"
                ))

        # Stop drift-causing behavior
        if drift_analysis.drift_detected and drift_analysis.drift_severity in ["moderate", "severe"]:
            stop_immediately.append(StrategyRecommendation(
                action="stop",
                target=f"Off-pillar content creation ({drift_analysis.off_pillar_percentage:.1f}%)",
                rationale=drift_analysis.correction_needed or "Drift from positioning charter",
                priority="high" if drift_analysis.drift_severity == "severe" else "medium",
                expected_impact="Restore positioning alignment"
            ))

        return double_down, stop_immediately

    def _evaluate_experiments(
        self,
        content_performance: List[ContentPerformance]
    ) -> List[Experiment]:
        """Evaluate active experiments and determine outcomes."""
        completed = []

        for experiment in self._experiments:
            if experiment.status != ExperimentStatus.ACTIVE:
                continue

            # Check if experiment period has ended
            if experiment.end_date:
                end = datetime.fromisoformat(experiment.end_date)
                if datetime.now() < end:
                    continue

            # Find content in this experiment
            experiment_content = [
                c for c in content_performance
                if c.content_id in experiment.content_ids
            ]

            if not experiment_content:
                experiment.status = ExperimentStatus.COMPLETED
                experiment.outcome = "inconclusive"
                experiment.learnings.append("No content data available for experiment period")
                completed.append(experiment)
                continue

            # Calculate experiment metrics
            if experiment.metric == "engagement_rate":
                values = [c.engagement_rate for c in experiment_content]
                experiment.result_value = sum(values) / len(values) if values else 0

            # Determine outcome
            if experiment.baseline_value and experiment.result_value:
                change = (experiment.result_value - experiment.baseline_value) / experiment.baseline_value * 100

                if change >= 10:  # 10% improvement threshold
                    experiment.outcome = "success"
                    experiment.learnings.append(
                        f"Hypothesis confirmed: {change:.1f}% improvement over baseline"
                    )
                elif change <= -10:
                    experiment.outcome = "failure"
                    experiment.learnings.append(
                        f"Hypothesis rejected: {abs(change):.1f}% decline from baseline"
                    )
                else:
                    experiment.outcome = "inconclusive"
                    experiment.learnings.append(
                        f"Results inconclusive: {change:.1f}% change within noise range"
                    )
            else:
                experiment.outcome = "inconclusive"
                experiment.learnings.append("Insufficient baseline data for comparison")

            experiment.status = ExperimentStatus.COMPLETED
            completed.append(experiment)

        return completed

    def _generate_strategic_questions(
        self,
        drift_analysis: DriftAnalysis,
        successes: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        specific_questions: Optional[List[str]]
    ) -> List[str]:
        """Generate strategic questions for Alfred/user consideration."""
        questions = []

        # Questions from drift
        if drift_analysis.drift_detected:
            if drift_analysis.drift_severity == "severe":
                questions.append(
                    f"Is the current drift ({drift_analysis.drift_direction}) intentional, "
                    "or should we course-correct immediately?"
                )
            elif drift_analysis.off_pillar_percentage > 20:
                questions.append(
                    "Should we add a new pillar to accommodate emerging content themes, "
                    "or recommit to existing pillars?"
                )

        # Questions from patterns
        if successes and failures:
            success_pillars = {s.get("pillar") for s in successes if s.get("pillar")}
            failure_pillars = {f.get("pillar") for f in failures if f.get("pillar")}

            struggling_pillars = failure_pillars - success_pillars
            if struggling_pillars:
                questions.append(
                    f"Pillar(s) {', '.join(struggling_pillars)} are underperforming - "
                    "should we adjust approach or reduce focus?"
                )

        # Questions about audience
        if self._positioning_charter:
            questions.append(
                f"Is '{self._positioning_charter.priority_audience}' still the right priority audience "
                "based on engagement patterns?"
            )

        # Include user's specific questions
        if specific_questions:
            questions.extend(specific_questions[:3])

        # Cap at 5 questions
        return questions[:5]

    def _build_scoreboard(
        self,
        content_performance: List[ContentPerformance],
        drift_analysis: DriftAnalysis,
        alignment_score: float
    ) -> Dict[str, Any]:
        """Build the positioning scoreboard."""
        # Calculate trends
        pillar_trends = {}
        if self._positioning_charter:
            for pillar in self._positioning_charter.content_pillars:
                pillar_content = [
                    c for c in content_performance
                    if c.pillar and c.pillar.lower() == pillar.name.lower()
                ]

                if pillar_content:
                    avg_engagement = sum(c.engagement_rate for c in pillar_content) / len(pillar_content)
                    pillar_trends[pillar.name] = {
                        "content_count": len(pillar_content),
                        "avg_engagement": avg_engagement,
                        "current_percentage": drift_analysis.pillar_distribution.get(pillar.name, 0),
                        "target_percentage": pillar.target_percentage
                    }

        # Count authority signals
        total_authority_signals = sum(
            len(c.authority_signals) for c in content_performance
        )

        # Calculate conversion metrics
        total_saves = sum(c.saves for c in content_performance)
        total_shares = sum(c.shares for c in content_performance)

        return {
            "alignment_score": alignment_score,
            "pillar_distribution": drift_analysis.pillar_distribution,
            "pillar_trends": pillar_trends,
            "voice_consistency_score": drift_analysis.voice_consistency_score,
            "audience_alignment_score": drift_analysis.audience_alignment_score,
            "authority_signals_count": total_authority_signals,
            "total_saves": total_saves,
            "total_shares": total_shares,
            "content_analyzed": len(content_performance),
            "off_pillar_percentage": drift_analysis.off_pillar_percentage
        }

    def _generate_executive_summary(
        self,
        alignment_score: float,
        drift_analysis: DriftAnalysis,
        successes: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        content_performance: List[ContentPerformance]
    ) -> str:
        """Generate 2-3 sentence executive summary."""
        parts = []

        # Overall health assessment
        if alignment_score >= 80:
            parts.append("Strong alignment with positioning charter this period.")
        elif alignment_score >= 60:
            parts.append("Moderate alignment with positioning charter - some areas need attention.")
        else:
            parts.append("Significant positioning concerns this period requiring immediate attention.")

        # Key highlight
        if successes:
            top_lever = successes[0].get("lever", "content")
            parts.append(
                f"Top performer driven by {top_lever}; "
                f"consider scaling this approach."
            )

        # Key concern
        if drift_analysis.drift_detected and drift_analysis.drift_severity != "minor":
            parts.append(
                f"Alert: {drift_analysis.drift_severity} drift detected - {drift_analysis.drift_direction}."
            )
        elif failures:
            failure_pattern = failures[0].get("diagnosis", "underperformance")
            parts.append(f"Address: {failure_pattern[:60]}...")

        return " ".join(parts)

    def _check_for_alerts(
        self,
        drift_analysis: DriftAnalysis,
        content_performance: List[ContentPerformance],
        alignment_score: float
    ) -> List[StrategyAlert]:
        """Check for conditions that warrant alerts."""
        alerts = []

        # Severe drift alert
        if drift_analysis.drift_severity == "severe":
            alerts.append(self.create_alert(
                alert_type=AlertType.DRIFT,
                issue=f"Severe positioning drift: {drift_analysis.drift_direction}",
                evidence=drift_analysis.evidence,
                urgency=AlertUrgency.HIGH,
                recommended_action=drift_analysis.correction_needed or "Review and correct positioning immediately"
            ))

        # Low alignment alert
        if alignment_score < 50:
            alerts.append(self.create_alert(
                alert_type=AlertType.POSITIONING_VIOLATION,
                issue=f"Critical alignment score: {alignment_score:.1f}/100",
                evidence=[
                    f"Voice consistency: {drift_analysis.voice_consistency_score:.1f}",
                    f"Audience alignment: {drift_analysis.audience_alignment_score:.1f}",
                    f"Off-pillar content: {drift_analysis.off_pillar_percentage:.1f}%"
                ],
                urgency=AlertUrgency.HIGH,
                recommended_action="Pause new content creation and realign with positioning charter"
            ))

        # Performance cliff detection
        if len(self._historical_memos) >= 2:
            previous_score = self._historical_memos[-2].alignment_score
            change = alignment_score - previous_score
            if change <= self.PERFORMANCE_CLIFF_THRESHOLD:
                alerts.append(self.create_alert(
                    alert_type=AlertType.PERFORMANCE_CLIFF,
                    issue=f"Performance cliff: {change:.1f}% decline from previous period",
                    evidence=[
                        f"Previous alignment: {previous_score:.1f}",
                        f"Current alignment: {alignment_score:.1f}"
                    ],
                    urgency=AlertUrgency.HIGH,
                    recommended_action="Investigate cause of rapid decline and implement recovery plan"
                ))

        return alerts


# Factory function for creating analyst with charter
def create_analyst_with_charter(charter_data: Dict[str, Any]) -> ContentStrategyAnalyst:
    """Create a Content Strategy Analyst with a pre-configured positioning charter."""
    analyst = ContentStrategyAnalyst()
    analyst.positioning_charter = PositioningCharter.from_dict(charter_data)
    return analyst


# Example positioning charter template
EXAMPLE_CHARTER_TEMPLATE = {
    "who_you_are": "A surgeon who educates patients about evidence-based approaches to common conditions",
    "who_you_are_not": [
        "A lifestyle influencer",
        "A supplement salesperson",
        "A medical advice hotline",
        "A controversy chaser"
    ],
    "priority_audience": "Patients considering surgery who want to understand their options",
    "secondary_audiences": ["Primary care physicians", "Medical students"],
    "content_pillars": [
        {
            "name": "Surgical Education",
            "description": "Explaining procedures, risks, and outcomes",
            "keywords": ["surgery", "procedure", "recovery", "risks"],
            "target_percentage": 35
        },
        {
            "name": "Evidence Translation",
            "description": "Making research accessible to patients",
            "keywords": ["study", "research", "evidence", "data"],
            "target_percentage": 25
        },
        {
            "name": "Patient Empowerment",
            "description": "Helping patients navigate the healthcare system",
            "keywords": ["questions", "second opinion", "decision", "advocate"],
            "target_percentage": 25
        },
        {
            "name": "Medical Myth Busting",
            "description": "Addressing common misconceptions",
            "keywords": ["myth", "truth", "actually", "misconception"],
            "target_percentage": 15
        }
    ],
    "voice_rules": [
        {
            "rule_type": "always",
            "description": "Acknowledge uncertainty when evidence is limited",
            "examples": ["The data here is limited...", "We don't yet have strong evidence..."]
        },
        {
            "rule_type": "always",
            "description": "Cite sources for medical claims",
            "examples": ["According to a 2023 study in JAMA...", "Guidelines recommend..."]
        },
        {
            "rule_type": "never",
            "description": "Provide specific medical advice to individuals",
            "examples": []
        },
        {
            "rule_type": "never",
            "description": "Attack or name specific colleagues",
            "examples": []
        }
    ],
    "platform_goals": [
        {
            "platform": "youtube",
            "success_definition": "Educational videos that patients reference before consultations",
            "primary_metric": "completion_rate",
            "target_value": 50,
            "secondary_metrics": ["saves", "comments_asking_questions"]
        },
        {
            "platform": "twitter",
            "success_definition": "Building authority among medical peers",
            "primary_metric": "engagement_rate",
            "target_value": 3.0,
            "secondary_metrics": ["peer_engagement", "saves"]
        },
        {
            "platform": "substack",
            "success_definition": "Deep-dive content that establishes thought leadership",
            "primary_metric": "open_rate",
            "target_value": 45,
            "secondary_metrics": ["read_time", "forwards"]
        }
    ]
}
