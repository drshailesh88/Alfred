# Alfred Sub-Agent Tools
# Each tool implements a specialized sub-agent that operates under Alfred's coordination

from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import json


class AlfredState(Enum):
    """Alfred's operational state affecting sub-agent behavior."""
    GREEN = "GREEN"   # Normal operations
    YELLOW = "YELLOW" # Elevated monitoring, restrict reactive content
    RED = "RED"       # Active threat, all public-facing output paused


class AgentCategory(Enum):
    """Categories of sub-agents."""
    SIGNAL = "signal"           # Reputation Sentinel, World Radar, Social Triage
    CONTENT = "content"         # Research, Substack, Twitter, YouTube
    LEARNING = "learning"       # Learning Curator, Scout, Distiller
    STRATEGY = "strategy"       # Social Metrics, Audience Signals, Content Strategy
    OPERATIONS = "operations"   # Intake, Patient Data, Scheduling, etc.


@dataclass
class AgentResponse:
    """Standard response wrapper for all sub-agent outputs."""
    agent_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    alfred_state: AlfredState = AlfredState.GREEN
    data: Dict[str, Any] = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "success": self.success,
            "alfred_state": self.alfred_state.value,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class BaseAgent:
    """Base class for all Alfred sub-agents."""

    def __init__(self, name: str, category: AgentCategory):
        self.name = name
        self.category = category
        self._alfred_state = AlfredState.GREEN

    @property
    def alfred_state(self) -> AlfredState:
        return self._alfred_state

    @alfred_state.setter
    def alfred_state(self, state: AlfredState):
        self._alfred_state = state

    def check_state_permission(self) -> tuple[bool, str]:
        """
        Check if the agent is permitted to operate in current state.
        Override in subclasses for state-specific behavior.
        """
        return True, "Operation permitted"

    def create_response(self, data: Dict[str, Any],
                       success: bool = True,
                       errors: list = None,
                       warnings: list = None) -> AgentResponse:
        """Create a standardized response."""
        return AgentResponse(
            agent_name=self.name,
            alfred_state=self.alfred_state,
            success=success,
            data=data,
            errors=errors or [],
            warnings=warnings or []
        )

    def blocked_response(self, reason: str) -> AgentResponse:
        """Return a blocked response when agent cannot operate."""
        return AgentResponse(
            agent_name=self.name,
            alfred_state=self.alfred_state,
            success=False,
            data={"status": "BLOCKED", "reason": reason},
            errors=[f"Agent blocked: {reason}"]
        )


class ContentAgent(BaseAgent):
    """Base class for content-generating agents (blocked in YELLOW/RED states)."""

    def __init__(self, name: str):
        super().__init__(name, AgentCategory.CONTENT)

    def check_state_permission(self) -> tuple[bool, str]:
        if self.alfred_state == AlfredState.RED:
            return False, "All content generation blocked in RED state"
        if self.alfred_state == AlfredState.YELLOW:
            return False, "Content generation restricted in YELLOW state"
        return True, "Operation permitted"


class SignalAgent(BaseAgent):
    """Base class for signal/awareness agents (heightened in YELLOW/RED)."""

    def __init__(self, name: str):
        super().__init__(name, AgentCategory.SIGNAL)

    def get_monitoring_level(self) -> str:
        if self.alfred_state == AlfredState.RED:
            return "CRITICAL"
        if self.alfred_state == AlfredState.YELLOW:
            return "HEIGHTENED"
        return "NORMAL"


class OperationsAgent(BaseAgent):
    """Base class for operational agents (mostly continue in all states)."""

    def __init__(self, name: str):
        super().__init__(name, AgentCategory.OPERATIONS)


class LearningAgent(BaseAgent):
    """Base class for learning pipeline agents."""

    def __init__(self, name: str):
        super().__init__(name, AgentCategory.LEARNING)

    def check_state_permission(self) -> tuple[bool, str]:
        if self.alfred_state == AlfredState.RED:
            return False, "Non-essential learning paused in RED state"
        return True, "Operation permitted"


class StrategyAgent(BaseAgent):
    """Base class for strategy/analytics agents."""

    def __init__(self, name: str):
        super().__init__(name, AgentCategory.STRATEGY)

    def check_state_permission(self) -> tuple[bool, str]:
        if self.alfred_state == AlfredState.RED:
            return False, "Strategy work paused in RED state - focus on recovery"
        return True, "Operation permitted"


# Import Learning Pipeline Agents
from .learning_curator import (
    LearningCurator,
    LearningQueueItem,
    LearningQueue,
    LinkedOutput,
    TimeWindow,
    TimeWindowType,
    OutputType,
    LearningUrgency,
)

from .learning_scout import (
    LearningScout,
    LearningCandidate,
    SearchResults,
    ResourceMetadata,
    AuthorProfile,
    CredibilityAssessment,
    TimestampSection,
    SourceType,
    CredibilityLevel,
    RelevanceScore,
)

from .learning_distiller import (
    LearningDistiller,
    LearningQuestion,
    LearningSignal,
    DistillerOutput,
    ExecutionContext,
    LinkedExecution,
    QuestionType,
    SignalType,
    Priority,
)

# Import Content Generation Agents
from .substack_agent import (
    SubstackAgent,
    LongformDraft,
    ContentSection,
    EvidenceCitation,
    UncertaintyDisclosure,
    EvidenceGap,
    QualityGateResult,
    ContentTone,
    EvidenceStrength,
)

from .twitter_thread_agent import (
    TwitterThreadAgent,
    ThreadDraft,
    Tweet,
    ToneCheckResult,
    SourceFidelityCheck,
    ToneClassification,
    SourceFidelity,
    TWITTER_CHAR_LIMIT,
    MAX_THREAD_SIZE,
    MIN_THREAD_SIZE,
)

from .youtube_script_agent import (
    YouTubeScriptAgent,
    ScriptDraft,
    ScriptSection,
    VisualSuggestion,
    UncertaintyMoment,
    TimingNote,
    ContentGateResult,
    ScriptType,
    VisualType,
)

# Import Daily Brief Agent
from .daily_brief import (
    DailyBriefAgent,
    DailyBrief,
    MorningBrief,
    EveningShutdown,
    PriorityItem,
    BlockedItem,
    CalendarBlock,
    DailyConstraint,
    ShippedItem,
    UnshippedItem,
    TomorrowNote,
    BriefType,
    ReputationStatus,
    PrioritySource,
    NotificationChannel,
    ConstraintType,
    NotificationDelivery,
    ConsoleNotification,
    EmailNotification,
    AgentDataCollector,
    create_daily_brief_agent,
)

# Import Weekly Brief Generator
from .weekly_brief import (
    WeeklyBriefGenerator,
    WeeklyBrief,
    StrategicSummary,
    ContentPlan,
    ContentPlanItem,
    WeeklyMetrics,
    WeeklyConstraint,
    LearningItem,
    BuilderWorkAssignment,
    PatternAlert,
    ContentItem,
    ReputationTrajectory,
    DriftSeverity,
    BuilderWorkStatus,
    ConstraintType as WeeklyConstraintType,
    TrendDirection as WeeklyTrendDirection,
    create_weekly_brief_generator,
)


# Export all public classes
__all__ = [
    # Base classes and core types
    "AlfredState",
    "AgentCategory",
    "AgentResponse",
    "BaseAgent",
    "ContentAgent",
    "SignalAgent",
    "OperationsAgent",
    "LearningAgent",
    "StrategyAgent",
    # Learning Curator
    "LearningCurator",
    "LearningQueueItem",
    "LearningQueue",
    "LinkedOutput",
    "TimeWindow",
    "TimeWindowType",
    "OutputType",
    "LearningUrgency",
    # Learning Scout
    "LearningScout",
    "LearningCandidate",
    "SearchResults",
    "ResourceMetadata",
    "AuthorProfile",
    "CredibilityAssessment",
    "TimestampSection",
    "SourceType",
    "CredibilityLevel",
    "RelevanceScore",
    # Learning Distiller
    "LearningDistiller",
    "LearningQuestion",
    "LearningSignal",
    "DistillerOutput",
    "ExecutionContext",
    "LinkedExecution",
    "QuestionType",
    "SignalType",
    "Priority",
    # Substack Agent (Authority Builder)
    "SubstackAgent",
    "LongformDraft",
    "ContentSection",
    "EvidenceCitation",
    "UncertaintyDisclosure",
    "EvidenceGap",
    "QualityGateResult",
    "ContentTone",
    "EvidenceStrength",
    # Twitter Thread Agent (Short-Form Translator)
    "TwitterThreadAgent",
    "ThreadDraft",
    "Tweet",
    "ToneCheckResult",
    "SourceFidelityCheck",
    "ToneClassification",
    "SourceFidelity",
    "TWITTER_CHAR_LIMIT",
    "MAX_THREAD_SIZE",
    "MIN_THREAD_SIZE",
    # YouTube Script Agent (Educator)
    "YouTubeScriptAgent",
    "ScriptDraft",
    "ScriptSection",
    "VisualSuggestion",
    "UncertaintyMoment",
    "TimingNote",
    "ContentGateResult",
    "ScriptType",
    "VisualType",
    # Daily Brief Agent
    "DailyBriefAgent",
    "DailyBrief",
    "MorningBrief",
    "EveningShutdown",
    "PriorityItem",
    "BlockedItem",
    "CalendarBlock",
    "DailyConstraint",
    "ShippedItem",
    "UnshippedItem",
    "TomorrowNote",
    "BriefType",
    "ReputationStatus",
    "PrioritySource",
    "NotificationChannel",
    "ConstraintType",
    "NotificationDelivery",
    "ConsoleNotification",
    "EmailNotification",
    "AgentDataCollector",
    "create_daily_brief_agent",
    # Weekly Brief Generator
    "WeeklyBriefGenerator",
    "WeeklyBrief",
    "StrategicSummary",
    "ContentPlan",
    "ContentPlanItem",
    "WeeklyMetrics",
    "WeeklyConstraint",
    "LearningItem",
    "BuilderWorkAssignment",
    "PatternAlert",
    "ContentItem",
    "ReputationTrajectory",
    "DriftSeverity",
    "BuilderWorkStatus",
    "WeeklyConstraintType",
    "WeeklyTrendDirection",
    "create_weekly_brief_generator",
]
