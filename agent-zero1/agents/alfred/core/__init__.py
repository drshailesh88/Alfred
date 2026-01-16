"""
ALFRED Core Module

Infrastructure components for Alfred's operation:
- State Manager: GREEN/YELLOW/RED state control
- Orchestrator: Sub-agent commissioning system
- Daily Brief: Morning and evening brief generation
- Weekly Brief: Strategic weekly brief generation
"""

# Daily Brief System - Implemented
from .daily_brief import (
    DailyBrief,
    DailyBriefTool,
    MorningBrief,
    EveningBrief,
    Priority,
    BlockedItem,
    ShippedItem,
    CalendarSnapshot,
    ReputationState,
    PriorityLevel
)

# Weekly Brief System - Implemented
from .weekly_brief import (
    WeeklyBrief,
    generate_weekly_brief,
    ReputationTrajectory,
    BuilderStatus,
    ReputationAnalysis,
    DriftAnalysis,
    PerformanceItem,
    ContentPlanItem,
    LearningQueueItem,
    BuilderConstraint,
    WeeklyBriefData
)

# State Manager - Implemented
from .state_manager import (
    StateManager,
    OperationalState,
    ActionType,
    AgentName,
    StateChangeRecord,
    AgentPermissions
)

# Orchestrator System - Implemented
from .orchestrator import (
    Orchestrator,
    create_orchestrator,
    get_agent_registry,
    get_agent_categories,
    get_commission_types,
    AgentCategory,
    CommissionType,
    AgentState,
    CommissionStatus,
    AlfredState,
    AgentDefinition,
    Commission,
    ScheduledCommission,
    AGENT_REGISTRY,
)

__all__ = [
    # Daily Brief System
    "DailyBrief",
    "DailyBriefTool",
    "MorningBrief",
    "EveningBrief",
    "Priority",
    "BlockedItem",
    "ShippedItem",
    "CalendarSnapshot",
    "ReputationState",
    "PriorityLevel",
    # Weekly Brief System
    "WeeklyBrief",
    "generate_weekly_brief",
    "ReputationTrajectory",
    "BuilderStatus",
    "ReputationAnalysis",
    "DriftAnalysis",
    "PerformanceItem",
    "ContentPlanItem",
    "LearningQueueItem",
    "BuilderConstraint",
    "WeeklyBriefData",
    # State Manager
    "StateManager",
    "OperationalState",
    "ActionType",
    "AgentName",
    "StateChangeRecord",
    "AgentPermissions",
    # Orchestrator System
    "Orchestrator",
    "create_orchestrator",
    "get_agent_registry",
    "get_agent_categories",
    "get_commission_types",
    "AgentCategory",
    "CommissionType",
    "AgentState",
    "CommissionStatus",
    "AlfredState",
    "AgentDefinition",
    "Commission",
    "ScheduledCommission",
    "AGENT_REGISTRY",
]
