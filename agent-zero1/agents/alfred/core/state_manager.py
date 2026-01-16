"""
Global State Manager for ALFRED

Manages Alfred's operational state (GREEN/YELLOW/RED) which controls what sub-agents can do.

State Definitions:
- GREEN: Normal operations, all agents can function
- YELLOW: Elevated monitoring, restrict reactive content, content agents produce drafts only
- RED: Active threat, all public-facing output paused

State Propagation Rules (from agent.system.subagents.md):
- YELLOW blocks: Twitter Thread Agent produces nothing
- RED blocks: All content agents (Substack, Twitter, YouTube, Social Triage) - queue only
- RED continues: Intake Agent, Patient Data Agent, Financial Sentinel
- RED heightens: Reputation Sentinel monitoring

This module provides:
- get_state() - returns current state
- set_state(new_state, reason, source) - changes state with logging
- is_action_allowed(action_type) - checks if action is allowed in current state
- get_state_history() - returns state change log
- get_agent_permissions(agent_name) - returns what agent can do in current state
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from python.helpers.tool import Tool, Response


class OperationalState(Enum):
    """Alfred's operational state levels."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class ActionType(Enum):
    """Types of actions that can be permission-checked."""
    # Content actions
    PUBLISH_TWITTER = "publish_twitter"
    PUBLISH_SUBSTACK = "publish_substack"
    PUBLISH_YOUTUBE = "publish_youtube"
    DRAFT_CONTENT = "draft_content"

    # Signal/monitoring actions
    MONITOR_REPUTATION = "monitor_reputation"
    SCAN_WORLD = "scan_world"
    TRIAGE_SOCIAL = "triage_social"

    # Operations actions
    PROCESS_INTAKE = "process_intake"
    ACCESS_PATIENT_DATA = "access_patient_data"
    MANAGE_SCHEDULE = "manage_schedule"
    MANAGE_CONTENT = "manage_content"
    GOVERN_SHIPPING = "govern_shipping"
    MONITOR_FINANCIAL = "monitor_financial"
    NUDGE_RELATIONSHIP = "nudge_relationship"

    # Learning actions
    CURATE_LEARNING = "curate_learning"
    SCOUT_LEARNING = "scout_learning"
    DISTILL_LEARNING = "distill_learning"

    # Strategy actions
    HARVEST_METRICS = "harvest_metrics"
    EXTRACT_SIGNALS = "extract_signals"
    ANALYZE_STRATEGY = "analyze_strategy"

    # Research actions
    RETRIEVE_EVIDENCE = "retrieve_evidence"


class AgentName(Enum):
    """All Alfred sub-agents."""
    # Signal/Awareness
    REPUTATION_SENTINEL = "reputation_sentinel"
    WORLD_RADAR = "world_radar"
    SOCIAL_TRIAGE = "social_triage"

    # Content Generation
    RESEARCH_AGENT = "research_agent"
    SUBSTACK_AGENT = "substack_agent"
    TWITTER_THREAD_AGENT = "twitter_thread_agent"
    YOUTUBE_SCRIPT_AGENT = "youtube_script_agent"

    # Learning Pipeline
    LEARNING_CURATOR = "learning_curator"
    LEARNING_SCOUT = "learning_scout"
    LEARNING_DISTILLER = "learning_distiller"

    # Operations
    INTAKE_AGENT = "intake_agent"
    PATIENT_DATA_AGENT = "patient_data_agent"
    SCHEDULING_AGENT = "scheduling_agent"
    CONTENT_MANAGER = "content_manager"
    SHIPPING_GOVERNOR = "shipping_governor"
    FINANCIAL_SENTINEL = "financial_sentinel"
    RELATIONSHIP_NUDGE_AGENT = "relationship_nudge_agent"

    # Strategy
    SOCIAL_METRICS_HARVESTER = "social_metrics_harvester"
    AUDIENCE_SIGNALS_EXTRACTOR = "audience_signals_extractor"
    CONTENT_STRATEGY_ANALYST = "content_strategy_analyst"


@dataclass
class StateChangeRecord:
    """Record of a state change."""
    timestamp: str
    previous_state: str
    new_state: str
    reason: str
    source: str  # Who/what triggered the change (e.g., "reputation_sentinel", "user", "alfred")
    risk_score: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentPermissions:
    """Permissions for an agent in current state."""
    agent_name: str
    can_operate: bool
    can_produce_output: bool
    restrictions: List[str]
    behavior_modifications: List[str]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StateManager(Tool):
    """
    Global State Manager for ALFRED

    Manages the operational state (GREEN/YELLOW/RED) and controls what sub-agents can do.

    Methods:
    - get_state: Returns current operational state
    - set_state: Changes state with logging and propagation
    - is_action_allowed: Checks if an action is allowed in current state
    - get_state_history: Returns state change log
    - get_agent_permissions: Returns what an agent can do in current state
    - request_state_change: Request a state change (requires confirmation)
    - confirm_state_change: Confirm a pending state change
    - get_all_agent_permissions: Get permissions for all agents
    """

    # Agent permissions by state
    # Format: {state: {agent: {can_operate, can_produce_output, restrictions, modifications}}}
    AGENT_PERMISSIONS = {
        OperationalState.GREEN.value: {
            # Signal/Awareness - all operate normally
            AgentName.REPUTATION_SENTINEL.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Normal monitoring frequency"],
                "notes": "Standard monitoring operations"
            },
            AgentName.WORLD_RADAR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Standard scan operations"
            },
            AgentName.SOCIAL_TRIAGE.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal triage operations"
            },
            # Content Generation - all operate normally
            AgentName.RESEARCH_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Full research capabilities"
            },
            AgentName.SUBSTACK_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Can draft and recommend publishing"
            },
            AgentName.TWITTER_THREAD_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Can generate threads for deployment"
            },
            AgentName.YOUTUBE_SCRIPT_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Can generate scripts for production"
            },
            # Learning Pipeline - all operate normally
            AgentName.LEARNING_CURATOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal learning curation"
            },
            AgentName.LEARNING_SCOUT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal resource discovery"
            },
            AgentName.LEARNING_DISTILLER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal question extraction"
            },
            # Operations - all operate normally
            AgentName.INTAKE_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal intake operations"
            },
            AgentName.PATIENT_DATA_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal data operations"
            },
            AgentName.SCHEDULING_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal scheduling"
            },
            AgentName.CONTENT_MANAGER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal content management"
            },
            AgentName.SHIPPING_GOVERNOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal shipping governance"
            },
            AgentName.FINANCIAL_SENTINEL.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal financial monitoring"
            },
            AgentName.RELATIONSHIP_NUDGE_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal relationship nudges"
            },
            # Strategy - all operate normally
            AgentName.SOCIAL_METRICS_HARVESTER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal metrics collection"
            },
            AgentName.AUDIENCE_SIGNALS_EXTRACTOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal signal extraction"
            },
            AgentName.CONTENT_STRATEGY_ANALYST.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal strategy analysis"
            },
        },
        OperationalState.YELLOW.value: {
            # Signal/Awareness - elevated monitoring
            AgentName.REPUTATION_SENTINEL.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Elevated monitoring frequency", "Heightened sensitivity"],
                "notes": "Increased monitoring cadence"
            },
            AgentName.WORLD_RADAR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Flag potentially reactive content"],
                "notes": "Normal with caution flags"
            },
            AgentName.SOCIAL_TRIAGE.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["No engagement recommendations"], "modifications": ["Caution flag active"],
                "notes": "Triage only, no engagement"
            },
            # Content Generation - restricted
            AgentName.RESEARCH_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Prioritize reputation-relevant evidence"],
                "notes": "Full capabilities, prioritized focus"
            },
            AgentName.SUBSTACK_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["No immediate publish recommendations"],
                "modifications": ["Include explicit caution review", "Draft only mode"],
                "notes": "Drafts require extra review before publishing"
            },
            AgentName.TWITTER_THREAD_AGENT.value: {
                "can_operate": False, "can_produce_output": False,
                "restrictions": ["BLOCKED - produces nothing"], "modifications": [],
                "notes": "HARD BLOCK: No Twitter content during YELLOW state"
            },
            AgentName.YOUTUBE_SCRIPT_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["No potentially reactive content"],
                "modifications": ["Flag any potentially reactive content"],
                "notes": "Scripts require review for reactive elements"
            },
            # Learning Pipeline - restricted
            AgentName.LEARNING_CURATOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Normal operation with caution flag"],
                "notes": "Normal with caution"
            },
            AgentName.LEARNING_SCOUT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Normal operation with caution flag"],
                "notes": "Normal with caution"
            },
            AgentName.LEARNING_DISTILLER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Normal operation with caution flag"],
                "notes": "Normal with caution"
            },
            # Operations - continue normally (critical operations)
            AgentName.INTAKE_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Critical operation - continues normally"
            },
            AgentName.PATIENT_DATA_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Critical operation - continues normally"
            },
            AgentName.SCHEDULING_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal scheduling"
            },
            AgentName.CONTENT_MANAGER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["Hold public-facing orchestration"], "modifications": [],
                "notes": "Internal coordination only"
            },
            AgentName.SHIPPING_GOVERNOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Caution flag on public outputs"],
                "notes": "Normal with caution"
            },
            AgentName.FINANCIAL_SENTINEL.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Critical operation - continues normally"
            },
            AgentName.RELATIONSHIP_NUDGE_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Normal relationship nudges"
            },
            # Strategy - restricted
            AgentName.SOCIAL_METRICS_HARVESTER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Monitoring continues"
            },
            AgentName.AUDIENCE_SIGNALS_EXTRACTOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Analysis continues"
            },
            AgentName.CONTENT_STRATEGY_ANALYST.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["Hold publication recommendations"], "modifications": [],
                "notes": "Analysis only, hold recommendations"
            },
        },
        OperationalState.RED.value: {
            # Signal/Awareness - heightened
            AgentName.REPUTATION_SENTINEL.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Continuous monitoring mode", "Maximum sensitivity"],
                "notes": "Continuous monitoring - heightened operations"
            },
            AgentName.WORLD_RADAR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Elevated scan frequency"],
                "notes": "Heightened scanning"
            },
            AgentName.SOCIAL_TRIAGE.value: {
                "can_operate": False, "can_produce_output": False,
                "restrictions": ["BLOCKED - all public engagement paused"], "modifications": [],
                "notes": "PAUSED: No social engagement during active threat"
            },
            # Content Generation - blocked/queue only
            AgentName.RESEARCH_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": ["Prioritize reputation-relevant evidence"],
                "notes": "Focus on crisis-relevant research"
            },
            AgentName.SUBSTACK_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["Queue only - no publish recommendations"], "modifications": [],
                "notes": "Can draft, cannot recommend publishing"
            },
            AgentName.TWITTER_THREAD_AGENT.value: {
                "can_operate": False, "can_produce_output": False,
                "restrictions": ["BLOCKED - produces nothing"], "modifications": [],
                "notes": "HARD BLOCK: No Twitter content during RED state"
            },
            AgentName.YOUTUBE_SCRIPT_AGENT.value: {
                "can_operate": False, "can_produce_output": False,
                "restrictions": ["No new content scripts"], "modifications": [],
                "notes": "PAUSED: No new video content during active threat"
            },
            # Learning Pipeline - emergency only
            AgentName.LEARNING_CURATOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["Emergency learning only"], "modifications": ["Crisis response mode"],
                "notes": "Only crisis-relevant learning"
            },
            AgentName.LEARNING_SCOUT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["Emergency resources only"], "modifications": [],
                "notes": "Only crisis-relevant resources"
            },
            AgentName.LEARNING_DISTILLER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Can process existing materials"
            },
            # Operations - critical operations CONTINUE
            AgentName.INTAKE_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "CRITICAL: Continues during RED - intake must not stop"
            },
            AgentName.PATIENT_DATA_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "CRITICAL: Continues during RED - patient data must be accessible"
            },
            AgentName.SCHEDULING_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["Clear public-facing appointments"], "modifications": [],
                "notes": "Internal scheduling only, clear public calendar"
            },
            AgentName.CONTENT_MANAGER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["All public orchestration paused"], "modifications": [],
                "notes": "Internal coordination only - no public content"
            },
            AgentName.SHIPPING_GOVERNOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["Pause non-essential projects"], "modifications": [],
                "notes": "Essential projects only"
            },
            AgentName.FINANCIAL_SENTINEL.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "CRITICAL: Continues during RED - financial monitoring must continue"
            },
            AgentName.RELATIONSHIP_NUDGE_AGENT.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["Private communications only"], "modifications": [],
                "notes": "Private relationship maintenance only"
            },
            # Strategy - monitoring continues
            AgentName.SOCIAL_METRICS_HARVESTER.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Monitoring continues for situational awareness"
            },
            AgentName.AUDIENCE_SIGNALS_EXTRACTOR.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": [], "modifications": [],
                "notes": "Analysis continues for crisis assessment"
            },
            AgentName.CONTENT_STRATEGY_ANALYST.value: {
                "can_operate": True, "can_produce_output": True,
                "restrictions": ["No publication planning"], "modifications": [],
                "notes": "Assessment only, no planning"
            },
        },
    }

    # Action permissions by state
    ACTION_PERMISSIONS = {
        OperationalState.GREEN.value: {
            ActionType.PUBLISH_TWITTER.value: True,
            ActionType.PUBLISH_SUBSTACK.value: True,
            ActionType.PUBLISH_YOUTUBE.value: True,
            ActionType.DRAFT_CONTENT.value: True,
            ActionType.MONITOR_REPUTATION.value: True,
            ActionType.SCAN_WORLD.value: True,
            ActionType.TRIAGE_SOCIAL.value: True,
            ActionType.PROCESS_INTAKE.value: True,
            ActionType.ACCESS_PATIENT_DATA.value: True,
            ActionType.MANAGE_SCHEDULE.value: True,
            ActionType.MANAGE_CONTENT.value: True,
            ActionType.GOVERN_SHIPPING.value: True,
            ActionType.MONITOR_FINANCIAL.value: True,
            ActionType.NUDGE_RELATIONSHIP.value: True,
            ActionType.CURATE_LEARNING.value: True,
            ActionType.SCOUT_LEARNING.value: True,
            ActionType.DISTILL_LEARNING.value: True,
            ActionType.HARVEST_METRICS.value: True,
            ActionType.EXTRACT_SIGNALS.value: True,
            ActionType.ANALYZE_STRATEGY.value: True,
            ActionType.RETRIEVE_EVIDENCE.value: True,
        },
        OperationalState.YELLOW.value: {
            ActionType.PUBLISH_TWITTER.value: False,  # BLOCKED
            ActionType.PUBLISH_SUBSTACK.value: False,  # Draft only
            ActionType.PUBLISH_YOUTUBE.value: True,  # With review
            ActionType.DRAFT_CONTENT.value: True,
            ActionType.MONITOR_REPUTATION.value: True,  # Elevated
            ActionType.SCAN_WORLD.value: True,
            ActionType.TRIAGE_SOCIAL.value: True,  # No engagement
            ActionType.PROCESS_INTAKE.value: True,  # Critical
            ActionType.ACCESS_PATIENT_DATA.value: True,  # Critical
            ActionType.MANAGE_SCHEDULE.value: True,
            ActionType.MANAGE_CONTENT.value: True,  # Internal only
            ActionType.GOVERN_SHIPPING.value: True,
            ActionType.MONITOR_FINANCIAL.value: True,  # Critical
            ActionType.NUDGE_RELATIONSHIP.value: True,
            ActionType.CURATE_LEARNING.value: True,
            ActionType.SCOUT_LEARNING.value: True,
            ActionType.DISTILL_LEARNING.value: True,
            ActionType.HARVEST_METRICS.value: True,
            ActionType.EXTRACT_SIGNALS.value: True,
            ActionType.ANALYZE_STRATEGY.value: True,
            ActionType.RETRIEVE_EVIDENCE.value: True,
        },
        OperationalState.RED.value: {
            ActionType.PUBLISH_TWITTER.value: False,  # BLOCKED
            ActionType.PUBLISH_SUBSTACK.value: False,  # Queue only
            ActionType.PUBLISH_YOUTUBE.value: False,  # PAUSED
            ActionType.DRAFT_CONTENT.value: True,  # Can draft
            ActionType.MONITOR_REPUTATION.value: True,  # Heightened
            ActionType.SCAN_WORLD.value: True,  # Elevated
            ActionType.TRIAGE_SOCIAL.value: False,  # PAUSED
            ActionType.PROCESS_INTAKE.value: True,  # CRITICAL
            ActionType.ACCESS_PATIENT_DATA.value: True,  # CRITICAL
            ActionType.MANAGE_SCHEDULE.value: True,  # Internal only
            ActionType.MANAGE_CONTENT.value: True,  # Internal only
            ActionType.GOVERN_SHIPPING.value: True,  # Essential only
            ActionType.MONITOR_FINANCIAL.value: True,  # CRITICAL
            ActionType.NUDGE_RELATIONSHIP.value: True,  # Private only
            ActionType.CURATE_LEARNING.value: True,  # Emergency only
            ActionType.SCOUT_LEARNING.value: True,  # Emergency only
            ActionType.DISTILL_LEARNING.value: True,
            ActionType.HARVEST_METRICS.value: True,  # Continue
            ActionType.EXTRACT_SIGNALS.value: True,  # Continue
            ActionType.ANALYZE_STRATEGY.value: True,  # Assessment only
            ActionType.RETRIEVE_EVIDENCE.value: True,  # Crisis focus
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._storage_path = self._get_storage_path()
        self._state_file = self._storage_path / "current_state.json"
        self._history_file = self._storage_path / "state_history.json"
        self._pending_change_file = self._storage_path / "pending_change.json"

        # Load current state
        self._current_state = self._load_current_state()
        self._state_history: List[StateChangeRecord] = self._load_history()
        self._pending_change: Optional[Dict] = self._load_pending_change()

    def _get_storage_path(self) -> Path:
        """Get the storage path for state data."""
        base_path = Path(__file__).parent.parent.parent.parent
        storage_path = base_path / "data" / "alfred" / "state"
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    def _load_current_state(self) -> OperationalState:
        """Load the current state from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    data = json.load(f)
                    state_value = data.get("state", OperationalState.GREEN.value)
                    return OperationalState(state_value)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        return OperationalState.GREEN

    def _save_current_state(self) -> None:
        """Save the current state to disk."""
        data = {
            "state": self._current_state.value,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._state_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_history(self) -> List[StateChangeRecord]:
        """Load state history from disk."""
        if self._history_file.exists():
            try:
                with open(self._history_file, "r") as f:
                    data = json.load(f)
                    return [StateChangeRecord(**record) for record in data.get("history", [])]
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def _save_history(self) -> None:
        """Save state history to disk."""
        data = {
            "history": [record.to_dict() for record in self._state_history]
        }
        with open(self._history_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_pending_change(self) -> Optional[Dict]:
        """Load pending state change from disk."""
        if self._pending_change_file.exists():
            try:
                with open(self._pending_change_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return None

    def _save_pending_change(self) -> None:
        """Save pending state change to disk."""
        if self._pending_change:
            with open(self._pending_change_file, "w") as f:
                json.dump(self._pending_change, f, indent=2)
        elif self._pending_change_file.exists():
            self._pending_change_file.unlink()

    def _clear_pending_change(self) -> None:
        """Clear pending state change."""
        self._pending_change = None
        if self._pending_change_file.exists():
            self._pending_change_file.unlink()

    async def execute(self, **kwargs) -> Response:
        """Execute a state manager method."""
        method = kwargs.get("method", "get_state")
        method_map = {
            "get_state": self.get_state,
            "set_state": self.set_state,
            "is_action_allowed": self.is_action_allowed,
            "get_state_history": self.get_state_history,
            "get_agent_permissions": self.get_agent_permissions,
            "request_state_change": self.request_state_change,
            "confirm_state_change": self.confirm_state_change,
            "reject_state_change": self.reject_state_change,
            "get_all_agent_permissions": self.get_all_agent_permissions,
            "get_pending_change": self.get_pending_change,
        }
        if method in method_map:
            return await method_map[method](**kwargs)
        return Response(
            message=f"Unknown method '{method}'. Available: {', '.join(method_map.keys())}",
            break_loop=False
        )

    async def get_state(self, **kwargs) -> Response:
        """
        Get the current operational state.

        Returns:
            Current state with metadata
        """
        result = {
            "current_state": self._current_state.value.upper(),
            "state_definitions": {
                "GREEN": "Normal operations - all agents can function",
                "YELLOW": "Elevated monitoring - restrict reactive content, content agents draft only",
                "RED": "Active threat - all public-facing output paused"
            },
            "last_change": self._state_history[-1].to_dict() if self._state_history else None,
            "pending_change": self._pending_change is not None,
        }
        return Response(message=json.dumps(result, indent=2), break_loop=False)

    async def set_state(
        self,
        new_state: str,
        reason: str,
        source: str,
        risk_score: int = None,
        metadata: Dict = None,
        **kwargs
    ) -> Response:
        """
        Set the operational state directly (used after confirmation).

        Args:
            new_state: The new state (green, yellow, red)
            reason: Why the state is changing
            source: Who/what triggered the change
            risk_score: Optional risk score (0-100)
            metadata: Optional additional metadata
        """
        if not new_state or not reason or not source:
            return Response(
                message="Error: 'new_state', 'reason', and 'source' are required",
                break_loop=False
            )

        try:
            new_state_enum = OperationalState(new_state.lower())
        except ValueError:
            return Response(
                message=f"Invalid state: {new_state}. Must be green, yellow, or red.",
                break_loop=False
            )

        previous_state = self._current_state

        # Create change record
        record = StateChangeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            previous_state=previous_state.value,
            new_state=new_state_enum.value,
            reason=reason,
            source=source,
            risk_score=risk_score,
            metadata=metadata
        )

        # Update state
        self._current_state = new_state_enum
        self._state_history.append(record)

        # Persist changes
        self._save_current_state()
        self._save_history()
        self._clear_pending_change()

        result = {
            "status": "STATE_CHANGED",
            "previous_state": previous_state.value.upper(),
            "new_state": new_state_enum.value.upper(),
            "reason": reason,
            "source": source,
            "timestamp": record.timestamp,
            "affected_agents": self._get_affected_agents(previous_state.value, new_state_enum.value)
        }

        return Response(message=json.dumps(result, indent=2), break_loop=False)

    async def request_state_change(
        self,
        recommended_state: str,
        reason: str,
        source: str,
        risk_score: int = None,
        metadata: Dict = None,
        **kwargs
    ) -> Response:
        """
        Request a state change (requires confirmation from Alfred or user).

        Args:
            recommended_state: The recommended new state
            reason: Why the state should change
            source: Who/what is recommending the change
            risk_score: Optional risk score
            metadata: Optional additional metadata
        """
        if not recommended_state or not reason or not source:
            return Response(
                message="Error: 'recommended_state', 'reason', and 'source' are required",
                break_loop=False
            )

        try:
            new_state_enum = OperationalState(recommended_state.lower())
        except ValueError:
            return Response(
                message=f"Invalid state: {recommended_state}. Must be green, yellow, or red.",
                break_loop=False
            )

        self._pending_change = {
            "recommended_state": new_state_enum.value,
            "current_state": self._current_state.value,
            "reason": reason,
            "source": source,
            "risk_score": risk_score,
            "metadata": metadata,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_pending_change()

        # Calculate impact
        affected = self._get_affected_agents(self._current_state.value, new_state_enum.value)

        result = {
            "status": "PENDING_CONFIRMATION",
            "current_state": self._current_state.value.upper(),
            "recommended_state": new_state_enum.value.upper(),
            "reason": reason,
            "source": source,
            "risk_score": risk_score,
            "impact": {
                "agents_to_block": affected.get("newly_blocked", []),
                "agents_with_restrictions": affected.get("newly_restricted", []),
                "agents_unaffected": affected.get("unaffected", [])
            },
            "confirmation_required": "Use confirm_state_change to apply or reject_state_change to cancel"
        }

        return Response(message=json.dumps(result, indent=2), break_loop=False)

    async def confirm_state_change(self, **kwargs) -> Response:
        """
        Confirm a pending state change.
        """
        if not self._pending_change:
            return Response(
                message="No pending state change to confirm",
                break_loop=False
            )

        return await self.set_state(
            new_state=self._pending_change["recommended_state"],
            reason=self._pending_change["reason"],
            source=self._pending_change["source"] + " (confirmed)",
            risk_score=self._pending_change.get("risk_score"),
            metadata=self._pending_change.get("metadata")
        )

    async def reject_state_change(self, reason: str = None, **kwargs) -> Response:
        """
        Reject a pending state change.

        Args:
            reason: Why the change was rejected
        """
        if not self._pending_change:
            return Response(
                message="No pending state change to reject",
                break_loop=False
            )

        rejected_change = self._pending_change.copy()
        self._clear_pending_change()

        result = {
            "status": "CHANGE_REJECTED",
            "rejected_state": rejected_change["recommended_state"].upper(),
            "current_state": self._current_state.value.upper(),
            "rejection_reason": reason or "Rejected by user or Alfred",
            "original_request": rejected_change
        }

        return Response(message=json.dumps(result, indent=2), break_loop=False)

    async def get_pending_change(self, **kwargs) -> Response:
        """
        Get any pending state change request.
        """
        if not self._pending_change:
            return Response(
                message=json.dumps({"pending_change": None}, indent=2),
                break_loop=False
            )

        return Response(
            message=json.dumps({"pending_change": self._pending_change}, indent=2),
            break_loop=False
        )

    async def is_action_allowed(self, action_type: str, **kwargs) -> Response:
        """
        Check if an action is allowed in the current state.

        Args:
            action_type: The type of action to check
        """
        if not action_type:
            return Response(
                message="Error: 'action_type' is required",
                break_loop=False
            )

        # Normalize action type
        action_type_normalized = action_type.lower()

        # Check if it's a valid action type
        valid_actions = [a.value for a in ActionType]
        if action_type_normalized not in valid_actions:
            return Response(
                message=f"Unknown action type: {action_type}. Valid types: {', '.join(valid_actions)}",
                break_loop=False
            )

        state_permissions = self.ACTION_PERMISSIONS.get(self._current_state.value, {})
        allowed = state_permissions.get(action_type_normalized, False)

        result = {
            "action_type": action_type_normalized,
            "current_state": self._current_state.value.upper(),
            "allowed": allowed,
            "reason": self._get_action_restriction_reason(action_type_normalized) if not allowed else "Action permitted in current state"
        }

        return Response(message=json.dumps(result, indent=2), break_loop=False)

    async def get_state_history(self, limit: int = 50, **kwargs) -> Response:
        """
        Get the state change history.

        Args:
            limit: Maximum number of records to return
        """
        history = self._state_history[-limit:] if limit > 0 else self._state_history

        result = {
            "total_changes": len(self._state_history),
            "showing": len(history),
            "current_state": self._current_state.value.upper(),
            "history": [record.to_dict() for record in history]
        }

        return Response(message=json.dumps(result, indent=2), break_loop=False)

    async def get_agent_permissions(self, agent_name: str, **kwargs) -> Response:
        """
        Get permissions for a specific agent in the current state.

        Args:
            agent_name: The name of the agent to check
        """
        if not agent_name:
            return Response(
                message="Error: 'agent_name' is required",
                break_loop=False
            )

        # Normalize agent name
        agent_name_normalized = agent_name.lower().replace(" ", "_").replace("-", "_")

        # Check if it's a valid agent
        valid_agents = [a.value for a in AgentName]
        if agent_name_normalized not in valid_agents:
            return Response(
                message=f"Unknown agent: {agent_name}. Valid agents: {', '.join(valid_agents)}",
                break_loop=False
            )

        state_agents = self.AGENT_PERMISSIONS.get(self._current_state.value, {})
        agent_perms = state_agents.get(agent_name_normalized, {})

        permissions = AgentPermissions(
            agent_name=agent_name_normalized,
            can_operate=agent_perms.get("can_operate", False),
            can_produce_output=agent_perms.get("can_produce_output", False),
            restrictions=agent_perms.get("restrictions", []),
            behavior_modifications=agent_perms.get("modifications", []),
            notes=agent_perms.get("notes", "")
        )

        result = {
            "agent": agent_name_normalized,
            "current_state": self._current_state.value.upper(),
            "permissions": permissions.to_dict()
        }

        return Response(message=json.dumps(result, indent=2), break_loop=False)

    async def get_all_agent_permissions(self, **kwargs) -> Response:
        """
        Get permissions for all agents in the current state.
        """
        state_agents = self.AGENT_PERMISSIONS.get(self._current_state.value, {})

        # Categorize agents
        blocked = []
        restricted = []
        normal = []
        heightened = []

        for agent_name, perms in state_agents.items():
            if not perms.get("can_operate", True):
                blocked.append({
                    "agent": agent_name,
                    "reason": perms.get("restrictions", ["Blocked"])[0] if perms.get("restrictions") else "Blocked",
                    "notes": perms.get("notes", "")
                })
            elif perms.get("restrictions"):
                restricted.append({
                    "agent": agent_name,
                    "restrictions": perms.get("restrictions", []),
                    "notes": perms.get("notes", "")
                })
            elif perms.get("modifications") and any("elevated" in m.lower() or "heighten" in m.lower() or "continuous" in m.lower() for m in perms.get("modifications", [])):
                heightened.append({
                    "agent": agent_name,
                    "modifications": perms.get("modifications", []),
                    "notes": perms.get("notes", "")
                })
            else:
                normal.append({
                    "agent": agent_name,
                    "notes": perms.get("notes", "")
                })

        result = {
            "current_state": self._current_state.value.upper(),
            "summary": {
                "blocked_count": len(blocked),
                "restricted_count": len(restricted),
                "heightened_count": len(heightened),
                "normal_count": len(normal)
            },
            "agents": {
                "blocked": blocked,
                "restricted": restricted,
                "heightened": heightened,
                "normal": normal
            }
        }

        return Response(message=json.dumps(result, indent=2), break_loop=False)

    def _get_affected_agents(self, from_state: str, to_state: str) -> Dict[str, List[str]]:
        """Get agents affected by a state transition."""
        from_perms = self.AGENT_PERMISSIONS.get(from_state, {})
        to_perms = self.AGENT_PERMISSIONS.get(to_state, {})

        newly_blocked = []
        newly_restricted = []
        unaffected = []
        newly_unblocked = []

        for agent_name in AgentName:
            name = agent_name.value
            from_perm = from_perms.get(name, {})
            to_perm = to_perms.get(name, {})

            from_can_operate = from_perm.get("can_operate", True)
            to_can_operate = to_perm.get("can_operate", True)
            from_restrictions = from_perm.get("restrictions", [])
            to_restrictions = to_perm.get("restrictions", [])

            if from_can_operate and not to_can_operate:
                newly_blocked.append(name)
            elif not from_can_operate and to_can_operate:
                newly_unblocked.append(name)
            elif len(to_restrictions) > len(from_restrictions):
                newly_restricted.append(name)
            else:
                unaffected.append(name)

        return {
            "newly_blocked": newly_blocked,
            "newly_restricted": newly_restricted,
            "newly_unblocked": newly_unblocked,
            "unaffected": unaffected
        }

    def _get_action_restriction_reason(self, action_type: str) -> str:
        """Get the reason why an action is restricted in the current state."""
        reasons = {
            OperationalState.YELLOW.value: {
                ActionType.PUBLISH_TWITTER.value: "Twitter publishing blocked during YELLOW state - elevated risk",
                ActionType.PUBLISH_SUBSTACK.value: "Substack publishing requires additional review during YELLOW state",
            },
            OperationalState.RED.value: {
                ActionType.PUBLISH_TWITTER.value: "All Twitter activity blocked during RED state - active threat",
                ActionType.PUBLISH_SUBSTACK.value: "All public-facing content paused during RED state",
                ActionType.PUBLISH_YOUTUBE.value: "All public-facing content paused during RED state",
                ActionType.TRIAGE_SOCIAL.value: "All social engagement paused during RED state",
            }
        }

        state_reasons = reasons.get(self._current_state.value, {})
        return state_reasons.get(action_type, f"Action restricted in {self._current_state.value.upper()} state")
