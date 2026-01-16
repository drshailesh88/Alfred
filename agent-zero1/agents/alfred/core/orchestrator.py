"""
Alfred Orchestration Engine

The commissioning system that Alfred uses to invoke sub-agents.

Commission Protocol:
- Alfred commissions sub-agents based on user request + context + state
- Sub-agents return structured packets to Alfred
- Sub-agents NEVER communicate with each other directly
- Sub-agents NEVER communicate with user directly

Commission Rules:
- Always Active (background): Reputation Sentinel, World Radar, Intake Agent
- On-Demand: All content agents, Patient Data, Scheduling, Learning, Financial, Relationship
- Periodic (cadence): Social Metrics Harvester (weekly), Audience Signals (weekly),
                       Content Strategy Analyst (weekly), Financial Sentinel (monthly)
- Conditional (triggered): Shipping Governor (when stalls), Learning Distiller (after stuck points)

Dependencies:
- Content Strategy Analyst <- Social Metrics Harvester + Audience Signals Extractor
- Learning Curator <- Learning Scout + Learning Distiller
- Content Manager <- Research Agent + Content Agents
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import uuid
import asyncio
from pathlib import Path


class AgentCategory(Enum):
    SIGNAL = "signal"
    CONTENT = "content"
    LEARNING = "learning"
    STRATEGY = "strategy"
    OPERATIONS = "operations"


class CommissionType(Enum):
    ALWAYS_ACTIVE = "always_active"
    ON_DEMAND = "on_demand"
    PERIODIC = "periodic"
    CONDITIONAL = "conditional"


class AgentState(Enum):
    AVAILABLE = "available"
    BLOCKED = "blocked"
    RUNNING = "running"
    INACTIVE = "inactive"


class CommissionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class AlfredState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class AgentDefinition:
    """Definition of a sub-agent with its configuration."""
    name: str
    category: AgentCategory
    commission_type: CommissionType
    output_format: str
    dependencies: List[str] = field(default_factory=list)
    cadence_days: Optional[int] = None  # For periodic agents
    trigger_conditions: List[str] = field(default_factory=list)  # For conditional agents
    state_restrictions: Dict[str, List[str]] = field(default_factory=dict)  # state -> blocked actions


@dataclass
class Commission:
    """A commission request for a sub-agent."""
    commission_id: str
    agent_name: str
    request_data: Dict[str, Any]
    created_at: str
    status: CommissionStatus
    alfred_state: str
    priority: str = "normal"
    context: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    parent_commission_id: Optional[str] = None  # For chained commissions


@dataclass
class ScheduledCommission:
    """A scheduled periodic commission."""
    schedule_id: str
    agent_name: str
    cadence_days: int
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    enabled: bool = True
    request_template: Dict[str, Any] = field(default_factory=dict)


# Complete Agent Registry based on specification
AGENT_REGISTRY: Dict[str, AgentDefinition] = {
    # Signal/Awareness Agents (3)
    "reputation_sentinel": AgentDefinition(
        name="reputation_sentinel",
        category=AgentCategory.SIGNAL,
        commission_type=CommissionType.ALWAYS_ACTIVE,
        output_format="REPUTATION_PACKET",
        state_restrictions={
            "yellow": ["heightened_monitoring"],
            "red": ["critical_monitoring"]
        }
    ),
    "world_radar": AgentDefinition(
        name="world_radar",
        category=AgentCategory.SIGNAL,
        commission_type=CommissionType.ALWAYS_ACTIVE,
        output_format="WORLD_SIGNAL",
        state_restrictions={
            "red": ["priority_reputation_signals"]
        }
    ),
    "social_triage": AgentDefinition(
        name="social_triage",
        category=AgentCategory.SIGNAL,
        commission_type=CommissionType.ON_DEMAND,
        output_format="SOCIAL_TRIAGE_REPORT",
        state_restrictions={
            "yellow": ["no_engagement_recommendations"],
            "red": ["blocked"]
        }
    ),

    # Content Generation Agents (4)
    "research_agent": AgentDefinition(
        name="research_agent",
        category=AgentCategory.CONTENT,
        commission_type=CommissionType.ON_DEMAND,
        output_format="EVIDENCE_BRIEF",
        state_restrictions={
            "red": ["prioritize_reputation_evidence"]
        }
    ),
    "substack_agent": AgentDefinition(
        name="substack_agent",
        category=AgentCategory.CONTENT,
        commission_type=CommissionType.ON_DEMAND,
        output_format="LONGFORM_DRAFT",
        dependencies=["research_agent"],
        state_restrictions={
            "yellow": ["draft_only"],
            "red": ["blocked"]
        }
    ),
    "twitter_thread_agent": AgentDefinition(
        name="twitter_thread_agent",
        category=AgentCategory.CONTENT,
        commission_type=CommissionType.ON_DEMAND,
        output_format="THREAD_DRAFT",
        dependencies=["substack_agent"],
        state_restrictions={
            "yellow": ["blocked"],
            "red": ["blocked"]
        }
    ),
    "youtube_script_agent": AgentDefinition(
        name="youtube_script_agent",
        category=AgentCategory.CONTENT,
        commission_type=CommissionType.ON_DEMAND,
        output_format="SCRIPT_DRAFT",
        dependencies=["research_agent"],
        state_restrictions={
            "yellow": ["draft_only"],
            "red": ["blocked"]
        }
    ),

    # Learning Pipeline Agents (3)
    "learning_curator": AgentDefinition(
        name="learning_curator",
        category=AgentCategory.LEARNING,
        commission_type=CommissionType.ON_DEMAND,
        output_format="LEARNING_QUEUE",
        dependencies=["learning_scout", "learning_distiller"],
        state_restrictions={
            "red": ["emergency_learning_only"]
        }
    ),
    "learning_scout": AgentDefinition(
        name="learning_scout",
        category=AgentCategory.LEARNING,
        commission_type=CommissionType.ON_DEMAND,
        output_format="LEARNING_CANDIDATES"
    ),
    "learning_distiller": AgentDefinition(
        name="learning_distiller",
        category=AgentCategory.LEARNING,
        commission_type=CommissionType.CONDITIONAL,
        output_format="LEARNING_QUESTIONS",
        trigger_conditions=["stuck_point_detected", "recent_blocker"]
    ),

    # Social/Content Strategy Agents (3)
    "social_metrics_harvester": AgentDefinition(
        name="social_metrics_harvester",
        category=AgentCategory.STRATEGY,
        commission_type=CommissionType.PERIODIC,
        output_format="METRICS_REPORT",
        cadence_days=7,
        state_restrictions={
            "red": ["continue_data_collection"]
        }
    ),
    "audience_signals_extractor": AgentDefinition(
        name="audience_signals_extractor",
        category=AgentCategory.STRATEGY,
        commission_type=CommissionType.PERIODIC,
        output_format="AUDIENCE_SIGNALS",
        dependencies=["social_metrics_harvester"],
        cadence_days=7
    ),
    "content_strategy_analyst": AgentDefinition(
        name="content_strategy_analyst",
        category=AgentCategory.STRATEGY,
        commission_type=CommissionType.PERIODIC,
        output_format="STRATEGY_MEMO",
        dependencies=["social_metrics_harvester", "audience_signals_extractor"],
        cadence_days=7,
        state_restrictions={
            "red": ["pause_strategy_focus_recovery"]
        }
    ),

    # Operations/Infrastructure Agents (7)
    "intake_agent": AgentDefinition(
        name="intake_agent",
        category=AgentCategory.OPERATIONS,
        commission_type=CommissionType.ALWAYS_ACTIVE,
        output_format="INBOUND_BATCH",
        state_restrictions={
            "red": ["continue"]  # Alfred needs information
        }
    ),
    "patient_data_agent": AgentDefinition(
        name="patient_data_agent",
        category=AgentCategory.OPERATIONS,
        commission_type=CommissionType.ON_DEMAND,
        output_format="PATIENT_DATA_RESPONSE",
        state_restrictions={
            "red": ["continue"]  # Clinical never stops
        }
    ),
    "scheduling_agent": AgentDefinition(
        name="scheduling_agent",
        category=AgentCategory.OPERATIONS,
        commission_type=CommissionType.ON_DEMAND,
        output_format="CALENDAR_REPORT",
        state_restrictions={
            "red": ["suggest_clearing_calendar"]
        }
    ),
    "content_manager": AgentDefinition(
        name="content_manager",
        category=AgentCategory.OPERATIONS,
        commission_type=CommissionType.ON_DEMAND,
        output_format="CONTENT_STATUS",
        dependencies=["research_agent", "substack_agent", "twitter_thread_agent", "youtube_script_agent"],
        state_restrictions={
            "red": ["halt_pipeline"]
        }
    ),
    "shipping_governor": AgentDefinition(
        name="shipping_governor",
        category=AgentCategory.OPERATIONS,
        commission_type=CommissionType.CONDITIONAL,
        output_format="SHIPPING_ALERT",
        trigger_conditions=["project_stall_detected", "days_without_output_exceeded"],
        state_restrictions={
            "red": ["pause_shipping_pressure"]
        }
    ),
    "financial_sentinel": AgentDefinition(
        name="financial_sentinel",
        category=AgentCategory.OPERATIONS,
        commission_type=CommissionType.PERIODIC,
        output_format="FINANCIAL_REPORT",
        cadence_days=30,
        state_restrictions={
            "red": ["continue"]
        }
    ),
    "relationship_nudge_agent": AgentDefinition(
        name="relationship_nudge_agent",
        category=AgentCategory.OPERATIONS,
        commission_type=CommissionType.ON_DEMAND,
        output_format="RELATIONSHIP_NUDGE",
        state_restrictions={
            "red": ["defer_non_urgent"]
        }
    ),
}


class Orchestrator:
    """
    Alfred's Orchestration Engine

    Manages sub-agent commissioning, state-based access control, and commission tracking.

    Key Features:
    - Commission Protocol: Structured request/response with sub-agents
    - State Integration: Block/restrict agents based on GREEN/YELLOW/RED state
    - Dependency Resolution: Ensure required agents run before dependent agents
    - Periodic Scheduling: Manage weekly/monthly agent cadences
    - Commission Logging: Full audit trail of all sub-agent interactions
    """

    STORAGE_PATH = "data/alfred/orchestration"
    COMMISSION_LOG_FILE = "commission_log.json"
    SCHEDULE_FILE = "schedules.json"
    STATE_FILE = "orchestrator_state.json"

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent.parent.parent.parent
        self.storage_path = self.base_path / self.STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self.current_state: AlfredState = AlfredState.GREEN
        self.active_commissions: Dict[str, Commission] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.scheduled_commissions: Dict[str, ScheduledCommission] = {}

        # Initialize
        self._load_state()
        self._initialize_agent_states()
        self._initialize_schedules()

    def _load_state(self) -> None:
        """Load persisted orchestrator state."""
        state_file = self.storage_path / self.STATE_FILE
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    self.current_state = AlfredState(data.get("current_state", "green"))
            except (json.JSONDecodeError, ValueError):
                pass

    def _save_state(self) -> None:
        """Persist orchestrator state."""
        state_file = self.storage_path / self.STATE_FILE
        with open(state_file, "w") as f:
            json.dump({
                "current_state": self.current_state.value,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)

    def _initialize_agent_states(self) -> None:
        """Initialize agent availability states."""
        for agent_name in AGENT_REGISTRY:
            self.agent_states[agent_name] = AgentState.AVAILABLE

    def _initialize_schedules(self) -> None:
        """Initialize periodic commission schedules."""
        schedule_file = self.storage_path / self.SCHEDULE_FILE

        if schedule_file.exists():
            try:
                with open(schedule_file, "r") as f:
                    data = json.load(f)
                    for schedule_data in data.get("schedules", []):
                        schedule = ScheduledCommission(
                            schedule_id=schedule_data["schedule_id"],
                            agent_name=schedule_data["agent_name"],
                            cadence_days=schedule_data["cadence_days"],
                            last_run=schedule_data.get("last_run"),
                            next_run=schedule_data.get("next_run"),
                            enabled=schedule_data.get("enabled", True),
                            request_template=schedule_data.get("request_template", {})
                        )
                        self.scheduled_commissions[schedule.schedule_id] = schedule
                return
            except (json.JSONDecodeError, KeyError):
                pass

        # Create default schedules for periodic agents
        for agent_name, agent_def in AGENT_REGISTRY.items():
            if agent_def.commission_type == CommissionType.PERIODIC and agent_def.cadence_days:
                schedule_id = f"SCHED_{agent_name.upper()}"
                self.scheduled_commissions[schedule_id] = ScheduledCommission(
                    schedule_id=schedule_id,
                    agent_name=agent_name,
                    cadence_days=agent_def.cadence_days,
                    next_run=datetime.now().isoformat(),
                    enabled=True
                )

        self._save_schedules()

    def _save_schedules(self) -> None:
        """Persist schedule state."""
        schedule_file = self.storage_path / self.SCHEDULE_FILE
        schedules_data = []

        for schedule in self.scheduled_commissions.values():
            schedules_data.append({
                "schedule_id": schedule.schedule_id,
                "agent_name": schedule.agent_name,
                "cadence_days": schedule.cadence_days,
                "last_run": schedule.last_run,
                "next_run": schedule.next_run,
                "enabled": schedule.enabled,
                "request_template": schedule.request_template
            })

        with open(schedule_file, "w") as f:
            json.dump({"schedules": schedules_data, "updated": datetime.now().isoformat()}, f, indent=2)

    def _log_commission(self, commission: Commission) -> None:
        """Log commission to persistent storage."""
        log_file = self.storage_path / self.COMMISSION_LOG_FILE

        # Load existing log
        log_data = {"commissions": []}
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                pass

        # Add commission
        commission_dict = {
            "commission_id": commission.commission_id,
            "agent_name": commission.agent_name,
            "request_data": commission.request_data,
            "created_at": commission.created_at,
            "status": commission.status.value,
            "alfred_state": commission.alfred_state,
            "priority": commission.priority,
            "context": commission.context,
            "result": commission.result,
            "completed_at": commission.completed_at,
            "error": commission.error,
            "parent_commission_id": commission.parent_commission_id
        }

        # Update if exists, otherwise append
        found = False
        for i, c in enumerate(log_data["commissions"]):
            if c["commission_id"] == commission.commission_id:
                log_data["commissions"][i] = commission_dict
                found = True
                break

        if not found:
            log_data["commissions"].append(commission_dict)

        # Keep only last 1000 commissions
        if len(log_data["commissions"]) > 1000:
            log_data["commissions"] = log_data["commissions"][-1000:]

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def update_alfred_state(self, new_state: AlfredState) -> Dict[str, Any]:
        """
        Update Alfred's state and propagate to agent availability.

        Returns dict with state change effects.
        """
        old_state = self.current_state
        self.current_state = new_state

        effects = {
            "previous_state": old_state.value,
            "new_state": new_state.value,
            "timestamp": datetime.now().isoformat(),
            "blocked_agents": [],
            "restricted_agents": []
        }

        # Update agent states based on new Alfred state
        for agent_name, agent_def in AGENT_REGISTRY.items():
            restrictions = agent_def.state_restrictions.get(new_state.value, [])

            if "blocked" in restrictions:
                self.agent_states[agent_name] = AgentState.BLOCKED
                effects["blocked_agents"].append(agent_name)
            elif restrictions:
                effects["restricted_agents"].append({
                    "agent": agent_name,
                    "restrictions": restrictions
                })
            else:
                self.agent_states[agent_name] = AgentState.AVAILABLE

        self._save_state()
        return effects

    def _check_agent_available(self, agent_name: str) -> Tuple[bool, str]:
        """
        Check if an agent is available for commissioning.

        Returns (is_available, reason).
        """
        if agent_name not in AGENT_REGISTRY:
            return False, f"Unknown agent: {agent_name}"

        agent_state = self.agent_states.get(agent_name, AgentState.AVAILABLE)

        if agent_state == AgentState.BLOCKED:
            return False, f"Agent {agent_name} is blocked in {self.current_state.value.upper()} state"

        if agent_state == AgentState.RUNNING:
            return False, f"Agent {agent_name} is already running a commission"

        # Check state restrictions
        agent_def = AGENT_REGISTRY[agent_name]
        restrictions = agent_def.state_restrictions.get(self.current_state.value, [])

        if "blocked" in restrictions:
            return False, f"Agent {agent_name} is blocked in {self.current_state.value.upper()} state"

        return True, "available"

    def _check_dependencies(self, agent_name: str) -> Tuple[bool, List[str]]:
        """
        Check if agent dependencies are satisfied.

        Returns (dependencies_met, missing_dependencies).
        """
        if agent_name not in AGENT_REGISTRY:
            return False, [f"Unknown agent: {agent_name}"]

        agent_def = AGENT_REGISTRY[agent_name]
        missing = []

        for dep_name in agent_def.dependencies:
            dep_available, reason = self._check_agent_available(dep_name)
            if not dep_available:
                missing.append(f"{dep_name}: {reason}")

        return len(missing) == 0, missing

    def commission_agent(
        self,
        agent_name: str,
        request_data: Dict[str, Any],
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None,
        force: bool = False,
        parent_commission_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Commission a sub-agent to perform work.

        Args:
            agent_name: Name of the agent to commission
            request_data: Structured request data for the agent
            priority: Commission priority (normal, elevated, urgent)
            context: Additional context from Alfred
            force: Override state restrictions (use with caution)
            parent_commission_id: For chained commissions

        Returns:
            Dict with commission_id, status, and result (if synchronous)
        """
        timestamp = datetime.now().isoformat()
        commission_id = f"COM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Check agent availability
        available, reason = self._check_agent_available(agent_name)
        if not available and not force:
            commission = Commission(
                commission_id=commission_id,
                agent_name=agent_name,
                request_data=request_data,
                created_at=timestamp,
                status=CommissionStatus.BLOCKED,
                alfred_state=self.current_state.value,
                priority=priority,
                context=context,
                error=reason,
                parent_commission_id=parent_commission_id
            )
            self._log_commission(commission)

            return {
                "commission_id": commission_id,
                "status": CommissionStatus.BLOCKED.value,
                "agent_name": agent_name,
                "error": reason,
                "alfred_state": self.current_state.value,
                "timestamp": timestamp
            }

        # Check dependencies
        deps_met, missing_deps = self._check_dependencies(agent_name)
        if not deps_met and not force:
            commission = Commission(
                commission_id=commission_id,
                agent_name=agent_name,
                request_data=request_data,
                created_at=timestamp,
                status=CommissionStatus.BLOCKED,
                alfred_state=self.current_state.value,
                priority=priority,
                context=context,
                error=f"Missing dependencies: {', '.join(missing_deps)}",
                parent_commission_id=parent_commission_id
            )
            self._log_commission(commission)

            return {
                "commission_id": commission_id,
                "status": CommissionStatus.BLOCKED.value,
                "agent_name": agent_name,
                "error": f"Missing dependencies: {', '.join(missing_deps)}",
                "alfred_state": self.current_state.value,
                "timestamp": timestamp
            }

        # Get state-based restrictions to include in request
        agent_def = AGENT_REGISTRY.get(agent_name)
        restrictions = []
        if agent_def:
            restrictions = agent_def.state_restrictions.get(self.current_state.value, [])

        # Create commission
        commission = Commission(
            commission_id=commission_id,
            agent_name=agent_name,
            request_data=request_data,
            created_at=timestamp,
            status=CommissionStatus.PENDING,
            alfred_state=self.current_state.value,
            priority=priority,
            context={
                **(context or {}),
                "state_restrictions": restrictions,
                "force_override": force
            },
            parent_commission_id=parent_commission_id
        )

        # Track active commission
        self.active_commissions[commission_id] = commission
        self.agent_states[agent_name] = AgentState.RUNNING

        # Log commission
        self._log_commission(commission)

        return {
            "commission_id": commission_id,
            "status": CommissionStatus.PENDING.value,
            "agent_name": agent_name,
            "alfred_state": self.current_state.value,
            "state_restrictions": restrictions,
            "timestamp": timestamp,
            "request_format": self._format_commission_request(agent_name, request_data, commission_id)
        }

    def _format_commission_request(
        self,
        agent_name: str,
        request_data: Dict[str, Any],
        commission_id: str
    ) -> str:
        """Format the commission request in the standard protocol format."""
        agent_def = AGENT_REGISTRY.get(agent_name)
        if not agent_def:
            return ""

        lines = [
            "COMMISSION",
            f"- Agent: {agent_name}",
            f"- Task ID: {commission_id}",
            f"- Priority: {request_data.get('priority', 'normal')}",
            f"- Alfred State: {self.current_state.value.upper()}",
            "- Request:"
        ]

        for key, value in request_data.items():
            if key != "priority":
                lines.append(f"  {key}: {value}")

        restrictions = agent_def.state_restrictions.get(self.current_state.value, [])
        if restrictions:
            lines.append(f"- State Restrictions: {', '.join(restrictions)}")

        return "\n".join(lines)

    def complete_commission(
        self,
        commission_id: str,
        result: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mark a commission as complete and log results.

        Args:
            commission_id: The commission to complete
            result: The structured result from the sub-agent
            success: Whether the commission succeeded
            error: Error message if failed

        Returns:
            Dict with completion status
        """
        if commission_id not in self.active_commissions:
            return {
                "status": "error",
                "error": f"Commission {commission_id} not found in active commissions"
            }

        commission = self.active_commissions[commission_id]
        commission.completed_at = datetime.now().isoformat()
        commission.result = result

        if success:
            commission.status = CommissionStatus.COMPLETED
        else:
            commission.status = CommissionStatus.FAILED
            commission.error = error

        # Update agent state
        self.agent_states[commission.agent_name] = AgentState.AVAILABLE

        # Log completion
        self._log_commission(commission)

        # Remove from active
        del self.active_commissions[commission_id]

        return {
            "commission_id": commission_id,
            "status": commission.status.value,
            "agent_name": commission.agent_name,
            "completed_at": commission.completed_at,
            "result_summary": self._summarize_result(result)
        }

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """Create a brief summary of commission result."""
        if not result:
            return "No result data"

        if "status" in result:
            return f"Status: {result['status']}"

        return f"Result contains {len(result)} fields"

    def get_available_agents(self) -> Dict[str, Any]:
        """
        Get list of agents available in the current state.

        Returns dict with available, blocked, and restricted agents.
        """
        available = []
        blocked = []
        restricted = []
        running = []

        for agent_name, agent_def in AGENT_REGISTRY.items():
            state = self.agent_states.get(agent_name, AgentState.AVAILABLE)
            restrictions = agent_def.state_restrictions.get(self.current_state.value, [])

            if state == AgentState.RUNNING:
                running.append(agent_name)
            elif state == AgentState.BLOCKED or "blocked" in restrictions:
                blocked.append({
                    "agent": agent_name,
                    "reason": f"Blocked in {self.current_state.value.upper()} state"
                })
            elif restrictions:
                restricted.append({
                    "agent": agent_name,
                    "restrictions": restrictions
                })
                available.append(agent_name)  # Still available but with restrictions
            else:
                available.append(agent_name)

        return {
            "alfred_state": self.current_state.value,
            "timestamp": datetime.now().isoformat(),
            "available": available,
            "blocked": blocked,
            "restricted": restricted,
            "running": running,
            "total_agents": len(AGENT_REGISTRY)
        }

    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """
        Get detailed status of a specific agent.

        Returns status, restrictions, dependencies, and recent commissions.
        """
        if agent_name not in AGENT_REGISTRY:
            return {
                "error": f"Unknown agent: {agent_name}",
                "available_agents": list(AGENT_REGISTRY.keys())
            }

        agent_def = AGENT_REGISTRY[agent_name]
        state = self.agent_states.get(agent_name, AgentState.AVAILABLE)
        restrictions = agent_def.state_restrictions.get(self.current_state.value, [])

        # Check if blocked
        is_blocked = state == AgentState.BLOCKED or "blocked" in restrictions

        # Check dependencies
        deps_met, missing_deps = self._check_dependencies(agent_name)

        # Get recent commissions for this agent
        recent_commissions = self._get_agent_commission_history(agent_name, limit=5)

        return {
            "agent_name": agent_name,
            "category": agent_def.category.value,
            "commission_type": agent_def.commission_type.value,
            "output_format": agent_def.output_format,
            "state": state.value,
            "is_blocked": is_blocked,
            "alfred_state": self.current_state.value,
            "restrictions": restrictions,
            "dependencies": agent_def.dependencies,
            "dependencies_met": deps_met,
            "missing_dependencies": missing_deps,
            "cadence_days": agent_def.cadence_days,
            "trigger_conditions": agent_def.trigger_conditions,
            "recent_commissions": recent_commissions,
            "timestamp": datetime.now().isoformat()
        }

    def _get_agent_commission_history(
        self,
        agent_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent commission history for an agent."""
        log_file = self.storage_path / self.COMMISSION_LOG_FILE

        if not log_file.exists():
            return []

        try:
            with open(log_file, "r") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            return []

        agent_commissions = [
            c for c in log_data.get("commissions", [])
            if c.get("agent_name") == agent_name
        ]

        # Return most recent
        return agent_commissions[-limit:]

    def schedule_periodic_agent(
        self,
        agent_name: str,
        cadence_days: int,
        request_template: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Set up or update periodic commission schedule for an agent.

        Args:
            agent_name: Agent to schedule
            cadence_days: Days between runs
            request_template: Default request data for scheduled runs

        Returns:
            Dict with schedule details
        """
        if agent_name not in AGENT_REGISTRY:
            return {
                "error": f"Unknown agent: {agent_name}",
                "available_agents": list(AGENT_REGISTRY.keys())
            }

        schedule_id = f"SCHED_{agent_name.upper()}"
        now = datetime.now()
        next_run = (now + timedelta(days=cadence_days)).isoformat()

        schedule = ScheduledCommission(
            schedule_id=schedule_id,
            agent_name=agent_name,
            cadence_days=cadence_days,
            next_run=next_run,
            enabled=True,
            request_template=request_template or {}
        )

        self.scheduled_commissions[schedule_id] = schedule
        self._save_schedules()

        return {
            "schedule_id": schedule_id,
            "agent_name": agent_name,
            "cadence_days": cadence_days,
            "next_run": next_run,
            "status": "scheduled",
            "timestamp": datetime.now().isoformat()
        }

    def get_pending_commissions(self) -> Dict[str, Any]:
        """
        Get all pending and scheduled work.

        Returns active commissions and upcoming scheduled runs.
        """
        now = datetime.now()

        # Active commissions
        active = []
        for commission in self.active_commissions.values():
            active.append({
                "commission_id": commission.commission_id,
                "agent_name": commission.agent_name,
                "status": commission.status.value,
                "created_at": commission.created_at,
                "priority": commission.priority
            })

        # Upcoming scheduled
        upcoming = []
        overdue = []

        for schedule in self.scheduled_commissions.values():
            if not schedule.enabled:
                continue

            if schedule.next_run:
                next_run_dt = datetime.fromisoformat(schedule.next_run)
                if next_run_dt <= now:
                    overdue.append({
                        "schedule_id": schedule.schedule_id,
                        "agent_name": schedule.agent_name,
                        "scheduled_for": schedule.next_run,
                        "cadence_days": schedule.cadence_days,
                        "overdue_hours": round((now - next_run_dt).total_seconds() / 3600, 1)
                    })
                else:
                    upcoming.append({
                        "schedule_id": schedule.schedule_id,
                        "agent_name": schedule.agent_name,
                        "next_run": schedule.next_run,
                        "cadence_days": schedule.cadence_days,
                        "days_until": (next_run_dt - now).days
                    })

        return {
            "timestamp": datetime.now().isoformat(),
            "alfred_state": self.current_state.value,
            "active_commissions": active,
            "active_count": len(active),
            "overdue_schedules": overdue,
            "overdue_count": len(overdue),
            "upcoming_schedules": sorted(upcoming, key=lambda x: x["next_run"])[:10],
            "upcoming_count": len(upcoming)
        }

    def run_scheduled_commissions(self) -> List[Dict[str, Any]]:
        """
        Check and run any overdue scheduled commissions.

        Returns list of commissions that were triggered.
        """
        now = datetime.now()
        triggered = []

        for schedule_id, schedule in self.scheduled_commissions.items():
            if not schedule.enabled:
                continue

            if schedule.next_run:
                next_run_dt = datetime.fromisoformat(schedule.next_run)
                if next_run_dt <= now:
                    # Commission the agent
                    result = self.commission_agent(
                        agent_name=schedule.agent_name,
                        request_data=schedule.request_template,
                        priority="normal",
                        context={"scheduled": True, "schedule_id": schedule_id}
                    )

                    # Update schedule
                    schedule.last_run = now.isoformat()
                    schedule.next_run = (now + timedelta(days=schedule.cadence_days)).isoformat()

                    triggered.append({
                        "schedule_id": schedule_id,
                        "agent_name": schedule.agent_name,
                        "commission_result": result,
                        "next_run": schedule.next_run
                    })

        if triggered:
            self._save_schedules()

        return triggered

    def get_agents_by_category(self, category: AgentCategory) -> List[Dict[str, Any]]:
        """Get all agents in a specific category."""
        agents = []

        for agent_name, agent_def in AGENT_REGISTRY.items():
            if agent_def.category == category:
                state = self.agent_states.get(agent_name, AgentState.AVAILABLE)
                agents.append({
                    "name": agent_name,
                    "commission_type": agent_def.commission_type.value,
                    "output_format": agent_def.output_format,
                    "state": state.value,
                    "dependencies": agent_def.dependencies
                })

        return agents

    def get_agents_by_commission_type(self, commission_type: CommissionType) -> List[str]:
        """Get all agents with a specific commission type."""
        return [
            name for name, agent_def in AGENT_REGISTRY.items()
            if agent_def.commission_type == commission_type
        ]

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the complete agent dependency graph."""
        return {
            name: agent_def.dependencies
            for name, agent_def in AGENT_REGISTRY.items()
            if agent_def.dependencies
        }

    def cancel_commission(self, commission_id: str, reason: str = "") -> Dict[str, Any]:
        """Cancel an active commission."""
        if commission_id not in self.active_commissions:
            return {
                "status": "error",
                "error": f"Commission {commission_id} not found in active commissions"
            }

        commission = self.active_commissions[commission_id]
        commission.status = CommissionStatus.CANCELLED
        commission.completed_at = datetime.now().isoformat()
        commission.error = f"Cancelled: {reason}" if reason else "Cancelled by user"

        # Update agent state
        self.agent_states[commission.agent_name] = AgentState.AVAILABLE

        # Log
        self._log_commission(commission)

        # Remove from active
        del self.active_commissions[commission_id]

        return {
            "commission_id": commission_id,
            "status": "cancelled",
            "agent_name": commission.agent_name,
            "timestamp": datetime.now().isoformat()
        }

    def get_commission_log(
        self,
        agent_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get commission history with optional filters.

        Args:
            agent_name: Filter by agent name
            status: Filter by status
            limit: Maximum entries to return

        Returns:
            List of commission records
        """
        log_file = self.storage_path / self.COMMISSION_LOG_FILE

        if not log_file.exists():
            return []

        try:
            with open(log_file, "r") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            return []

        commissions = log_data.get("commissions", [])

        # Apply filters
        if agent_name:
            commissions = [c for c in commissions if c.get("agent_name") == agent_name]

        if status:
            commissions = [c for c in commissions if c.get("status") == status]

        # Return most recent
        return commissions[-limit:]

    def get_orchestration_summary(self) -> Dict[str, Any]:
        """
        Get complete orchestration status summary.

        Returns comprehensive state of the orchestration system.
        """
        available = self.get_available_agents()
        pending = self.get_pending_commissions()

        # Count by category
        by_category = {}
        for cat in AgentCategory:
            agents = self.get_agents_by_category(cat)
            by_category[cat.value] = {
                "total": len(agents),
                "available": sum(1 for a in agents if a["state"] == "available"),
                "blocked": sum(1 for a in agents if a["state"] == "blocked")
            }

        # Count by commission type
        by_type = {}
        for ct in CommissionType:
            agents = self.get_agents_by_commission_type(ct)
            by_type[ct.value] = len(agents)

        return {
            "timestamp": datetime.now().isoformat(),
            "alfred_state": self.current_state.value,
            "total_agents": len(AGENT_REGISTRY),
            "available_count": len(available["available"]),
            "blocked_count": len(available["blocked"]),
            "restricted_count": len(available["restricted"]),
            "running_count": len(available["running"]),
            "active_commissions": pending["active_count"],
            "overdue_schedules": pending["overdue_count"],
            "agents_by_category": by_category,
            "agents_by_commission_type": by_type,
            "dependency_graph": self.get_dependency_graph()
        }


# Convenience functions for direct use
def create_orchestrator(base_path: Optional[Path] = None) -> Orchestrator:
    """Create and return an Orchestrator instance."""
    return Orchestrator(base_path)


def get_agent_registry() -> Dict[str, AgentDefinition]:
    """Get the complete agent registry."""
    return AGENT_REGISTRY


def get_agent_categories() -> List[str]:
    """Get all agent category names."""
    return [cat.value for cat in AgentCategory]


def get_commission_types() -> List[str]:
    """Get all commission type names."""
    return [ct.value for ct in CommissionType]
