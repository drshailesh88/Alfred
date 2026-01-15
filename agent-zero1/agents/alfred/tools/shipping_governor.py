# Shipping Governor - Anti-Procrastination Enforcer
# "Building without shipping is procrastination. Tools without output are toys."

from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json

from . import OperationsAgent, AgentResponse, AlfredState


class ShippingHealth(Enum):
    """Overall shipping health status."""
    HEALTHY = "HEALTHY"      # Regular output cadence, projects progressing
    WARNING = "WARNING"      # Some stalls detected, intervention recommended
    CRITICAL = "CRITICAL"    # Output has stopped, building addiction likely


class ProjectAction(Enum):
    """Recommended actions for projects."""
    SHIP = "SHIP"        # Ship it now, even if imperfect
    KILL = "KILL"        # Kill the project, it's zombie work
    PAUSE = "PAUSE"      # Pause until conditions change
    CONTINUE = "CONTINUE" # Keep working, on track


class BlockerValidity(Enum):
    """Assessment of claimed blockers."""
    VALID = "VALID"            # Legitimate blocker, genuinely blocking
    QUESTIONABLE = "QUESTIONABLE"  # May be excuse, needs scrutiny
    INVALID = "INVALID"        # Not a real blocker, likely procrastination


@dataclass
class Project:
    """Represents an active project being tracked."""
    name: str
    started: date
    description: str = ""
    last_milestone: Optional[str] = None
    last_milestone_date: Optional[date] = None
    claimed_blockers: List[str] = field(default_factory=list)
    outputs_shipped: List[Dict[str, Any]] = field(default_factory=list)
    is_paused: bool = False
    pause_reason: Optional[str] = None

    @property
    def days_since_start(self) -> int:
        """Days since project started."""
        return (date.today() - self.started).days

    @property
    def days_without_output(self) -> int:
        """Days since last shipped output."""
        if not self.outputs_shipped:
            return self.days_since_start

        last_output_date = max(
            datetime.fromisoformat(o.get("shipped_date", self.started.isoformat())).date()
            if isinstance(o.get("shipped_date"), str)
            else o.get("shipped_date", self.started)
            for o in self.outputs_shipped
        )
        return (date.today() - last_output_date).days

    @property
    def days_since_milestone(self) -> int:
        """Days since last milestone."""
        if not self.last_milestone_date:
            return self.days_since_start
        return (date.today() - self.last_milestone_date).days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "started": self.started.isoformat(),
            "description": self.description,
            "last_milestone": self.last_milestone,
            "last_milestone_date": self.last_milestone_date.isoformat() if self.last_milestone_date else None,
            "claimed_blockers": self.claimed_blockers,
            "outputs_shipped": self.outputs_shipped,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "days_since_start": self.days_since_start,
            "days_without_output": self.days_without_output,
        }


@dataclass
class BuildItem:
    """Represents a tool or system being built."""
    name: str
    started: date
    description: str = ""
    linked_output: Optional[str] = None  # What output this tool will produce
    linked_project: Optional[str] = None  # Which project this serves
    progress_percent: int = 0
    last_activity: Optional[date] = None

    @property
    def has_linked_output(self) -> bool:
        """Check if build has a linked output (shipping justification)."""
        return self.linked_output is not None and len(self.linked_output.strip()) > 0

    @property
    def days_in_progress(self) -> int:
        """Days since build started."""
        return (date.today() - self.started).days

    @property
    def days_since_activity(self) -> int:
        """Days since last activity on this build."""
        if not self.last_activity:
            return self.days_in_progress
        return (date.today() - self.last_activity).days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "started": self.started.isoformat(),
            "description": self.description,
            "linked_output": self.linked_output,
            "linked_project": self.linked_project,
            "progress_percent": self.progress_percent,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "has_linked_output": self.has_linked_output,
            "days_in_progress": self.days_in_progress,
        }


@dataclass
class ProjectAssessment:
    """Assessment result for a single project."""
    project_name: str
    started: date
    days_without_output: int
    last_milestone: Optional[str]
    last_milestone_date: Optional[date]
    blocker_validity: BlockerValidity
    blocker_analysis: str
    recommended_action: ProjectAction
    rationale: str
    urgency: str  # "low", "medium", "high", "critical"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output format."""
        return {
            "project": self.project_name,
            "started": self.started.isoformat(),
            "days_without_output": self.days_without_output,
            "last_milestone": f"{self.last_milestone} ({self.last_milestone_date.isoformat()})" if self.last_milestone else "None",
            "blocker_validity": self.blocker_validity.value.lower(),
            "blocker_analysis": self.blocker_analysis,
            "recommended_action": self.recommended_action.value,
            "rationale": self.rationale,
            "urgency": self.urgency,
        }


@dataclass
class ShippingAlert:
    """Urgent shipping alert for a specific project."""
    project_name: str
    days_without_output: int
    pattern: Optional[str]  # Recurring stall pattern if detected
    recommended_action: ProjectAction
    deadline_recommendation: Optional[str]
    consequence_of_inaction: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to alert format."""
        return {
            "project": self.project_name,
            "days_without_output": self.days_without_output,
            "pattern": self.pattern if self.pattern else "None detected",
            "recommended_action": self.recommended_action.value,
            "deadline_recommendation": self.deadline_recommendation if self.deadline_recommendation else "N/A",
            "consequence_of_inaction": self.consequence_of_inaction,
        }


class ShippingGovernor(OperationsAgent):
    """
    Anti-procrastination enforcer that prevents endless preparation and
    building without output. Tracks projects, enforces shipping, kills zombie work.

    SHIPPING RULE: "Building without shipping is procrastination.
                    Tools without output are toys."

    DOES NOT:
    - Encourage building or tool creation
    - Allow new projects when shipping is stalled
    - Extend deadlines without justification
    - Accept "almost done" as status
    - Permit tool creation without linked output
    - Enable perfectionism

    DOES:
    - Track all unfinished projects
    - Count days without shipped output
    - Recommend killing stalled projects
    - Freeze building when shipping stalls
    - Enforce output cadence
    - Distinguish productive work from productive-feeling work
    - Call out building addiction
    """

    # Thresholds for shipping health assessment
    DAYS_WARNING_THRESHOLD = 7      # Days without output before WARNING
    DAYS_CRITICAL_THRESHOLD = 14    # Days without output before CRITICAL
    DAYS_KILL_THRESHOLD = 30        # Days without output before recommending KILL
    BUILD_WITHOUT_OUTPUT_WARNING = 3  # Number of builds without output to flag

    # Blocker validity indicators
    INVALID_BLOCKER_PHRASES = [
        "almost done",
        "just need to",
        "one more thing",
        "polishing",
        "perfecting",
        "refining",
        "tweaking",
        "waiting for inspiration",
        "not ready yet",
        "needs more work",
        "making it better",
    ]

    QUESTIONABLE_BLOCKER_PHRASES = [
        "waiting for feedback",
        "need more research",
        "blocked by",
        "depends on",
        "scheduling conflict",
        "too busy",
        "other priorities",
    ]

    def __init__(self):
        super().__init__(name="Shipping Governor")
        self._projects: Dict[str, Project] = {}
        self._builds: Dict[str, BuildItem] = {}
        self._shipped_outputs: List[Dict[str, Any]] = []
        self._stall_patterns: Dict[str, List[str]] = {}  # Track patterns per project

    # ==================== Project Management ====================

    def add_project(self, name: str, started: date, description: str = "",
                   claimed_blockers: List[str] = None) -> None:
        """Register a new project for tracking."""
        self._projects[name] = Project(
            name=name,
            started=started,
            description=description,
            claimed_blockers=claimed_blockers or [],
        )

    def update_project(self, name: str, **kwargs) -> bool:
        """Update project attributes."""
        if name not in self._projects:
            return False
        project = self._projects[name]
        for key, value in kwargs.items():
            if hasattr(project, key):
                setattr(project, key, value)
        return True

    def record_milestone(self, project_name: str, milestone: str,
                        milestone_date: date = None) -> bool:
        """Record a milestone for a project."""
        if project_name not in self._projects:
            return False
        project = self._projects[project_name]
        project.last_milestone = milestone
        project.last_milestone_date = milestone_date or date.today()
        return True

    def record_output(self, project_name: str, output_name: str,
                     shipped_date: date = None, description: str = "") -> bool:
        """Record a shipped output for a project."""
        if project_name not in self._projects:
            return False

        output = {
            "name": output_name,
            "shipped_date": (shipped_date or date.today()).isoformat(),
            "description": description,
            "project": project_name,
        }

        self._projects[project_name].outputs_shipped.append(output)
        self._shipped_outputs.append(output)

        # Clear stall pattern on successful ship
        if project_name in self._stall_patterns:
            self._stall_patterns[project_name] = []

        return True

    def set_blocker(self, project_name: str, blockers: List[str]) -> bool:
        """Set claimed blockers for a project."""
        if project_name not in self._projects:
            return False
        self._projects[project_name].claimed_blockers = blockers
        return True

    def pause_project(self, project_name: str, reason: str) -> bool:
        """Pause a project with a reason."""
        if project_name not in self._projects:
            return False
        self._projects[project_name].is_paused = True
        self._projects[project_name].pause_reason = reason
        return True

    def resume_project(self, project_name: str) -> bool:
        """Resume a paused project."""
        if project_name not in self._projects:
            return False
        self._projects[project_name].is_paused = False
        self._projects[project_name].pause_reason = None
        return True

    def kill_project(self, project_name: str) -> bool:
        """Remove a project (kill it)."""
        if project_name not in self._projects:
            return False
        del self._projects[project_name]
        return True

    # ==================== Build Management ====================

    def add_build(self, name: str, started: date, description: str = "",
                 linked_output: str = None, linked_project: str = None) -> None:
        """Register a new build (tool/system) for tracking."""
        self._builds[name] = BuildItem(
            name=name,
            started=started,
            description=description,
            linked_output=linked_output,
            linked_project=linked_project,
            last_activity=started,
        )

    def update_build_progress(self, name: str, progress: int,
                             last_activity: date = None) -> bool:
        """Update build progress."""
        if name not in self._builds:
            return False
        self._builds[name].progress_percent = min(100, max(0, progress))
        self._builds[name].last_activity = last_activity or date.today()
        return True

    def link_build_output(self, name: str, linked_output: str) -> bool:
        """Link a build to an expected output."""
        if name not in self._builds:
            return False
        self._builds[name].linked_output = linked_output
        return True

    def complete_build(self, name: str) -> bool:
        """Mark a build as complete and remove from tracking."""
        if name not in self._builds:
            return False
        del self._builds[name]
        return True

    # ==================== Core Assessment Methods ====================

    def validate_blockers(self, blockers: List[str]) -> tuple[BlockerValidity, str]:
        """
        Assess validity of claimed blockers.

        Returns:
            Tuple of (BlockerValidity, analysis_string)
        """
        if not blockers:
            return BlockerValidity.INVALID, "No blockers claimed but not shipping"

        invalid_count = 0
        questionable_count = 0
        analyses = []

        for blocker in blockers:
            blocker_lower = blocker.lower()

            # Check for invalid blocker phrases (procrastination indicators)
            for phrase in self.INVALID_BLOCKER_PHRASES:
                if phrase in blocker_lower:
                    invalid_count += 1
                    analyses.append(f"'{blocker}' contains procrastination pattern '{phrase}'")
                    break
            else:
                # Check for questionable blocker phrases
                for phrase in self.QUESTIONABLE_BLOCKER_PHRASES:
                    if phrase in blocker_lower:
                        questionable_count += 1
                        analyses.append(f"'{blocker}' needs verification")
                        break
                else:
                    analyses.append(f"'{blocker}' may be valid")

        if invalid_count > 0:
            return BlockerValidity.INVALID, "; ".join(analyses)
        elif questionable_count > len(blockers) // 2:
            return BlockerValidity.QUESTIONABLE, "; ".join(analyses)
        else:
            return BlockerValidity.VALID, "; ".join(analyses)

    def assess_project(self, project: Project) -> ProjectAssessment:
        """
        Assess a single project and determine recommended action.
        """
        # Validate blockers
        blocker_validity, blocker_analysis = self.validate_blockers(project.claimed_blockers)

        days_without = project.days_without_output

        # Determine urgency
        if days_without >= self.DAYS_CRITICAL_THRESHOLD:
            urgency = "critical"
        elif days_without >= self.DAYS_WARNING_THRESHOLD:
            urgency = "high"
        elif days_without >= 3:
            urgency = "medium"
        else:
            urgency = "low"

        # Determine recommended action
        if project.is_paused:
            action = ProjectAction.PAUSE
            rationale = f"Project paused: {project.pause_reason}"
        elif days_without >= self.DAYS_KILL_THRESHOLD and blocker_validity != BlockerValidity.VALID:
            action = ProjectAction.KILL
            rationale = f"No output for {days_without} days with invalid/questionable blockers. This is zombie work."
        elif days_without >= self.DAYS_CRITICAL_THRESHOLD:
            action = ProjectAction.SHIP
            rationale = f"Ship something NOW. {days_without} days without output is unacceptable."
        elif days_without >= self.DAYS_WARNING_THRESHOLD:
            if blocker_validity == BlockerValidity.VALID:
                action = ProjectAction.CONTINUE
                rationale = f"Valid blockers but {days_without} days stalled. Address blockers urgently."
            else:
                action = ProjectAction.SHIP
                rationale = f"Blockers are {blocker_validity.value.lower()}. Ship imperfect work."
        else:
            action = ProjectAction.CONTINUE
            rationale = "On track. Keep moving toward output."

        # Track stall patterns
        if days_without >= self.DAYS_WARNING_THRESHOLD and project.name not in self._stall_patterns:
            self._stall_patterns[project.name] = []
        if days_without >= self.DAYS_WARNING_THRESHOLD:
            pattern_entry = f"{date.today().isoformat()}: stalled at {days_without} days"
            if project.name in self._stall_patterns:
                self._stall_patterns[project.name].append(pattern_entry)

        return ProjectAssessment(
            project_name=project.name,
            started=project.started,
            days_without_output=days_without,
            last_milestone=project.last_milestone,
            last_milestone_date=project.last_milestone_date,
            blocker_validity=blocker_validity,
            blocker_analysis=blocker_analysis,
            recommended_action=action,
            rationale=rationale,
            urgency=urgency,
        )

    def check_shipping_health(self) -> ShippingHealth:
        """
        Determine overall shipping health across all projects.
        """
        if not self._projects:
            return ShippingHealth.HEALTHY

        active_projects = [p for p in self._projects.values() if not p.is_paused]

        if not active_projects:
            return ShippingHealth.WARNING  # All paused is suspicious

        # Count projects by days without output
        critical_count = sum(1 for p in active_projects
                            if p.days_without_output >= self.DAYS_CRITICAL_THRESHOLD)
        warning_count = sum(1 for p in active_projects
                           if self.DAYS_WARNING_THRESHOLD <= p.days_without_output < self.DAYS_CRITICAL_THRESHOLD)

        # Check builds without output
        builds_without_output = sum(1 for b in self._builds.values() if not b.has_linked_output)

        # Determine overall health
        if critical_count > 0 or builds_without_output >= self.BUILD_WITHOUT_OUTPUT_WARNING:
            return ShippingHealth.CRITICAL
        elif warning_count > 0:
            return ShippingHealth.WARNING
        else:
            return ShippingHealth.HEALTHY

    def recommend_actions(self) -> Dict[str, Any]:
        """
        Generate recommended actions for all tracked projects and builds.
        """
        project_assessments = []
        for project in self._projects.values():
            assessment = self.assess_project(project)
            project_assessments.append(assessment)

        # Identify builds that should be frozen
        recommended_freezes = []
        shipping_health = self.check_shipping_health()

        if shipping_health == ShippingHealth.CRITICAL:
            # Freeze ALL building that isn't directly linked to shipping
            for build in self._builds.values():
                if not build.has_linked_output:
                    recommended_freezes.append({
                        "build": build.name,
                        "reason": "Building without linked output during shipping crisis",
                        "action": "FREEZE until shipping resumes"
                    })
        elif shipping_health == ShippingHealth.WARNING:
            # Freeze builds without output links
            for build in self._builds.values():
                if not build.has_linked_output:
                    recommended_freezes.append({
                        "build": build.name,
                        "reason": "No linked output - tool without purpose",
                        "action": "Link to output or FREEZE"
                    })

        return {
            "assessments": project_assessments,
            "freezes": recommended_freezes,
            "health": shipping_health,
        }

    def generate_alerts(self) -> List[ShippingAlert]:
        """
        Generate urgent shipping alerts for projects needing immediate attention.
        """
        alerts = []

        for project in self._projects.values():
            if project.is_paused:
                continue

            days_without = project.days_without_output

            if days_without >= self.DAYS_WARNING_THRESHOLD:
                # Detect recurring stall pattern
                pattern = None
                if project.name in self._stall_patterns and len(self._stall_patterns[project.name]) >= 2:
                    pattern = "Recurring stall pattern detected - this project stalls repeatedly"

                # Determine action and deadline
                if days_without >= self.DAYS_KILL_THRESHOLD:
                    action = ProjectAction.KILL
                    deadline = None
                    consequence = "Continued investment in zombie work. Energy drain without output."
                elif days_without >= self.DAYS_CRITICAL_THRESHOLD:
                    action = ProjectAction.SHIP
                    deadline = f"Within {3} days or KILL"
                    consequence = "Project becomes zombie work. Recommend killing if no output by deadline."
                else:
                    action = ProjectAction.SHIP
                    deadline = f"Within {self.DAYS_CRITICAL_THRESHOLD - days_without} days"
                    consequence = "Escalation to CRITICAL status and potential project kill."

                alerts.append(ShippingAlert(
                    project_name=project.name,
                    days_without_output=days_without,
                    pattern=pattern,
                    recommended_action=action,
                    deadline_recommendation=deadline,
                    consequence_of_inaction=consequence,
                ))

        return alerts

    def generate_report(self) -> AgentResponse:
        """
        Generate complete SHIPPING_REPORT for Alfred.

        Output Format:
        SHIPPING_REPORT
        - Report Generated: [timestamp]
        - Overall Shipping Health: HEALTHY | WARNING | CRITICAL
        - Project Assessments: [detailed per project]
        - Building Inventory: [tools in progress]
        - Shipping Alerts: [urgent flags]
        - Recommended Freezes: [building that should stop]
        """
        # Check state permission for RED state behavior
        if self.alfred_state == AlfredState.RED:
            # Pause shipping pressure during crisis
            return self.create_response(
                data={
                    "report_type": "SHIPPING_REPORT",
                    "status": "PAUSED",
                    "reason": "Shipping pressure paused during RED state - focus on recovery",
                    "timestamp": datetime.now().isoformat(),
                },
                warnings=["Shipping Governor paused during RED state"]
            )

        # Generate assessments and recommendations
        recommendations = self.recommend_actions()
        alerts = self.generate_alerts()

        # Build project assessments list
        project_assessments = [a.to_dict() for a in recommendations["assessments"]]

        # Build building inventory
        tools_with_output = [b for b in self._builds.values() if b.has_linked_output]
        tools_without_output = [b for b in self._builds.values() if not b.has_linked_output]

        building_inventory = {
            "tools_in_progress": len(self._builds),
            "tools_with_linked_output": len(tools_with_output),
            "tools_without_output_link": [
                {
                    "name": b.name,
                    "days_in_progress": b.days_in_progress,
                    "description": b.description,
                    "status": "FLAGGED - no linked output"
                }
                for b in tools_without_output
            ],
        }

        # Build alerts list
        shipping_alerts = [a.to_dict() for a in alerts]

        # Build recommended freezes
        recommended_freezes = recommendations["freezes"]

        # Calculate summary stats
        active_projects = [p for p in self._projects.values() if not p.is_paused]
        total_days_stalled = sum(p.days_without_output for p in active_projects) if active_projects else 0
        avg_days_stalled = total_days_stalled / len(active_projects) if active_projects else 0

        # Recent outputs (last 30 days)
        thirty_days_ago = date.today() - timedelta(days=30)
        recent_outputs = [
            o for o in self._shipped_outputs
            if datetime.fromisoformat(o["shipped_date"]).date() >= thirty_days_ago
        ]

        # Compile the full report data
        report_data = {
            "report_type": "SHIPPING_REPORT",
            "report_generated": datetime.now().isoformat(),
            "overall_shipping_health": recommendations["health"].value,

            "summary": {
                "active_projects": len(active_projects),
                "paused_projects": len(self._projects) - len(active_projects),
                "builds_in_progress": len(self._builds),
                "recent_outputs_30d": len(recent_outputs),
                "average_days_without_output": round(avg_days_stalled, 1),
            },

            "project_assessments": project_assessments,

            "building_inventory": building_inventory,

            "shipping_alerts": shipping_alerts,

            "recommended_freezes": recommended_freezes,

            "recent_outputs": recent_outputs[-5:] if recent_outputs else [],  # Last 5 outputs

            "governance_message": self._get_governance_message(recommendations["health"]),
        }

        # Determine warnings
        warnings = []
        if recommendations["health"] == ShippingHealth.CRITICAL:
            warnings.append("CRITICAL: Shipping has stalled. All building should freeze until output resumes.")
        if tools_without_output:
            warnings.append(f"WARNING: {len(tools_without_output)} tools have no linked output - these are toys, not tools.")
        if alerts:
            warnings.append(f"ALERT: {len(alerts)} projects require immediate shipping action.")

        return self.create_response(
            data=report_data,
            warnings=warnings,
        )

    def _get_governance_message(self, health: ShippingHealth) -> str:
        """Get appropriate governance message based on health."""
        if health == ShippingHealth.CRITICAL:
            return (
                "SHIPPING CRISIS DETECTED. "
                "Building without shipping is procrastination. "
                "FREEZE all tool creation. SHIP something today or KILL zombie projects. "
                "No new projects until current work ships."
            )
        elif health == ShippingHealth.WARNING:
            return (
                "SHIPPING WARNING. "
                "Output cadence is slipping. Review stalled projects. "
                "Ship imperfect work rather than perfect nothing. "
                "Distinguish productive work from productive-feeling work."
            )
        else:
            return (
                "Shipping health is GOOD. "
                "Maintain output cadence. "
                "Remember: Tools without output are toys."
            )

    # ==================== Request Processing ====================

    def process_shipping_check(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process a SHIPPING_CHECK_REQUEST from Alfred.

        Expected request format:
        {
            "active_projects": [{"name": str, "start_date": str, "blockers": [str]}],
            "recent_outputs": [{"name": str, "shipped_date": str, "project": str}],
            "pending_builds": [{"name": str, "start_date": str, "linked_output": str | None}],
            "claimed_blockers": [str]  # General blockers
        }
        """
        # Parse and register active projects
        for proj_data in request.get("active_projects", []):
            name = proj_data.get("name")
            if not name:
                continue

            start_date = date.fromisoformat(proj_data.get("start_date", date.today().isoformat()))

            if name not in self._projects:
                self.add_project(
                    name=name,
                    started=start_date,
                    description=proj_data.get("description", ""),
                    claimed_blockers=proj_data.get("blockers", []),
                )
            else:
                self.set_blocker(name, proj_data.get("blockers", []))

        # Record recent outputs
        for output_data in request.get("recent_outputs", []):
            project_name = output_data.get("project")
            if project_name and project_name in self._projects:
                self.record_output(
                    project_name=project_name,
                    output_name=output_data.get("name", "unnamed"),
                    shipped_date=date.fromisoformat(output_data.get("shipped_date", date.today().isoformat())),
                    description=output_data.get("description", ""),
                )

        # Register pending builds
        for build_data in request.get("pending_builds", []):
            name = build_data.get("name")
            if not name:
                continue

            if name not in self._builds:
                start_date = date.fromisoformat(build_data.get("start_date", date.today().isoformat()))
                self.add_build(
                    name=name,
                    started=start_date,
                    description=build_data.get("description", ""),
                    linked_output=build_data.get("linked_output"),
                    linked_project=build_data.get("linked_project"),
                )

        # Generate and return report
        return self.generate_report()

    def generate_alert_packet(self, project_name: str) -> AgentResponse:
        """
        Generate a SHIPPING_ALERT for a specific project.

        Used for urgent escalation to Alfred.
        """
        if project_name not in self._projects:
            return self.create_response(
                data={"error": f"Project '{project_name}' not found"},
                success=False,
                errors=[f"Unknown project: {project_name}"]
            )

        project = self._projects[project_name]
        assessment = self.assess_project(project)

        # Only generate alert if project needs attention
        if project.days_without_output < self.DAYS_WARNING_THRESHOLD:
            return self.create_response(
                data={
                    "report_type": "SHIPPING_ALERT",
                    "status": "NO_ALERT_NEEDED",
                    "project": project_name,
                    "days_without_output": project.days_without_output,
                    "message": "Project is shipping adequately"
                }
            )

        # Generate alert
        pattern = None
        if project_name in self._stall_patterns and len(self._stall_patterns[project_name]) >= 2:
            pattern = "Recurring stall pattern - this project has stalled before"

        # Determine deadline
        if project.days_without_output >= self.DAYS_KILL_THRESHOLD:
            deadline = "IMMEDIATE - KILL recommended"
            consequence = "Continued zombie work draining energy without output"
        elif project.days_without_output >= self.DAYS_CRITICAL_THRESHOLD:
            deadline = "3 days or KILL"
            consequence = "Escalation to zombie work status and project termination"
        else:
            days_to_critical = self.DAYS_CRITICAL_THRESHOLD - project.days_without_output
            deadline = f"{days_to_critical} days before critical"
            consequence = "Escalation to CRITICAL status"

        alert_data = {
            "report_type": "SHIPPING_ALERT",
            "project": project_name,
            "days_without_output": project.days_without_output,
            "pattern": pattern if pattern else "None detected",
            "recommended_action": assessment.recommended_action.value,
            "deadline_recommendation": deadline,
            "consequence_of_inaction": consequence,
            "blocker_validity": assessment.blocker_validity.value,
            "blocker_analysis": assessment.blocker_analysis,
            "urgency": assessment.urgency,
        }

        return self.create_response(
            data=alert_data,
            warnings=[f"SHIPPING ALERT: {project_name} needs immediate attention"]
        )

    def can_start_new_project(self) -> tuple[bool, str]:
        """
        Check if starting a new project is allowed.

        Rule: No new projects when shipping is stalled.
        """
        health = self.check_shipping_health()

        if health == ShippingHealth.CRITICAL:
            return False, (
                "BLOCKED: Cannot start new projects during shipping crisis. "
                "Ship or kill existing projects first. "
                "Building without shipping is procrastination."
            )
        elif health == ShippingHealth.WARNING:
            return False, (
                "BLOCKED: Cannot start new projects with shipping warnings. "
                "Address stalled projects before adding new work. "
                "New projects are often procrastination in disguise."
            )
        else:
            return True, "New project permitted - shipping health is good."

    def can_create_tool(self, linked_output: str = None) -> tuple[bool, str]:
        """
        Check if creating a new tool/build is allowed.

        Rule: No tool creation without linked output.
        """
        if not linked_output or not linked_output.strip():
            return False, (
                "BLOCKED: Cannot create tools without linked output. "
                "Tools without output are toys. "
                "Specify what output this tool will produce."
            )

        health = self.check_shipping_health()

        if health == ShippingHealth.CRITICAL:
            return False, (
                "BLOCKED: Cannot create tools during shipping crisis. "
                "Ship existing work before building new tools. "
                "Building is often procrastination disguised as productivity."
            )

        return True, f"Tool creation permitted with linked output: {linked_output}"

    # ==================== State Export/Import ====================

    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        return {
            "projects": {name: proj.to_dict() for name, proj in self._projects.items()},
            "builds": {name: build.to_dict() for name, build in self._builds.items()},
            "shipped_outputs": self._shipped_outputs,
            "stall_patterns": self._stall_patterns,
            "exported_at": datetime.now().isoformat(),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state from persistence."""
        # Import projects
        self._projects = {}
        for name, proj_data in state.get("projects", {}).items():
            self._projects[name] = Project(
                name=proj_data["name"],
                started=date.fromisoformat(proj_data["started"]),
                description=proj_data.get("description", ""),
                last_milestone=proj_data.get("last_milestone"),
                last_milestone_date=date.fromisoformat(proj_data["last_milestone_date"]) if proj_data.get("last_milestone_date") else None,
                claimed_blockers=proj_data.get("claimed_blockers", []),
                outputs_shipped=proj_data.get("outputs_shipped", []),
                is_paused=proj_data.get("is_paused", False),
                pause_reason=proj_data.get("pause_reason"),
            )

        # Import builds
        self._builds = {}
        for name, build_data in state.get("builds", {}).items():
            self._builds[name] = BuildItem(
                name=build_data["name"],
                started=date.fromisoformat(build_data["started"]),
                description=build_data.get("description", ""),
                linked_output=build_data.get("linked_output"),
                linked_project=build_data.get("linked_project"),
                progress_percent=build_data.get("progress_percent", 0),
                last_activity=date.fromisoformat(build_data["last_activity"]) if build_data.get("last_activity") else None,
            )

        # Import shipped outputs and stall patterns
        self._shipped_outputs = state.get("shipped_outputs", [])
        self._stall_patterns = state.get("stall_patterns", {})


# Convenience function for quick access
def create_shipping_governor() -> ShippingGovernor:
    """Create and return a new ShippingGovernor instance."""
    return ShippingGovernor()
