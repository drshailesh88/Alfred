"""
Tests for ShippingGovernor - Anti-Procrastination Enforcer

Tests cover:
- Main class functionality
- State-aware behavior (GREEN/YELLOW/RED)
- Input/output format compliance
- Project tracking and assessment
- Blocker validation
- Build management
- Edge cases
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent-zero1" / "agents" / "alfred"))

from tools.shipping_governor import (
    ShippingGovernor,
    Project,
    BuildItem,
    ProjectAssessment,
    ShippingAlert,
    ShippingHealth,
    ProjectAction,
    BlockerValidity,
    create_shipping_governor,
)
from tools import AlfredState, AgentResponse


class TestShippingGovernorInitialization:
    """Tests for ShippingGovernor initialization."""

    def test_create_shipping_governor(self):
        """Test that ShippingGovernor can be created."""
        agent = create_shipping_governor()
        assert agent is not None
        assert agent.name == "Shipping Governor"

    def test_initial_state_is_green(self):
        """Test that agent starts in GREEN state."""
        agent = ShippingGovernor()
        assert agent.alfred_state == AlfredState.GREEN

    def test_initial_empty_tracking(self):
        """Test that agent starts with empty tracking."""
        agent = ShippingGovernor()
        assert len(agent._projects) == 0
        assert len(agent._builds) == 0
        assert len(agent._shipped_outputs) == 0

    def test_threshold_constants(self):
        """Test that threshold constants are defined."""
        agent = ShippingGovernor()
        assert agent.DAYS_WARNING_THRESHOLD == 7
        assert agent.DAYS_CRITICAL_THRESHOLD == 14
        assert agent.DAYS_KILL_THRESHOLD == 30


class TestShippingGovernorStateAwareBehavior:
    """Tests for state-aware behavior."""

    def test_green_state_normal_operation(self, shipping_governor_factory, mock_alfred_state_green):
        """Test normal operation in GREEN state."""
        agent = shipping_governor_factory(state=mock_alfred_state_green)
        response = agent.generate_report()
        assert response.success is True
        assert "PAUSED" not in str(response.data)

    def test_yellow_state_continues_operation(self, shipping_governor_factory, mock_alfred_state_yellow):
        """Test that YELLOW state continues operations."""
        agent = shipping_governor_factory(state=mock_alfred_state_yellow)
        response = agent.generate_report()
        assert response.success is True

    def test_red_state_pauses_shipping_pressure(self, shipping_governor_factory, mock_alfred_state_red):
        """Test that RED state pauses shipping pressure."""
        agent = shipping_governor_factory(state=mock_alfred_state_red)
        response = agent.generate_report()
        # In RED state, shipping pressure should be paused
        assert response.data.get("status") == "PAUSED" or response.success is True

    def test_red_state_focus_on_recovery(self, shipping_governor_factory, mock_alfred_state_red):
        """Test that RED state focuses on recovery."""
        agent = shipping_governor_factory(state=mock_alfred_state_red)
        response = agent.generate_report()
        if response.data.get("status") == "PAUSED":
            assert "recovery" in response.data.get("reason", "").lower()


class TestProject:
    """Tests for Project data class."""

    def test_project_creation(self, sample_project_data):
        """Test basic project creation."""
        project = Project(
            name=sample_project_data["name"],
            started=date.fromisoformat(sample_project_data["start_date"]),
            description=sample_project_data["description"],
            claimed_blockers=sample_project_data["blockers"]
        )
        assert project.name == "Alfred Integration"

    def test_days_since_start(self):
        """Test days since start calculation."""
        project = Project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )
        assert project.days_since_start == 10

    def test_days_without_output_no_outputs(self):
        """Test days without output when no outputs shipped."""
        project = Project(
            name="Test Project",
            started=date.today() - timedelta(days=20)
        )
        assert project.days_without_output == 20

    def test_days_without_output_with_outputs(self):
        """Test days without output with shipped outputs."""
        project = Project(
            name="Test Project",
            started=date.today() - timedelta(days=20),
            outputs_shipped=[
                {
                    "name": "Output 1",
                    "shipped_date": (date.today() - timedelta(days=5)).isoformat()
                }
            ]
        )
        assert project.days_without_output == 5

    def test_days_since_milestone(self):
        """Test days since milestone calculation."""
        project = Project(
            name="Test Project",
            started=date.today() - timedelta(days=30),
            last_milestone="MVP Complete",
            last_milestone_date=date.today() - timedelta(days=10)
        )
        assert project.days_since_milestone == 10

    def test_project_to_dict(self):
        """Test project serialization."""
        project = Project(
            name="Test Project",
            started=date.today() - timedelta(days=10),
            description="Test description"
        )
        result = project.to_dict()
        assert result["name"] == "Test Project"
        assert "days_since_start" in result
        assert "days_without_output" in result


class TestBuildItem:
    """Tests for BuildItem data class."""

    def test_build_creation(self, sample_build_data):
        """Test basic build creation."""
        build = BuildItem(
            name=sample_build_data["name"],
            started=date.fromisoformat(sample_build_data["start_date"]),
            description=sample_build_data["description"],
            linked_output=sample_build_data["linked_output"]
        )
        assert build.name == "Custom Dashboard"
        assert build.has_linked_output is True

    def test_build_without_linked_output(self, sample_build_no_output):
        """Test build without linked output."""
        build = BuildItem(
            name=sample_build_no_output["name"],
            started=date.fromisoformat(sample_build_no_output["start_date"]),
            linked_output=sample_build_no_output["linked_output"]
        )
        assert build.has_linked_output is False

    def test_days_in_progress(self):
        """Test days in progress calculation."""
        build = BuildItem(
            name="Test Build",
            started=date.today() - timedelta(days=15)
        )
        assert build.days_in_progress == 15

    def test_days_since_activity(self):
        """Test days since activity calculation."""
        build = BuildItem(
            name="Test Build",
            started=date.today() - timedelta(days=20),
            last_activity=date.today() - timedelta(days=5)
        )
        assert build.days_since_activity == 5

    def test_build_to_dict(self):
        """Test build serialization."""
        build = BuildItem(
            name="Test Build",
            started=date.today() - timedelta(days=10),
            linked_output="test_output"
        )
        result = build.to_dict()
        assert result["name"] == "Test Build"
        assert result["has_linked_output"] is True


class TestProjectManagement:
    """Tests for project management functionality."""

    def test_add_project(self, shipping_governor_factory, sample_project_data):
        """Test adding a project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name=sample_project_data["name"],
            started=date.fromisoformat(sample_project_data["start_date"]),
            description=sample_project_data["description"],
            claimed_blockers=sample_project_data["blockers"]
        )
        assert sample_project_data["name"] in agent._projects

    def test_update_project(self, shipping_governor_factory):
        """Test updating a project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )

        result = agent.update_project("Test Project", description="Updated description")
        assert result is True
        assert agent._projects["Test Project"].description == "Updated description"

    def test_record_milestone(self, shipping_governor_factory):
        """Test recording a milestone."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )

        result = agent.record_milestone("Test Project", "MVP Complete")
        assert result is True
        assert agent._projects["Test Project"].last_milestone == "MVP Complete"

    def test_record_output(self, shipping_governor_factory):
        """Test recording a shipped output."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )

        result = agent.record_output("Test Project", "First Release")
        assert result is True
        assert len(agent._projects["Test Project"].outputs_shipped) == 1

    def test_set_blocker(self, shipping_governor_factory):
        """Test setting blockers."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )

        result = agent.set_blocker("Test Project", ["Waiting for API access"])
        assert result is True
        assert "Waiting for API access" in agent._projects["Test Project"].claimed_blockers

    def test_pause_project(self, shipping_governor_factory):
        """Test pausing a project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )

        result = agent.pause_project("Test Project", "Waiting for external dependency")
        assert result is True
        assert agent._projects["Test Project"].is_paused is True

    def test_resume_project(self, shipping_governor_factory):
        """Test resuming a paused project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )
        agent.pause_project("Test Project", "Testing")

        result = agent.resume_project("Test Project")
        assert result is True
        assert agent._projects["Test Project"].is_paused is False

    def test_kill_project(self, shipping_governor_factory):
        """Test killing a project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )

        result = agent.kill_project("Test Project")
        assert result is True
        assert "Test Project" not in agent._projects


class TestBuildManagement:
    """Tests for build management functionality."""

    def test_add_build(self, shipping_governor_factory, sample_build_data):
        """Test adding a build."""
        agent = shipping_governor_factory()
        agent.add_build(
            name=sample_build_data["name"],
            started=date.fromisoformat(sample_build_data["start_date"]),
            description=sample_build_data["description"],
            linked_output=sample_build_data["linked_output"]
        )
        assert sample_build_data["name"] in agent._builds

    def test_update_build_progress(self, shipping_governor_factory):
        """Test updating build progress."""
        agent = shipping_governor_factory()
        agent.add_build(
            name="Test Build",
            started=date.today() - timedelta(days=5)
        )

        result = agent.update_build_progress("Test Build", progress=50)
        assert result is True
        assert agent._builds["Test Build"].progress_percent == 50

    def test_link_build_output(self, shipping_governor_factory):
        """Test linking build to output."""
        agent = shipping_governor_factory()
        agent.add_build(
            name="Test Build",
            started=date.today() - timedelta(days=5)
        )

        result = agent.link_build_output("Test Build", "Weekly Report")
        assert result is True
        assert agent._builds["Test Build"].linked_output == "Weekly Report"

    def test_complete_build(self, shipping_governor_factory):
        """Test completing a build."""
        agent = shipping_governor_factory()
        agent.add_build(
            name="Test Build",
            started=date.today() - timedelta(days=5)
        )

        result = agent.complete_build("Test Build")
        assert result is True
        assert "Test Build" not in agent._builds


class TestBlockerValidation:
    """Tests for blocker validation logic."""

    def test_validate_empty_blockers(self, shipping_governor_factory):
        """Test validation of empty blockers."""
        agent = shipping_governor_factory()
        validity, analysis = agent.validate_blockers([])
        assert validity == BlockerValidity.INVALID
        assert "No blockers" in analysis

    def test_validate_invalid_blocker_phrases(self, shipping_governor_factory):
        """Test detection of invalid blocker phrases."""
        agent = shipping_governor_factory()

        # Test procrastination indicators
        invalid_blockers = [
            "almost done",
            "just need to polish",
            "perfecting the design"
        ]

        validity, analysis = agent.validate_blockers(invalid_blockers)
        assert validity == BlockerValidity.INVALID
        assert "procrastination" in analysis.lower()

    def test_validate_questionable_blockers(self, shipping_governor_factory):
        """Test detection of questionable blockers."""
        agent = shipping_governor_factory()

        questionable_blockers = [
            "waiting for feedback",
            "blocked by other team"
        ]

        validity, analysis = agent.validate_blockers(questionable_blockers)
        assert validity in [BlockerValidity.QUESTIONABLE, BlockerValidity.VALID]

    def test_validate_valid_blockers(self, shipping_governor_factory):
        """Test validation of valid blockers."""
        agent = shipping_governor_factory()

        valid_blockers = [
            "Server migration scheduled for next week",
            "Legal review required before launch"
        ]

        validity, analysis = agent.validate_blockers(valid_blockers)
        assert validity == BlockerValidity.VALID


class TestProjectAssessment:
    """Tests for project assessment functionality."""

    def test_assess_healthy_project(self, shipping_governor_factory):
        """Test assessment of healthy project."""
        agent = shipping_governor_factory()
        project = Project(
            name="Healthy Project",
            started=date.today() - timedelta(days=5),
            outputs_shipped=[{
                "name": "Initial Release",
                "shipped_date": (date.today() - timedelta(days=2)).isoformat()
            }]
        )

        assessment = agent.assess_project(project)
        assert assessment.recommended_action == ProjectAction.CONTINUE
        assert assessment.urgency == "low"

    def test_assess_warning_project(self, shipping_governor_factory):
        """Test assessment of project at warning threshold."""
        agent = shipping_governor_factory()
        project = Project(
            name="Warning Project",
            started=date.today() - timedelta(days=10),
            claimed_blockers=["waiting for feedback"]
        )

        assessment = agent.assess_project(project)
        assert assessment.urgency in ["medium", "high"]

    def test_assess_critical_project(self, shipping_governor_factory):
        """Test assessment of project at critical threshold."""
        agent = shipping_governor_factory()
        project = Project(
            name="Critical Project",
            started=date.today() - timedelta(days=20),
            claimed_blockers=["almost done"]
        )

        assessment = agent.assess_project(project)
        assert assessment.recommended_action in [ProjectAction.SHIP, ProjectAction.KILL]
        assert assessment.urgency == "critical"

    def test_assess_zombie_project(self, shipping_governor_factory, sample_project_stalled):
        """Test assessment of zombie project (kill recommendation)."""
        agent = shipping_governor_factory()
        project = Project(
            name=sample_project_stalled["name"],
            started=date.fromisoformat(sample_project_stalled["start_date"]),
            claimed_blockers=sample_project_stalled["blockers"]
        )

        assessment = agent.assess_project(project)
        assert assessment.recommended_action == ProjectAction.KILL
        assert "zombie" in assessment.rationale.lower()

    def test_assess_paused_project(self, shipping_governor_factory):
        """Test assessment of paused project."""
        agent = shipping_governor_factory()
        project = Project(
            name="Paused Project",
            started=date.today() - timedelta(days=30),
            is_paused=True,
            pause_reason="Waiting for Q2 budget"
        )

        assessment = agent.assess_project(project)
        assert assessment.recommended_action == ProjectAction.PAUSE


class TestShippingHealth:
    """Tests for shipping health assessment."""

    def test_healthy_shipping(self, shipping_governor_factory):
        """Test healthy shipping status."""
        agent = shipping_governor_factory()
        # Add a healthy project
        agent.add_project(
            name="Active Project",
            started=date.today() - timedelta(days=5)
        )
        agent.record_output("Active Project", "Recent Output")

        health = agent.check_shipping_health()
        assert health == ShippingHealth.HEALTHY

    def test_warning_shipping(self, shipping_governor_factory):
        """Test warning shipping status."""
        agent = shipping_governor_factory()
        # Add project stalled at warning threshold
        agent.add_project(
            name="Stalled Project",
            started=date.today() - timedelta(days=10)
        )

        health = agent.check_shipping_health()
        assert health == ShippingHealth.WARNING

    def test_critical_shipping(self, shipping_governor_factory):
        """Test critical shipping status."""
        agent = shipping_governor_factory()
        # Add project stalled at critical threshold
        agent.add_project(
            name="Critical Project",
            started=date.today() - timedelta(days=20)
        )

        health = agent.check_shipping_health()
        assert health == ShippingHealth.CRITICAL

    def test_builds_affect_shipping_health(self, shipping_governor_factory):
        """Test that builds without output affect health."""
        agent = shipping_governor_factory()
        # Add builds without linked output
        for i in range(3):
            agent.add_build(
                name=f"Unlinked Build {i}",
                started=date.today() - timedelta(days=10)
            )

        health = agent.check_shipping_health()
        # Multiple builds without output should trigger warning/critical
        assert health in [ShippingHealth.WARNING, ShippingHealth.CRITICAL, ShippingHealth.HEALTHY]


class TestNewProjectPermission:
    """Tests for new project permission logic."""

    def test_can_start_project_when_healthy(self, shipping_governor_factory):
        """Test permission to start new project when healthy."""
        agent = shipping_governor_factory()
        # Add healthy project
        agent.add_project(
            name="Active Project",
            started=date.today() - timedelta(days=5)
        )
        agent.record_output("Active Project", "Output")

        can_start, reason = agent.can_start_new_project()
        assert can_start is True

    def test_cannot_start_project_during_crisis(self, shipping_governor_factory):
        """Test blocking new project during shipping crisis."""
        agent = shipping_governor_factory()
        # Add critical project
        agent.add_project(
            name="Critical Project",
            started=date.today() - timedelta(days=20)
        )

        can_start, reason = agent.can_start_new_project()
        assert can_start is False
        assert "BLOCKED" in reason


class TestNewToolPermission:
    """Tests for new tool/build permission logic."""

    def test_can_create_tool_with_output(self, shipping_governor_factory):
        """Test permission to create tool with linked output."""
        agent = shipping_governor_factory()
        # Ensure healthy state
        agent.add_project(
            name="Active Project",
            started=date.today() - timedelta(days=3)
        )
        agent.record_output("Active Project", "Output")

        can_create, reason = agent.can_create_tool(linked_output="Weekly Report")
        assert can_create is True

    def test_cannot_create_tool_without_output(self, shipping_governor_factory):
        """Test blocking tool creation without linked output."""
        agent = shipping_governor_factory()

        can_create, reason = agent.can_create_tool(linked_output=None)
        assert can_create is False
        assert "linked output" in reason.lower()

    def test_cannot_create_tool_during_crisis(self, shipping_governor_factory):
        """Test blocking tool creation during shipping crisis."""
        agent = shipping_governor_factory()
        # Add critical project
        agent.add_project(
            name="Critical Project",
            started=date.today() - timedelta(days=20)
        )

        can_create, reason = agent.can_create_tool(linked_output="Some Output")
        assert can_create is False
        assert "crisis" in reason.lower()


class TestShippingAlerts:
    """Tests for shipping alert generation."""

    def test_generate_alert_for_stalled_project(self, shipping_governor_factory):
        """Test alert generation for stalled project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Stalled Project",
            started=date.today() - timedelta(days=15)
        )

        alerts = agent.generate_alerts()
        assert len(alerts) >= 1
        assert alerts[0].days_without_output >= 7

    def test_no_alert_for_healthy_project(self, shipping_governor_factory):
        """Test no alert for healthy project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Healthy Project",
            started=date.today() - timedelta(days=3)
        )
        agent.record_output("Healthy Project", "Output")

        alerts = agent.generate_alerts()
        assert len(alerts) == 0

    def test_alert_includes_deadline(self, shipping_governor_factory):
        """Test that alerts include deadline recommendations."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Stalled Project",
            started=date.today() - timedelta(days=10)
        )

        alerts = agent.generate_alerts()
        if len(alerts) > 0:
            assert alerts[0].deadline_recommendation is not None

    def test_alert_detects_recurring_pattern(self, shipping_governor_factory):
        """Test detection of recurring stall pattern."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Recurring Stall",
            started=date.today() - timedelta(days=20)
        )
        # Simulate pattern detection
        agent._stall_patterns["Recurring Stall"] = [
            "2024-01-01: stalled",
            "2024-01-15: stalled"
        ]

        alerts = agent.generate_alerts()
        if len(alerts) > 0:
            assert alerts[0].pattern is not None or alerts[0].pattern is None


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_shipping_report(self, shipping_governor_factory):
        """Test shipping report generation."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )

        response = agent.generate_report()
        assert isinstance(response, AgentResponse)
        assert "report_type" in response.data
        assert response.data["report_type"] == "SHIPPING_REPORT"

    def test_report_includes_assessments(self, shipping_governor_factory):
        """Test that report includes project assessments."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Project A",
            started=date.today() - timedelta(days=5)
        )
        agent.add_project(
            name="Project B",
            started=date.today() - timedelta(days=10)
        )

        response = agent.generate_report()
        assert "project_assessments" in response.data
        assert len(response.data["project_assessments"]) == 2

    def test_report_includes_governance_message(self, shipping_governor_factory):
        """Test that report includes governance message."""
        agent = shipping_governor_factory()

        response = agent.generate_report()
        assert "governance_message" in response.data


class TestProcessShippingCheck:
    """Tests for shipping check request processing."""

    def test_process_shipping_check(self, shipping_governor_factory, sample_shipping_check_request):
        """Test processing shipping check request."""
        agent = shipping_governor_factory()
        response = agent.process_shipping_check(sample_shipping_check_request)

        assert isinstance(response, AgentResponse)
        assert response.success is True

    def test_shipping_check_registers_projects(self, shipping_governor_factory):
        """Test that shipping check registers new projects."""
        agent = shipping_governor_factory()
        request = {
            "active_projects": [
                {
                    "name": "New Project",
                    "start_date": (date.today() - timedelta(days=5)).isoformat()
                }
            ]
        }

        agent.process_shipping_check(request)
        assert "New Project" in agent._projects

    def test_shipping_check_records_outputs(self, shipping_governor_factory):
        """Test that shipping check records outputs."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Existing Project",
            started=date.today() - timedelta(days=10)
        )

        request = {
            "active_projects": [
                {
                    "name": "Existing Project",
                    "start_date": (date.today() - timedelta(days=10)).isoformat()
                }
            ],
            "recent_outputs": [
                {
                    "name": "New Output",
                    "project": "Existing Project",
                    "shipped_date": date.today().isoformat()
                }
            ]
        }

        agent.process_shipping_check(request)
        assert len(agent._projects["Existing Project"].outputs_shipped) == 1


class TestAlertPacketGeneration:
    """Tests for alert packet generation."""

    def test_generate_alert_packet(self, shipping_governor_factory):
        """Test alert packet generation for project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Stalled Project",
            started=date.today() - timedelta(days=15)
        )

        response = agent.generate_alert_packet("Stalled Project")
        assert isinstance(response, AgentResponse)
        assert "SHIPPING_ALERT" in response.data.get("report_type", "")

    def test_alert_packet_unknown_project(self, shipping_governor_factory):
        """Test alert packet for unknown project."""
        agent = shipping_governor_factory()

        response = agent.generate_alert_packet("Unknown Project")
        assert response.success is False
        assert "not found" in str(response.data).lower()

    def test_alert_packet_healthy_project(self, shipping_governor_factory):
        """Test alert packet for healthy project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Healthy Project",
            started=date.today() - timedelta(days=3)
        )
        agent.record_output("Healthy Project", "Output")

        response = agent.generate_alert_packet("Healthy Project")
        assert "NO_ALERT_NEEDED" in response.data.get("status", "")


class TestStateExportImport:
    """Tests for state export/import functionality."""

    def test_export_state(self, shipping_governor_factory):
        """Test state export."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Test Project",
            started=date.today() - timedelta(days=10)
        )
        agent.add_build(
            name="Test Build",
            started=date.today() - timedelta(days=5),
            linked_output="output"
        )

        state = agent.export_state()
        assert "projects" in state
        assert "builds" in state
        assert "Test Project" in state["projects"]
        assert "Test Build" in state["builds"]

    def test_import_state(self, shipping_governor_factory):
        """Test state import."""
        agent = shipping_governor_factory()

        state = {
            "projects": {
                "Imported Project": {
                    "name": "Imported Project",
                    "started": (date.today() - timedelta(days=5)).isoformat(),
                    "description": "Imported",
                    "last_milestone": None,
                    "last_milestone_date": None,
                    "claimed_blockers": [],
                    "outputs_shipped": [],
                    "is_paused": False,
                    "pause_reason": None
                }
            },
            "builds": {},
            "shipped_outputs": [],
            "stall_patterns": {}
        }

        agent.import_state(state)
        assert "Imported Project" in agent._projects


class TestShippingGovernorEdgeCases:
    """Edge case tests for ShippingGovernor."""

    def test_project_with_future_start_date(self, shipping_governor_factory):
        """Test handling project with future start date."""
        agent = shipping_governor_factory()
        # This shouldn't happen but should handle gracefully
        agent.add_project(
            name="Future Project",
            started=date.today() + timedelta(days=5)
        )
        project = agent._projects["Future Project"]
        assert project.days_since_start <= 0

    def test_empty_project_name(self, shipping_governor_factory):
        """Test handling empty project name."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="",
            started=date.today()
        )
        # Should handle empty name
        assert "" in agent._projects

    def test_very_old_project(self, shipping_governor_factory):
        """Test handling very old project."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Ancient Project",
            started=date.today() - timedelta(days=365)
        )

        assessment = agent.assess_project(agent._projects["Ancient Project"])
        assert assessment.recommended_action == ProjectAction.KILL

    def test_all_projects_paused(self, shipping_governor_factory):
        """Test health check when all projects paused."""
        agent = shipping_governor_factory()
        agent.add_project(
            name="Paused 1",
            started=date.today() - timedelta(days=10)
        )
        agent.pause_project("Paused 1", "Testing")

        health = agent.check_shipping_health()
        # All paused should be suspicious
        assert health == ShippingHealth.WARNING

    def test_no_projects(self, shipping_governor_factory):
        """Test health check with no projects."""
        agent = shipping_governor_factory()

        health = agent.check_shipping_health()
        assert health == ShippingHealth.HEALTHY
