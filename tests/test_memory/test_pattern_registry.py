"""
Tests for PatternRegistry - Behavioral Pattern Tracking System

Tests cover:
- Main class functionality
- Pattern lifecycle (add, record, update, resolve)
- Trajectory tracking and auto-update
- Intervention effectiveness tracking
- Query functionality
- Edge cases
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent-zero1" / "agents" / "alfred"))

from memory.pattern_registry import (
    PatternRegistry,
    PatternType,
    Trajectory,
)
from memory import MemoryType, MemoryEntry, BaseMemorySystem


class TestPatternRegistryInitialization:
    """Tests for PatternRegistry initialization."""

    def test_create_pattern_registry(self, pattern_registry_factory):
        """Test that PatternRegistry can be created."""
        registry = pattern_registry_factory()
        assert registry is not None
        assert registry.memory_type == MemoryType.PATTERN

    def test_initial_empty_registry(self, pattern_registry_factory):
        """Test that registry starts empty."""
        registry = pattern_registry_factory()
        assert len(registry.list_all()) == 0

    def test_storage_path_set(self, pattern_registry_factory, temp_memory_path):
        """Test that storage path is set correctly."""
        registry = pattern_registry_factory()
        assert registry.storage_path is not None


class TestPatternType:
    """Tests for PatternType enum."""

    def test_pattern_types_defined(self):
        """Test that all pattern types are defined."""
        assert PatternType.OBSESSION_LOOP.value == "obsession_loop"
        assert PatternType.AVOIDANCE.value == "avoidance"
        assert PatternType.EGO_OVERREACH.value == "ego_overreach"
        assert PatternType.DEPLETION.value == "depletion"
        assert PatternType.THRESHOLD_APPROACH.value == "threshold_approach"


class TestTrajectory:
    """Tests for Trajectory enum."""

    def test_trajectory_values(self):
        """Test trajectory values."""
        assert Trajectory.IMPROVING.value == "improving"
        assert Trajectory.STABLE.value == "stable"
        assert Trajectory.WORSENING.value == "worsening"
        assert Trajectory.RESOLVED.value == "resolved"


class TestAddPattern:
    """Tests for adding patterns."""

    def test_add_pattern_basic(self, pattern_registry_factory, sample_pattern_data):
        """Test adding a basic pattern."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.OBSESSION_LOOP,
            description=sample_pattern_data["description"],
            initial_context=sample_pattern_data["initial_context"],
            severity=sample_pattern_data["severity"]
        )

        assert pattern_id is not None
        assert pattern_id.startswith("pattern_")

    def test_add_pattern_without_context(self, pattern_registry_factory):
        """Test adding pattern without initial context."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.AVOIDANCE,
            description="Avoiding financial review tasks"
        )

        assert pattern_id is not None
        entry = registry.get(pattern_id)
        assert entry is not None
        assert len(entry.data["occurrences"]) == 0

    def test_add_pattern_stores_correctly(self, pattern_registry_factory, sample_pattern_data):
        """Test that pattern data is stored correctly."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.OBSESSION_LOOP,
            description=sample_pattern_data["description"],
            initial_context=sample_pattern_data["initial_context"],
            severity=sample_pattern_data["severity"]
        )

        entry = registry.get(pattern_id)
        assert entry.data["pattern_type"] == PatternType.OBSESSION_LOOP.value
        assert entry.data["description"] == sample_pattern_data["description"]
        assert entry.data["active"] is True

    def test_add_pattern_with_initial_occurrence(self, pattern_registry_factory):
        """Test that initial context creates occurrence."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.DEPLETION,
            description="Energy depletion pattern",
            initial_context="Working 12 hour days for 2 weeks",
            severity=8
        )

        entry = registry.get(pattern_id)
        assert len(entry.data["occurrences"]) == 1
        assert entry.data["occurrence_count"] == 1

    def test_severity_bounds(self, pattern_registry_factory):
        """Test that severity is bounded 1-10."""
        registry = pattern_registry_factory()

        # Test upper bound
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.EGO_OVERREACH,
            description="Test",
            initial_context="Context",
            severity=15  # Over 10
        )
        entry = registry.get(pattern_id)
        assert entry.data["occurrences"][0]["severity"] == 10

        # Test lower bound
        pattern_id_2 = registry.add_pattern(
            pattern_type=PatternType.AVOIDANCE,
            description="Test 2",
            initial_context="Context",
            severity=-5  # Under 1
        )
        entry_2 = registry.get(pattern_id_2)
        assert entry_2.data["occurrences"][0]["severity"] == 1


class TestRecordOccurrence:
    """Tests for recording pattern occurrences."""

    def test_record_occurrence(self, pattern_registry_factory, sample_occurrence_data):
        """Test recording a pattern occurrence."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.OBSESSION_LOOP,
            description="Test pattern"
        )

        result = registry.record_occurrence(
            pattern_id=pattern_id,
            context=sample_occurrence_data["context"],
            severity=sample_occurrence_data["severity"],
            notes=sample_occurrence_data["notes"],
            trigger=sample_occurrence_data["trigger"]
        )

        assert result is True
        entry = registry.get(pattern_id)
        assert len(entry.data["occurrences"]) == 1

    def test_record_multiple_occurrences(self, pattern_registry_factory):
        """Test recording multiple occurrences."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.AVOIDANCE,
            description="Test pattern"
        )

        for i in range(5):
            registry.record_occurrence(
                pattern_id=pattern_id,
                context=f"Occurrence {i}",
                severity=5
            )

        entry = registry.get(pattern_id)
        assert entry.data["occurrence_count"] == 5

    def test_record_occurrence_updates_average_severity(self, pattern_registry_factory):
        """Test that recording updates average severity."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.DEPLETION,
            description="Test",
            initial_context="First",
            severity=4
        )

        registry.record_occurrence(pattern_id, "Second", severity=8)
        registry.record_occurrence(pattern_id, "Third", severity=6)

        entry = registry.get(pattern_id)
        # Average of 4, 8, 6 = 6
        assert entry.data["average_severity"] == 6.0

    def test_record_occurrence_updates_last_occurrence(self, pattern_registry_factory):
        """Test that last_occurrence timestamp is updated."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.THRESHOLD_APPROACH,
            description="Test"
        )

        before = datetime.now().isoformat()
        registry.record_occurrence(pattern_id, "Context", severity=5)
        after = datetime.now().isoformat()

        entry = registry.get(pattern_id)
        assert before <= entry.data["last_occurrence"] <= after

    def test_record_occurrence_invalid_pattern(self, pattern_registry_factory):
        """Test recording to non-existent pattern."""
        registry = pattern_registry_factory()
        result = registry.record_occurrence(
            pattern_id="nonexistent_id",
            context="Test",
            severity=5
        )
        assert result is False


class TestTrajectoryUpdate:
    """Tests for trajectory update functionality."""

    def test_manual_trajectory_update(self, pattern_registry_factory):
        """Test manual trajectory update."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.OBSESSION_LOOP,
            description="Test"
        )

        result = registry.update_trajectory(
            pattern_id=pattern_id,
            trajectory=Trajectory.IMPROVING,
            reason="Started new routine"
        )

        assert result is True
        entry = registry.get(pattern_id)
        assert entry.data["trajectory"] == Trajectory.IMPROVING.value

    def test_trajectory_history_tracked(self, pattern_registry_factory):
        """Test that trajectory history is tracked."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.AVOIDANCE,
            description="Test"
        )

        registry.update_trajectory(pattern_id, Trajectory.WORSENING, "Getting worse")
        registry.update_trajectory(pattern_id, Trajectory.IMPROVING, "Intervention worked")

        entry = registry.get(pattern_id)
        assert len(entry.data["trajectory_history"]) == 2

    def test_resolved_trajectory_deactivates_pattern(self, pattern_registry_factory):
        """Test that RESOLVED trajectory deactivates pattern."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.DEPLETION,
            description="Test"
        )

        registry.update_trajectory(pattern_id, Trajectory.RESOLVED, "Pattern resolved")

        entry = registry.get(pattern_id)
        assert entry.data["active"] is False
        assert "resolved_at" in entry.data

    def test_auto_trajectory_update_worsening(self, pattern_registry_factory):
        """Test automatic trajectory detection for worsening pattern."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.EGO_OVERREACH,
            description="Test",
            initial_context="First",
            severity=3
        )

        # Add occurrences with increasing severity
        severities = [4, 5, 6, 7, 8]
        for i, sev in enumerate(severities):
            registry.record_occurrence(pattern_id, f"Occurrence {i}", severity=sev)

        entry = registry.get(pattern_id)
        assert entry.data["trajectory"] == Trajectory.WORSENING.value

    def test_auto_trajectory_update_improving(self, pattern_registry_factory):
        """Test automatic trajectory detection for improving pattern."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.OBSESSION_LOOP,
            description="Test",
            initial_context="First",
            severity=9
        )

        # Add occurrences with decreasing severity
        severities = [8, 6, 4, 3, 2]
        for i, sev in enumerate(severities):
            registry.record_occurrence(pattern_id, f"Occurrence {i}", severity=sev)

        entry = registry.get(pattern_id)
        assert entry.data["trajectory"] == Trajectory.IMPROVING.value


class TestInterventionTracking:
    """Tests for intervention effectiveness tracking."""

    def test_record_intervention(self, pattern_registry_factory, sample_intervention_data):
        """Test recording an intervention."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.OBSESSION_LOOP,
            description="Test"
        )

        result = registry.record_intervention(
            pattern_id=pattern_id,
            intervention_id=sample_intervention_data["intervention_id"],
            intervention_description=sample_intervention_data["intervention_description"],
            effectiveness=sample_intervention_data["effectiveness"]
        )

        assert result is True
        entry = registry.get(pattern_id)
        assert sample_intervention_data["intervention_id"] in entry.data["intervention_effectiveness"]

    def test_intervention_stores_details(self, pattern_registry_factory):
        """Test that intervention details are stored."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.AVOIDANCE,
            description="Test"
        )

        registry.record_intervention(
            pattern_id=pattern_id,
            intervention_id="int_001",
            intervention_description="Implemented daily check-in",
            effectiveness=7
        )

        entry = registry.get(pattern_id)
        intervention = entry.data["intervention_effectiveness"]["int_001"]
        assert intervention["description"] == "Implemented daily check-in"
        assert intervention["effectiveness"] == 7
        assert "timestamp" in intervention

    def test_multiple_interventions(self, pattern_registry_factory):
        """Test recording multiple interventions."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            pattern_type=PatternType.DEPLETION,
            description="Test"
        )

        registry.record_intervention(pattern_id, "int_001", "First intervention", 5)
        registry.record_intervention(pattern_id, "int_002", "Second intervention", 8)

        entry = registry.get(pattern_id)
        assert len(entry.data["intervention_effectiveness"]) == 2


class TestQueryPatterns:
    """Tests for pattern query functionality."""

    def test_get_patterns_by_type(self, pattern_registry_factory):
        """Test getting patterns by type."""
        registry = pattern_registry_factory()

        # Add patterns of different types
        registry.add_pattern(PatternType.OBSESSION_LOOP, "Pattern 1")
        registry.add_pattern(PatternType.OBSESSION_LOOP, "Pattern 2")
        registry.add_pattern(PatternType.AVOIDANCE, "Pattern 3")

        obsession_patterns = registry.get_patterns_by_type(PatternType.OBSESSION_LOOP)
        assert len(obsession_patterns) == 2

    def test_get_active_patterns(self, pattern_registry_factory):
        """Test getting active patterns only."""
        registry = pattern_registry_factory()

        pattern_1 = registry.add_pattern(PatternType.OBSESSION_LOOP, "Active")
        pattern_2 = registry.add_pattern(PatternType.AVOIDANCE, "Resolved")
        registry.update_trajectory(pattern_2, Trajectory.RESOLVED)

        active = registry.get_active_patterns()
        assert len(active) == 1
        assert active[0]["pattern_id"] == pattern_1

    def test_get_worsening_patterns(self, pattern_registry_factory):
        """Test getting worsening patterns."""
        registry = pattern_registry_factory()

        pattern_1 = registry.add_pattern(PatternType.DEPLETION, "Worsening")
        pattern_2 = registry.add_pattern(PatternType.EGO_OVERREACH, "Stable")

        registry.update_trajectory(pattern_1, Trajectory.WORSENING)

        worsening = registry.get_worsening_patterns()
        assert len(worsening) == 1
        assert worsening[0]["pattern_id"] == pattern_1

    def test_get_pattern_summary(self, pattern_registry_factory):
        """Test getting pattern summary."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            PatternType.OBSESSION_LOOP,
            "Test pattern",
            "Initial context",
            severity=5
        )
        registry.record_occurrence(pattern_id, "Second", severity=7)
        registry.record_intervention(pattern_id, "int_001", "Test intervention", 6)

        summary = registry.get_pattern_summary(pattern_id)

        assert summary is not None
        assert summary["pattern_id"] == pattern_id
        assert summary["occurrence_count"] == 2
        assert summary["average_severity"] == 6.0  # (5+7)/2
        assert summary["intervention_count"] == 1

    def test_get_pattern_summary_nonexistent(self, pattern_registry_factory):
        """Test summary for non-existent pattern."""
        registry = pattern_registry_factory()
        summary = registry.get_pattern_summary("nonexistent_id")
        assert summary is None


class TestPatternLinking:
    """Tests for pattern linking functionality."""

    def test_link_patterns(self, pattern_registry_factory):
        """Test linking two patterns."""
        registry = pattern_registry_factory()

        pattern_1 = registry.add_pattern(PatternType.OBSESSION_LOOP, "Pattern 1")
        pattern_2 = registry.add_pattern(PatternType.DEPLETION, "Pattern 2")

        result = registry.link_patterns(pattern_1, pattern_2, "causes")

        assert result is True
        entry_1 = registry.get(pattern_1)
        entry_2 = registry.get(pattern_2)

        assert len(entry_1.data["related_patterns"]) == 1
        assert len(entry_2.data["related_patterns"]) == 1

    def test_link_bidirectional(self, pattern_registry_factory):
        """Test that linking is bidirectional."""
        registry = pattern_registry_factory()

        pattern_1 = registry.add_pattern(PatternType.AVOIDANCE, "Pattern 1")
        pattern_2 = registry.add_pattern(PatternType.EGO_OVERREACH, "Pattern 2")

        registry.link_patterns(pattern_1, pattern_2)

        entry_1 = registry.get(pattern_1)
        assert entry_1.data["related_patterns"][0]["pattern_id"] == pattern_2

        entry_2 = registry.get(pattern_2)
        assert entry_2.data["related_patterns"][0]["pattern_id"] == pattern_1

    def test_link_nonexistent_pattern(self, pattern_registry_factory):
        """Test linking with non-existent pattern."""
        registry = pattern_registry_factory()
        pattern_1 = registry.add_pattern(PatternType.DEPLETION, "Pattern 1")

        result = registry.link_patterns(pattern_1, "nonexistent_id")
        assert result is False


class TestRecentOccurrences:
    """Tests for recent occurrences query."""

    def test_get_recent_occurrences(self, pattern_registry_factory):
        """Test getting recent occurrences."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            PatternType.OBSESSION_LOOP,
            "Test",
            "Recent context",
            severity=5
        )

        recent = registry.get_recent_occurrences(days=7)

        assert len(recent) == 1
        assert recent[0]["pattern_id"] == pattern_id

    def test_recent_occurrences_sorted(self, pattern_registry_factory):
        """Test that recent occurrences are sorted by timestamp."""
        registry = pattern_registry_factory()

        pattern_1 = registry.add_pattern(PatternType.AVOIDANCE, "First", "Context 1", 5)
        pattern_2 = registry.add_pattern(PatternType.DEPLETION, "Second", "Context 2", 6)

        recent = registry.get_recent_occurrences(days=7)

        # Should be sorted descending by timestamp
        assert len(recent) == 2
        assert recent[0]["timestamp"] >= recent[1]["timestamp"]


class TestPatternRegistryEdgeCases:
    """Edge case tests for PatternRegistry."""

    def test_empty_description(self, pattern_registry_factory):
        """Test pattern with empty description."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            PatternType.OBSESSION_LOOP,
            description=""
        )
        assert pattern_id is not None
        entry = registry.get(pattern_id)
        assert entry.data["description"] == ""

    def test_very_long_description(self, pattern_registry_factory):
        """Test pattern with very long description."""
        registry = pattern_registry_factory()
        long_desc = "A" * 10000
        pattern_id = registry.add_pattern(
            PatternType.AVOIDANCE,
            description=long_desc
        )
        entry = registry.get(pattern_id)
        assert entry.data["description"] == long_desc

    def test_many_occurrences(self, pattern_registry_factory):
        """Test pattern with many occurrences."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            PatternType.DEPLETION,
            description="High frequency pattern"
        )

        for i in range(100):
            registry.record_occurrence(pattern_id, f"Occurrence {i}", severity=5)

        entry = registry.get(pattern_id)
        assert entry.data["occurrence_count"] == 100

    def test_delete_pattern(self, pattern_registry_factory):
        """Test deleting a pattern."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            PatternType.OBSESSION_LOOP,
            description="To delete"
        )

        result = registry.delete(pattern_id)
        assert result is True
        assert registry.get(pattern_id) is None

    def test_clear_registry(self, pattern_registry_factory):
        """Test clearing all patterns."""
        registry = pattern_registry_factory()
        registry.add_pattern(PatternType.OBSESSION_LOOP, "Pattern 1")
        registry.add_pattern(PatternType.AVOIDANCE, "Pattern 2")

        registry.clear()
        assert len(registry.list_all()) == 0

    def test_persistence_after_add(self, pattern_registry_factory, temp_memory_path):
        """Test that patterns persist to storage."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            PatternType.THRESHOLD_APPROACH,
            description="Persistent pattern"
        )

        # Create new registry with same path
        storage_path = temp_memory_path / "pattern_test.json"
        from memory.pattern_registry import PatternRegistry
        new_registry = PatternRegistry(storage_path=storage_path)

        entry = new_registry.get(pattern_id)
        assert entry is not None
        assert entry.data["description"] == "Persistent pattern"


class TestMostEffectiveIntervention:
    """Tests for finding most effective intervention."""

    def test_find_most_effective(self, pattern_registry_factory):
        """Test finding most effective intervention."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            PatternType.OBSESSION_LOOP,
            description="Test"
        )

        registry.record_intervention(pattern_id, "int_001", "Less effective", 4)
        registry.record_intervention(pattern_id, "int_002", "Most effective", 9)
        registry.record_intervention(pattern_id, "int_003", "Medium effective", 6)

        summary = registry.get_pattern_summary(pattern_id)
        best = summary["most_effective_intervention"]

        assert best["id"] == "int_002"
        assert best["effectiveness"] == 9

    def test_no_interventions(self, pattern_registry_factory):
        """Test summary when no interventions recorded."""
        registry = pattern_registry_factory()
        pattern_id = registry.add_pattern(
            PatternType.AVOIDANCE,
            description="No interventions"
        )

        summary = registry.get_pattern_summary(pattern_id)
        assert summary["most_effective_intervention"] is None
        assert summary["intervention_count"] == 0
