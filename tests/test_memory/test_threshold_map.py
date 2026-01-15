"""
Tests for ThresholdMap - Critical Threshold Tracking System

Tests cover:
- Main class functionality
- Threshold lifecycle (register, update, track)
- Proximity and trend tracking
- Alert level transitions
- Warning generation
- Intervention tracking
- Edge cases
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent-zero1" / "agents" / "alfred"))

from memory.threshold_map import (
    ThresholdMap,
    ThresholdType,
    TrendDirection,
    AlertLevel,
)
from memory import MemoryType, MemoryEntry, BaseMemorySystem


class TestThresholdMapInitialization:
    """Tests for ThresholdMap initialization."""

    def test_create_threshold_map(self, threshold_map_factory):
        """Test that ThresholdMap can be created."""
        threshold_map = threshold_map_factory()
        assert threshold_map is not None
        assert threshold_map.memory_type == MemoryType.THRESHOLD

    def test_initial_empty_map(self, threshold_map_factory):
        """Test that map starts empty."""
        threshold_map = threshold_map_factory()
        assert len(threshold_map.list_all()) == 0

    def test_storage_path_set(self, threshold_map_factory, temp_memory_path):
        """Test that storage path is set correctly."""
        threshold_map = threshold_map_factory()
        assert threshold_map.storage_path is not None


class TestThresholdType:
    """Tests for ThresholdType enum."""

    def test_threshold_types_defined(self):
        """Test that all threshold types are defined."""
        assert ThresholdType.HEALTH.value == "health"
        assert ThresholdType.FINANCIAL.value == "financial"
        assert ThresholdType.RELATIONSHIP.value == "relationship"
        assert ThresholdType.PROFESSIONAL.value == "professional"
        assert ThresholdType.ETHICAL.value == "ethical"
        assert ThresholdType.EMOTIONAL.value == "emotional"
        assert ThresholdType.TIME.value == "time"
        assert ThresholdType.ENERGY.value == "energy"
        assert ThresholdType.LEGAL.value == "legal"
        assert ThresholdType.IDENTITY.value == "identity"


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_trend_values(self):
        """Test trend direction values."""
        assert TrendDirection.APPROACHING.value == "approaching"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.RECEDING.value == "receding"
        assert TrendDirection.CRITICAL.value == "critical"


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_level_values(self):
        """Test alert level values."""
        assert AlertLevel.SAFE.value == "safe"
        assert AlertLevel.MONITORING.value == "monitoring"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.DANGER.value == "danger"
        assert AlertLevel.CRITICAL.value == "critical"


class TestRegisterThreshold:
    """Tests for threshold registration."""

    def test_register_threshold_basic(self, threshold_map_factory, sample_threshold_data):
        """Test basic threshold registration."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name=sample_threshold_data["threshold_name"],
            threshold_type=ThresholdType.HEALTH,
            description=sample_threshold_data["description"],
            initial_proximity=sample_threshold_data["initial_proximity"]
        )

        assert threshold_id is not None
        assert threshold_id.startswith("threshold_")

    def test_register_threshold_stores_data(self, threshold_map_factory, sample_threshold_data):
        """Test that threshold data is stored correctly."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name=sample_threshold_data["threshold_name"],
            threshold_type=ThresholdType.HEALTH,
            description=sample_threshold_data["description"],
            initial_proximity=sample_threshold_data["initial_proximity"],
            crossing_consequences=sample_threshold_data["crossing_consequences"]
        )

        entry = threshold_map.get(threshold_id)
        assert entry.data["threshold_name"] == sample_threshold_data["threshold_name"]
        assert entry.data["threshold_type"] == ThresholdType.HEALTH.value
        assert entry.data["current_proximity"] == sample_threshold_data["initial_proximity"]

    def test_register_threshold_initial_history(self, threshold_map_factory):
        """Test that initial registration creates history entry."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test Threshold",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=25
        )

        entry = threshold_map.get(threshold_id)
        assert len(entry.data["history"]) == 1
        assert entry.data["history"][0]["notes"] == "Initial registration"

    def test_proximity_bounds(self, threshold_map_factory):
        """Test that proximity is bounded 0-100."""
        threshold_map = threshold_map_factory()

        # Test upper bound
        threshold_id = threshold_map.register_threshold(
            threshold_name="Over 100",
            threshold_type=ThresholdType.ENERGY,
            description="Test",
            initial_proximity=150
        )
        entry = threshold_map.get(threshold_id)
        assert entry.data["current_proximity"] == 100

        # Test lower bound
        threshold_id_2 = threshold_map.register_threshold(
            threshold_name="Under 0",
            threshold_type=ThresholdType.TIME,
            description="Test",
            initial_proximity=-20
        )
        entry_2 = threshold_map.get(threshold_id_2)
        assert entry_2.data["current_proximity"] == 0


class TestAlertLevelCalculation:
    """Tests for alert level calculation."""

    def test_safe_level(self, threshold_map_factory):
        """Test SAFE alert level (0-30)."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Safe",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=20
        )
        entry = threshold_map.get(threshold_id)
        assert entry.data["alert_level"] == AlertLevel.SAFE.value

    def test_monitoring_level(self, threshold_map_factory):
        """Test MONITORING alert level (31-60)."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Monitoring",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=45
        )
        entry = threshold_map.get(threshold_id)
        assert entry.data["alert_level"] == AlertLevel.MONITORING.value

    def test_warning_level(self, threshold_map_factory):
        """Test WARNING alert level (61-80)."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Warning",
            threshold_type=ThresholdType.RELATIONSHIP,
            description="Test",
            initial_proximity=70
        )
        entry = threshold_map.get(threshold_id)
        assert entry.data["alert_level"] == AlertLevel.WARNING.value

    def test_danger_level(self, threshold_map_factory):
        """Test DANGER alert level (81-95)."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Danger",
            threshold_type=ThresholdType.ETHICAL,
            description="Test",
            initial_proximity=90
        )
        entry = threshold_map.get(threshold_id)
        assert entry.data["alert_level"] == AlertLevel.DANGER.value

    def test_critical_level(self, threshold_map_factory):
        """Test CRITICAL alert level (96-100)."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Critical",
            threshold_type=ThresholdType.LEGAL,
            description="Test",
            initial_proximity=98
        )
        entry = threshold_map.get(threshold_id)
        assert entry.data["alert_level"] == AlertLevel.CRITICAL.value


class TestUpdateProximity:
    """Tests for proximity updates."""

    def test_update_proximity(self, threshold_map_factory, sample_proximity_update):
        """Test basic proximity update."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=50
        )

        result = threshold_map.update_proximity(
            threshold_id=threshold_id,
            new_proximity=sample_proximity_update["new_proximity"],
            notes=sample_proximity_update["notes"],
            evidence=sample_proximity_update["evidence"]
        )

        assert result["success"] is True
        assert result["new_proximity"] == 65
        assert result["delta"] == 15

    def test_update_adds_history(self, threshold_map_factory):
        """Test that updates add to history."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=30
        )

        threshold_map.update_proximity(threshold_id, 40, "Update 1")
        threshold_map.update_proximity(threshold_id, 50, "Update 2")

        entry = threshold_map.get(threshold_id)
        assert len(entry.data["history"]) == 3  # Initial + 2 updates

    def test_update_tracks_delta(self, threshold_map_factory):
        """Test that delta is tracked in history."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.ENERGY,
            description="Test",
            initial_proximity=40
        )

        threshold_map.update_proximity(threshold_id, 55)

        entry = threshold_map.get(threshold_id)
        latest = entry.data["history"][-1]
        assert latest["delta"] == 15

    def test_update_tracks_peak(self, threshold_map_factory):
        """Test that peak proximity is tracked."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.EMOTIONAL,
            description="Test",
            initial_proximity=30
        )

        threshold_map.update_proximity(threshold_id, 70)
        threshold_map.update_proximity(threshold_id, 50)

        entry = threshold_map.get(threshold_id)
        assert entry.data["peak_proximity"] == 70

    def test_update_nonexistent_threshold(self, threshold_map_factory):
        """Test updating non-existent threshold."""
        threshold_map = threshold_map_factory()
        result = threshold_map.update_proximity("nonexistent", 50)
        assert result["success"] is False


class TestTrendDetection:
    """Tests for trend detection."""

    def test_approaching_trend(self, threshold_map_factory):
        """Test detection of approaching trend."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=30
        )

        result = threshold_map.update_proximity(threshold_id, 45)  # +15
        assert result["trend"] == TrendDirection.APPROACHING.value

    def test_receding_trend(self, threshold_map_factory):
        """Test detection of receding trend."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=70
        )

        result = threshold_map.update_proximity(threshold_id, 55)  # -15
        assert result["trend"] == TrendDirection.RECEDING.value

    def test_stable_trend(self, threshold_map_factory):
        """Test detection of stable trend."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.RELATIONSHIP,
            description="Test",
            initial_proximity=50
        )

        result = threshold_map.update_proximity(threshold_id, 51)  # +1
        assert result["trend"] == TrendDirection.STABLE.value

    def test_critical_trend(self, threshold_map_factory):
        """Test detection of critical trend."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.LEGAL,
            description="Test",
            initial_proximity=90
        )

        result = threshold_map.update_proximity(threshold_id, 98)
        assert result["trend"] == TrendDirection.CRITICAL.value

    def test_velocity_calculation(self, threshold_map_factory):
        """Test trend velocity calculation over multiple updates."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.ENERGY,
            description="Test",
            initial_proximity=20
        )

        # Multiple updates with consistent increase
        threshold_map.update_proximity(threshold_id, 30)
        threshold_map.update_proximity(threshold_id, 40)
        threshold_map.update_proximity(threshold_id, 50)

        entry = threshold_map.get(threshold_id)
        assert entry.data["trend_velocity"] > 0


class TestThresholdCrossing:
    """Tests for threshold crossing detection."""

    def test_crossing_detected(self, threshold_map_factory):
        """Test that crossing at 100 is detected."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.ETHICAL,
            description="Test",
            initial_proximity=95
        )

        result = threshold_map.update_proximity(threshold_id, 100)

        assert result["crossed"] is True
        entry = threshold_map.get(threshold_id)
        assert entry.data["crossed"] is True
        assert entry.data["crossed_at"] is not None

    def test_recovery_tracked(self, threshold_map_factory):
        """Test that recovery from crossing is tracked."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=95
        )

        # Cross threshold
        threshold_map.update_proximity(threshold_id, 100)
        # Recover
        threshold_map.update_proximity(threshold_id, 85)

        entry = threshold_map.get(threshold_id)
        assert entry.data["recovery_attempts"] == 1

    def test_get_crossed_thresholds(self, threshold_map_factory):
        """Test getting all crossed thresholds."""
        threshold_map = threshold_map_factory()

        # Crossed threshold
        threshold_1 = threshold_map.register_threshold(
            threshold_name="Crossed",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=100
        )

        # Not crossed
        threshold_2 = threshold_map.register_threshold(
            threshold_name="Not Crossed",
            threshold_type=ThresholdType.TIME,
            description="Test",
            initial_proximity=50
        )

        crossed = threshold_map.get_crossed_thresholds()
        assert len(crossed) == 1
        assert crossed[0]["threshold_id"] == threshold_1


class TestWarningGeneration:
    """Tests for warning generation."""

    def test_warning_on_level_escalation(self, threshold_map_factory):
        """Test warning generation on level escalation."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=25  # SAFE
        )

        # Escalate to WARNING
        result = threshold_map.update_proximity(threshold_id, 65)

        assert result["warning_generated"] is not None
        assert "warning" in result["alert_level"].lower() or \
               "monitoring" in result["alert_level"].lower()

    def test_no_warning_on_decrease(self, threshold_map_factory):
        """Test no warning when level decreases."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=75  # WARNING
        )

        # Decrease to SAFE
        result = threshold_map.update_proximity(threshold_id, 20)

        assert result["warning_generated"] is None

    def test_manual_warning_recording(self, threshold_map_factory):
        """Test manual warning recording."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.RELATIONSHIP,
            description="Test",
            initial_proximity=50
        )

        warning_id = threshold_map.record_warning(
            threshold_id=threshold_id,
            warning_message="Manual warning for testing",
            evidence="Observed behavior change"
        )

        assert warning_id is not None
        entry = threshold_map.get(threshold_id)
        assert len(entry.data["warnings"]) == 1

    def test_acknowledge_warning(self, threshold_map_factory):
        """Test acknowledging a warning."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.EMOTIONAL,
            description="Test",
            initial_proximity=50
        )

        warning_id = threshold_map.record_warning(threshold_id, "Test warning")
        result = threshold_map.acknowledge_warning(
            threshold_id=threshold_id,
            warning_id=warning_id,
            notes="Acknowledged and addressed"
        )

        assert result is True
        entry = threshold_map.get(threshold_id)
        assert entry.data["warnings"][0]["acknowledged"] is True

    def test_get_unacknowledged_warnings(self, threshold_map_factory):
        """Test getting unacknowledged warnings."""
        threshold_map = threshold_map_factory()

        threshold_1 = threshold_map.register_threshold(
            threshold_name="Test 1",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=50
        )
        threshold_map.record_warning(threshold_1, "Warning 1")

        threshold_2 = threshold_map.register_threshold(
            threshold_name="Test 2",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=50
        )
        warning_id = threshold_map.record_warning(threshold_2, "Warning 2")
        threshold_map.acknowledge_warning(threshold_2, warning_id)

        unacked = threshold_map.get_unacknowledged_warnings()
        assert len(unacked) == 1
        assert unacked[0]["threshold_name"] == "Test 1"


class TestInterventionTracking:
    """Tests for intervention tracking."""

    def test_record_intervention(self, threshold_map_factory):
        """Test recording an intervention."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=70
        )

        intervention_id = threshold_map.record_intervention(
            threshold_id=threshold_id,
            intervention_description="Implemented sleep schedule",
            expected_impact=-20
        )

        assert intervention_id is not None
        entry = threshold_map.get(threshold_id)
        assert len(entry.data["interventions"]) == 1

    def test_record_intervention_outcome(self, threshold_map_factory):
        """Test recording intervention outcome."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.ENERGY,
            description="Test",
            initial_proximity=70
        )

        intervention_id = threshold_map.record_intervention(
            threshold_id=threshold_id,
            intervention_description="Rest day",
            expected_impact=-15
        )

        result = threshold_map.record_intervention_outcome(
            threshold_id=threshold_id,
            intervention_id=intervention_id,
            actual_impact=-20,
            notes="Better than expected"
        )

        assert result is True
        entry = threshold_map.get(threshold_id)
        intervention = entry.data["interventions"][0]
        assert intervention["actual_impact"] == -20
        assert intervention["effectiveness"] is not None


class TestQueryThresholds:
    """Tests for threshold query functionality."""

    def test_get_approaching_thresholds(self, threshold_map_factory):
        """Test getting thresholds approaching limits."""
        threshold_map = threshold_map_factory()

        # High proximity
        threshold_map.register_threshold(
            threshold_name="High",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=75
        )

        # Low proximity
        threshold_map.register_threshold(
            threshold_name="Low",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=20
        )

        approaching = threshold_map.get_approaching_thresholds(min_proximity=50)
        assert len(approaching) == 1
        assert approaching[0]["threshold_name"] == "High"

    def test_get_thresholds_by_type(self, threshold_map_factory):
        """Test getting thresholds by type."""
        threshold_map = threshold_map_factory()

        threshold_map.register_threshold(
            threshold_name="Health 1",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=30
        )
        threshold_map.register_threshold(
            threshold_name="Health 2",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=50
        )
        threshold_map.register_threshold(
            threshold_name="Financial",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=40
        )

        health = threshold_map.get_thresholds_by_type(ThresholdType.HEALTH)
        assert len(health) == 2

    def test_get_thresholds_by_alert_level(self, threshold_map_factory):
        """Test getting thresholds by alert level."""
        threshold_map = threshold_map_factory()

        # WARNING level
        threshold_map.register_threshold(
            threshold_name="Warning",
            threshold_type=ThresholdType.ENERGY,
            description="Test",
            initial_proximity=70
        )

        # SAFE level
        threshold_map.register_threshold(
            threshold_name="Safe",
            threshold_type=ThresholdType.TIME,
            description="Test",
            initial_proximity=20
        )

        warning = threshold_map.get_thresholds_by_alert_level(AlertLevel.WARNING)
        assert len(warning) == 1
        assert warning[0]["threshold_name"] == "Warning"

    def test_get_critical_thresholds(self, threshold_map_factory):
        """Test getting critical thresholds (danger + critical)."""
        threshold_map = threshold_map_factory()

        threshold_map.register_threshold(
            threshold_name="Danger",
            threshold_type=ThresholdType.LEGAL,
            description="Test",
            initial_proximity=85
        )
        threshold_map.register_threshold(
            threshold_name="Critical",
            threshold_type=ThresholdType.ETHICAL,
            description="Test",
            initial_proximity=98
        )
        threshold_map.register_threshold(
            threshold_name="Safe",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=20
        )

        critical = threshold_map.get_critical_thresholds()
        assert len(critical) == 2


class TestTrendAnalysis:
    """Tests for trend analysis functionality."""

    def test_get_threshold_trend(self, threshold_map_factory):
        """Test getting threshold trend analysis."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=30
        )

        # Add some history
        threshold_map.update_proximity(threshold_id, 35)
        threshold_map.update_proximity(threshold_id, 40)
        threshold_map.update_proximity(threshold_id, 45)

        trend = threshold_map.get_threshold_trend(threshold_id)

        assert trend is not None
        assert "average_proximity" in trend
        assert "average_change_rate" in trend
        assert trend["average_change_rate"] > 0

    def test_trend_insufficient_data(self, threshold_map_factory):
        """Test trend with insufficient data."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=50
        )

        trend = threshold_map.get_threshold_trend(threshold_id)
        assert trend["insufficient_data"] is True

    def test_trend_projected_next(self, threshold_map_factory):
        """Test projected next proximity."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test",
            threshold_type=ThresholdType.ENERGY,
            description="Test",
            initial_proximity=20
        )

        # Consistent +10 increases
        threshold_map.update_proximity(threshold_id, 30)
        threshold_map.update_proximity(threshold_id, 40)

        trend = threshold_map.get_threshold_trend(threshold_id)
        # Projected should be around 50 (40 + ~10)
        assert trend["projected_next"] >= 40


class TestThresholdSummary:
    """Tests for threshold summary functionality."""

    def test_get_threshold_summary(self, threshold_map_factory):
        """Test getting comprehensive summary."""
        threshold_map = threshold_map_factory()

        # Add various thresholds
        threshold_map.register_threshold(
            threshold_name="Safe",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=20
        )
        threshold_map.register_threshold(
            threshold_name="Danger",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=90
        )

        summary = threshold_map.get_threshold_summary()

        assert summary["total_thresholds"] == 2
        assert "by_alert_level" in summary
        assert "by_type" in summary
        assert "most_critical" in summary

    def test_summary_counts(self, threshold_map_factory):
        """Test summary count accuracy."""
        threshold_map = threshold_map_factory()

        # Add crossed threshold
        crossed_id = threshold_map.register_threshold(
            threshold_name="Crossed",
            threshold_type=ThresholdType.LEGAL,
            description="Test",
            initial_proximity=100
        )

        # Add approaching threshold
        approaching_id = threshold_map.register_threshold(
            threshold_name="Approaching",
            threshold_type=ThresholdType.ETHICAL,
            description="Test",
            initial_proximity=50
        )
        threshold_map.update_proximity(approaching_id, 65)

        summary = threshold_map.get_threshold_summary()
        assert summary["crossed_count"] == 1

    def test_get_all_thresholds(self, threshold_map_factory):
        """Test getting all thresholds sorted by proximity."""
        threshold_map = threshold_map_factory()

        threshold_map.register_threshold(
            threshold_name="Low",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=20
        )
        threshold_map.register_threshold(
            threshold_name="High",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=80
        )
        threshold_map.register_threshold(
            threshold_name="Medium",
            threshold_type=ThresholdType.TIME,
            description="Test",
            initial_proximity=50
        )

        all_thresholds = threshold_map.get_all_thresholds()
        assert len(all_thresholds) == 3
        # Should be sorted by proximity descending
        assert all_thresholds[0]["threshold_name"] == "High"
        assert all_thresholds[2]["threshold_name"] == "Low"


class TestThresholdMapEdgeCases:
    """Edge case tests for ThresholdMap."""

    def test_empty_threshold_name(self, threshold_map_factory):
        """Test threshold with empty name."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="",
            threshold_type=ThresholdType.HEALTH,
            description="Empty name threshold",
            initial_proximity=30
        )
        assert threshold_id is not None

    def test_very_long_description(self, threshold_map_factory):
        """Test threshold with very long description."""
        threshold_map = threshold_map_factory()
        long_desc = "A" * 10000
        threshold_id = threshold_map.register_threshold(
            threshold_name="Long Desc",
            threshold_type=ThresholdType.FINANCIAL,
            description=long_desc,
            initial_proximity=50
        )
        entry = threshold_map.get(threshold_id)
        assert entry.data["description"] == long_desc

    def test_many_updates(self, threshold_map_factory):
        """Test threshold with many proximity updates."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Frequent Updates",
            threshold_type=ThresholdType.ENERGY,
            description="Test",
            initial_proximity=50
        )

        for i in range(100):
            new_prox = 50 + (i % 10)
            threshold_map.update_proximity(threshold_id, new_prox)

        entry = threshold_map.get(threshold_id)
        assert len(entry.data["history"]) == 101  # Initial + 100 updates

    def test_delete_threshold(self, threshold_map_factory):
        """Test deleting a threshold."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="To Delete",
            threshold_type=ThresholdType.TIME,
            description="Test",
            initial_proximity=40
        )

        result = threshold_map.delete(threshold_id)
        assert result is True
        assert threshold_map.get(threshold_id) is None

    def test_clear_map(self, threshold_map_factory):
        """Test clearing all thresholds."""
        threshold_map = threshold_map_factory()
        threshold_map.register_threshold(
            threshold_name="Threshold 1",
            threshold_type=ThresholdType.HEALTH,
            description="Test",
            initial_proximity=30
        )
        threshold_map.register_threshold(
            threshold_name="Threshold 2",
            threshold_type=ThresholdType.FINANCIAL,
            description="Test",
            initial_proximity=50
        )

        threshold_map.clear()
        assert len(threshold_map.list_all()) == 0

    def test_persistence_after_update(self, threshold_map_factory, temp_memory_path):
        """Test that updates persist to storage."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Persistent",
            threshold_type=ThresholdType.RELATIONSHIP,
            description="Test",
            initial_proximity=40
        )
        threshold_map.update_proximity(threshold_id, 60)

        # Create new map with same path
        storage_path = temp_memory_path / "threshold_test.json"
        from memory.threshold_map import ThresholdMap
        new_map = ThresholdMap(storage_path=storage_path)

        entry = new_map.get(threshold_id)
        assert entry is not None
        assert entry.data["current_proximity"] == 60

    def test_warning_message_content(self, threshold_map_factory):
        """Test warning message content by level."""
        threshold_map = threshold_map_factory()
        threshold_id = threshold_map.register_threshold(
            threshold_name="Test Threshold",
            threshold_type=ThresholdType.IDENTITY,
            description="Test",
            initial_proximity=25
        )

        # Escalate through levels
        result = threshold_map.update_proximity(threshold_id, 85)

        if result["warning_generated"]:
            message = result["warning_generated"]["message"]
            assert "Test Threshold" in message
