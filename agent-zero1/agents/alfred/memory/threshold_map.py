# Threshold Map Memory System
# Tracks approaching critical thresholds and boundaries

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

from . import BaseMemorySystem, MemoryType, MemoryEntry


class ThresholdType(Enum):
    """Types of thresholds tracked."""
    HEALTH = "health"               # Physical/mental health boundaries
    FINANCIAL = "financial"         # Financial boundaries
    RELATIONSHIP = "relationship"   # Relationship capacity/quality
    PROFESSIONAL = "professional"   # Career/work boundaries
    ETHICAL = "ethical"             # Ethical/moral boundaries
    EMOTIONAL = "emotional"         # Emotional capacity
    TIME = "time"                   # Time/commitment capacity
    ENERGY = "energy"               # Energy/bandwidth
    LEGAL = "legal"                 # Legal boundaries
    IDENTITY = "identity"           # Personal identity boundaries


class TrendDirection(Enum):
    """Direction of threshold proximity trend."""
    APPROACHING = "approaching"     # Getting closer to threshold
    STABLE = "stable"               # Maintaining distance
    RECEDING = "receding"           # Moving away from threshold
    CRITICAL = "critical"           # At or past threshold


class AlertLevel(Enum):
    """Alert levels for thresholds."""
    SAFE = "safe"                   # Proximity 0-30
    MONITORING = "monitoring"       # Proximity 31-60
    WARNING = "warning"             # Proximity 61-80
    DANGER = "danger"               # Proximity 81-95
    CRITICAL = "critical"           # Proximity 96-100


class ThresholdMap(BaseMemorySystem):
    """
    Tracks the user's proximity to critical thresholds.

    People often don't notice they're approaching dangerous boundaries
    until they've crossed them. This system monitors approach velocity
    and provides early warning.

    Schema per threshold:
    - threshold_id: Unique identifier
    - threshold_name: Name of the threshold
    - threshold_type: Type category
    - description: What crossing this threshold means
    - current_proximity: 0-100 scale (100 = at threshold)
    - trend: Current trend direction
    - trend_velocity: How fast approaching/receding
    - warnings: List of warnings issued
    - history: Proximity history over time
    - interventions: Interventions attempted
    """

    def __init__(self, storage_path=None):
        super().__init__(MemoryType.THRESHOLD, storage_path)

    def register_threshold(
        self,
        threshold_name: str,
        threshold_type: ThresholdType,
        description: str,
        initial_proximity: int = 0,
        crossing_consequences: Optional[str] = None
    ) -> str:
        """
        Register a new threshold to track.

        Args:
            threshold_name: Name of the threshold
            threshold_type: Type category
            description: What this threshold represents
            initial_proximity: Starting proximity (0-100)
            crossing_consequences: What happens if threshold is crossed

        Returns:
            threshold_id: Unique identifier
        """
        threshold_id = f"threshold_{uuid.uuid4().hex[:12]}"

        proximity = max(0, min(100, initial_proximity))

        threshold_data = {
            "threshold_name": threshold_name,
            "threshold_type": threshold_type.value,
            "description": description,
            "crossing_consequences": crossing_consequences,
            "current_proximity": proximity,
            "peak_proximity": proximity,
            "trend": TrendDirection.STABLE.value,
            "trend_velocity": 0,  # Rate of change per update
            "alert_level": self._calculate_alert_level(proximity).value,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "warnings": [],
            "history": [{
                "timestamp": datetime.now().isoformat(),
                "proximity": proximity,
                "notes": "Initial registration"
            }],
            "interventions": [],
            "crossed": False,
            "crossed_at": None,
            "recovery_attempts": 0,
            "active": True
        }

        self.add(threshold_id, threshold_data)
        return threshold_id

    def _calculate_alert_level(self, proximity: int) -> AlertLevel:
        """Calculate alert level based on proximity."""
        if proximity <= 30:
            return AlertLevel.SAFE
        elif proximity <= 60:
            return AlertLevel.MONITORING
        elif proximity <= 80:
            return AlertLevel.WARNING
        elif proximity <= 95:
            return AlertLevel.DANGER
        else:
            return AlertLevel.CRITICAL

    def update_proximity(
        self,
        threshold_id: str,
        new_proximity: int,
        notes: Optional[str] = None,
        evidence: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the proximity to a threshold.

        Args:
            threshold_id: ID of the threshold
            new_proximity: New proximity value (0-100)
            notes: Notes about this update
            evidence: Evidence for the change

        Returns:
            Update result with alert info
        """
        entry = self.get(threshold_id)
        if not entry:
            return {"success": False, "error": "Threshold not found"}

        old_proximity = entry.data["current_proximity"]
        new_proximity = max(0, min(100, new_proximity))

        # Calculate trend
        delta = new_proximity - old_proximity
        if len(entry.data["history"]) >= 2:
            # Calculate velocity over last few updates
            recent = entry.data["history"][-3:]
            avg_delta = sum(
                entry.data["history"][i+1]["proximity"] - entry.data["history"][i]["proximity"]
                for i in range(len(recent)-1)
            ) / max(1, len(recent)-1)
            entry.data["trend_velocity"] = round(avg_delta, 2)
        else:
            entry.data["trend_velocity"] = delta

        # Determine trend direction
        if new_proximity >= 96:
            trend = TrendDirection.CRITICAL
        elif delta > 2:
            trend = TrendDirection.APPROACHING
        elif delta < -2:
            trend = TrendDirection.RECEDING
        else:
            trend = TrendDirection.STABLE

        # Update data
        entry.data["current_proximity"] = new_proximity
        entry.data["trend"] = trend.value
        entry.data["peak_proximity"] = max(entry.data["peak_proximity"], new_proximity)
        entry.data["alert_level"] = self._calculate_alert_level(new_proximity).value
        entry.data["last_updated"] = datetime.now().isoformat()

        # Add to history
        entry.data["history"].append({
            "timestamp": datetime.now().isoformat(),
            "proximity": new_proximity,
            "delta": delta,
            "notes": notes,
            "evidence": evidence
        })

        # Check if threshold was crossed
        if new_proximity >= 100 and not entry.data["crossed"]:
            entry.data["crossed"] = True
            entry.data["crossed_at"] = datetime.now().isoformat()
        elif new_proximity < 100 and entry.data["crossed"]:
            entry.data["recovery_attempts"] += 1

        # Generate warning if needed
        warning_generated = None
        old_alert = self._calculate_alert_level(old_proximity)
        new_alert = self._calculate_alert_level(new_proximity)

        if new_alert.value != old_alert.value:
            alert_order = [AlertLevel.SAFE, AlertLevel.MONITORING, AlertLevel.WARNING, AlertLevel.DANGER, AlertLevel.CRITICAL]
            if alert_order.index(new_alert) > alert_order.index(old_alert):
                warning_generated = self._generate_warning(entry, old_alert, new_alert, evidence)

        entry.updated_at = datetime.now().isoformat()
        self._save()

        return {
            "success": True,
            "threshold_name": entry.data["threshold_name"],
            "old_proximity": old_proximity,
            "new_proximity": new_proximity,
            "delta": delta,
            "trend": trend.value,
            "alert_level": new_alert.value,
            "warning_generated": warning_generated,
            "crossed": entry.data["crossed"]
        }

    def _generate_warning(
        self,
        entry: MemoryEntry,
        old_alert: AlertLevel,
        new_alert: AlertLevel,
        evidence: Optional[str]
    ) -> Dict[str, Any]:
        """Generate and store a warning."""
        warning = {
            "warning_id": f"warn_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "previous_level": old_alert.value,
            "new_level": new_alert.value,
            "proximity": entry.data["current_proximity"],
            "evidence": evidence,
            "acknowledged": False,
            "message": self._get_warning_message(entry.data["threshold_name"], new_alert)
        }

        entry.data["warnings"].append(warning)
        return warning

    def _get_warning_message(self, threshold_name: str, alert_level: AlertLevel) -> str:
        """Generate appropriate warning message."""
        messages = {
            AlertLevel.MONITORING: f"'{threshold_name}' threshold entering monitoring zone. Early attention recommended.",
            AlertLevel.WARNING: f"Warning: '{threshold_name}' threshold approaching. Take preventive action.",
            AlertLevel.DANGER: f"DANGER: '{threshold_name}' threshold critically close. Immediate attention required.",
            AlertLevel.CRITICAL: f"CRITICAL: '{threshold_name}' threshold reached or exceeded. Emergency intervention needed."
        }
        return messages.get(alert_level, f"'{threshold_name}' threshold status changed.")

    def record_warning(
        self,
        threshold_id: str,
        warning_message: str,
        evidence: Optional[str] = None
    ) -> str:
        """
        Manually record a warning for a threshold.

        Args:
            threshold_id: ID of the threshold
            warning_message: The warning message
            evidence: Supporting evidence

        Returns:
            warning_id: ID of the warning
        """
        entry = self.get(threshold_id)
        if not entry:
            return None

        warning_id = f"warn_{uuid.uuid4().hex[:8]}"

        warning = {
            "warning_id": warning_id,
            "timestamp": datetime.now().isoformat(),
            "previous_level": entry.data.get("alert_level"),
            "new_level": entry.data.get("alert_level"),
            "proximity": entry.data["current_proximity"],
            "evidence": evidence,
            "message": warning_message,
            "acknowledged": False,
            "manual": True
        }

        entry.data["warnings"].append(warning)
        entry.updated_at = datetime.now().isoformat()
        self._save()

        return warning_id

    def acknowledge_warning(
        self,
        threshold_id: str,
        warning_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Acknowledge a warning.

        Args:
            threshold_id: ID of the threshold
            warning_id: ID of the warning
            notes: Acknowledgment notes

        Returns:
            Success boolean
        """
        entry = self.get(threshold_id)
        if not entry:
            return False

        for warning in entry.data["warnings"]:
            if warning["warning_id"] == warning_id:
                warning["acknowledged"] = True
                warning["acknowledged_at"] = datetime.now().isoformat()
                warning["acknowledgment_notes"] = notes
                self._save()
                return True

        return False

    def record_intervention(
        self,
        threshold_id: str,
        intervention_description: str,
        expected_impact: int = 0
    ) -> str:
        """
        Record an intervention to address a threshold.

        Args:
            threshold_id: ID of the threshold
            intervention_description: What was done
            expected_impact: Expected proximity reduction

        Returns:
            intervention_id: ID of the intervention
        """
        entry = self.get(threshold_id)
        if not entry:
            return None

        intervention_id = f"int_{uuid.uuid4().hex[:8]}"

        intervention = {
            "intervention_id": intervention_id,
            "description": intervention_description,
            "timestamp": datetime.now().isoformat(),
            "proximity_at_intervention": entry.data["current_proximity"],
            "expected_impact": expected_impact,
            "actual_impact": None,
            "effectiveness": None
        }

        entry.data["interventions"].append(intervention)
        entry.updated_at = datetime.now().isoformat()
        self._save()

        return intervention_id

    def record_intervention_outcome(
        self,
        threshold_id: str,
        intervention_id: str,
        actual_impact: int,
        notes: Optional[str] = None
    ) -> bool:
        """
        Record the outcome of an intervention.

        Args:
            threshold_id: ID of the threshold
            intervention_id: ID of the intervention
            actual_impact: Actual proximity change
            notes: Notes about the outcome

        Returns:
            Success boolean
        """
        entry = self.get(threshold_id)
        if not entry:
            return False

        for intervention in entry.data["interventions"]:
            if intervention["intervention_id"] == intervention_id:
                intervention["actual_impact"] = actual_impact
                intervention["outcome_notes"] = notes
                intervention["outcome_recorded_at"] = datetime.now().isoformat()

                # Calculate effectiveness
                expected = intervention.get("expected_impact", 0)
                if expected != 0:
                    intervention["effectiveness"] = min(100, max(0, (actual_impact / expected) * 100))
                else:
                    intervention["effectiveness"] = 100 if actual_impact < 0 else 0

                self._save()
                return True

        return False

    def get_approaching_thresholds(
        self,
        min_proximity: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get thresholds that are approaching critical levels.

        Args:
            min_proximity: Minimum proximity to include

        Returns:
            List of approaching thresholds
        """
        results = []
        for entry in self._entries.values():
            proximity = entry.data.get("current_proximity", 0)
            trend = entry.data.get("trend")

            if proximity >= min_proximity:
                results.append({
                    "threshold_id": entry.id,
                    **entry.data
                })
            elif trend == TrendDirection.APPROACHING.value and proximity >= min_proximity - 20:
                # Include if approaching even if not yet at threshold
                results.append({
                    "threshold_id": entry.id,
                    **entry.data
                })

        # Sort by proximity descending
        results.sort(key=lambda x: x.get("current_proximity", 0), reverse=True)
        return results

    def get_thresholds_by_type(
        self,
        threshold_type: ThresholdType
    ) -> List[Dict[str, Any]]:
        """
        Get all thresholds of a specific type.

        Args:
            threshold_type: Type to filter by

        Returns:
            List of thresholds
        """
        results = []
        for entry in self._entries.values():
            if entry.data.get("threshold_type") == threshold_type.value:
                results.append({
                    "threshold_id": entry.id,
                    **entry.data
                })

        results.sort(key=lambda x: x.get("current_proximity", 0), reverse=True)
        return results

    def get_thresholds_by_alert_level(
        self,
        alert_level: AlertLevel
    ) -> List[Dict[str, Any]]:
        """
        Get all thresholds at a specific alert level.

        Args:
            alert_level: Alert level to filter by

        Returns:
            List of thresholds
        """
        results = []
        for entry in self._entries.values():
            if entry.data.get("alert_level") == alert_level.value:
                results.append({
                    "threshold_id": entry.id,
                    **entry.data
                })

        return results

    def get_critical_thresholds(self) -> List[Dict[str, Any]]:
        """Get all thresholds at danger or critical level."""
        results = []
        critical_levels = [AlertLevel.DANGER.value, AlertLevel.CRITICAL.value]

        for entry in self._entries.values():
            if entry.data.get("alert_level") in critical_levels:
                results.append({
                    "threshold_id": entry.id,
                    **entry.data
                })

        results.sort(key=lambda x: x.get("current_proximity", 0), reverse=True)
        return results

    def get_crossed_thresholds(self) -> List[Dict[str, Any]]:
        """Get all thresholds that have been crossed."""
        results = []
        for entry in self._entries.values():
            if entry.data.get("crossed"):
                results.append({
                    "threshold_id": entry.id,
                    **entry.data
                })

        return results

    def get_unacknowledged_warnings(self) -> List[Dict[str, Any]]:
        """Get all unacknowledged warnings across all thresholds."""
        results = []
        for entry in self._entries.values():
            for warning in entry.data.get("warnings", []):
                if not warning.get("acknowledged"):
                    results.append({
                        "threshold_id": entry.id,
                        "threshold_name": entry.data.get("threshold_name"),
                        **warning
                    })

        # Sort by timestamp descending
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    def get_threshold_trend(
        self,
        threshold_id: str,
        periods: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Get trend analysis for a threshold.

        Args:
            threshold_id: ID of the threshold
            periods: Number of historical periods to analyze

        Returns:
            Trend analysis dictionary
        """
        entry = self.get(threshold_id)
        if not entry:
            return None

        history = entry.data.get("history", [])[-periods:]

        if len(history) < 2:
            return {
                "threshold_id": threshold_id,
                "threshold_name": entry.data.get("threshold_name"),
                "insufficient_data": True
            }

        proximities = [h["proximity"] for h in history]
        deltas = [h.get("delta", 0) for h in history[1:]]

        # Calculate statistics
        avg_proximity = sum(proximities) / len(proximities)
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        max_proximity = max(proximities)
        min_proximity = min(proximities)

        # Predict next proximity (simple linear projection)
        projected_next = proximities[-1] + avg_delta

        # Time to threshold crossing estimate
        if avg_delta > 0:
            remaining = 100 - proximities[-1]
            updates_to_crossing = remaining / avg_delta if avg_delta > 0 else float('inf')
        else:
            updates_to_crossing = float('inf')

        return {
            "threshold_id": threshold_id,
            "threshold_name": entry.data.get("threshold_name"),
            "current_proximity": entry.data.get("current_proximity"),
            "history_periods": len(history),
            "average_proximity": round(avg_proximity, 2),
            "average_change_rate": round(avg_delta, 2),
            "max_proximity": max_proximity,
            "min_proximity": min_proximity,
            "projected_next": max(0, min(100, round(projected_next, 2))),
            "estimated_updates_to_crossing": (
                round(updates_to_crossing, 1) if updates_to_crossing != float('inf') else None
            ),
            "trend_direction": entry.data.get("trend"),
            "volatility": round(max_proximity - min_proximity, 2)
        }

    def get_threshold_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all thresholds.

        Returns:
            Summary dictionary
        """
        all_thresholds = list(self._entries.values())

        # Count by alert level
        by_alert = {}
        for entry in all_thresholds:
            level = entry.data.get("alert_level", "unknown")
            by_alert[level] = by_alert.get(level, 0) + 1

        # Count by type
        by_type = {}
        for entry in all_thresholds:
            t = entry.data.get("threshold_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        # Find most critical
        critical = sorted(
            all_thresholds,
            key=lambda x: x.data.get("current_proximity", 0),
            reverse=True
        )[:5]

        # Count warnings
        total_warnings = sum(
            len(entry.data.get("warnings", []))
            for entry in all_thresholds
        )
        unacked = len(self.get_unacknowledged_warnings())

        return {
            "total_thresholds": len(all_thresholds),
            "by_alert_level": by_alert,
            "by_type": by_type,
            "crossed_count": len([e for e in all_thresholds if e.data.get("crossed")]),
            "approaching_count": len([e for e in all_thresholds if e.data.get("trend") == TrendDirection.APPROACHING.value]),
            "most_critical": [
                {
                    "name": e.data.get("threshold_name"),
                    "proximity": e.data.get("current_proximity"),
                    "type": e.data.get("threshold_type"),
                    "trend": e.data.get("trend")
                }
                for e in critical
            ],
            "total_warnings_issued": total_warnings,
            "unacknowledged_warnings": unacked,
            "total_interventions": sum(
                len(e.data.get("interventions", []))
                for e in all_thresholds
            )
        }

    def get_all_thresholds(self) -> List[Dict[str, Any]]:
        """Get all registered thresholds."""
        results = []
        for entry in self._entries.values():
            results.append({
                "threshold_id": entry.id,
                **entry.data
            })

        results.sort(key=lambda x: x.get("current_proximity", 0), reverse=True)
        return results
