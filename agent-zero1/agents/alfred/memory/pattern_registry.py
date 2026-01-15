# Pattern Registry Memory System
# Tracks behavioral patterns: obsession loops, avoidance, ego overreach, depletion, threshold approach

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

from . import BaseMemorySystem, MemoryType, MemoryEntry


class PatternType(Enum):
    """Types of patterns Alfred tracks."""
    OBSESSION_LOOP = "obsession_loop"       # Recursive fixation on a topic/goal
    AVOIDANCE = "avoidance"                  # Systematic avoidance of important topics
    EGO_OVERREACH = "ego_overreach"          # Overestimating capabilities/importance
    DEPLETION = "depletion"                  # Resource/energy depletion patterns
    THRESHOLD_APPROACH = "threshold_approach" # Approaching critical boundaries


class Trajectory(Enum):
    """Trajectory of a pattern over time."""
    IMPROVING = "improving"     # Pattern frequency decreasing
    STABLE = "stable"           # Pattern remains consistent
    WORSENING = "worsening"     # Pattern frequency increasing
    RESOLVED = "resolved"       # Pattern no longer active


class PatternRegistry(BaseMemorySystem):
    """
    Tracks behavioral patterns in the user.

    Patterns include:
    - Obsession loops: Recursive fixation that consumes disproportionate attention
    - Avoidance: Systematic avoidance of topics the user needs to address
    - Ego overreach: Overestimating one's abilities or importance
    - Depletion: Running down reserves (energy, money, relationships)
    - Threshold approach: Getting close to boundaries that shouldn't be crossed

    Schema per pattern:
    - pattern_id: Unique identifier
    - pattern_type: Type from PatternType enum
    - description: Human-readable description
    - occurrences: List of {timestamp, context, severity, notes}
    - trajectory: Current trajectory (improving/stable/worsening/resolved)
    - intervention_effectiveness: Dict of {intervention_id: effectiveness_score}
    """

    def __init__(self, storage_path=None):
        super().__init__(MemoryType.PATTERN, storage_path)

    def add_pattern(
        self,
        pattern_type: PatternType,
        description: str,
        initial_context: Optional[str] = None,
        severity: int = 5
    ) -> str:
        """
        Add a new pattern to the registry.

        Args:
            pattern_type: Type of pattern from PatternType enum
            description: Human-readable description of the pattern
            initial_context: Context of first observation
            severity: Initial severity (1-10)

        Returns:
            pattern_id: Unique identifier for the pattern
        """
        pattern_id = f"pattern_{uuid.uuid4().hex[:12]}"

        initial_occurrence = None
        if initial_context:
            initial_occurrence = {
                "timestamp": datetime.now().isoformat(),
                "context": initial_context,
                "severity": max(1, min(10, severity)),
                "notes": "Initial observation"
            }

        data = {
            "pattern_type": pattern_type.value,
            "description": description,
            "occurrences": [initial_occurrence] if initial_occurrence else [],
            "trajectory": Trajectory.STABLE.value,
            "intervention_effectiveness": {},
            "first_observed": datetime.now().isoformat(),
            "last_occurrence": datetime.now().isoformat() if initial_occurrence else None,
            "occurrence_count": 1 if initial_occurrence else 0,
            "average_severity": severity if initial_occurrence else 0,
            "tags": [],
            "related_patterns": [],
            "active": True
        }

        self.add(pattern_id, data)
        return pattern_id

    def record_occurrence(
        self,
        pattern_id: str,
        context: str,
        severity: int = 5,
        notes: Optional[str] = None,
        trigger: Optional[str] = None
    ) -> bool:
        """
        Record a new occurrence of an existing pattern.

        Args:
            pattern_id: ID of the pattern
            context: Context of this occurrence
            severity: Severity of this occurrence (1-10)
            notes: Optional notes about this occurrence
            trigger: What triggered this occurrence

        Returns:
            Success boolean
        """
        entry = self.get(pattern_id)
        if not entry:
            return False

        occurrence = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "severity": max(1, min(10, severity)),
            "notes": notes,
            "trigger": trigger
        }

        entry.data["occurrences"].append(occurrence)
        entry.data["last_occurrence"] = datetime.now().isoformat()
        entry.data["occurrence_count"] = len(entry.data["occurrences"])

        # Recalculate average severity
        severities = [o["severity"] for o in entry.data["occurrences"]]
        entry.data["average_severity"] = sum(severities) / len(severities)

        # Auto-update trajectory based on recent occurrences
        self._auto_update_trajectory(entry)

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def _auto_update_trajectory(self, entry: MemoryEntry):
        """Automatically update trajectory based on occurrence patterns."""
        occurrences = entry.data["occurrences"]
        if len(occurrences) < 3:
            return

        # Look at last 5 occurrences (or all if fewer)
        recent = occurrences[-5:]
        severities = [o["severity"] for o in recent]

        # Calculate trend
        if len(severities) >= 2:
            first_half = severities[:len(severities)//2]
            second_half = severities[len(severities)//2:]

            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            diff = avg_second - avg_first
            if diff > 1:
                entry.data["trajectory"] = Trajectory.WORSENING.value
            elif diff < -1:
                entry.data["trajectory"] = Trajectory.IMPROVING.value
            else:
                entry.data["trajectory"] = Trajectory.STABLE.value

    def update_trajectory(
        self,
        pattern_id: str,
        trajectory: Trajectory,
        reason: Optional[str] = None
    ) -> bool:
        """
        Manually update the trajectory of a pattern.

        Args:
            pattern_id: ID of the pattern
            trajectory: New trajectory
            reason: Reason for the update

        Returns:
            Success boolean
        """
        entry = self.get(pattern_id)
        if not entry:
            return False

        old_trajectory = entry.data["trajectory"]
        entry.data["trajectory"] = trajectory.value

        # Log trajectory change
        if "trajectory_history" not in entry.data:
            entry.data["trajectory_history"] = []

        entry.data["trajectory_history"].append({
            "timestamp": datetime.now().isoformat(),
            "from": old_trajectory,
            "to": trajectory.value,
            "reason": reason
        })

        if trajectory == Trajectory.RESOLVED:
            entry.data["active"] = False
            entry.data["resolved_at"] = datetime.now().isoformat()

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def record_intervention(
        self,
        pattern_id: str,
        intervention_id: str,
        intervention_description: str,
        effectiveness: int
    ) -> bool:
        """
        Record an intervention attempt and its effectiveness.

        Args:
            pattern_id: ID of the pattern
            intervention_id: Unique ID for the intervention
            intervention_description: What was tried
            effectiveness: Effectiveness score (1-10)

        Returns:
            Success boolean
        """
        entry = self.get(pattern_id)
        if not entry:
            return False

        entry.data["intervention_effectiveness"][intervention_id] = {
            "description": intervention_description,
            "effectiveness": max(1, min(10, effectiveness)),
            "timestamp": datetime.now().isoformat()
        }

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[Dict[str, Any]]:
        """
        Get all patterns of a specific type.

        Args:
            pattern_type: Type of patterns to retrieve

        Returns:
            List of pattern data dictionaries
        """
        results = []
        for entry in self._entries.values():
            if entry.data.get("pattern_type") == pattern_type.value:
                results.append({
                    "pattern_id": entry.id,
                    **entry.data
                })
        return results

    def get_active_patterns(self) -> List[Dict[str, Any]]:
        """Get all active (non-resolved) patterns."""
        results = []
        for entry in self._entries.values():
            if entry.data.get("active", True):
                results.append({
                    "pattern_id": entry.id,
                    **entry.data
                })
        return results

    def get_worsening_patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns with worsening trajectory."""
        results = []
        for entry in self._entries.values():
            if entry.data.get("trajectory") == Trajectory.WORSENING.value:
                results.append({
                    "pattern_id": entry.id,
                    **entry.data
                })
        return results

    def get_pattern_summary(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a pattern including statistics.

        Args:
            pattern_id: ID of the pattern

        Returns:
            Summary dictionary or None if not found
        """
        entry = self.get(pattern_id)
        if not entry:
            return None

        occurrences = entry.data.get("occurrences", [])
        interventions = entry.data.get("intervention_effectiveness", {})

        # Find most effective intervention
        best_intervention = None
        if interventions:
            best_id = max(interventions.keys(), key=lambda k: interventions[k]["effectiveness"])
            best_intervention = {
                "id": best_id,
                **interventions[best_id]
            }

        return {
            "pattern_id": pattern_id,
            "pattern_type": entry.data.get("pattern_type"),
            "description": entry.data.get("description"),
            "active": entry.data.get("active", True),
            "trajectory": entry.data.get("trajectory"),
            "occurrence_count": len(occurrences),
            "average_severity": entry.data.get("average_severity", 0),
            "first_observed": entry.data.get("first_observed"),
            "last_occurrence": entry.data.get("last_occurrence"),
            "most_effective_intervention": best_intervention,
            "intervention_count": len(interventions)
        }

    def link_patterns(self, pattern_id_1: str, pattern_id_2: str, relationship: str = "related") -> bool:
        """
        Link two patterns as related.

        Args:
            pattern_id_1: First pattern ID
            pattern_id_2: Second pattern ID
            relationship: Type of relationship

        Returns:
            Success boolean
        """
        entry1 = self.get(pattern_id_1)
        entry2 = self.get(pattern_id_2)

        if not entry1 or not entry2:
            return False

        link1 = {"pattern_id": pattern_id_2, "relationship": relationship}
        link2 = {"pattern_id": pattern_id_1, "relationship": relationship}

        if "related_patterns" not in entry1.data:
            entry1.data["related_patterns"] = []
        if "related_patterns" not in entry2.data:
            entry2.data["related_patterns"] = []

        if link1 not in entry1.data["related_patterns"]:
            entry1.data["related_patterns"].append(link1)
        if link2 not in entry2.data["related_patterns"]:
            entry2.data["related_patterns"].append(link2)

        self._save()
        return True

    def get_recent_occurrences(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get all pattern occurrences in the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of occurrences with pattern info
        """
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        results = []
        for entry in self._entries.values():
            for occ in entry.data.get("occurrences", []):
                if occ.get("timestamp", "") >= cutoff:
                    results.append({
                        "pattern_id": entry.id,
                        "pattern_type": entry.data.get("pattern_type"),
                        "description": entry.data.get("description"),
                        **occ
                    })

        # Sort by timestamp descending
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results
