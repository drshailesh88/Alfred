# Values Hierarchy Memory System
# Tracks stated vs revealed values and detects conflicts between them

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

from . import BaseMemorySystem, MemoryType, MemoryEntry


class ValueSource(Enum):
    """How a value was identified."""
    STATED = "stated"           # User explicitly stated this value
    INFERRED = "inferred"       # Inferred from behavior
    CONFLICT = "conflict"       # Identified through value conflict


class ValueStrength(Enum):
    """Strength of a value based on evidence."""
    WEAK = "weak"               # Few supporting instances
    MODERATE = "moderate"       # Some supporting evidence
    STRONG = "strong"           # Consistent supporting evidence
    CORE = "core"               # Fundamental, rarely violated


class ValuesHierarchy(BaseMemorySystem):
    """
    Tracks the user's stated vs revealed values and conflicts.

    Key insight: What people say they value often differs from what their
    behavior reveals they actually value. This system tracks both and
    identifies conflicts that may cause internal dissonance.

    Schema:
    - stated_values: Values the user has explicitly claimed
      - value_id, value_name, description, stated_at, priority, supporting_statements[]
    - revealed_values: Values inferred from behavior
      - value_id, value_name, description, inferred_at, evidence[], confidence
    - conflicts: Detected conflicts between stated and revealed values
      - conflict_id, stated_value_id, revealed_value_id, instances[], severity, resolved
    """

    def __init__(self, storage_path=None):
        super().__init__(MemoryType.VALUES, storage_path)
        # Initialize structure if empty
        if not self._entries:
            self._initialize_structure()

    def _initialize_structure(self):
        """Initialize the values hierarchy structure."""
        # Create main structure entries
        self.add("stated_values", {"values": {}})
        self.add("revealed_values", {"values": {}})
        self.add("conflicts", {"conflicts": {}})
        self.add("value_rankings", {"rankings": [], "last_updated": None})

    def add_stated_value(
        self,
        value_name: str,
        description: str,
        priority: int = 5,
        statement: Optional[str] = None
    ) -> str:
        """
        Add a value that the user has explicitly stated.

        Args:
            value_name: Name of the value (e.g., "family", "honesty")
            description: Description of what this value means to the user
            priority: User's stated priority for this value (1-10)
            statement: The user's statement that revealed this value

        Returns:
            value_id: Unique identifier for this value
        """
        stated_entry = self.get("stated_values")
        if not stated_entry:
            self._initialize_structure()
            stated_entry = self.get("stated_values")

        value_id = f"stated_{uuid.uuid4().hex[:8]}"

        value_data = {
            "value_name": value_name,
            "description": description,
            "stated_at": datetime.now().isoformat(),
            "priority": max(1, min(10, priority)),
            "supporting_statements": [],
            "strength": ValueStrength.MODERATE.value,
            "active": True,
            "related_revealed_values": []
        }

        if statement:
            value_data["supporting_statements"].append({
                "statement": statement,
                "timestamp": datetime.now().isoformat(),
                "context": None
            })

        stated_entry.data["values"][value_id] = value_data
        stated_entry.updated_at = datetime.now().isoformat()
        self._save()

        return value_id

    def add_statement_to_value(
        self,
        value_id: str,
        statement: str,
        context: Optional[str] = None
    ) -> bool:
        """
        Add a supporting statement to an existing stated value.

        Args:
            value_id: ID of the stated value
            statement: The supporting statement
            context: Context in which it was said

        Returns:
            Success boolean
        """
        stated_entry = self.get("stated_values")
        if not stated_entry or value_id not in stated_entry.data["values"]:
            return False

        value = stated_entry.data["values"][value_id]
        value["supporting_statements"].append({
            "statement": statement,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })

        # Strengthen value based on evidence
        num_statements = len(value["supporting_statements"])
        if num_statements >= 5:
            value["strength"] = ValueStrength.CORE.value
        elif num_statements >= 3:
            value["strength"] = ValueStrength.STRONG.value

        stated_entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def infer_revealed_value(
        self,
        value_name: str,
        description: str,
        evidence: str,
        confidence: float = 0.5,
        behavior_context: Optional[str] = None
    ) -> str:
        """
        Infer a revealed value from user behavior.

        Args:
            value_name: Name of the inferred value
            description: What this value appears to be
            evidence: The behavioral evidence
            confidence: Confidence in inference (0-1)
            behavior_context: Context of the observed behavior

        Returns:
            value_id: Unique identifier for this revealed value
        """
        revealed_entry = self.get("revealed_values")
        if not revealed_entry:
            self._initialize_structure()
            revealed_entry = self.get("revealed_values")

        # Check if similar revealed value exists
        for vid, vdata in revealed_entry.data["values"].items():
            if vdata["value_name"].lower() == value_name.lower():
                # Add to existing value's evidence
                vdata["evidence"].append({
                    "observation": evidence,
                    "timestamp": datetime.now().isoformat(),
                    "context": behavior_context
                })
                # Update confidence (weighted average)
                old_conf = vdata["confidence"]
                vdata["confidence"] = (old_conf + confidence) / 2
                self._update_revealed_strength(vdata)
                self._save()
                return vid

        value_id = f"revealed_{uuid.uuid4().hex[:8]}"

        value_data = {
            "value_name": value_name,
            "description": description,
            "inferred_at": datetime.now().isoformat(),
            "evidence": [{
                "observation": evidence,
                "timestamp": datetime.now().isoformat(),
                "context": behavior_context
            }],
            "confidence": max(0, min(1, confidence)),
            "strength": ValueStrength.WEAK.value,
            "active": True,
            "contradicting_evidence": []
        }

        revealed_entry.data["values"][value_id] = value_data
        revealed_entry.updated_at = datetime.now().isoformat()
        self._save()

        # Check for potential conflicts with stated values
        self._check_for_conflicts(value_id, value_name)

        return value_id

    def _update_revealed_strength(self, value_data: Dict[str, Any]):
        """Update strength of revealed value based on evidence."""
        evidence_count = len(value_data["evidence"])
        confidence = value_data["confidence"]

        if evidence_count >= 5 and confidence >= 0.7:
            value_data["strength"] = ValueStrength.CORE.value
        elif evidence_count >= 3 and confidence >= 0.5:
            value_data["strength"] = ValueStrength.STRONG.value
        elif evidence_count >= 2 or confidence >= 0.4:
            value_data["strength"] = ValueStrength.MODERATE.value
        else:
            value_data["strength"] = ValueStrength.WEAK.value

    def _check_for_conflicts(self, revealed_value_id: str, revealed_value_name: str):
        """Check if a revealed value conflicts with any stated values."""
        stated_entry = self.get("stated_values")
        revealed_entry = self.get("revealed_values")

        if not stated_entry or not revealed_entry:
            return

        revealed_value = revealed_entry.data["values"].get(revealed_value_id)
        if not revealed_value:
            return

        # Define known value oppositions
        oppositions = {
            "family": ["career obsession", "workaholism", "isolation"],
            "health": ["overwork", "neglect", "substance use"],
            "honesty": ["deception", "image management", "people pleasing"],
            "freedom": ["security seeking", "control", "dependency"],
            "creativity": ["conformity", "risk aversion", "perfectionism"],
            "connection": ["isolation", "independence", "self-reliance"],
            "growth": ["comfort seeking", "avoidance", "stagnation"],
            "balance": ["obsession", "extremism", "workaholism"]
        }

        revealed_lower = revealed_value_name.lower()

        for stated_id, stated_value in stated_entry.data["values"].items():
            stated_lower = stated_value["value_name"].lower()

            # Check if revealed value opposes stated value
            opposing_values = oppositions.get(stated_lower, [])
            if any(opp in revealed_lower for opp in opposing_values):
                self._create_conflict(stated_id, revealed_value_id)

    def _create_conflict(self, stated_value_id: str, revealed_value_id: str):
        """Create a conflict record between stated and revealed values."""
        conflicts_entry = self.get("conflicts")
        stated_entry = self.get("stated_values")
        revealed_entry = self.get("revealed_values")

        if not all([conflicts_entry, stated_entry, revealed_entry]):
            return

        stated_value = stated_entry.data["values"].get(stated_value_id)
        revealed_value = revealed_entry.data["values"].get(revealed_value_id)

        if not stated_value or not revealed_value:
            return

        conflict_id = f"conflict_{uuid.uuid4().hex[:8]}"

        conflict_data = {
            "stated_value_id": stated_value_id,
            "stated_value_name": stated_value["value_name"],
            "revealed_value_id": revealed_value_id,
            "revealed_value_name": revealed_value["value_name"],
            "detected_at": datetime.now().isoformat(),
            "instances": [{
                "timestamp": datetime.now().isoformat(),
                "description": f"Detected conflict: stated '{stated_value['value_name']}' vs revealed '{revealed_value['value_name']}'"
            }],
            "severity": 5,  # Default severity
            "resolved": False,
            "resolution_notes": None,
            "user_acknowledged": False
        }

        conflicts_entry.data["conflicts"][conflict_id] = conflict_data
        conflicts_entry.updated_at = datetime.now().isoformat()
        self._save()

    def detect_conflict(
        self,
        stated_value_id: str,
        revealed_value_id: str,
        instance_description: str,
        severity: int = 5
    ) -> str:
        """
        Manually detect and record a conflict between values.

        Args:
            stated_value_id: ID of the stated value
            revealed_value_id: ID of the revealed value
            instance_description: Description of the conflict instance
            severity: Severity of the conflict (1-10)

        Returns:
            conflict_id: Unique identifier for this conflict
        """
        conflicts_entry = self.get("conflicts")
        stated_entry = self.get("stated_values")
        revealed_entry = self.get("revealed_values")

        if not all([conflicts_entry, stated_entry, revealed_entry]):
            self._initialize_structure()
            conflicts_entry = self.get("conflicts")
            stated_entry = self.get("stated_values")
            revealed_entry = self.get("revealed_values")

        stated_value = stated_entry.data["values"].get(stated_value_id)
        revealed_value = revealed_entry.data["values"].get(revealed_value_id)

        if not stated_value or not revealed_value:
            return None

        # Check if conflict already exists
        for cid, conflict in conflicts_entry.data["conflicts"].items():
            if (conflict["stated_value_id"] == stated_value_id and
                conflict["revealed_value_id"] == revealed_value_id):
                # Add instance to existing conflict
                conflict["instances"].append({
                    "timestamp": datetime.now().isoformat(),
                    "description": instance_description
                })
                conflict["severity"] = max(conflict["severity"], severity)
                self._save()
                return cid

        # Create new conflict
        conflict_id = f"conflict_{uuid.uuid4().hex[:8]}"

        conflict_data = {
            "stated_value_id": stated_value_id,
            "stated_value_name": stated_value["value_name"],
            "revealed_value_id": revealed_value_id,
            "revealed_value_name": revealed_value["value_name"],
            "detected_at": datetime.now().isoformat(),
            "instances": [{
                "timestamp": datetime.now().isoformat(),
                "description": instance_description
            }],
            "severity": max(1, min(10, severity)),
            "resolved": False,
            "resolution_notes": None,
            "user_acknowledged": False
        }

        conflicts_entry.data["conflicts"][conflict_id] = conflict_data
        conflicts_entry.updated_at = datetime.now().isoformat()
        self._save()

        return conflict_id

    def get_value_conflicts(
        self,
        include_resolved: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all value conflicts.

        Args:
            include_resolved: Whether to include resolved conflicts

        Returns:
            List of conflict dictionaries
        """
        conflicts_entry = self.get("conflicts")
        if not conflicts_entry:
            return []

        results = []
        for conflict_id, conflict in conflicts_entry.data["conflicts"].items():
            if include_resolved or not conflict["resolved"]:
                results.append({
                    "conflict_id": conflict_id,
                    **conflict
                })

        # Sort by severity descending
        results.sort(key=lambda x: x["severity"], reverse=True)
        return results

    def get_stated_values(self) -> List[Dict[str, Any]]:
        """Get all stated values."""
        stated_entry = self.get("stated_values")
        if not stated_entry:
            return []

        results = []
        for value_id, value in stated_entry.data["values"].items():
            results.append({
                "value_id": value_id,
                **value
            })

        # Sort by priority descending
        results.sort(key=lambda x: x.get("priority", 0), reverse=True)
        return results

    def get_revealed_values(self) -> List[Dict[str, Any]]:
        """Get all revealed values."""
        revealed_entry = self.get("revealed_values")
        if not revealed_entry:
            return []

        results = []
        for value_id, value in revealed_entry.data["values"].items():
            results.append({
                "value_id": value_id,
                **value
            })

        # Sort by confidence descending
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results

    def acknowledge_conflict(
        self,
        conflict_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Mark a conflict as acknowledged by the user.

        Args:
            conflict_id: ID of the conflict
            notes: User's notes about the conflict

        Returns:
            Success boolean
        """
        conflicts_entry = self.get("conflicts")
        if not conflicts_entry or conflict_id not in conflicts_entry.data["conflicts"]:
            return False

        conflict = conflicts_entry.data["conflicts"][conflict_id]
        conflict["user_acknowledged"] = True
        conflict["acknowledged_at"] = datetime.now().isoformat()
        if notes:
            conflict["acknowledgment_notes"] = notes

        self._save()
        return True

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution_notes: str
    ) -> bool:
        """
        Mark a conflict as resolved.

        Args:
            conflict_id: ID of the conflict
            resolution_notes: How the conflict was resolved

        Returns:
            Success boolean
        """
        conflicts_entry = self.get("conflicts")
        if not conflicts_entry or conflict_id not in conflicts_entry.data["conflicts"]:
            return False

        conflict = conflicts_entry.data["conflicts"][conflict_id]
        conflict["resolved"] = True
        conflict["resolution_notes"] = resolution_notes
        conflict["resolved_at"] = datetime.now().isoformat()

        self._save()
        return True

    def get_values_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the values hierarchy.

        Returns:
            Summary dictionary with statistics and insights
        """
        stated = self.get_stated_values()
        revealed = self.get_revealed_values()
        conflicts = self.get_value_conflicts()

        return {
            "stated_values_count": len(stated),
            "revealed_values_count": len(revealed),
            "active_conflicts_count": len([c for c in conflicts if not c.get("resolved")]),
            "unacknowledged_conflicts": len([c for c in conflicts if not c.get("user_acknowledged")]),
            "top_stated_values": [v["value_name"] for v in stated[:5]],
            "high_confidence_revealed": [v["value_name"] for v in revealed if v.get("confidence", 0) >= 0.7],
            "severe_conflicts": [c for c in conflicts if c.get("severity", 0) >= 7],
            "alignment_score": self._calculate_alignment_score(stated, revealed, conflicts)
        }

    def _calculate_alignment_score(
        self,
        stated: List[Dict],
        revealed: List[Dict],
        conflicts: List[Dict]
    ) -> float:
        """Calculate overall alignment between stated and revealed values (0-100)."""
        if not stated or not revealed:
            return 100.0  # No misalignment if no data

        # Start with 100, subtract for conflicts
        score = 100.0

        for conflict in conflicts:
            if not conflict.get("resolved"):
                severity = conflict.get("severity", 5)
                instances = len(conflict.get("instances", []))
                # More instances and higher severity = bigger impact
                impact = (severity / 10) * min(instances, 5) * 4
                score -= impact

        return max(0, min(100, score))

    def compare_stated_vs_revealed(self, value_name: str) -> Optional[Dict[str, Any]]:
        """
        Compare stated and revealed versions of a value.

        Args:
            value_name: Name of the value to compare

        Returns:
            Comparison dictionary or None if not found
        """
        stated_entry = self.get("stated_values")
        revealed_entry = self.get("revealed_values")

        stated_match = None
        revealed_match = None
        value_lower = value_name.lower()

        if stated_entry:
            for vid, value in stated_entry.data["values"].items():
                if value["value_name"].lower() == value_lower:
                    stated_match = {"value_id": vid, **value}
                    break

        if revealed_entry:
            for vid, value in revealed_entry.data["values"].items():
                if value["value_name"].lower() == value_lower:
                    revealed_match = {"value_id": vid, **value}
                    break

        if not stated_match and not revealed_match:
            return None

        return {
            "value_name": value_name,
            "stated": stated_match,
            "revealed": revealed_match,
            "alignment": "aligned" if (stated_match and revealed_match) else
                        ("stated_only" if stated_match else "revealed_only")
        }
