# Self-Violation Log Memory System
# Tracks when the user violates their own stated standards

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

from . import BaseMemorySystem, MemoryType, MemoryEntry


class ViolationSeverity(Enum):
    """Severity levels of self-violations."""
    MINOR = "minor"           # Small inconsistency
    MODERATE = "moderate"     # Notable violation
    SIGNIFICANT = "significant"  # Major departure from standards
    CRITICAL = "critical"     # Severe violation with consequences


class ViolationCategory(Enum):
    """Categories of self-violations."""
    BEHAVIOR = "behavior"           # Actions that violate standards
    COMMITMENT = "commitment"       # Breaking commitments to self
    VALUE = "value"                 # Acting against stated values
    BOUNDARY = "boundary"           # Crossing self-set boundaries
    RELATIONSHIP = "relationship"   # Violating relationship standards
    HEALTH = "health"              # Violating health/wellness standards
    PROFESSIONAL = "professional"   # Violating professional standards
    FINANCIAL = "financial"        # Violating financial standards


class SelfViolationLog(BaseMemorySystem):
    """
    Tracks when the user violates their own stated standards.

    This is not about judgment but about awareness. People often
    violate their own standards and either don't notice or rationalize
    it away. This system maintains a compassionate but honest record.

    Schema per violation:
    - violation_id: Unique identifier
    - standard_violated: The standard that was violated
    - category: Category of violation
    - context: What was happening when the violation occurred
    - justification_given: How the user justified/rationalized it
    - outcome: What resulted from the violation
    - severity: How significant the violation was
    - acknowledged: Whether user acknowledged the violation
    - lesson_learned: Any insight gained
    """

    def __init__(self, storage_path=None):
        super().__init__(MemoryType.VIOLATION, storage_path)
        # Initialize standards tracker if needed
        if "standards" not in [e.id for e in self._entries.values()]:
            self.add("standards", {"standards": {}})

    def register_standard(
        self,
        standard_name: str,
        description: str,
        category: ViolationCategory,
        importance: int = 5
    ) -> str:
        """
        Register a standard the user holds themselves to.

        Args:
            standard_name: Name of the standard
            description: What this standard means
            category: Category of the standard
            importance: How important this standard is (1-10)

        Returns:
            standard_id: Unique identifier for this standard
        """
        standards_entry = self.get("standards")
        if not standards_entry:
            self.add("standards", {"standards": {}})
            standards_entry = self.get("standards")

        standard_id = f"std_{uuid.uuid4().hex[:8]}"

        standard_data = {
            "name": standard_name,
            "description": description,
            "category": category.value,
            "importance": max(1, min(10, importance)),
            "created_at": datetime.now().isoformat(),
            "violation_count": 0,
            "last_violated": None,
            "active": True
        }

        standards_entry.data["standards"][standard_id] = standard_data
        standards_entry.updated_at = datetime.now().isoformat()
        self._save()

        return standard_id

    def log_violation(
        self,
        standard_violated: str,
        context: str,
        justification_given: Optional[str] = None,
        outcome: Optional[str] = None,
        severity: ViolationSeverity = ViolationSeverity.MODERATE,
        category: ViolationCategory = ViolationCategory.BEHAVIOR,
        trigger: Optional[str] = None
    ) -> str:
        """
        Log a self-violation.

        Args:
            standard_violated: The standard that was violated (name or ID)
            context: What was happening when the violation occurred
            justification_given: How the user justified/rationalized it
            outcome: What resulted from the violation
            severity: Severity level of the violation
            category: Category of the violation
            trigger: What triggered the violation

        Returns:
            violation_id: Unique identifier for this violation
        """
        violation_id = f"viol_{uuid.uuid4().hex[:12]}"

        # Try to find standard ID
        standard_id = self._find_standard_id(standard_violated)

        violation_data = {
            "standard_violated": standard_violated,
            "standard_id": standard_id,
            "category": category.value,
            "context": context,
            "justification_given": justification_given,
            "outcome": outcome,
            "severity": severity.value,
            "trigger": trigger,
            "occurred_at": datetime.now().isoformat(),
            "acknowledged": False,
            "acknowledged_at": None,
            "lesson_learned": None,
            "pattern_detected": False,
            "related_violations": []
        }

        self.add(violation_id, violation_data)

        # Update standard's violation count
        if standard_id:
            self._update_standard_violation_count(standard_id)

        # Check for patterns
        self._detect_violation_patterns(violation_id)

        return violation_id

    def _find_standard_id(self, standard_ref: str) -> Optional[str]:
        """Find standard ID by name or ID."""
        standards_entry = self.get("standards")
        if not standards_entry:
            return None

        # Check if it's already an ID
        if standard_ref in standards_entry.data["standards"]:
            return standard_ref

        # Search by name
        for sid, standard in standards_entry.data["standards"].items():
            if standard["name"].lower() == standard_ref.lower():
                return sid

        return None

    def _update_standard_violation_count(self, standard_id: str):
        """Update the violation count for a standard."""
        standards_entry = self.get("standards")
        if standards_entry and standard_id in standards_entry.data["standards"]:
            standards_entry.data["standards"][standard_id]["violation_count"] += 1
            standards_entry.data["standards"][standard_id]["last_violated"] = datetime.now().isoformat()
            self._save()

    def _detect_violation_patterns(self, violation_id: str):
        """Detect patterns in violations."""
        entry = self.get(violation_id)
        if not entry:
            return

        standard = entry.data.get("standard_violated")
        category = entry.data.get("category")

        # Find related violations
        related = []
        for e in self._entries.values():
            if e.id == violation_id or e.id == "standards":
                continue

            if (e.data.get("standard_violated") == standard or
                e.data.get("category") == category):
                related.append(e.id)

        if len(related) >= 2:
            entry.data["pattern_detected"] = True
            entry.data["related_violations"] = related[-5:]  # Keep last 5
            self._save()

    def acknowledge_violation(
        self,
        violation_id: str,
        acknowledgment_notes: Optional[str] = None
    ) -> bool:
        """
        Acknowledge a violation.

        Args:
            violation_id: ID of the violation
            acknowledgment_notes: Notes about the acknowledgment

        Returns:
            Success boolean
        """
        entry = self.get(violation_id)
        if not entry or entry.id == "standards":
            return False

        entry.data["acknowledged"] = True
        entry.data["acknowledged_at"] = datetime.now().isoformat()
        if acknowledgment_notes:
            entry.data["acknowledgment_notes"] = acknowledgment_notes

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def record_lesson(
        self,
        violation_id: str,
        lesson: str
    ) -> bool:
        """
        Record a lesson learned from a violation.

        Args:
            violation_id: ID of the violation
            lesson: The lesson learned

        Returns:
            Success boolean
        """
        entry = self.get(violation_id)
        if not entry or entry.id == "standards":
            return False

        entry.data["lesson_learned"] = lesson
        entry.data["lesson_recorded_at"] = datetime.now().isoformat()

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def update_outcome(
        self,
        violation_id: str,
        outcome: str
    ) -> bool:
        """
        Update the outcome of a violation.

        Args:
            violation_id: ID of the violation
            outcome: The outcome to record

        Returns:
            Success boolean
        """
        entry = self.get(violation_id)
        if not entry or entry.id == "standards":
            return False

        entry.data["outcome"] = outcome
        entry.data["outcome_recorded_at"] = datetime.now().isoformat()

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def get_violations_by_standard(
        self,
        standard_ref: str,
        include_acknowledged: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all violations of a specific standard.

        Args:
            standard_ref: Standard name or ID
            include_acknowledged: Whether to include acknowledged violations

        Returns:
            List of violation dictionaries
        """
        standard_id = self._find_standard_id(standard_ref)

        results = []
        for entry in self._entries.values():
            if entry.id == "standards":
                continue

            # Match by name or ID
            if (entry.data.get("standard_violated", "").lower() == standard_ref.lower() or
                entry.data.get("standard_id") == standard_id):

                if include_acknowledged or not entry.data.get("acknowledged"):
                    results.append({
                        "violation_id": entry.id,
                        **entry.data
                    })

        # Sort by occurred_at descending
        results.sort(key=lambda x: x.get("occurred_at", ""), reverse=True)
        return results

    def get_violations_by_category(
        self,
        category: ViolationCategory,
        include_acknowledged: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all violations in a specific category.

        Args:
            category: Category to filter by
            include_acknowledged: Whether to include acknowledged violations

        Returns:
            List of violation dictionaries
        """
        results = []
        for entry in self._entries.values():
            if entry.id == "standards":
                continue

            if entry.data.get("category") == category.value:
                if include_acknowledged or not entry.data.get("acknowledged"):
                    results.append({
                        "violation_id": entry.id,
                        **entry.data
                    })

        results.sort(key=lambda x: x.get("occurred_at", ""), reverse=True)
        return results

    def get_unacknowledged_violations(self) -> List[Dict[str, Any]]:
        """Get all violations that haven't been acknowledged."""
        results = []
        for entry in self._entries.values():
            if entry.id == "standards":
                continue

            if not entry.data.get("acknowledged"):
                results.append({
                    "violation_id": entry.id,
                    **entry.data
                })

        # Sort by severity then by date
        severity_order = {
            ViolationSeverity.CRITICAL.value: 4,
            ViolationSeverity.SIGNIFICANT.value: 3,
            ViolationSeverity.MODERATE.value: 2,
            ViolationSeverity.MINOR.value: 1
        }

        results.sort(
            key=lambda x: (
                severity_order.get(x.get("severity"), 0),
                x.get("occurred_at", "")
            ),
            reverse=True
        )
        return results

    def get_recurring_violations(self, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Get standards that have been violated multiple times.

        Args:
            min_occurrences: Minimum number of violations to be considered recurring

        Returns:
            List of recurring violation patterns
        """
        standards_entry = self.get("standards")
        if not standards_entry:
            return []

        results = []
        for std_id, standard in standards_entry.data["standards"].items():
            if standard.get("violation_count", 0) >= min_occurrences:
                violations = self.get_violations_by_standard(std_id)

                # Analyze justifications
                justifications = [v.get("justification_given") for v in violations if v.get("justification_given")]

                results.append({
                    "standard_id": std_id,
                    "standard_name": standard["name"],
                    "violation_count": standard["violation_count"],
                    "category": standard["category"],
                    "importance": standard["importance"],
                    "last_violated": standard.get("last_violated"),
                    "common_justifications": justifications[:5],
                    "lessons_learned": [v.get("lesson_learned") for v in violations if v.get("lesson_learned")]
                })

        results.sort(key=lambda x: x["violation_count"], reverse=True)
        return results

    def get_justification_patterns(self) -> Dict[str, List[str]]:
        """
        Analyze patterns in justifications given for violations.

        Returns:
            Dictionary mapping categories to common justifications
        """
        patterns = {}

        for entry in self._entries.values():
            if entry.id == "standards":
                continue

            category = entry.data.get("category", "unknown")
            justification = entry.data.get("justification_given")

            if justification:
                if category not in patterns:
                    patterns[category] = []
                patterns[category].append(justification)

        return patterns

    def get_violation_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of self-violations.

        Returns:
            Summary dictionary with statistics and insights
        """
        violations = []
        for entry in self._entries.values():
            if entry.id != "standards":
                violations.append(entry.data)

        standards_entry = self.get("standards")
        standards = standards_entry.data["standards"] if standards_entry else {}

        # Count by category
        by_category = {}
        for v in violations:
            cat = v.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1

        # Count by severity
        by_severity = {}
        for v in violations:
            sev = v.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # Most violated standards
        most_violated = sorted(
            standards.items(),
            key=lambda x: x[1].get("violation_count", 0),
            reverse=True
        )[:5]

        return {
            "total_violations": len(violations),
            "unacknowledged_count": len([v for v in violations if not v.get("acknowledged")]),
            "lessons_extracted": len([v for v in violations if v.get("lesson_learned")]),
            "violations_by_category": by_category,
            "violations_by_severity": by_severity,
            "most_violated_standards": [
                {"name": s[1]["name"], "count": s[1].get("violation_count", 0)}
                for s in most_violated
            ],
            "pattern_violations": len([v for v in violations if v.get("pattern_detected")]),
            "registered_standards_count": len(standards)
        }

    def get_standards(self) -> List[Dict[str, Any]]:
        """Get all registered standards."""
        standards_entry = self.get("standards")
        if not standards_entry:
            return []

        results = []
        for std_id, standard in standards_entry.data["standards"].items():
            results.append({
                "standard_id": std_id,
                **standard
            })

        results.sort(key=lambda x: x.get("importance", 0), reverse=True)
        return results

    def get_recent_violations(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get violations from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of recent violations
        """
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        results = []
        for entry in self._entries.values():
            if entry.id == "standards":
                continue

            if entry.data.get("occurred_at", "") >= cutoff:
                results.append({
                    "violation_id": entry.id,
                    **entry.data
                })

        results.sort(key=lambda x: x.get("occurred_at", ""), reverse=True)
        return results
