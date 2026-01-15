# Optionality Register Memory System
# Tracks exit options and whether they're being burned

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

from . import BaseMemorySystem, MemoryType, MemoryEntry


class OptionStatus(Enum):
    """Status of an option."""
    OPEN = "open"               # Fully available
    NARROWING = "narrowing"     # Being constrained
    AT_RISK = "at_risk"         # Significant danger of closure
    CLOSED = "closed"           # No longer available
    RECOVERED = "recovered"     # Was closed but reopened


class OptionCategory(Enum):
    """Categories of options."""
    CAREER = "career"           # Career paths and opportunities
    RELATIONSHIP = "relationship"  # Relationship options
    FINANCIAL = "financial"     # Financial flexibility
    GEOGRAPHIC = "geographic"   # Location options
    LIFESTYLE = "lifestyle"     # Lifestyle choices
    HEALTH = "health"           # Health-related options
    EDUCATIONAL = "educational"  # Learning/skill paths
    SOCIAL = "social"           # Social connections/networks
    CREATIVE = "creative"       # Creative pursuits
    EXIT = "exit"               # Exit strategies


class ClosureReason(Enum):
    """Reasons for option closure."""
    DECISION = "decision"       # Active choice closed it
    NEGLECT = "neglect"         # Passive inaction closed it
    EXTERNAL = "external"       # External circumstances
    TIME = "time"               # Time-based expiration
    RESOURCE = "resource"       # Resource exhaustion
    BRIDGE_BURNED = "bridge_burned"  # Relationship damage
    COMMITMENT = "commitment"   # Other commitments preclude it


class OptionalityRegister(BaseMemorySystem):
    """
    Tracks exit options and whether they're being burned.

    Optionality is valuable - keeping options open is often better than
    prematurely committing. This system tracks what options exist and
    warns when decisions or behaviors are closing them.

    Schema per option:
    - option_id: Unique identifier
    - option_name: Name of the option
    - category: Category of option
    - description: What this option represents
    - status: Current status (open/narrowing/at_risk/closed)
    - value_score: How valuable this option is (1-10)
    - reversibility: How reversible closure is (1-10)
    - closure_date: When it was closed (if applicable)
    - closure_reason: Why it closed
    - decision_correlations: Decisions that affected this option
    - history: Status change history
    """

    def __init__(self, storage_path=None):
        super().__init__(MemoryType.OPTIONALITY, storage_path)
        # Initialize correlations tracker
        if "correlations" not in [e.id for e in self._entries.values()]:
            self.add("correlations", {
                "decision_to_options": {},  # decision_id -> [option_ids affected]
                "option_to_decisions": {}   # option_id -> [decision_ids that affected]
            })

    def register_option(
        self,
        option_name: str,
        category: OptionCategory,
        description: str,
        value_score: int = 5,
        reversibility: int = 5,
        expiration_date: Optional[str] = None,
        prerequisites: Optional[List[str]] = None
    ) -> str:
        """
        Register an option to track.

        Args:
            option_name: Name of the option
            category: Category of option
            description: What this option represents
            value_score: How valuable the option is (1-10)
            reversibility: How reversible closure is (1-10)
            expiration_date: When option naturally expires
            prerequisites: Things needed to exercise this option

        Returns:
            option_id: Unique identifier
        """
        option_id = f"option_{uuid.uuid4().hex[:12]}"

        option_data = {
            "option_name": option_name,
            "category": category.value,
            "description": description,
            "status": OptionStatus.OPEN.value,
            "value_score": max(1, min(10, value_score)),
            "reversibility": max(1, min(10, reversibility)),
            "created_at": datetime.now().isoformat(),
            "last_status_change": datetime.now().isoformat(),
            "expiration_date": expiration_date,
            "closure_date": None,
            "closure_reason": None,
            "closure_notes": None,
            "decision_correlations": [],
            "prerequisites": prerequisites or [],
            "prerequisites_status": {},
            "narrowing_factors": [],
            "recovery_attempts": [],
            "history": [{
                "timestamp": datetime.now().isoformat(),
                "status": OptionStatus.OPEN.value,
                "notes": "Option registered"
            }],
            "warnings_issued": []
        }

        self.add(option_id, option_data)
        return option_id

    def update_status(
        self,
        option_id: str,
        new_status: OptionStatus,
        reason: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the status of an option.

        Args:
            option_id: ID of the option
            new_status: New status
            reason: Reason for change
            notes: Additional notes

        Returns:
            Update result dictionary
        """
        entry = self.get(option_id)
        if not entry or entry.id == "correlations":
            return {"success": False, "error": "Option not found"}

        old_status = entry.data.get("status")

        if old_status == new_status.value:
            return {"success": True, "status": "unchanged"}

        # Update status
        entry.data["status"] = new_status.value
        entry.data["last_status_change"] = datetime.now().isoformat()

        # Add to history
        entry.data["history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": new_status.value,
            "previous_status": old_status,
            "reason": reason,
            "notes": notes
        })

        # Handle specific status changes
        warning_issued = None

        if new_status == OptionStatus.CLOSED:
            entry.data["closure_date"] = datetime.now().isoformat()
            entry.data["closure_reason"] = reason
            entry.data["closure_notes"] = notes

        elif new_status == OptionStatus.RECOVERED:
            entry.data["recovery_attempts"].append({
                "timestamp": datetime.now().isoformat(),
                "notes": notes,
                "success": True
            })

        elif new_status in [OptionStatus.NARROWING, OptionStatus.AT_RISK]:
            # Issue warning
            warning_issued = self._issue_warning(entry, old_status, new_status, reason)

        entry.updated_at = datetime.now().isoformat()
        self._save()

        return {
            "success": True,
            "option_name": entry.data["option_name"],
            "previous_status": old_status,
            "new_status": new_status.value,
            "warning_issued": warning_issued
        }

    def _issue_warning(
        self,
        entry: MemoryEntry,
        old_status: str,
        new_status: OptionStatus,
        reason: Optional[str]
    ) -> Dict[str, Any]:
        """Issue a warning about option status change."""
        warning = {
            "warning_id": f"optwarn_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "previous_status": old_status,
            "new_status": new_status.value,
            "reason": reason,
            "acknowledged": False,
            "message": self._get_warning_message(entry.data["option_name"], new_status)
        }

        entry.data["warnings_issued"].append(warning)
        return warning

    def _get_warning_message(self, option_name: str, status: OptionStatus) -> str:
        """Generate warning message for option status."""
        messages = {
            OptionStatus.NARROWING: f"Option '{option_name}' is narrowing. Constraints are developing.",
            OptionStatus.AT_RISK: f"WARNING: Option '{option_name}' is at serious risk of closure. Take action to preserve if valued.",
            OptionStatus.CLOSED: f"Option '{option_name}' has closed. Consider if this was intentional."
        }
        return messages.get(status, f"Option '{option_name}' status changed.")

    def add_narrowing_factor(
        self,
        option_id: str,
        factor: str,
        impact: int = 5,
        source: Optional[str] = None
    ) -> bool:
        """
        Add a factor that's causing an option to narrow.

        Args:
            option_id: ID of the option
            factor: Description of the narrowing factor
            impact: Severity of impact (1-10)
            source: Source of the narrowing (decision, external, etc.)

        Returns:
            Success boolean
        """
        entry = self.get(option_id)
        if not entry or entry.id == "correlations":
            return False

        entry.data["narrowing_factors"].append({
            "factor": factor,
            "impact": max(1, min(10, impact)),
            "source": source,
            "added_at": datetime.now().isoformat()
        })

        # Auto-update status if significant narrowing
        total_impact = sum(f["impact"] for f in entry.data["narrowing_factors"])
        current_status = entry.data["status"]

        if total_impact >= 20 and current_status == OptionStatus.OPEN.value:
            self.update_status(option_id, OptionStatus.AT_RISK, "Accumulated narrowing factors")
        elif total_impact >= 10 and current_status == OptionStatus.OPEN.value:
            self.update_status(option_id, OptionStatus.NARROWING, "Multiple narrowing factors")

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def record_closure(
        self,
        option_id: str,
        closure_reason: ClosureReason,
        notes: Optional[str] = None,
        correlated_decision_id: Optional[str] = None
    ) -> bool:
        """
        Record the closure of an option.

        Args:
            option_id: ID of the option
            closure_reason: Reason for closure
            notes: Additional notes
            correlated_decision_id: Decision that caused closure

        Returns:
            Success boolean
        """
        entry = self.get(option_id)
        if not entry or entry.id == "correlations":
            return False

        result = self.update_status(
            option_id,
            OptionStatus.CLOSED,
            reason=closure_reason.value,
            notes=notes
        )

        if result.get("success") and correlated_decision_id:
            self.record_decision_correlation(option_id, correlated_decision_id, "closure")

        return result.get("success", False)

    def record_decision_correlation(
        self,
        option_id: str,
        decision_id: str,
        impact_type: str = "affected",
        notes: Optional[str] = None
    ) -> bool:
        """
        Record correlation between a decision and an option.

        Args:
            option_id: ID of the option
            decision_id: ID of the decision (from RegretMemory or external)
            impact_type: Type of impact (affected/narrowed/closed/opened)
            notes: Notes about the correlation

        Returns:
            Success boolean
        """
        entry = self.get(option_id)
        correlations = self.get("correlations")

        if not entry or entry.id == "correlations" or not correlations:
            return False

        correlation = {
            "decision_id": decision_id,
            "impact_type": impact_type,
            "timestamp": datetime.now().isoformat(),
            "notes": notes
        }

        # Add to option's correlations
        entry.data["decision_correlations"].append(correlation)

        # Update correlations index
        if decision_id not in correlations.data["decision_to_options"]:
            correlations.data["decision_to_options"][decision_id] = []
        correlations.data["decision_to_options"][decision_id].append({
            "option_id": option_id,
            "impact_type": impact_type
        })

        if option_id not in correlations.data["option_to_decisions"]:
            correlations.data["option_to_decisions"][option_id] = []
        correlations.data["option_to_decisions"][option_id].append({
            "decision_id": decision_id,
            "impact_type": impact_type
        })

        self._save()
        return True

    def get_at_risk_options(self) -> List[Dict[str, Any]]:
        """Get all options that are at risk or narrowing."""
        results = []
        at_risk_statuses = [OptionStatus.NARROWING.value, OptionStatus.AT_RISK.value]

        for entry in self._entries.values():
            if entry.id == "correlations":
                continue

            if entry.data.get("status") in at_risk_statuses:
                results.append({
                    "option_id": entry.id,
                    **entry.data
                })

        # Sort by value score (most valuable at risk first)
        results.sort(key=lambda x: x.get("value_score", 0), reverse=True)
        return results

    def get_options_by_category(
        self,
        category: OptionCategory,
        include_closed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all options in a specific category.

        Args:
            category: Category to filter by
            include_closed: Whether to include closed options

        Returns:
            List of options
        """
        results = []
        for entry in self._entries.values():
            if entry.id == "correlations":
                continue

            if entry.data.get("category") != category.value:
                continue

            if not include_closed and entry.data.get("status") == OptionStatus.CLOSED.value:
                continue

            results.append({
                "option_id": entry.id,
                **entry.data
            })

        return results

    def get_options_by_status(
        self,
        status: OptionStatus
    ) -> List[Dict[str, Any]]:
        """
        Get all options with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of options
        """
        results = []
        for entry in self._entries.values():
            if entry.id == "correlations":
                continue

            if entry.data.get("status") == status.value:
                results.append({
                    "option_id": entry.id,
                    **entry.data
                })

        return results

    def get_closed_options(
        self,
        reason_filter: Optional[ClosureReason] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all closed options.

        Args:
            reason_filter: Filter by closure reason

        Returns:
            List of closed options
        """
        results = []
        for entry in self._entries.values():
            if entry.id == "correlations":
                continue

            if entry.data.get("status") != OptionStatus.CLOSED.value:
                continue

            if reason_filter and entry.data.get("closure_reason") != reason_filter.value:
                continue

            results.append({
                "option_id": entry.id,
                **entry.data
            })

        # Sort by closure date descending
        results.sort(key=lambda x: x.get("closure_date", ""), reverse=True)
        return results

    def get_expiring_soon(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get options expiring within N days.

        Args:
            days: Number of days to look ahead

        Returns:
            List of expiring options
        """
        from datetime import timedelta
        cutoff = (datetime.now() + timedelta(days=days)).isoformat()
        now = datetime.now().isoformat()

        results = []
        for entry in self._entries.values():
            if entry.id == "correlations":
                continue

            exp_date = entry.data.get("expiration_date")
            if exp_date and now <= exp_date <= cutoff:
                results.append({
                    "option_id": entry.id,
                    **entry.data
                })

        # Sort by expiration date
        results.sort(key=lambda x: x.get("expiration_date", ""))
        return results

    def get_high_value_at_risk(self, min_value: int = 7) -> List[Dict[str, Any]]:
        """
        Get high-value options that are at risk.

        Args:
            min_value: Minimum value score to consider high-value

        Returns:
            List of high-value at-risk options
        """
        at_risk = self.get_at_risk_options()
        return [o for o in at_risk if o.get("value_score", 0) >= min_value]

    def get_decisions_affecting_option(
        self,
        option_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all decisions that affected an option.

        Args:
            option_id: ID of the option

        Returns:
            List of decision correlations
        """
        entry = self.get(option_id)
        if not entry or entry.id == "correlations":
            return []

        return entry.data.get("decision_correlations", [])

    def get_options_affected_by_decision(
        self,
        decision_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all options affected by a specific decision.

        Args:
            decision_id: ID of the decision

        Returns:
            List of option impacts
        """
        correlations = self.get("correlations")
        if not correlations:
            return []

        impacts = correlations.data["decision_to_options"].get(decision_id, [])

        results = []
        for impact in impacts:
            entry = self.get(impact["option_id"])
            if entry:
                results.append({
                    "option_id": impact["option_id"],
                    "impact_type": impact["impact_type"],
                    **entry.data
                })

        return results

    def attempt_recovery(
        self,
        option_id: str,
        recovery_action: str,
        success: bool,
        notes: Optional[str] = None
    ) -> bool:
        """
        Record a recovery attempt for a closed option.

        Args:
            option_id: ID of the option
            recovery_action: What was attempted
            success: Whether it succeeded
            notes: Additional notes

        Returns:
            Success boolean
        """
        entry = self.get(option_id)
        if not entry or entry.id == "correlations":
            return False

        recovery = {
            "action": recovery_action,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "notes": notes
        }

        entry.data["recovery_attempts"].append(recovery)

        if success:
            self.update_status(option_id, OptionStatus.RECOVERED, "Recovery successful", notes)

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def get_optionality_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of optionality.

        Returns:
            Summary dictionary
        """
        all_options = [e for e in self._entries.values() if e.id != "correlations"]

        # Count by status
        by_status = {}
        for entry in all_options:
            status = entry.data.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1

        # Count by category
        by_category = {}
        for entry in all_options:
            cat = entry.data.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1

        # Calculate optionality score (weighted average of open options)
        total_value = sum(
            entry.data.get("value_score", 0)
            for entry in all_options
            if entry.data.get("status") in [OptionStatus.OPEN.value, OptionStatus.RECOVERED.value]
        )
        max_possible = len(all_options) * 10

        optionality_score = (total_value / max_possible * 100) if max_possible > 0 else 0

        # Recently closed
        closed = [e for e in all_options if e.data.get("status") == OptionStatus.CLOSED.value]
        closed.sort(key=lambda x: x.data.get("closure_date", ""), reverse=True)

        # Closure reasons distribution
        closure_reasons = {}
        for entry in closed:
            reason = entry.data.get("closure_reason", "unknown")
            closure_reasons[reason] = closure_reasons.get(reason, 0) + 1

        return {
            "total_options": len(all_options),
            "by_status": by_status,
            "by_category": by_category,
            "open_count": by_status.get(OptionStatus.OPEN.value, 0),
            "at_risk_count": by_status.get(OptionStatus.AT_RISK.value, 0) + by_status.get(OptionStatus.NARROWING.value, 0),
            "closed_count": by_status.get(OptionStatus.CLOSED.value, 0),
            "optionality_score": round(optionality_score, 1),
            "closure_reasons": closure_reasons,
            "recently_closed": [
                {"name": e.data.get("option_name"), "reason": e.data.get("closure_reason")}
                for e in closed[:5]
            ],
            "high_value_at_risk": len(self.get_high_value_at_risk()),
            "expiring_soon": len(self.get_expiring_soon(30))
        }

    def get_all_options(self) -> List[Dict[str, Any]]:
        """Get all registered options."""
        results = []
        for entry in self._entries.values():
            if entry.id == "correlations":
                continue
            results.append({
                "option_id": entry.id,
                **entry.data
            })

        return results

    def assess_decision_impact(
        self,
        decision_description: str,
        category_hint: Optional[OptionCategory] = None
    ) -> Dict[str, Any]:
        """
        Assess potential impact of a decision on open options.

        Args:
            decision_description: Description of the pending decision
            category_hint: Category most likely to be affected

        Returns:
            Impact assessment
        """
        keywords = decision_description.lower().split()

        potentially_affected = []
        for entry in self._entries.values():
            if entry.id == "correlations":
                continue

            if entry.data.get("status") == OptionStatus.CLOSED.value:
                continue

            relevance_score = 0

            # Category match
            if category_hint and entry.data.get("category") == category_hint.value:
                relevance_score += 3

            # Keyword matching
            option_text = (
                entry.data.get("option_name", "").lower() + " " +
                entry.data.get("description", "").lower()
            )
            for keyword in keywords:
                if keyword in option_text:
                    relevance_score += 1

            if relevance_score > 0:
                potentially_affected.append({
                    "option_id": entry.id,
                    "option_name": entry.data.get("option_name"),
                    "current_status": entry.data.get("status"),
                    "value_score": entry.data.get("value_score"),
                    "reversibility": entry.data.get("reversibility"),
                    "relevance_score": relevance_score
                })

        potentially_affected.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {
            "decision": decision_description,
            "potentially_affected_options": potentially_affected[:10],
            "high_value_at_risk": [
                o for o in potentially_affected
                if o.get("value_score", 0) >= 7
            ],
            "low_reversibility_risk": [
                o for o in potentially_affected
                if o.get("reversibility", 10) <= 3
            ],
            "recommendation": (
                "Proceed with caution - high-value, low-reversibility options may be affected"
                if any(o.get("value_score", 0) >= 7 and o.get("reversibility", 10) <= 3 for o in potentially_affected)
                else "Consider impact on identified options before deciding"
            )
        }
