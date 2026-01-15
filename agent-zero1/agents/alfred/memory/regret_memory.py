# Regret Memory System
# Tracks decisions, outcomes, and lessons learned from regrets

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

from . import BaseMemorySystem, MemoryType, MemoryEntry


class DecisionDomain(Enum):
    """Domains of decisions."""
    CAREER = "career"
    RELATIONSHIP = "relationship"
    FINANCIAL = "financial"
    HEALTH = "health"
    PERSONAL = "personal"
    CREATIVE = "creative"
    SOCIAL = "social"
    EDUCATIONAL = "educational"
    ETHICAL = "ethical"


class OutcomeType(Enum):
    """Types of outcomes."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class RegretIntensity(Enum):
    """Intensity of regret expressed."""
    MILD = "mild"               # Minor disappointment
    MODERATE = "moderate"       # Notable regret
    SIGNIFICANT = "significant"  # Strong regret
    PROFOUND = "profound"       # Deep, lasting regret


class RegretMemory(BaseMemorySystem):
    """
    Tracks decisions, their outcomes, regrets expressed, and lessons learned.

    The purpose is not to dwell on regrets but to:
    1. Extract actionable lessons
    2. Identify decision patterns that lead to regret
    3. Recognize when similar decision points arise
    4. Provide wisdom when facing analogous situations

    Schema per decision:
    - decision_id: Unique identifier
    - decision: Description of the decision made
    - domain: Domain of the decision
    - context: Situation when decision was made
    - alternatives_considered: What other options existed
    - expected_outcome: What was expected to happen
    - actual_outcome: What actually happened
    - outcome_type: Positive/negative/mixed
    - regret_expressed: Whether regret was expressed
    - regret_intensity: How strong the regret was
    - regret_statement: What was said about the regret
    - lesson_extracted: The lesson learned
    - applied_lessons: Times this lesson was applied
    """

    def __init__(self, storage_path=None):
        super().__init__(MemoryType.REGRET, storage_path)
        # Initialize lessons index
        if "lessons_index" not in [e.id for e in self._entries.values()]:
            self.add("lessons_index", {"lessons": {}, "lesson_applications": []})

    def record_decision(
        self,
        decision: str,
        domain: DecisionDomain,
        context: str,
        expected_outcome: str,
        alternatives_considered: Optional[List[str]] = None,
        reasoning: Optional[str] = None
    ) -> str:
        """
        Record a decision that was made.

        Args:
            decision: Description of the decision
            domain: Domain of the decision
            context: Situational context
            expected_outcome: What was expected
            alternatives_considered: Other options that existed
            reasoning: Why this choice was made

        Returns:
            decision_id: Unique identifier for this decision
        """
        decision_id = f"decision_{uuid.uuid4().hex[:12]}"

        decision_data = {
            "decision": decision,
            "domain": domain.value,
            "context": context,
            "expected_outcome": expected_outcome,
            "alternatives_considered": alternatives_considered or [],
            "reasoning": reasoning,
            "made_at": datetime.now().isoformat(),
            "actual_outcome": None,
            "outcome_type": OutcomeType.UNKNOWN.value,
            "outcome_recorded_at": None,
            "regret_expressed": False,
            "regret_intensity": None,
            "regret_statement": None,
            "regret_recorded_at": None,
            "lesson_extracted": None,
            "lesson_id": None,
            "tags": [],
            "similar_decisions": []
        }

        self.add(decision_id, decision_data)

        # Find similar past decisions
        self._link_similar_decisions(decision_id)

        return decision_id

    def _link_similar_decisions(self, decision_id: str):
        """Find and link similar past decisions."""
        entry = self.get(decision_id)
        if not entry:
            return

        domain = entry.data.get("domain")
        decision_text = entry.data.get("decision", "").lower()

        similar = []
        for e in self._entries.values():
            if e.id in [decision_id, "lessons_index"]:
                continue

            # Check for domain match
            if e.data.get("domain") == domain:
                # Simple keyword matching
                other_text = e.data.get("decision", "").lower()
                common_words = set(decision_text.split()) & set(other_text.split())
                if len(common_words) >= 2:
                    similar.append({
                        "decision_id": e.id,
                        "decision": e.data.get("decision"),
                        "outcome_type": e.data.get("outcome_type"),
                        "had_regret": e.data.get("regret_expressed")
                    })

        if similar:
            entry.data["similar_decisions"] = similar[:5]  # Keep top 5
            self._save()

    def record_outcome(
        self,
        decision_id: str,
        actual_outcome: str,
        outcome_type: OutcomeType
    ) -> bool:
        """
        Record the actual outcome of a decision.

        Args:
            decision_id: ID of the decision
            actual_outcome: What actually happened
            outcome_type: Type of outcome

        Returns:
            Success boolean
        """
        entry = self.get(decision_id)
        if not entry or entry.id == "lessons_index":
            return False

        entry.data["actual_outcome"] = actual_outcome
        entry.data["outcome_type"] = outcome_type.value
        entry.data["outcome_recorded_at"] = datetime.now().isoformat()

        # Calculate outcome deviation
        entry.data["outcome_matched_expectation"] = (
            outcome_type in [OutcomeType.POSITIVE, OutcomeType.NEUTRAL]
        )

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def record_regret(
        self,
        decision_id: str,
        regret_statement: str,
        intensity: RegretIntensity = RegretIntensity.MODERATE,
        what_would_do_differently: Optional[str] = None
    ) -> bool:
        """
        Record regret expressed about a decision.

        Args:
            decision_id: ID of the decision
            regret_statement: What was said about the regret
            intensity: How intense the regret is
            what_would_do_differently: Alternative the user wishes they'd chosen

        Returns:
            Success boolean
        """
        entry = self.get(decision_id)
        if not entry or entry.id == "lessons_index":
            return False

        entry.data["regret_expressed"] = True
        entry.data["regret_statement"] = regret_statement
        entry.data["regret_intensity"] = intensity.value
        entry.data["regret_recorded_at"] = datetime.now().isoformat()

        if what_would_do_differently:
            entry.data["what_would_do_differently"] = what_would_do_differently

        entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def extract_lesson(
        self,
        decision_id: str,
        lesson: str,
        applicability: Optional[str] = None,
        confidence: float = 0.7
    ) -> str:
        """
        Extract a lesson from a decision/regret.

        Args:
            decision_id: ID of the decision
            lesson: The lesson learned
            applicability: When this lesson applies
            confidence: How confident we are in this lesson (0-1)

        Returns:
            lesson_id: Unique identifier for this lesson
        """
        entry = self.get(decision_id)
        if not entry or entry.id == "lessons_index":
            return None

        lessons_entry = self.get("lessons_index")
        if not lessons_entry:
            self.add("lessons_index", {"lessons": {}, "lesson_applications": []})
            lessons_entry = self.get("lessons_index")

        lesson_id = f"lesson_{uuid.uuid4().hex[:8]}"

        lesson_data = {
            "lesson": lesson,
            "source_decision_id": decision_id,
            "source_domain": entry.data.get("domain"),
            "applicability": applicability,
            "confidence": max(0, min(1, confidence)),
            "extracted_at": datetime.now().isoformat(),
            "times_applied": 0,
            "applications": [],
            "reinforced": False,
            "reinforcement_count": 0
        }

        lessons_entry.data["lessons"][lesson_id] = lesson_data
        lessons_entry.updated_at = datetime.now().isoformat()

        # Update decision entry
        entry.data["lesson_extracted"] = lesson
        entry.data["lesson_id"] = lesson_id
        entry.updated_at = datetime.now().isoformat()

        self._save()
        return lesson_id

    def apply_lesson(
        self,
        lesson_id: str,
        application_context: str,
        outcome: Optional[str] = None
    ) -> bool:
        """
        Record that a lesson was applied in a new situation.

        Args:
            lesson_id: ID of the lesson
            application_context: Context where it was applied
            outcome: Result of applying the lesson

        Returns:
            Success boolean
        """
        lessons_entry = self.get("lessons_index")
        if not lessons_entry or lesson_id not in lessons_entry.data["lessons"]:
            return False

        lesson = lessons_entry.data["lessons"][lesson_id]
        lesson["times_applied"] += 1
        lesson["applications"].append({
            "context": application_context,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        })

        # Update confidence based on applications
        if outcome and "positive" in outcome.lower():
            lesson["confidence"] = min(1.0, lesson["confidence"] + 0.1)
            lesson["reinforced"] = True
            lesson["reinforcement_count"] += 1

        lessons_entry.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def get_lessons_for_similar(
        self,
        domain: DecisionDomain,
        context_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get lessons relevant to a similar situation.

        Args:
            domain: Domain of the current decision
            context_keywords: Keywords describing the situation

        Returns:
            List of relevant lessons
        """
        lessons_entry = self.get("lessons_index")
        if not lessons_entry:
            return []

        results = []
        keywords = [k.lower() for k in (context_keywords or [])]

        for lesson_id, lesson in lessons_entry.data["lessons"].items():
            relevance_score = 0

            # Domain match
            if lesson.get("source_domain") == domain.value:
                relevance_score += 3

            # Keyword matching
            lesson_text = (
                lesson.get("lesson", "").lower() + " " +
                lesson.get("applicability", "").lower()
            )
            for keyword in keywords:
                if keyword in lesson_text:
                    relevance_score += 1

            # Confidence and reinforcement boost
            relevance_score += lesson.get("confidence", 0) * 2
            if lesson.get("reinforced"):
                relevance_score += 1

            if relevance_score > 0:
                results.append({
                    "lesson_id": lesson_id,
                    "relevance_score": relevance_score,
                    **lesson
                })

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:10]  # Return top 10

    def get_decisions_with_regret(
        self,
        domain: Optional[DecisionDomain] = None,
        min_intensity: Optional[RegretIntensity] = None
    ) -> List[Dict[str, Any]]:
        """
        Get decisions where regret was expressed.

        Args:
            domain: Filter by domain
            min_intensity: Minimum regret intensity

        Returns:
            List of decisions with regret
        """
        intensity_order = {
            RegretIntensity.MILD.value: 1,
            RegretIntensity.MODERATE.value: 2,
            RegretIntensity.SIGNIFICANT.value: 3,
            RegretIntensity.PROFOUND.value: 4
        }

        min_level = intensity_order.get(min_intensity.value, 0) if min_intensity else 0

        results = []
        for entry in self._entries.values():
            if entry.id == "lessons_index":
                continue

            if not entry.data.get("regret_expressed"):
                continue

            # Domain filter
            if domain and entry.data.get("domain") != domain.value:
                continue

            # Intensity filter
            entry_intensity = intensity_order.get(entry.data.get("regret_intensity"), 0)
            if entry_intensity < min_level:
                continue

            results.append({
                "decision_id": entry.id,
                **entry.data
            })

        # Sort by intensity then by date
        results.sort(
            key=lambda x: (
                intensity_order.get(x.get("regret_intensity"), 0),
                x.get("regret_recorded_at", "")
            ),
            reverse=True
        )
        return results

    def get_decisions_by_domain(
        self,
        domain: DecisionDomain,
        include_outcomes: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all decisions in a specific domain.

        Args:
            domain: Domain to filter by
            include_outcomes: Only include decisions with recorded outcomes

        Returns:
            List of decisions
        """
        results = []
        for entry in self._entries.values():
            if entry.id == "lessons_index":
                continue

            if entry.data.get("domain") != domain.value:
                continue

            if include_outcomes and not entry.data.get("actual_outcome"):
                continue

            results.append({
                "decision_id": entry.id,
                **entry.data
            })

        results.sort(key=lambda x: x.get("made_at", ""), reverse=True)
        return results

    def get_all_lessons(self, min_confidence: float = 0) -> List[Dict[str, Any]]:
        """
        Get all extracted lessons.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of lessons
        """
        lessons_entry = self.get("lessons_index")
        if not lessons_entry:
            return []

        results = []
        for lesson_id, lesson in lessons_entry.data["lessons"].items():
            if lesson.get("confidence", 0) >= min_confidence:
                results.append({
                    "lesson_id": lesson_id,
                    **lesson
                })

        # Sort by confidence then by times applied
        results.sort(
            key=lambda x: (x.get("confidence", 0), x.get("times_applied", 0)),
            reverse=True
        )
        return results

    def get_decision_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in decisions and regrets.

        Returns:
            Analysis of decision patterns
        """
        decisions = []
        for entry in self._entries.values():
            if entry.id != "lessons_index":
                decisions.append(entry.data)

        # Count by domain
        by_domain = {}
        regrets_by_domain = {}
        for d in decisions:
            domain = d.get("domain", "unknown")
            by_domain[domain] = by_domain.get(domain, 0) + 1
            if d.get("regret_expressed"):
                regrets_by_domain[domain] = regrets_by_domain.get(domain, 0) + 1

        # Calculate regret rate by domain
        regret_rates = {}
        for domain, count in by_domain.items():
            regrets = regrets_by_domain.get(domain, 0)
            regret_rates[domain] = round(regrets / count * 100, 1) if count > 0 else 0

        # Outcome analysis
        outcomes = {}
        for d in decisions:
            ot = d.get("outcome_type", "unknown")
            outcomes[ot] = outcomes.get(ot, 0) + 1

        # Find riskiest domain (highest regret rate)
        riskiest_domain = max(regret_rates.items(), key=lambda x: x[1])[0] if regret_rates else None

        return {
            "total_decisions": len(decisions),
            "decisions_by_domain": by_domain,
            "regrets_by_domain": regrets_by_domain,
            "regret_rates_by_domain": regret_rates,
            "riskiest_domain": riskiest_domain,
            "outcome_distribution": outcomes,
            "total_regrets": len([d for d in decisions if d.get("regret_expressed")]),
            "lessons_extracted": len([d for d in decisions if d.get("lesson_extracted")])
        }

    def get_regret_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of regrets and lessons.

        Returns:
            Summary dictionary
        """
        lessons_entry = self.get("lessons_index")
        lessons = lessons_entry.data["lessons"] if lessons_entry else {}

        patterns = self.get_decision_patterns()

        # Most applied lessons
        top_lessons = sorted(
            lessons.items(),
            key=lambda x: x[1].get("times_applied", 0),
            reverse=True
        )[:5]

        # Highest confidence lessons
        confident_lessons = sorted(
            lessons.items(),
            key=lambda x: x[1].get("confidence", 0),
            reverse=True
        )[:5]

        return {
            **patterns,
            "total_lessons": len(lessons),
            "reinforced_lessons": len([l for l in lessons.values() if l.get("reinforced")]),
            "total_lesson_applications": sum(l.get("times_applied", 0) for l in lessons.values()),
            "most_applied_lessons": [
                {"lesson": l[1]["lesson"], "times_applied": l[1]["times_applied"]}
                for l in top_lessons
            ],
            "highest_confidence_lessons": [
                {"lesson": l[1]["lesson"], "confidence": l[1]["confidence"]}
                for l in confident_lessons
            ]
        }

    def get_wisdom_for_decision(
        self,
        domain: DecisionDomain,
        decision_description: str
    ) -> Dict[str, Any]:
        """
        Get accumulated wisdom relevant to a pending decision.

        Args:
            domain: Domain of the decision
            decision_description: Description of what's being decided

        Returns:
            Wisdom package with lessons and warnings
        """
        # Get relevant lessons
        keywords = decision_description.lower().split()
        lessons = self.get_lessons_for_similar(domain, keywords)

        # Get past regrets in this domain
        regrets = self.get_decisions_with_regret(domain)

        # Get similar past decisions
        similar_decisions = []
        for entry in self._entries.values():
            if entry.id == "lessons_index":
                continue
            if entry.data.get("domain") == domain.value:
                decision_text = entry.data.get("decision", "").lower()
                if any(kw in decision_text for kw in keywords):
                    similar_decisions.append({
                        "decision_id": entry.id,
                        "decision": entry.data.get("decision"),
                        "outcome": entry.data.get("actual_outcome"),
                        "outcome_type": entry.data.get("outcome_type"),
                        "had_regret": entry.data.get("regret_expressed"),
                        "lesson": entry.data.get("lesson_extracted")
                    })

        return {
            "domain": domain.value,
            "relevant_lessons": lessons[:5],
            "similar_past_decisions": similar_decisions[:5],
            "past_regrets_in_domain": len(regrets),
            "warnings": [
                r.get("what_would_do_differently")
                for r in regrets
                if r.get("what_would_do_differently")
            ][:3],
            "confidence_note": (
                "High confidence recommendations available" if any(l.get("confidence", 0) > 0.7 for l in lessons)
                else "Limited historical data - proceed with awareness"
            )
        }
