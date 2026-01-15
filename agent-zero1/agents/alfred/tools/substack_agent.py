"""
Substack Agent - Authority Builder

Long-form content generation for intellectual authority building.
Creates thoughtful, evidence-based articles that establish credibility.

DOES:
- Explain mechanisms clearly
- Cite evidence properly
- State uncertainty honestly
- Build arguments methodically
- Provide nuanced analysis

DOES NOT:
- Chase virality
- Dramatize for effect
- Debate personalities
- Write clickbait
- Use outrage framing
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from . import ContentAgent, AgentResponse, AlfredState


class ContentTone(Enum):
    """Acceptable tones for Substack content."""
    EDUCATIONAL = "educational"
    ANALYTICAL = "analytical"
    EXPLANATORY = "explanatory"
    REFLECTIVE = "reflective"


class EvidenceStrength(Enum):
    """Strength classification for cited evidence."""
    STRONG = "strong"           # RCTs, meta-analyses, systematic reviews
    MODERATE = "moderate"       # Observational studies, cohort studies
    LIMITED = "limited"         # Case studies, expert opinion
    THEORETICAL = "theoretical" # Mechanistic reasoning, hypothesis


@dataclass
class EvidenceCitation:
    """A citation with source and strength assessment."""
    claim: str
    source: str
    source_type: str
    strength: EvidenceStrength
    notes: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "source": self.source,
            "source_type": self.source_type,
            "strength": self.strength.value,
            "notes": self.notes,
            "url": self.url
        }


@dataclass
class UncertaintyDisclosure:
    """Explicit statement of uncertainty or limitation."""
    topic: str
    uncertainty_type: str  # "data_gap", "conflicting_evidence", "evolving_research", "personal_limitation"
    disclosure_text: str
    confidence_level: float  # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "uncertainty_type": self.uncertainty_type,
            "disclosure_text": self.disclosure_text,
            "confidence_level": self.confidence_level
        }


@dataclass
class EvidenceGap:
    """Identified gap in available evidence."""
    topic: str
    gap_description: str
    what_would_help: str
    impact_on_conclusions: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "gap_description": self.gap_description,
            "what_would_help": self.what_would_help,
            "impact_on_conclusions": self.impact_on_conclusions
        }


@dataclass
class ContentSection:
    """A section of the long-form content."""
    heading: str
    content: str
    citations: List[EvidenceCitation] = field(default_factory=list)
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "heading": self.heading,
            "content": self.content,
            "citations": [c.to_dict() for c in self.citations],
            "word_count": self.word_count
        }


@dataclass
class QualityGateResult:
    """Result of quality gate checks."""
    passed: bool
    gate_name: str
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gate_name": self.gate_name,
            "violations": self.violations,
            "warnings": self.warnings
        }


@dataclass
class LongformDraft:
    """Complete long-form content draft output."""
    title: str
    subtitle: str
    sections: List[ContentSection]
    uncertainty_disclosures: List[UncertaintyDisclosure]
    evidence_gaps: List[EvidenceGap]
    all_citations: List[EvidenceCitation]
    tone: ContentTone
    total_word_count: int
    estimated_read_time_minutes: int
    quality_gates_passed: List[QualityGateResult]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    draft_status: str = "pending_review"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_type": "LONGFORM_DRAFT",
            "title": self.title,
            "subtitle": self.subtitle,
            "sections": [s.to_dict() for s in self.sections],
            "uncertainty_disclosures": [u.to_dict() for u in self.uncertainty_disclosures],
            "evidence_gaps": [g.to_dict() for g in self.evidence_gaps],
            "all_citations": [c.to_dict() for c in self.all_citations],
            "tone": self.tone.value,
            "total_word_count": self.total_word_count,
            "estimated_read_time_minutes": self.estimated_read_time_minutes,
            "quality_gates_passed": [q.to_dict() for q in self.quality_gates_passed],
            "created_at": self.created_at,
            "draft_status": self.draft_status
        }


class SubstackAgent(ContentAgent):
    """
    Authority Builder - Long-form content for intellectual authority.

    Creates evidence-based, methodically argued content that builds
    credibility through transparency about uncertainty and limitations.
    """

    # Patterns that indicate problematic content
    SHOCK_HOOK_PATTERNS = [
        r'\b(shocking|unbelievable|mind-blowing|explosive|bombshell)\b',
        r'\b(you won\'t believe|this will shock you|jaw-dropping)\b',
        r'^(BREAKING|URGENT|WARNING)[:,!]',
        r'\b(they don\'t want you to know|the truth they\'re hiding)\b',
        r'^\d+\s+(reasons|ways|secrets|things)\s+.*\b(will|that)\b',  # Listicle clickbait
    ]

    OUTRAGE_PATTERNS = [
        r'\b(outrageous|disgraceful|shameful|disgusting)\b',
        r'\b(idiots?|morons?|fools?|clowns?)\b',
        r'\b(destroyed|demolished|eviscerated|obliterated)\b',  # Combat framing
        r'\b(slams?|blasts?|attacks?|rips?)\b\s+\w+',  # Conflict language
        r'\b(the left|the right|liberals|conservatives)\s+(always|never|are)\b',
    ]

    PERSONALITY_DEBATE_PATTERNS = [
        r'@\w+\s+(is wrong|doesn\'t understand|fails to)',
        r'\b(Dr\.|Mr\.|Ms\.)\s+\w+\s+(is|was)\s+(wrong|incompetent|dishonest)',
        r'\b(my critics|the haters|the mob)\b',
    ]

    CLICKBAIT_PATTERNS = [
        r'\b(secret|hidden|banned|forbidden|suppressed)\b',
        r'\b(miracle|revolutionary|game-?changer|breakthrough)\b',
        r'\b(this one trick|what happened next|number \d+ will)\b',
        r'[?!]{2,}',  # Excessive punctuation
    ]

    # Words requiring evidence
    CLAIM_TRIGGER_WORDS = [
        'proven', 'shows', 'demonstrates', 'evidence', 'studies',
        'research', 'data', 'statistics', 'according to', 'found that',
        'linked to', 'causes', 'prevents', 'treats', 'cures'
    ]

    def __init__(self):
        super().__init__(name="SubstackAgent")
        self.words_per_minute_reading = 200  # Average reading speed

    def check_shock_hooks(self, content: str) -> QualityGateResult:
        """Check for shock-value hooks and sensationalism."""
        violations = []
        warnings = []

        for pattern in self.SHOCK_HOOK_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                violations.append(f"Shock hook pattern detected: {matches[:3]}")

        # Check title specifically (first line often)
        lines = content.split('\n')
        if lines:
            title_line = lines[0]
            if title_line.isupper() and len(title_line) > 10:
                warnings.append("Title appears to be all caps - may seem aggressive")
            if title_line.count('!') > 1:
                warnings.append("Multiple exclamation marks in title")

        return QualityGateResult(
            passed=len(violations) == 0,
            gate_name="no_shock_hooks",
            violations=violations,
            warnings=warnings
        )

    def check_outrage_framing(self, content: str) -> QualityGateResult:
        """Check for outrage-based framing."""
        violations = []
        warnings = []

        for pattern in self.OUTRAGE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"Outrage framing detected: {matches[:3]}")

        # Check emotional temperature
        negative_markers = len(re.findall(r'\b(terrible|horrible|awful|worst)\b', content, re.IGNORECASE))
        if negative_markers > 3:
            warnings.append(f"High negative emotional language count: {negative_markers}")

        return QualityGateResult(
            passed=len(violations) == 0,
            gate_name="no_outrage_framing",
            violations=violations,
            warnings=warnings
        )

    def check_personality_debates(self, content: str) -> QualityGateResult:
        """Check for personality-based debates rather than idea-based."""
        violations = []
        warnings = []

        for pattern in self.PERSONALITY_DEBATE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"Personality debate detected: {matches[:2]}")

        # Check for excessive name mentions with criticism
        name_criticism = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+\s+(is wrong|fails|doesn\'t)', content)
        if len(name_criticism) > 2:
            warnings.append("Multiple personal criticisms detected - consider focusing on ideas")

        return QualityGateResult(
            passed=len(violations) == 0,
            gate_name="no_personality_debates",
            violations=violations,
            warnings=warnings
        )

    def check_clickbait(self, content: str) -> QualityGateResult:
        """Check for clickbait language and formatting."""
        violations = []
        warnings = []

        for pattern in self.CLICKBAIT_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"Clickbait pattern detected: {matches[:3]}")

        return QualityGateResult(
            passed=len(violations) == 0,
            gate_name="no_clickbait",
            violations=violations,
            warnings=warnings
        )

    def check_claims_supported(self, content: str, citations: List[EvidenceCitation]) -> QualityGateResult:
        """Check that claims have supporting evidence."""
        violations = []
        warnings = []

        # Find sentences with claim trigger words
        sentences = re.split(r'[.!?]', content)
        claim_sentences = []

        for sentence in sentences:
            for trigger in self.CLAIM_TRIGGER_WORDS:
                if trigger.lower() in sentence.lower():
                    claim_sentences.append(sentence.strip())
                    break

        # Check if we have citations for claims
        cited_claims = {c.claim.lower() for c in citations}

        unsupported_claims = []
        for claim_sentence in claim_sentences:
            # Check if any citation relates to this sentence
            is_supported = any(
                claim.lower() in claim_sentence.lower() or claim_sentence.lower() in claim.lower()
                for claim in cited_claims
            )
            if not is_supported and len(claim_sentence) > 20:
                unsupported_claims.append(claim_sentence[:80] + "...")

        if len(unsupported_claims) > len(citations) // 2:
            violations.append(f"Multiple unsupported claims detected: {len(unsupported_claims)}")
            for claim in unsupported_claims[:3]:
                warnings.append(f"Potentially unsupported: {claim}")
        elif unsupported_claims:
            for claim in unsupported_claims[:3]:
                warnings.append(f"Consider adding citation: {claim}")

        return QualityGateResult(
            passed=len(violations) == 0,
            gate_name="claims_supported",
            violations=violations,
            warnings=warnings
        )

    def check_uncertainty_stated(self, content: str, disclosures: List[UncertaintyDisclosure]) -> QualityGateResult:
        """Check that uncertainty is properly acknowledged."""
        violations = []
        warnings = []

        # Look for absolute certainty markers
        certainty_markers = re.findall(
            r'\b(definitely|certainly|undoubtedly|clearly proves|without question|absolute(ly)?)\b',
            content,
            re.IGNORECASE
        )

        if certainty_markers and not disclosures:
            violations.append(f"Absolute certainty expressed without uncertainty disclosures: {certainty_markers[:3]}")
        elif len(certainty_markers) > len(disclosures) * 2:
            warnings.append("High certainty language relative to uncertainty disclosures")

        # Check for hedging language (good sign)
        hedging = re.findall(
            r'\b(may|might|could|possibly|potentially|suggests|appears to|seems to)\b',
            content,
            re.IGNORECASE
        )

        if not hedging and len(content.split()) > 500:
            warnings.append("No hedging language found in long content - consider adding nuance")

        return QualityGateResult(
            passed=len(violations) == 0,
            gate_name="uncertainty_stated",
            violations=violations,
            warnings=warnings
        )

    def run_all_quality_gates(self, content: str,
                              citations: List[EvidenceCitation],
                              disclosures: List[UncertaintyDisclosure]) -> List[QualityGateResult]:
        """Run all quality gate checks."""
        return [
            self.check_shock_hooks(content),
            self.check_outrage_framing(content),
            self.check_personality_debates(content),
            self.check_clickbait(content),
            self.check_claims_supported(content, citations),
            self.check_uncertainty_stated(content, disclosures)
        ]

    def calculate_read_time(self, word_count: int) -> int:
        """Calculate estimated reading time in minutes."""
        return max(1, round(word_count / self.words_per_minute_reading))

    def validate_title(self, title: str) -> tuple[bool, List[str]]:
        """Validate title for appropriateness."""
        issues = []

        if len(title) > 100:
            issues.append("Title too long (max 100 characters)")
        if title.isupper():
            issues.append("Title should not be all caps")
        if re.search(r'[!?]{2,}', title):
            issues.append("Avoid excessive punctuation in title")

        # Check for clickbait patterns in title
        for pattern in self.CLICKBAIT_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                issues.append(f"Title contains clickbait pattern")
                break

        return len(issues) == 0, issues

    def create_draft(self,
                     title: str,
                     subtitle: str,
                     sections: List[ContentSection],
                     uncertainty_disclosures: List[UncertaintyDisclosure],
                     evidence_gaps: List[EvidenceGap],
                     tone: ContentTone = ContentTone.EDUCATIONAL) -> AgentResponse:
        """
        Create a long-form content draft.

        Args:
            title: Article title (must pass validation)
            subtitle: Article subtitle
            sections: List of content sections with citations
            uncertainty_disclosures: Explicit uncertainty statements
            evidence_gaps: Identified gaps in evidence
            tone: Content tone (must be appropriate)

        Returns:
            AgentResponse with LONGFORM_DRAFT or blocked status
        """
        # Check state permission first
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        errors = []
        warnings = []

        # Validate title
        title_valid, title_issues = self.validate_title(title)
        if not title_valid:
            errors.extend([f"Title issue: {issue}" for issue in title_issues])

        # Aggregate all citations
        all_citations = []
        for section in sections:
            all_citations.extend(section.citations)

        # Aggregate all content for quality checks
        full_content = title + "\n" + subtitle + "\n"
        full_content += "\n\n".join([s.heading + "\n" + s.content for s in sections])

        # Run quality gates
        quality_results = self.run_all_quality_gates(
            full_content,
            all_citations,
            uncertainty_disclosures
        )

        # Check for gate failures
        failed_gates = [g for g in quality_results if not g.passed]
        if failed_gates:
            for gate in failed_gates:
                errors.extend([f"Quality gate '{gate.gate_name}' failed: {v}" for v in gate.violations])

        # Collect warnings from all gates
        for gate in quality_results:
            warnings.extend(gate.warnings)

        # Calculate totals
        total_word_count = sum(s.word_count for s in sections)
        read_time = self.calculate_read_time(total_word_count)

        # Check for evidence gap acknowledgment
        if not evidence_gaps and len(all_citations) > 0:
            warnings.append("No evidence gaps identified - consider what limitations exist")

        # Check for uncertainty disclosure
        if not uncertainty_disclosures:
            errors.append("No uncertainty disclosures provided - must acknowledge limitations")

        # If there are errors, return failure
        if errors:
            return self.create_response(
                data={
                    "output_type": "LONGFORM_DRAFT",
                    "draft_status": "quality_gate_failed",
                    "title": title,
                    "quality_gates": [g.to_dict() for g in quality_results],
                    "failed_gates": [g.gate_name for g in failed_gates]
                },
                success=False,
                errors=errors,
                warnings=warnings
            )

        # Create successful draft
        draft = LongformDraft(
            title=title,
            subtitle=subtitle,
            sections=sections,
            uncertainty_disclosures=uncertainty_disclosures,
            evidence_gaps=evidence_gaps,
            all_citations=all_citations,
            tone=tone,
            total_word_count=total_word_count,
            estimated_read_time_minutes=read_time,
            quality_gates_passed=quality_results,
            draft_status="ready_for_review"
        )

        return self.create_response(
            data=draft.to_dict(),
            success=True,
            warnings=warnings
        )

    def analyze_topic(self, topic: str, existing_research: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Analyze a topic for long-form content potential.

        Args:
            topic: The topic to analyze
            existing_research: Optional research data from Research Agent

        Returns:
            Analysis of topic suitability and structure suggestions
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        analysis = {
            "topic": topic,
            "suitability_score": 0.0,
            "suggested_angles": [],
            "required_evidence_types": [],
            "potential_uncertainty_areas": [],
            "recommended_structure": [],
            "warnings": []
        }

        # Basic topic analysis
        if existing_research:
            analysis["has_research_base"] = True
            analysis["suitability_score"] = 0.7

            # Check research quality
            if existing_research.get("citations"):
                analysis["suitability_score"] += 0.2
                analysis["required_evidence_types"] = [
                    "primary_sources",
                    "peer_reviewed",
                    "expert_opinion"
                ]
        else:
            analysis["has_research_base"] = False
            analysis["suitability_score"] = 0.3
            analysis["warnings"].append("No research base provided - will need significant research")

        # Suggest structure based on topic type
        analysis["recommended_structure"] = [
            {"section": "Introduction", "purpose": "Frame the question/problem"},
            {"section": "Background", "purpose": "Provide necessary context"},
            {"section": "Key Evidence", "purpose": "Present main findings"},
            {"section": "Analysis", "purpose": "Interpret the evidence"},
            {"section": "Limitations", "purpose": "Acknowledge gaps and uncertainty"},
            {"section": "Implications", "purpose": "Discuss what this means"},
            {"section": "Conclusion", "purpose": "Summarize key takeaways"}
        ]

        return self.create_response(data=analysis)

    def validate_content(self, content: str) -> AgentResponse:
        """
        Validate content against quality gates without creating a draft.

        Args:
            content: Raw content to validate

        Returns:
            Validation results with specific issues identified
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Run all pattern-based checks
        results = {
            "shock_hooks": self.check_shock_hooks(content).to_dict(),
            "outrage_framing": self.check_outrage_framing(content).to_dict(),
            "personality_debates": self.check_personality_debates(content).to_dict(),
            "clickbait": self.check_clickbait(content).to_dict(),
        }

        all_passed = all(r["passed"] for r in results.values())
        total_violations = sum(len(r["violations"]) for r in results.values())
        total_warnings = sum(len(r["warnings"]) for r in results.values())

        return self.create_response(
            data={
                "validation_results": results,
                "all_passed": all_passed,
                "total_violations": total_violations,
                "total_warnings": total_warnings,
                "word_count": len(content.split()),
                "estimated_read_time": self.calculate_read_time(len(content.split()))
            },
            success=all_passed
        )
