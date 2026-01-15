"""
YouTube Script Agent - Educator

Creates calm, evidence-anchored video scripts for educational content.
Focuses on clear teaching, tradeoff explanation, and uncertainty acknowledgment.

DOES:
- Teach clearly with structured explanations
- Explain tradeoffs and nuance
- Respect uncertainty explicitly
- Include visual suggestions
- Provide timing notes for pacing

DOES NOT:
- Write for performance/entertainment
- Provoke or create controversy
- Speculate beyond evidence
- Include hot takes
- Write original Shorts/Reels

SHORTS/REELS RULE: Only repurposed from long-form, never original.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from . import ContentAgent, AgentResponse, AlfredState


class ScriptType(Enum):
    """Types of video scripts."""
    EDUCATIONAL = "educational"      # Full educational video
    EXPLAINER = "explainer"          # Concept explanation
    CASE_STUDY = "case_study"        # Case analysis
    REVIEW = "review"                # Evidence review
    SHORT_REPURPOSED = "short_repurposed"  # Repurposed Short from long-form


class VisualType(Enum):
    """Types of visual suggestions."""
    B_ROLL = "b_roll"              # Supporting footage
    GRAPHIC = "graphic"            # Custom graphic/illustration
    CHART = "chart"                # Data visualization
    TEXT_OVERLAY = "text_overlay"  # On-screen text
    ANIMATION = "animation"        # Animated explanation
    TALKING_HEAD = "talking_head"  # Speaker on camera


@dataclass
class VisualSuggestion:
    """A visual element suggestion for the script."""
    timestamp_start: str  # "MM:SS" format
    duration_seconds: int
    visual_type: VisualType
    description: str
    purpose: str  # Why this visual helps
    priority: str = "medium"  # "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_start": self.timestamp_start,
            "duration_seconds": self.duration_seconds,
            "visual_type": self.visual_type.value,
            "description": self.description,
            "purpose": self.purpose,
            "priority": self.priority
        }


@dataclass
class UncertaintyMoment:
    """An explicit moment of uncertainty acknowledgment in the script."""
    section_name: str
    topic: str
    uncertainty_statement: str
    suggested_phrasing: str
    visual_treatment: Optional[str] = None  # How to visually represent uncertainty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_name": self.section_name,
            "topic": self.topic,
            "uncertainty_statement": self.uncertainty_statement,
            "suggested_phrasing": self.suggested_phrasing,
            "visual_treatment": self.visual_treatment
        }


@dataclass
class TimingNote:
    """Timing and pacing guidance for a script section."""
    target_duration_seconds: int
    pacing: str  # "slow", "moderate", "brisk"
    emphasis_points: List[str]  # Key moments to emphasize
    pause_suggestions: List[str]  # Where to pause for effect

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_duration_seconds": self.target_duration_seconds,
            "pacing": self.pacing,
            "emphasis_points": self.emphasis_points,
            "pause_suggestions": self.pause_suggestions
        }


@dataclass
class ScriptSection:
    """A section of the video script."""
    name: str
    order: int
    content: str  # The actual script text
    timing: TimingNote
    visuals: List[VisualSuggestion] = field(default_factory=list)
    speaker_notes: Optional[str] = None  # Notes for delivery
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "order": self.order,
            "content": self.content,
            "timing": self.timing.to_dict(),
            "visuals": [v.to_dict() for v in self.visuals],
            "speaker_notes": self.speaker_notes,
            "word_count": self.word_count
        }


@dataclass
class ContentGateResult:
    """Result of content quality gate checks."""
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
class ScriptDraft:
    """Complete video script draft output."""
    title: str
    description: str
    script_type: ScriptType
    sections: List[ScriptSection]
    uncertainty_moments: List[UncertaintyMoment]
    total_word_count: int
    estimated_duration_minutes: int
    all_visuals: List[VisualSuggestion]
    quality_gates: List[ContentGateResult]
    source_reference: Optional[str]  # For repurposed content
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    draft_status: str = "pending_review"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_type": "SCRIPT_DRAFT",
            "title": self.title,
            "description": self.description,
            "script_type": self.script_type.value,
            "sections": [s.to_dict() for s in self.sections],
            "uncertainty_moments": [u.to_dict() for u in self.uncertainty_moments],
            "total_word_count": self.total_word_count,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "all_visuals": [v.to_dict() for v in self.all_visuals],
            "quality_gates": [g.to_dict() for g in self.quality_gates],
            "source_reference": self.source_reference,
            "created_at": self.created_at,
            "draft_status": self.draft_status
        }


class YouTubeScriptAgent(ContentAgent):
    """
    Educator - Calm, evidence-anchored video scripts.

    Creates educational content that teaches clearly without
    performance-driven framing or controversy-seeking.
    """

    # Average speaking rate for video (words per minute)
    WORDS_PER_MINUTE = 150

    # Performance/entertainment patterns (NOT acceptable)
    PERFORMANCE_PATTERNS = [
        r'\b(crazy|insane|wild|mind-?blowing|epic)\b',
        r'\b(smash that|hit that|destroy that)\s+(like|subscribe)',
        r'\b(you guys|what\'s up|hey everyone)\b',  # Overly casual YouTuber speak
        r'[!]{2,}',  # Excessive exclamation
        r'\b(literally|actually|basically)\b.*\b(literally|actually|basically)\b',  # Filler overuse
    ]

    # Provocative/controversial patterns (NOT acceptable)
    PROVOCATIVE_PATTERNS = [
        r'\b(controversial|unpopular opinion|hot take)\b',
        r'\b(truth bomb|reality check|wake up)\b',
        r'\b(they|them|those people)\s+don\'t want you to\b',
        r'\b(exposed|busted|caught)\b',
        r'\b(debate|fight|battle|war)\b.*\b(against|with)\b',
    ]

    # Speculation patterns (NOT acceptable)
    SPECULATION_PATTERNS = [
        r'\b(I bet|I suspect|probably|likely|maybe)\b.*\b(will|going to)\b',
        r'\b(prediction|predict|predicting)\b',
        r'\b(in the future|soon|eventually)\b.*\b(will)\b',
        r'\b(could|might)\b.*\b(conspiracy|cover-?up|hiding)\b',
    ]

    # Hot take patterns (NOT acceptable)
    HOT_TAKE_PATTERNS = [
        r'^(Unpopular|Hot|Controversial)\s+(opinion|take):',
        r'\b(nobody talks about|everyone ignores)\b',
        r'\b(the real reason|the truth is|what they won\'t tell you)\b',
        r'\b(actually|secretly)\b.*\b(wrong|bad|terrible)\b',
    ]

    # Educational markers (good signs)
    EDUCATIONAL_PATTERNS = [
        r'\b(let\'s (explore|examine|look at|understand))\b',
        r'\b(research (shows|suggests|indicates))\b',
        r'\b(the evidence|the data|studies)\b',
        r'\b(tradeoff|nuance|complexity|context)\b',
        r'\b(on one hand|on the other|however|that said)\b',
    ]

    def __init__(self):
        super().__init__(name="YouTubeScriptAgent")

    def estimate_duration(self, word_count: int) -> int:
        """Estimate video duration in minutes from word count."""
        return max(1, round(word_count / self.WORDS_PER_MINUTE))

    def check_no_performance_writing(self, content: str) -> ContentGateResult:
        """Check that content isn't written for performance/entertainment."""
        violations = []
        warnings = []

        for pattern in self.PERFORMANCE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"Performance language detected: {matches[:3]}")

        # Check for excessive caps (shouting)
        caps_words = re.findall(r'\b[A-Z]{4,}\b', content)
        caps_words = [w for w in caps_words if w not in ['THIS', 'THAT', 'THESE', 'THOSE']]
        if len(caps_words) > 2:
            warnings.append(f"Excessive caps detected (shouting): {caps_words[:3]}")

        return ContentGateResult(
            passed=len(violations) == 0,
            gate_name="no_performance_writing",
            violations=violations,
            warnings=warnings
        )

    def check_no_provocation(self, content: str) -> ContentGateResult:
        """Check that content doesn't provoke or seek controversy."""
        violations = []
        warnings = []

        for pattern in self.PROVOCATIVE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"Provocative content detected: {matches[:3]}")

        # Check for conflict framing
        conflict_words = re.findall(r'\b(versus|vs\.?|against|fight)\b', content, re.IGNORECASE)
        if len(conflict_words) > 2:
            warnings.append("High conflict framing detected - consider more neutral framing")

        return ContentGateResult(
            passed=len(violations) == 0,
            gate_name="no_provocation",
            violations=violations,
            warnings=warnings
        )

    def check_no_speculation(self, content: str) -> ContentGateResult:
        """Check that content doesn't speculate beyond evidence."""
        violations = []
        warnings = []

        for pattern in self.SPECULATION_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"Speculation detected: {matches[:2]}")

        # Check for future predictions
        future_predictions = re.findall(
            r'\b(will|going to)\b.*\b(happen|occur|change|become)\b',
            content,
            re.IGNORECASE
        )
        if len(future_predictions) > 2:
            warnings.append("Multiple future predictions - ensure these are evidence-based")

        return ContentGateResult(
            passed=len(violations) == 0,
            gate_name="no_speculation",
            violations=violations,
            warnings=warnings
        )

    def check_no_hot_takes(self, content: str) -> ContentGateResult:
        """Check that content doesn't include hot takes."""
        violations = []
        warnings = []

        for pattern in self.HOT_TAKE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                violations.append(f"Hot take detected: {matches[:2]}")

        return ContentGateResult(
            passed=len(violations) == 0,
            gate_name="no_hot_takes",
            violations=violations,
            warnings=warnings
        )

    def check_educational_quality(self, content: str) -> ContentGateResult:
        """Check that content maintains educational quality."""
        violations = []
        warnings = []

        # Count educational markers
        educational_count = 0
        for pattern in self.EDUCATIONAL_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            educational_count += len(matches)

        word_count = len(content.split())

        # Expect at least some educational markers for longer content
        if word_count > 500 and educational_count < 3:
            warnings.append("Low educational marker density - consider adding more explanatory framing")

        if word_count > 1000 and educational_count < 5:
            violations.append("Insufficient educational framing for content length")

        return ContentGateResult(
            passed=len(violations) == 0,
            gate_name="educational_quality",
            violations=violations,
            warnings=warnings
        )

    def run_all_quality_gates(self, content: str) -> List[ContentGateResult]:
        """Run all content quality gates."""
        return [
            self.check_no_performance_writing(content),
            self.check_no_provocation(content),
            self.check_no_speculation(content),
            self.check_no_hot_takes(content),
            self.check_educational_quality(content)
        ]

    def validate_short_is_repurposed(self,
                                     short_content: str,
                                     source_reference: Optional[str]) -> tuple[bool, List[str]]:
        """
        Validate that a Short/Reel is repurposed from long-form, not original.

        SHORTS/REELS RULE: Only repurposed from long-form, never original.
        """
        errors = []

        if not source_reference:
            errors.append("BLOCKED: Short must reference source long-form content")
            errors.append("Original Shorts are not permitted - only repurposed content")

        # Check length (Shorts should be under 60 seconds, roughly 150 words)
        word_count = len(short_content.split())
        if word_count > 180:  # Allow some buffer
            errors.append(f"Content too long for Short: {word_count} words (max ~150)")

        return len(errors) == 0, errors

    def create_script(self,
                      title: str,
                      description: str,
                      sections: List[ScriptSection],
                      uncertainty_moments: List[UncertaintyMoment],
                      script_type: ScriptType = ScriptType.EDUCATIONAL,
                      source_reference: Optional[str] = None) -> AgentResponse:
        """
        Create a video script draft.

        Args:
            title: Video title
            description: Video description for YouTube
            sections: List of script sections with timing and visuals
            uncertainty_moments: Explicit uncertainty acknowledgments
            script_type: Type of script being created
            source_reference: For repurposed Shorts, the source URL

        Returns:
            AgentResponse with SCRIPT_DRAFT or blocked status
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        errors = []
        warnings = []

        # SHORTS/REELS RULE: Check if Short is repurposed
        if script_type == ScriptType.SHORT_REPURPOSED:
            full_content = " ".join([s.content for s in sections])
            valid, short_errors = self.validate_short_is_repurposed(full_content, source_reference)
            if not valid:
                errors.extend(short_errors)

        # Aggregate all content for quality checks
        full_content = title + "\n" + description + "\n"
        full_content += "\n\n".join([s.content for s in sections])

        # Run quality gates
        quality_results = self.run_all_quality_gates(full_content)

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
        estimated_duration = self.estimate_duration(total_word_count)

        # Aggregate all visuals
        all_visuals = []
        for section in sections:
            all_visuals.extend(section.visuals)

        # Check uncertainty moments
        if not uncertainty_moments and script_type != ScriptType.SHORT_REPURPOSED:
            errors.append("No uncertainty moments defined - must acknowledge limitations")

        # Check for visual suggestions in longer content
        if estimated_duration > 3 and len(all_visuals) < 3:
            warnings.append("Few visual suggestions for video length - consider adding more")

        # If there are errors, return failure
        if errors:
            return self.create_response(
                data={
                    "output_type": "SCRIPT_DRAFT",
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
        draft = ScriptDraft(
            title=title,
            description=description,
            script_type=script_type,
            sections=sections,
            uncertainty_moments=uncertainty_moments,
            total_word_count=total_word_count,
            estimated_duration_minutes=estimated_duration,
            all_visuals=all_visuals,
            quality_gates=quality_results,
            source_reference=source_reference,
            draft_status="ready_for_review"
        )

        return self.create_response(
            data=draft.to_dict(),
            success=True,
            warnings=warnings
        )

    def create_standard_sections(self,
                                 hook: str,
                                 introduction: str,
                                 main_points: List[Dict[str, str]],
                                 uncertainty_section: str,
                                 conclusion: str,
                                 call_to_action: str) -> List[ScriptSection]:
        """
        Create a standard educational video structure.

        Args:
            hook: Opening hook (15-30 seconds)
            introduction: Topic introduction (30-60 seconds)
            main_points: List of {"title": str, "content": str} for main sections
            uncertainty_section: Explicit limitations discussion
            conclusion: Summary and takeaways
            call_to_action: Appropriate CTA (not engagement-bait)

        Returns:
            List of ScriptSection objects with timing estimates
        """
        sections = []
        order = 1

        # Hook section
        hook_word_count = len(hook.split())
        sections.append(ScriptSection(
            name="Hook",
            order=order,
            content=hook,
            timing=TimingNote(
                target_duration_seconds=max(15, hook_word_count // 2),
                pacing="moderate",
                emphasis_points=["opening question or statement"],
                pause_suggestions=["after hook, before transition"]
            ),
            speaker_notes="Deliver with calm confidence, not excitement"
        ))
        order += 1

        # Introduction
        intro_word_count = len(introduction.split())
        sections.append(ScriptSection(
            name="Introduction",
            order=order,
            content=introduction,
            timing=TimingNote(
                target_duration_seconds=max(30, intro_word_count // 2),
                pacing="moderate",
                emphasis_points=["topic statement", "why this matters"],
                pause_suggestions=["after stating the topic"]
            ),
            speaker_notes="Set context clearly, avoid hype"
        ))
        order += 1

        # Main points
        for i, point in enumerate(main_points, 1):
            point_content = point.get("content", "")
            point_word_count = len(point_content.split())
            sections.append(ScriptSection(
                name=f"Main Point {i}: {point.get('title', f'Point {i}')}",
                order=order,
                content=point_content,
                timing=TimingNote(
                    target_duration_seconds=max(60, point_word_count // 2),
                    pacing="slow" if "complex" in point.get("title", "").lower() else "moderate",
                    emphasis_points=[point.get("title", f"Point {i}")],
                    pause_suggestions=["after key evidence", "before transition"]
                ),
                speaker_notes=f"Main point {i} - maintain teaching tone"
            ))
            order += 1

        # Uncertainty section
        unc_word_count = len(uncertainty_section.split())
        sections.append(ScriptSection(
            name="Limitations and Uncertainty",
            order=order,
            content=uncertainty_section,
            timing=TimingNote(
                target_duration_seconds=max(45, unc_word_count // 2),
                pacing="slow",
                emphasis_points=["what we don't know", "where evidence is limited"],
                pause_suggestions=["after each limitation"]
            ),
            speaker_notes="Deliver with honesty, not apologetically"
        ))
        order += 1

        # Conclusion
        conc_word_count = len(conclusion.split())
        sections.append(ScriptSection(
            name="Conclusion",
            order=order,
            content=conclusion,
            timing=TimingNote(
                target_duration_seconds=max(30, conc_word_count // 2),
                pacing="moderate",
                emphasis_points=["key takeaways"],
                pause_suggestions=["before final statement"]
            ),
            speaker_notes="Summarize clearly, reinforce key learning"
        ))
        order += 1

        # Call to Action (if provided)
        if call_to_action:
            cta_word_count = len(call_to_action.split())
            sections.append(ScriptSection(
                name="Call to Action",
                order=order,
                content=call_to_action,
                timing=TimingNote(
                    target_duration_seconds=max(15, cta_word_count // 2),
                    pacing="moderate",
                    emphasis_points=["what viewer should do next"],
                    pause_suggestions=[]
                ),
                speaker_notes="Keep brief, don't beg for engagement"
            ))

        return sections

    def repurpose_for_short(self,
                            source_title: str,
                            source_url: str,
                            key_insight: str,
                            supporting_context: str) -> AgentResponse:
        """
        Create a Short/Reel repurposed from existing long-form content.

        CRITICAL: This is the ONLY way to create Shorts - never original.

        Args:
            source_title: Title of the source long-form content
            source_url: URL to the source content
            key_insight: The main insight to highlight (keep brief)
            supporting_context: Brief context for the insight

        Returns:
            AgentResponse with SHORT script or blocked status
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Validate source reference exists
        if not source_url:
            return self.create_response(
                data={
                    "output_type": "SCRIPT_DRAFT",
                    "script_type": "short_repurposed",
                    "draft_status": "blocked"
                },
                success=False,
                errors=["BLOCKED: Cannot create original Shorts - must have source reference"]
            )

        # Create Short content
        short_content = f"{key_insight}\n\n{supporting_context}\n\nFull explanation in the full video."

        word_count = len(short_content.split())
        if word_count > 150:
            return self.create_response(
                data={},
                success=False,
                errors=[f"Short content too long: {word_count} words. Max ~150 for 60-second Short."]
            )

        # Create single section for Short
        section = ScriptSection(
            name="Short Content",
            order=1,
            content=short_content,
            timing=TimingNote(
                target_duration_seconds=min(60, word_count // 2),
                pacing="brisk",
                emphasis_points=[key_insight[:50] + "..."],
                pause_suggestions=[]
            ),
            speaker_notes="Quick, clear delivery - no filler"
        )

        return self.create_script(
            title=f"Short: {source_title[:50]}",
            description=f"Clip from: {source_title}\n\nFull video: {source_url}",
            sections=[section],
            uncertainty_moments=[],  # Shorts don't need separate uncertainty section
            script_type=ScriptType.SHORT_REPURPOSED,
            source_reference=source_url
        )

    def validate_script_content(self, content: str) -> AgentResponse:
        """
        Validate script content against quality gates without creating a draft.

        Args:
            content: Raw script content to validate

        Returns:
            Validation results with specific issues identified
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        quality_results = self.run_all_quality_gates(content)

        all_passed = all(r.passed for r in quality_results)
        total_violations = sum(len(r.violations) for r in quality_results)
        total_warnings = sum(len(r.warnings) for r in quality_results)

        word_count = len(content.split())

        return self.create_response(
            data={
                "validation_results": {r.gate_name: r.to_dict() for r in quality_results},
                "all_passed": all_passed,
                "total_violations": total_violations,
                "total_warnings": total_warnings,
                "word_count": word_count,
                "estimated_duration_minutes": self.estimate_duration(word_count)
            },
            success=all_passed
        )

    def suggest_visuals(self,
                        content: str,
                        script_type: ScriptType = ScriptType.EDUCATIONAL) -> AgentResponse:
        """
        Analyze content and suggest visual elements.

        Args:
            content: Script content to analyze
            script_type: Type of video for context

        Returns:
            List of visual suggestions
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        suggestions = []
        current_time = 0
        paragraphs = content.split('\n\n')

        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue

            para_words = len(para.split())
            para_duration = max(10, para_words // 2)

            # Detect content that benefits from visuals
            timestamp = f"{current_time // 60:02d}:{current_time % 60:02d}"

            # Data/statistics -> chart
            if re.search(r'\b(percent|statistics|data|numbers|study found)\b', para, re.IGNORECASE):
                suggestions.append(VisualSuggestion(
                    timestamp_start=timestamp,
                    duration_seconds=para_duration,
                    visual_type=VisualType.CHART,
                    description="Data visualization for statistics mentioned",
                    purpose="Make data tangible and memorable",
                    priority="high"
                ))

            # Process/steps -> animation
            elif re.search(r'\b(step|process|how.*works|mechanism)\b', para, re.IGNORECASE):
                suggestions.append(VisualSuggestion(
                    timestamp_start=timestamp,
                    duration_seconds=para_duration,
                    visual_type=VisualType.ANIMATION,
                    description="Animated explanation of process",
                    purpose="Visual learning for complex processes",
                    priority="high"
                ))

            # Key term definition -> text overlay
            elif re.search(r'\b(means|defined as|refers to|is called)\b', para, re.IGNORECASE):
                suggestions.append(VisualSuggestion(
                    timestamp_start=timestamp,
                    duration_seconds=10,
                    visual_type=VisualType.TEXT_OVERLAY,
                    description="Key term and definition on screen",
                    purpose="Reinforce terminology",
                    priority="medium"
                ))

            # Lists/comparisons -> graphic
            elif re.search(r'\b(first|second|third|versus|compared to|on one hand)\b', para, re.IGNORECASE):
                suggestions.append(VisualSuggestion(
                    timestamp_start=timestamp,
                    duration_seconds=para_duration,
                    visual_type=VisualType.GRAPHIC,
                    description="Comparison graphic or list visualization",
                    purpose="Organize multiple points visually",
                    priority="medium"
                ))

            # Default -> talking head with occasional B-roll
            else:
                if i % 3 == 0:  # Every third paragraph suggest B-roll
                    suggestions.append(VisualSuggestion(
                        timestamp_start=timestamp,
                        duration_seconds=min(15, para_duration),
                        visual_type=VisualType.B_ROLL,
                        description="Relevant B-roll footage",
                        purpose="Visual variety and engagement",
                        priority="low"
                    ))

            current_time += para_duration

        return self.create_response(
            data={
                "visual_suggestions": [s.to_dict() for s in suggestions],
                "total_suggestions": len(suggestions),
                "estimated_video_duration_seconds": current_time,
                "high_priority_count": len([s for s in suggestions if s.priority == "high"]),
                "medium_priority_count": len([s for s in suggestions if s.priority == "medium"]),
                "low_priority_count": len([s for s in suggestions if s.priority == "low"])
            }
        )
