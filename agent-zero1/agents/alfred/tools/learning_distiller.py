"""
Learning Distiller Agent - Implicit Question Extraction Pipeline

Role: Extract implicit learning questions from conversations and stuck points.

DOES:
- Monitor for learning signals ("I don't know how to...", confusion markers)
- Extract implicit questions from conversations
- Link questions to pending execution tasks
- Flag curiosity-driven questions as potential avoidance
- Categorize questions by type and priority

DOES NOT:
- Answer the questions (that's the Scout + Curator's job)
- Recommend resources directly
- Generate questions without execution link

Question Types:
- BLOCKER: Directly blocking current execution
- OPTIMIZATION: Would improve quality/speed of execution
- CURIOSITY: Interesting but not execution-linked (flag as potential avoidance)
- GAP: Missing knowledge needed for planned work
- DECISION: Need information to make execution decision
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import json

from . import LearningAgent, AgentResponse, AlfredState


class QuestionType(Enum):
    """Types of learning questions by their relationship to execution."""
    BLOCKER = "blocker"           # Directly blocking current work
    OPTIMIZATION = "optimization" # Would improve quality/speed
    CURIOSITY = "curiosity"       # Interesting but not execution-linked
    GAP = "gap"                   # Missing knowledge for planned work
    DECISION = "decision"         # Need info to make execution decision


class SignalType(Enum):
    """Types of learning signals detected in conversation."""
    EXPLICIT_UNKNOWN = "explicit_unknown"     # "I don't know..."
    CONFUSION = "confusion"                    # "I'm confused about..."
    STUCK = "stuck"                           # "I'm stuck on..."
    QUESTION = "question"                     # Direct questions
    UNCERTAINTY = "uncertainty"               # "I'm not sure..."
    BLOCKED = "blocked"                       # "I can't proceed because..."
    RESEARCH_NEEDED = "research_needed"       # "I need to research..."
    CURIOSITY = "curiosity"                   # "I wonder...", "What if..."


class Priority(Enum):
    """Priority levels for learning questions."""
    CRITICAL = "critical"         # Blocking active work, needs immediate attention
    HIGH = "high"                 # Needed soon for upcoming work
    MEDIUM = "medium"             # Would help but not blocking
    LOW = "low"                   # Nice to know
    FLAGGED = "flagged"           # Curiosity - potential avoidance behavior


@dataclass
class ExecutionContext:
    """Context about current execution state."""
    active_tasks: List[str] = field(default_factory=list)
    pending_outputs: List[Dict[str, Any]] = field(default_factory=list)
    deadlines: Dict[str, datetime] = field(default_factory=dict)
    current_blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_tasks": self.active_tasks,
            "pending_outputs": self.pending_outputs,
            "deadlines": {k: v.isoformat() for k, v in self.deadlines.items()},
            "current_blockers": self.current_blockers
        }


@dataclass
class LearningSignal:
    """A detected learning signal from conversation."""
    signal_type: SignalType
    raw_text: str
    confidence: float              # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""               # Where detected (conversation, log, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


@dataclass
class LinkedExecution:
    """An execution task/output that a question is linked to."""
    task_id: Optional[str] = None
    task_description: str = ""
    output_type: str = ""          # blog_post, video, implementation, etc.
    deadline: Optional[datetime] = None
    is_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "output_type": self.output_type,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "is_active": self.is_active
        }


@dataclass
class LearningQuestion:
    """An extracted learning question."""
    question: str
    question_type: QuestionType
    priority: Priority
    signal: LearningSignal
    linked_execution: Optional[LinkedExecution] = None
    is_curiosity_flag: bool = False  # Flagged as potential avoidance
    avoidance_risk: str = ""         # Explanation if flagged
    extracted_at: datetime = field(default_factory=datetime.now)
    context: str = ""                # Additional context
    related_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "question_type": self.question_type.value,
            "priority": self.priority.value,
            "signal": self.signal.to_dict(),
            "linked_execution": self.linked_execution.to_dict() if self.linked_execution else None,
            "is_curiosity_flag": self.is_curiosity_flag,
            "avoidance_risk": self.avoidance_risk,
            "extracted_at": self.extracted_at.isoformat(),
            "context": self.context,
            "related_questions": self.related_questions
        }


@dataclass
class DistillerOutput:
    """Output from the Learning Distiller."""
    questions: List[LearningQuestion] = field(default_factory=list)
    curiosity_flags: List[LearningQuestion] = field(default_factory=list)
    signals_detected: int = 0
    questions_extracted: int = 0
    questions_without_link: int = 0  # Rejected questions
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        # Separate priority questions from curiosity flags
        priority_questions = [q for q in self.questions if not q.is_curiosity_flag]

        return {
            "LEARNING_QUESTIONS": {
                "priority_questions": [q.to_dict() for q in priority_questions],
                "curiosity_flags": [q.to_dict() for q in self.curiosity_flags],
                "summary": {
                    "total_signals_detected": self.signals_detected,
                    "questions_extracted": self.questions_extracted,
                    "priority_count": len(priority_questions),
                    "curiosity_count": len(self.curiosity_flags),
                    "rejected_no_link": self.questions_without_link,
                    "by_type": self._count_by_type(priority_questions),
                    "by_priority": self._count_by_priority(priority_questions)
                }
            },
            "generated_at": self.generated_at.isoformat()
        }

    def _count_by_type(self, questions: List[LearningQuestion]) -> Dict[str, int]:
        counts = {}
        for q in questions:
            key = q.question_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_by_priority(self, questions: List[LearningQuestion]) -> Dict[str, int]:
        counts = {}
        for q in questions:
            key = q.priority.value
            counts[key] = counts.get(key, 0) + 1
        return counts


class LearningDistiller(LearningAgent):
    """
    Learning Distiller - Implicit Question Extraction

    Monitors conversations and execution context for learning signals,
    extracts implicit questions, and links them to execution tasks.

    Key Principle: Questions without execution links are flagged as
    potential avoidance behavior (curiosity over shipping).
    """

    # Patterns to detect learning signals
    SIGNAL_PATTERNS = {
        SignalType.EXPLICIT_UNKNOWN: [
            r"i don'?t know (?:how to|what|why|when|where|if)",
            r"i'?m not sure (?:how to|what|why|when|where|if)",
            r"i have no idea (?:how to|what|why)",
            r"no clue (?:how to|what|why)",
        ],
        SignalType.CONFUSION: [
            r"i'?m confused (?:about|by)",
            r"(?:this|that|it) (?:is|seems) confusing",
            r"i don'?t understand",
            r"(?:what|how) does .+ (?:mean|work)",
        ],
        SignalType.STUCK: [
            r"i'?m stuck (?:on|with|at)",
            r"(?:can'?t|cannot) (?:figure out|get past|move forward)",
            r"hit a (?:wall|block|roadblock)",
            r"(?:spinning|going in) circles",
        ],
        SignalType.BLOCKED: [
            r"(?:can'?t|cannot) proceed (?:because|until|without)",
            r"blocked (?:by|on|because)",
            r"waiting (?:on|for) .+ (?:to|before)",
            r"need .+ (?:before|to) (?:proceed|continue)",
        ],
        SignalType.RESEARCH_NEEDED: [
            r"need to (?:research|look into|investigate)",
            r"should (?:research|look into|investigate)",
            r"have to (?:find out|learn|figure out)",
            r"need (?:more|some) information (?:about|on)",
        ],
        SignalType.UNCERTAINTY: [
            r"i'?m not (?:certain|sure) (?:if|whether|about)",
            r"(?:might|may|could) be (?:wrong|mistaken)",
            r"(?:unclear|unsure) (?:if|whether|about)",
        ],
        SignalType.CURIOSITY: [
            r"i wonder (?:if|what|how|why|whether)",
            r"(?:would be|it'?d be) (?:interesting|cool|nice) to (?:know|learn|understand)",
            r"(?:curious|wondering) (?:about|if|whether)",
            r"what if",
            r"(?:just|purely) (?:curious|wondering)",
        ],
        SignalType.QUESTION: [
            r"^(?:how|what|why|when|where|who|which|can|could|should|would|is|are|do|does) .+\?$",
        ],
    }

    # Avoidance risk indicators
    AVOIDANCE_INDICATORS = [
        "just curious",
        "purely academic",
        "for fun",
        "interesting tangent",
        "rabbit hole",
        "side quest",
        "not urgent",
        "someday",
        "when I have time",
    ]

    def __init__(self):
        super().__init__("LearningDistiller")
        self._execution_context: Optional[ExecutionContext] = None
        self._detected_signals: List[LearningSignal] = []
        self._extracted_questions: List[LearningQuestion] = []

    def check_state_permission(self) -> tuple[bool, str]:
        """Learning distiller is paused in RED state."""
        if self.alfred_state == AlfredState.RED:
            return False, "Learning distillation paused in RED state"
        return True, "Operation permitted"

    def set_execution_context(self, context: ExecutionContext) -> None:
        """Set the current execution context for linking questions."""
        self._execution_context = context

    def detect_signals(
        self,
        text: str,
        source: str = "conversation"
    ) -> List[LearningSignal]:
        """
        Detect learning signals in text.

        Args:
            text: Text to analyze for learning signals
            source: Source of the text (conversation, log, etc.)

        Returns:
            List of detected LearningSignal objects
        """
        signals = []
        text_lower = text.lower()

        for signal_type, patterns in self.SIGNAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_signal_confidence(
                        signal_type, match.group(), text
                    )

                    signal = LearningSignal(
                        signal_type=signal_type,
                        raw_text=match.group(),
                        confidence=confidence,
                        source=source
                    )
                    signals.append(signal)

        self._detected_signals.extend(signals)
        return signals

    def _calculate_signal_confidence(
        self,
        signal_type: SignalType,
        matched_text: str,
        full_text: str
    ) -> float:
        """Calculate confidence score for a detected signal."""
        confidence = 0.7  # Base confidence

        # Boost for explicit signals
        if signal_type in [SignalType.EXPLICIT_UNKNOWN, SignalType.BLOCKED]:
            confidence += 0.2

        # Reduce for curiosity (might be avoidance)
        if signal_type == SignalType.CURIOSITY:
            confidence -= 0.1

        # Boost for longer/more specific matches
        if len(matched_text) > 20:
            confidence += 0.1

        return min(1.0, max(0.0, confidence))

    def extract_question(
        self,
        signal: LearningSignal,
        surrounding_context: str = ""
    ) -> Optional[LearningQuestion]:
        """
        Extract a learning question from a signal.

        Args:
            signal: The detected learning signal
            surrounding_context: Additional context around the signal

        Returns:
            LearningQuestion or None if extraction fails
        """
        # Formulate the question from the signal
        question_text = self._formulate_question(signal, surrounding_context)
        if not question_text:
            return None

        # Determine question type
        question_type = self._determine_question_type(signal, surrounding_context)

        # Try to link to execution
        linked_execution = self._find_execution_link(
            question_text, surrounding_context
        )

        # Determine if this is a curiosity flag (potential avoidance)
        is_curiosity, avoidance_risk = self._check_avoidance_risk(
            signal, question_text, linked_execution, surrounding_context
        )

        # Determine priority
        priority = self._determine_priority(
            question_type, linked_execution, is_curiosity
        )

        question = LearningQuestion(
            question=question_text,
            question_type=question_type,
            priority=priority,
            signal=signal,
            linked_execution=linked_execution,
            is_curiosity_flag=is_curiosity,
            avoidance_risk=avoidance_risk,
            context=surrounding_context
        )

        self._extracted_questions.append(question)
        return question

    def _formulate_question(
        self,
        signal: LearningSignal,
        context: str
    ) -> Optional[str]:
        """Convert a signal into a clear question."""
        raw = signal.raw_text

        # If already a question, clean it up
        if signal.signal_type == SignalType.QUESTION:
            return raw.strip().capitalize()

        # Convert "I don't know how to X" -> "How do I X?"
        patterns = [
            (r"i don'?t know how to (.+)", r"How do I \1?"),
            (r"i don'?t know what (.+)", r"What is \1?"),
            (r"i don'?t know why (.+)", r"Why does \1?"),
            (r"i'?m not sure how to (.+)", r"How do I \1?"),
            (r"i'?m confused about (.+)", r"How does \1 work?"),
            (r"i'?m stuck on (.+)", r"How do I solve \1?"),
            (r"can'?t figure out (.+)", r"How do I figure out \1?"),
            (r"need to research (.+)", r"What do I need to know about \1?"),
            (r"i wonder (?:if|what|how|why) (.+)", r"What about \1?"),
        ]

        raw_lower = raw.lower()
        for pattern, replacement in patterns:
            match = re.match(pattern, raw_lower)
            if match:
                result = re.sub(pattern, replacement, raw_lower)
                return result.strip().capitalize()

        # Fallback: wrap in "How do I understand X?"
        return f"What do I need to know about: {raw}?"

    def _determine_question_type(
        self,
        signal: LearningSignal,
        context: str
    ) -> QuestionType:
        """Determine the type of learning question."""
        if signal.signal_type == SignalType.BLOCKED:
            return QuestionType.BLOCKER

        if signal.signal_type == SignalType.STUCK:
            return QuestionType.BLOCKER

        if signal.signal_type == SignalType.CURIOSITY:
            return QuestionType.CURIOSITY

        if signal.signal_type == SignalType.RESEARCH_NEEDED:
            return QuestionType.GAP

        # Check context for decision markers
        decision_keywords = ["decide", "choose", "select", "option", "alternative"]
        if any(kw in context.lower() for kw in decision_keywords):
            return QuestionType.DECISION

        # Check for optimization context
        opt_keywords = ["better", "improve", "optimize", "faster", "easier"]
        if any(kw in context.lower() for kw in opt_keywords):
            return QuestionType.OPTIMIZATION

        return QuestionType.GAP

    def _find_execution_link(
        self,
        question: str,
        context: str
    ) -> Optional[LinkedExecution]:
        """Try to link question to an execution task/output."""
        if not self._execution_context:
            return None

        combined_text = f"{question} {context}".lower()

        # Check active tasks
        for task in self._execution_context.active_tasks:
            task_words = set(task.lower().split())
            question_words = set(combined_text.split())
            overlap = len(task_words & question_words)
            if overlap >= 2:  # Reasonable overlap
                deadline = self._execution_context.deadlines.get(task)
                return LinkedExecution(
                    task_description=task,
                    deadline=deadline,
                    is_active=True
                )

        # Check pending outputs
        for output in self._execution_context.pending_outputs:
            output_title = output.get("title", "").lower()
            output_type = output.get("type", "")
            if any(word in combined_text for word in output_title.split()):
                return LinkedExecution(
                    task_id=output.get("id"),
                    task_description=output_title,
                    output_type=output_type,
                    deadline=output.get("deadline"),
                    is_active=False
                )

        return None

    def _check_avoidance_risk(
        self,
        signal: LearningSignal,
        question: str,
        linked_execution: Optional[LinkedExecution],
        context: str
    ) -> tuple[bool, str]:
        """Check if this question might be avoidance behavior."""
        is_curiosity = False
        avoidance_risk = ""

        # Curiosity signal type is a flag
        if signal.signal_type == SignalType.CURIOSITY:
            is_curiosity = True
            avoidance_risk = "Question originated from curiosity signal, not execution need"

        # No execution link is a flag
        if not linked_execution:
            is_curiosity = True
            avoidance_risk = "No linked execution task - question not tied to shipping"

        # Check for avoidance indicators
        combined = f"{question} {context}".lower()
        for indicator in self.AVOIDANCE_INDICATORS:
            if indicator in combined:
                is_curiosity = True
                avoidance_risk = f"Contains avoidance indicator: '{indicator}'"
                break

        return is_curiosity, avoidance_risk

    def _determine_priority(
        self,
        question_type: QuestionType,
        linked_execution: Optional[LinkedExecution],
        is_curiosity: bool
    ) -> Priority:
        """Determine question priority."""
        if is_curiosity:
            return Priority.FLAGGED

        if question_type == QuestionType.BLOCKER:
            return Priority.CRITICAL

        if linked_execution and linked_execution.is_active:
            return Priority.HIGH

        if question_type == QuestionType.DECISION:
            return Priority.HIGH

        if question_type == QuestionType.GAP:
            return Priority.MEDIUM

        if question_type == QuestionType.OPTIMIZATION:
            return Priority.MEDIUM

        return Priority.LOW

    def distill_from_conversation(
        self,
        conversation: List[Dict[str, str]],
        execution_context: Optional[ExecutionContext] = None
    ) -> AgentResponse:
        """
        Extract learning questions from a conversation.

        Args:
            conversation: List of message dicts with 'role' and 'content'
            execution_context: Current execution context for linking

        Returns:
            AgentResponse with LEARNING_QUESTIONS
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        if execution_context:
            self._execution_context = execution_context

        all_signals = []
        all_questions = []
        curiosity_flags = []
        rejected_count = 0

        for message in conversation:
            content = message.get("content", "")
            role = message.get("role", "user")

            # Detect signals
            signals = self.detect_signals(content, source=f"conversation:{role}")
            all_signals.extend(signals)

            # Extract questions from signals
            for signal in signals:
                question = self.extract_question(signal, content)
                if question:
                    if question.is_curiosity_flag:
                        curiosity_flags.append(question)
                    elif question.linked_execution:
                        all_questions.append(question)
                    else:
                        # Question without link - count as rejected
                        rejected_count += 1
                        curiosity_flags.append(question)

        # Sort questions by priority
        priority_order = [
            Priority.CRITICAL,
            Priority.HIGH,
            Priority.MEDIUM,
            Priority.LOW,
            Priority.FLAGGED
        ]
        all_questions.sort(key=lambda q: priority_order.index(q.priority))

        output = DistillerOutput(
            questions=all_questions,
            curiosity_flags=curiosity_flags,
            signals_detected=len(all_signals),
            questions_extracted=len(all_questions) + len(curiosity_flags),
            questions_without_link=rejected_count
        )

        warnings = []
        if curiosity_flags:
            warnings.append(
                f"{len(curiosity_flags)} questions flagged as potential avoidance - "
                "consider if learning serves shipping"
            )

        return self.create_response(
            data=output.to_dict(),
            warnings=warnings
        )

    def distill_from_stuck_point(
        self,
        stuck_description: str,
        current_task: str,
        attempted_solutions: List[str] = None
    ) -> AgentResponse:
        """
        Extract learning questions from a stuck point.

        Args:
            stuck_description: Description of what you're stuck on
            current_task: The task being worked on
            attempted_solutions: What has already been tried

        Returns:
            AgentResponse with LEARNING_QUESTIONS
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Create execution link
        linked_execution = LinkedExecution(
            task_description=current_task,
            is_active=True
        )

        # Detect signals in stuck description
        signals = self.detect_signals(stuck_description, source="stuck_point")

        # Create a blocker signal if none detected
        if not signals:
            signals = [
                LearningSignal(
                    signal_type=SignalType.STUCK,
                    raw_text=stuck_description,
                    confidence=0.9,
                    source="stuck_point"
                )
            ]

        questions = []
        for signal in signals:
            question = self.extract_question(signal, stuck_description)
            if question:
                # Force link to current task
                question.linked_execution = linked_execution
                question.question_type = QuestionType.BLOCKER
                question.priority = Priority.CRITICAL
                question.is_curiosity_flag = False
                questions.append(question)

        # Add questions about failed solutions
        if attempted_solutions:
            for solution in attempted_solutions:
                question = LearningQuestion(
                    question=f"Why didn't '{solution}' work for {current_task}?",
                    question_type=QuestionType.BLOCKER,
                    priority=Priority.HIGH,
                    signal=LearningSignal(
                        signal_type=SignalType.STUCK,
                        raw_text=solution,
                        confidence=0.8,
                        source="attempted_solution"
                    ),
                    linked_execution=linked_execution,
                    context=f"Attempted solution: {solution}"
                )
                questions.append(question)

        output = DistillerOutput(
            questions=questions,
            curiosity_flags=[],
            signals_detected=len(signals),
            questions_extracted=len(questions),
            questions_without_link=0
        )

        return self.create_response(
            data=output.to_dict()
        )

    def get_blocker_questions(self) -> List[LearningQuestion]:
        """Get all questions that are blocking execution."""
        return [
            q for q in self._extracted_questions
            if q.question_type == QuestionType.BLOCKER
        ]

    def get_curiosity_flags(self) -> List[LearningQuestion]:
        """Get all curiosity-flagged questions (potential avoidance)."""
        return [
            q for q in self._extracted_questions
            if q.is_curiosity_flag
        ]

    def clear_extracted_questions(self) -> None:
        """Clear the extracted questions cache."""
        self._extracted_questions = []
        self._detected_signals = []

    def generate_learning_needs_summary(self) -> Dict[str, Any]:
        """Generate a summary of current learning needs."""
        blocker_count = sum(
            1 for q in self._extracted_questions
            if q.question_type == QuestionType.BLOCKER
        )
        curiosity_count = sum(
            1 for q in self._extracted_questions
            if q.is_curiosity_flag
        )
        linked_count = sum(
            1 for q in self._extracted_questions
            if q.linked_execution is not None
        )

        return {
            "total_questions": len(self._extracted_questions),
            "blocker_questions": blocker_count,
            "curiosity_flags": curiosity_count,
            "execution_linked": linked_count,
            "signals_detected": len(self._detected_signals),
            "health_check": {
                "learning_serves_shipping": linked_count > curiosity_count,
                "blockers_identified": blocker_count > 0,
                "avoidance_risk": curiosity_count > linked_count
            }
        }
