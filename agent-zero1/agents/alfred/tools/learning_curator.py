"""
Learning Curator Agent - Just-in-Time Learning Pipeline

Role: Match learning to execution problems. Learning serves shipping.

DOES:
- Extract learning questions from current execution context
- Match learning opportunities to available time windows
- Require every queued item to have a linked output
- Prioritize learning that unblocks current work

DOES NOT:
- Curate by topic preference
- Recommend tools or products
- Suggest review-style content
- Queue learning without linked output
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from . import LearningAgent, AgentResponse, AlfredState


class TimeWindowType(Enum):
    """Types of time windows available for learning."""
    COMMUTE = "commute"          # 15-45 min, audio-friendly
    BREAK = "break"              # 5-15 min, quick reads
    DEEP_WORK = "deep_work"      # 60+ min, tutorials/courses
    WIND_DOWN = "wind_down"      # 20-30 min, lighter content
    WAITING = "waiting"          # Variable, interruptible content
    EXERCISE = "exercise"        # 30-60 min, podcasts/audio


class OutputType(Enum):
    """Types of linked outputs that justify learning."""
    BLOG_POST = "blog_post"
    TWEET_THREAD = "tweet_thread"
    VIDEO_SCRIPT = "video_script"
    IMPLEMENTATION = "implementation"
    PATIENT_EDUCATION = "patient_education"
    PRESENTATION = "presentation"
    NEWSLETTER = "newsletter"
    DOCUMENTATION = "documentation"


class LearningUrgency(Enum):
    """Urgency level for learning queue items."""
    BLOCKING = "blocking"        # Execution blocked until learned
    NEXT_UP = "next_up"          # Needed for upcoming work
    OPTIMIZATION = "optimization" # Would improve quality
    BACKGROUND = "background"    # Nice to know for future


@dataclass
class LinkedOutput:
    """An output that requires this learning."""
    output_type: OutputType
    title: str
    deadline: Optional[datetime] = None
    status: str = "pending"
    output_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_type": self.output_type.value,
            "title": self.title,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status,
            "output_id": self.output_id
        }


@dataclass
class TimeWindow:
    """An available time window for learning."""
    window_type: TimeWindowType
    start_time: Optional[datetime] = None
    duration_minutes: int = 30
    context: str = ""  # e.g., "morning commute", "lunch break"
    audio_only: bool = False
    interruptible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_type": self.window_type.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration_minutes": self.duration_minutes,
            "context": self.context,
            "audio_only": self.audio_only,
            "interruptible": self.interruptible
        }


@dataclass
class LearningQueueItem:
    """A curated learning item ready for consumption."""
    question: str                    # The learning question this answers
    resource_title: str              # Title of the resource
    resource_url: str                # URL or location
    resource_type: str               # video, article, podcast, etc.
    duration_minutes: int            # Estimated time to consume
    linked_output: LinkedOutput      # Required: what output this serves
    urgency: LearningUrgency = LearningUrgency.NEXT_UP
    assigned_window: Optional[TimeWindow] = None
    key_sections: List[str] = field(default_factory=list)  # Specific chapters/timestamps
    extraction_prompt: str = ""       # What to extract while consuming
    queued_at: datetime = field(default_factory=datetime.now)
    consumed: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "resource_title": self.resource_title,
            "resource_url": self.resource_url,
            "resource_type": self.resource_type,
            "duration_minutes": self.duration_minutes,
            "linked_output": self.linked_output.to_dict(),
            "urgency": self.urgency.value,
            "assigned_window": self.assigned_window.to_dict() if self.assigned_window else None,
            "key_sections": self.key_sections,
            "extraction_prompt": self.extraction_prompt,
            "queued_at": self.queued_at.isoformat(),
            "consumed": self.consumed,
            "notes": self.notes
        }


@dataclass
class LearningQueue:
    """The complete learning queue output."""
    items: List[LearningQueueItem] = field(default_factory=list)
    total_duration_minutes: int = 0
    blocking_count: int = 0
    linked_outputs_count: int = 0
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        unique_outputs = set()
        for item in self.items:
            unique_outputs.add(item.linked_output.title)

        return {
            "items": [item.to_dict() for item in self.items],
            "total_duration_minutes": sum(item.duration_minutes for item in self.items),
            "blocking_count": sum(1 for item in self.items if item.urgency == LearningUrgency.BLOCKING),
            "linked_outputs_count": len(unique_outputs),
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "by_urgency": self._count_by_urgency(),
                "by_window_type": self._count_by_window(),
                "total_items": len(self.items)
            }
        }

    def _count_by_urgency(self) -> Dict[str, int]:
        counts = {}
        for item in self.items:
            key = item.urgency.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_by_window(self) -> Dict[str, int]:
        counts = {"unassigned": 0}
        for item in self.items:
            if item.assigned_window:
                key = item.assigned_window.window_type.value
                counts[key] = counts.get(key, 0) + 1
            else:
                counts["unassigned"] += 1
        return counts


class LearningCurator(LearningAgent):
    """
    Just-in-Time Learning Curator

    Learning Rule: "No learning queued without a linked output. Learning serves shipping."

    This agent matches learning opportunities to execution problems, ensuring
    that all learning is purposeful and tied to concrete outputs.
    """

    LEARNING_RULE = "No learning queued without a linked output. Learning serves shipping."

    def __init__(self):
        super().__init__("LearningCurator")
        self._queue: List[LearningQueueItem] = []
        self._time_windows: List[TimeWindow] = []
        self._pending_outputs: List[LinkedOutput] = []

    def check_state_permission(self) -> tuple[bool, str]:
        """Learning curator is paused in RED state."""
        if self.alfred_state == AlfredState.RED:
            return False, "Learning curation paused in RED state - focus on crisis management"
        return True, "Operation permitted"

    def register_time_window(self, window: TimeWindow) -> None:
        """Register an available time window for learning."""
        self._time_windows.append(window)

    def register_pending_output(self, output: LinkedOutput) -> None:
        """Register a pending output that may require learning."""
        self._pending_outputs.append(output)

    def clear_time_windows(self) -> None:
        """Clear all registered time windows."""
        self._time_windows = []

    def extract_learning_questions(
        self,
        execution_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract learning questions from current execution context.

        Args:
            execution_context: Contains current_task, blockers, and pending_outputs

        Returns:
            List of learning questions with linked outputs
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return []

        questions = []

        # Extract from blockers
        blockers = execution_context.get("blockers", [])
        for blocker in blockers:
            question = {
                "question": blocker.get("description", ""),
                "source": "blocker",
                "urgency": LearningUrgency.BLOCKING,
                "linked_output": blocker.get("blocks_output"),
                "context": blocker.get("context", "")
            }
            if question["linked_output"]:  # Only include if linked
                questions.append(question)

        # Extract from knowledge gaps
        gaps = execution_context.get("knowledge_gaps", [])
        for gap in gaps:
            question = {
                "question": gap.get("question", ""),
                "source": "knowledge_gap",
                "urgency": LearningUrgency.NEXT_UP,
                "linked_output": gap.get("needed_for"),
                "context": gap.get("context", "")
            }
            if question["linked_output"]:
                questions.append(question)

        # Extract from optimization opportunities
        optimizations = execution_context.get("optimization_opportunities", [])
        for opt in optimizations:
            question = {
                "question": opt.get("question", ""),
                "source": "optimization",
                "urgency": LearningUrgency.OPTIMIZATION,
                "linked_output": opt.get("improves_output"),
                "context": opt.get("context", "")
            }
            if question["linked_output"]:
                questions.append(question)

        return questions

    def match_to_time_windows(
        self,
        candidates: List[Dict[str, Any]],
        available_windows: Optional[List[TimeWindow]] = None
    ) -> List[LearningQueueItem]:
        """
        Match learning candidates to available time windows.

        Args:
            candidates: Learning resource candidates (from Scout)
            available_windows: Time windows, defaults to registered windows

        Returns:
            List of queue items with assigned windows
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return []

        windows = available_windows or self._time_windows
        matched_items = []

        for candidate in candidates:
            # Validate linked output exists
            linked_output = candidate.get("linked_output")
            if not linked_output:
                continue  # Skip - violates learning rule

            duration = candidate.get("duration_minutes", 30)
            is_audio = candidate.get("audio_friendly", False)

            # Find best matching window
            best_window = None
            for window in windows:
                if window.duration_minutes >= duration:
                    if is_audio or not window.audio_only:
                        best_window = window
                        break

            # Create queue item
            item = LearningQueueItem(
                question=candidate.get("question", ""),
                resource_title=candidate.get("title", ""),
                resource_url=candidate.get("url", ""),
                resource_type=candidate.get("type", "unknown"),
                duration_minutes=duration,
                linked_output=LinkedOutput(
                    output_type=OutputType(linked_output.get("type", "implementation")),
                    title=linked_output.get("title", ""),
                    deadline=linked_output.get("deadline"),
                    output_id=linked_output.get("id")
                ),
                urgency=LearningUrgency(candidate.get("urgency", "next_up")),
                assigned_window=best_window,
                key_sections=candidate.get("key_sections", []),
                extraction_prompt=candidate.get("extraction_prompt", "")
            )
            matched_items.append(item)

        return matched_items

    def curate_queue(
        self,
        questions: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        time_windows: Optional[List[TimeWindow]] = None,
        max_items: int = 10
    ) -> AgentResponse:
        """
        Curate the learning queue from questions and candidates.

        This is the main entry point for just-in-time learning curation.

        Args:
            questions: Learning questions from Distiller
            candidates: Resource candidates from Scout
            time_windows: Available time windows
            max_items: Maximum items to queue

        Returns:
            AgentResponse with LEARNING_QUEUE
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Match questions to candidates
        matched_candidates = []
        for question in questions:
            for candidate in candidates:
                if self._matches_question(question, candidate):
                    candidate["question"] = question.get("question", "")
                    candidate["linked_output"] = question.get("linked_output")
                    candidate["urgency"] = question.get("urgency", LearningUrgency.NEXT_UP)
                    if isinstance(candidate["urgency"], LearningUrgency):
                        candidate["urgency"] = candidate["urgency"].value
                    matched_candidates.append(candidate)
                    break

        # Filter out candidates without linked outputs
        valid_candidates = [
            c for c in matched_candidates
            if c.get("linked_output")
        ]

        # Generate warnings for rejected candidates
        warnings = []
        rejected_count = len(matched_candidates) - len(valid_candidates)
        if rejected_count > 0:
            warnings.append(
                f"Rejected {rejected_count} candidates: no linked output. "
                f"Learning rule: '{self.LEARNING_RULE}'"
            )

        # Match to time windows
        queue_items = self.match_to_time_windows(valid_candidates, time_windows)

        # Sort by urgency and limit
        queue_items.sort(key=lambda x: list(LearningUrgency).index(x.urgency))
        queue_items = queue_items[:max_items]

        # Build queue
        queue = LearningQueue(items=queue_items)

        return self.create_response(
            data={
                "LEARNING_QUEUE": queue.to_dict(),
                "learning_rule": self.LEARNING_RULE,
                "curation_stats": {
                    "questions_processed": len(questions),
                    "candidates_evaluated": len(candidates),
                    "items_queued": len(queue_items),
                    "rejected_no_output_link": rejected_count
                }
            },
            warnings=warnings
        )

    def _matches_question(
        self,
        question: Dict[str, Any],
        candidate: Dict[str, Any]
    ) -> bool:
        """Check if a candidate resource matches a learning question."""
        question_text = question.get("question", "").lower()
        title = candidate.get("title", "").lower()
        topics = [t.lower() for t in candidate.get("topics", [])]
        description = candidate.get("description", "").lower()

        # Simple keyword matching (could be enhanced with embeddings)
        question_words = set(question_text.split())
        title_words = set(title.split())
        desc_words = set(description.split())
        topic_words = set(" ".join(topics).split())

        all_resource_words = title_words | desc_words | topic_words
        overlap = question_words & all_resource_words

        # Require at least some overlap
        return len(overlap) >= 2 or any(t in question_text for t in topics)

    def validate_queue_item(self, item: LearningQueueItem) -> tuple[bool, List[str]]:
        """
        Validate a queue item against learning rules.

        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []

        # Must have linked output
        if not item.linked_output:
            violations.append("Missing linked output - learning must serve shipping")

        if not item.linked_output.title:
            violations.append("Linked output must have a title")

        # Must have a question
        if not item.question:
            violations.append("Missing learning question")

        # Must have resource
        if not item.resource_url:
            violations.append("Missing resource URL")

        # Duration must be positive
        if item.duration_minutes <= 0:
            violations.append("Duration must be positive")

        return len(violations) == 0, violations

    def get_blocking_items(self) -> List[LearningQueueItem]:
        """Get all items that are blocking current execution."""
        return [
            item for item in self._queue
            if item.urgency == LearningUrgency.BLOCKING and not item.consumed
        ]

    def mark_consumed(self, resource_url: str, notes: str = "") -> bool:
        """Mark a queue item as consumed."""
        for item in self._queue:
            if item.resource_url == resource_url:
                item.consumed = True
                item.notes = notes
                return True
        return False

    def get_queue_for_window(
        self,
        window: TimeWindow
    ) -> AgentResponse:
        """
        Get queue items appropriate for a specific time window.

        Args:
            window: The time window to fill

        Returns:
            AgentResponse with filtered queue items
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        matching_items = []
        for item in self._queue:
            if item.consumed:
                continue
            if item.duration_minutes > window.duration_minutes:
                continue
            if window.audio_only and item.resource_type not in ["podcast", "audiobook", "audio"]:
                continue
            matching_items.append(item)

        # Sort by urgency
        matching_items.sort(key=lambda x: list(LearningUrgency).index(x.urgency))

        return self.create_response(
            data={
                "window": window.to_dict(),
                "available_items": [item.to_dict() for item in matching_items],
                "total_available": len(matching_items),
                "recommended": matching_items[0].to_dict() if matching_items else None
            }
        )

    def generate_extraction_prompt(
        self,
        item: LearningQueueItem
    ) -> str:
        """Generate a prompt for extracting relevant information while consuming."""
        output_type = item.linked_output.output_type.value
        output_title = item.linked_output.title

        return f"""While consuming this resource, extract information for:
OUTPUT: {output_title} ({output_type})
QUESTION: {item.question}

Look for:
1. Key concepts that directly answer the question
2. Specific examples or case studies
3. Quotes or data points usable in output
4. Counter-arguments or nuances to address
5. Related questions that emerged

Capture in format ready for {output_type} creation."""
