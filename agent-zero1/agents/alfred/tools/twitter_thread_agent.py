"""
Twitter Thread Agent - Short-Form Translator

Translates approved long-form content into neutral educational threads.
Maintains fidelity to source material while optimizing for thread format.

DOES:
- Translate long-form to thread format
- Maintain neutral tone
- Compress without distortion
- Preserve key evidence references
- Track character counts precisely

DOES NOT:
- Reply to others
- Quote tweet
- Express opinion
- Write emotionally
- Produce ANY content in YELLOW/RED states

HARD BLOCK: If Alfred state is YELLOW or RED, produces NOTHING.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from . import ContentAgent, AgentResponse, AlfredState


# Twitter character limit (as of 2024 for standard users)
TWITTER_CHAR_LIMIT = 280

# Thread size limits
MAX_THREAD_SIZE = 25  # Maximum tweets in a thread
MIN_THREAD_SIZE = 3   # Minimum for a proper thread


class ToneClassification(Enum):
    """Classification of detected tone."""
    NEUTRAL = "neutral"         # Acceptable - factual, educational
    INFORMATIVE = "informative" # Acceptable - teaching focus
    ANALYTICAL = "analytical"   # Acceptable - breaking down concepts
    OPINION = "opinion"         # NOT ACCEPTABLE - personal views
    EMOTIONAL = "emotional"     # NOT ACCEPTABLE - appeals to emotion
    PROVOCATIVE = "provocative" # NOT ACCEPTABLE - designed to provoke


class SourceFidelity(Enum):
    """How faithfully the thread represents the source."""
    HIGH = "high"       # All key points preserved accurately
    MODERATE = "moderate"  # Main points preserved, some nuance lost
    LOW = "low"         # Significant distortion or omission
    UNKNOWN = "unknown" # Cannot assess without source


@dataclass
class Tweet:
    """A single tweet in a thread."""
    content: str
    position: int  # 1-indexed position in thread
    character_count: int = 0
    has_link: bool = False
    has_media_placeholder: bool = False
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.character_count = len(self.content)
        self._validate()

    def _validate(self):
        """Validate tweet constraints."""
        self.validation_errors = []

        if self.character_count > TWITTER_CHAR_LIMIT:
            self.is_valid = False
            self.validation_errors.append(
                f"Exceeds character limit: {self.character_count}/{TWITTER_CHAR_LIMIT}"
            )

        if self.character_count == 0:
            self.is_valid = False
            self.validation_errors.append("Tweet is empty")

        # Check for problematic patterns
        if re.search(r'^@\w+', self.content):
            self.is_valid = False
            self.validation_errors.append("Tweet appears to be a reply (starts with @)")

        if re.search(r'https://twitter\.com/\w+/status/', self.content):
            self.is_valid = False
            self.validation_errors.append("Tweet contains quote tweet URL")

        self.has_link = bool(re.search(r'https?://', self.content))
        self.has_media_placeholder = '[image]' in self.content.lower() or '[media]' in self.content.lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "position": self.position,
            "character_count": self.character_count,
            "has_link": self.has_link,
            "has_media_placeholder": self.has_media_placeholder,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors
        }


@dataclass
class ToneCheckResult:
    """Result of tone analysis."""
    classification: ToneClassification
    is_acceptable: bool
    confidence: float  # 0.0-1.0
    markers_found: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "is_acceptable": self.is_acceptable,
            "confidence": self.confidence,
            "markers_found": self.markers_found,
            "recommendations": self.recommendations
        }


@dataclass
class SourceFidelityCheck:
    """Assessment of how faithfully thread represents source."""
    fidelity_level: SourceFidelity
    source_title: Optional[str]
    key_points_preserved: List[str]
    key_points_missing: List[str]
    distortions_detected: List[str]
    compression_ratio: float  # Thread word count / source word count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fidelity_level": self.fidelity_level.value,
            "source_title": self.source_title,
            "key_points_preserved": self.key_points_preserved,
            "key_points_missing": self.key_points_missing,
            "distortions_detected": self.distortions_detected,
            "compression_ratio": self.compression_ratio
        }


@dataclass
class ThreadDraft:
    """Complete thread draft output."""
    tweets: List[Tweet]
    total_character_count: int
    tone_check: ToneCheckResult
    source_fidelity: SourceFidelityCheck
    thread_length: int
    all_tweets_valid: bool
    source_reference: Optional[str]  # Link to source article
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    draft_status: str = "pending_review"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_type": "THREAD_DRAFT",
            "tweets": [t.to_dict() for t in self.tweets],
            "total_character_count": self.total_character_count,
            "tone_check": self.tone_check.to_dict(),
            "source_fidelity": self.source_fidelity.to_dict(),
            "thread_length": self.thread_length,
            "all_tweets_valid": self.all_tweets_valid,
            "source_reference": self.source_reference,
            "created_at": self.created_at,
            "draft_status": self.draft_status
        }


class TwitterThreadAgent(ContentAgent):
    """
    Short-Form Translator - Twitter threads from approved long-form content.

    CRITICAL: This agent has a HARD BLOCK in YELLOW and RED states.
    It produces NOTHING if Alfred is not in GREEN state.
    """

    # Emotional language markers (NOT acceptable)
    EMOTIONAL_PATTERNS = [
        r'\b(amazing|incredible|unbelievable|shocking|terrifying)\b',
        r'\b(love|hate|angry|furious|excited|thrilled)\b',
        r'\b(beautiful|horrible|disgusting|wonderful)\b',
        r'[!]{2,}',  # Multiple exclamation marks
        r'\b(OMG|WTF|LOL|LMAO)\b',
    ]

    # Opinion markers (NOT acceptable)
    OPINION_PATTERNS = [
        r'\bI (think|believe|feel|suspect|bet)\b',
        r'\b(in my opinion|personally|to me|my view)\b',
        r'\b(should|must|need to)\b.*\b(everyone|you all|people)\b',
        r'\b(obviously|clearly|undeniably)\b',  # Subjective certainty
    ]

    # Provocative patterns (NOT acceptable)
    PROVOCATIVE_PATTERNS = [
        r'\b(wake up|open your eyes|sheeple)\b',
        r'\b(fight|battle|war|destroy|crush)\b',
        r'\b(they|them|those people)\s+(always|never|want to)\b',
        r'^Hot take:',
        r'\b(controversial|unpopular) opinion\b',
    ]

    # Neutral/informative markers (acceptable)
    NEUTRAL_PATTERNS = [
        r'\b(research shows|studies indicate|data suggests)\b',
        r'\b(according to|based on|evidence for)\b',
        r'\b(here\'s what|let\'s look at|breaking down)\b',
        r'\b(context|nuance|tradeoff|complexity)\b',
    ]

    def __init__(self):
        super().__init__(name="TwitterThreadAgent")

    def check_state_permission(self) -> tuple[bool, str]:
        """
        HARD BLOCK: Twitter agent produces NOTHING in YELLOW or RED.

        This override ensures stricter blocking than the base ContentAgent.
        """
        if self.alfred_state == AlfredState.RED:
            return False, "HARD BLOCK: All Twitter output suspended in RED state"
        if self.alfred_state == AlfredState.YELLOW:
            return False, "HARD BLOCK: All Twitter output suspended in YELLOW state"
        return True, "Operation permitted"

    def analyze_tone(self, content: str) -> ToneCheckResult:
        """
        Analyze content for tone classification.

        Returns classification and whether tone is acceptable for posting.
        """
        markers_found = []
        recommendations = []

        # Check emotional patterns
        emotional_count = 0
        for pattern in self.EMOTIONAL_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                emotional_count += len(matches)
                markers_found.extend([f"emotional: {m}" for m in matches[:2]])

        # Check opinion patterns
        opinion_count = 0
        for pattern in self.OPINION_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                opinion_count += len(matches)
                markers_found.extend([f"opinion: {m}" for m in matches[:2]])

        # Check provocative patterns
        provocative_count = 0
        for pattern in self.PROVOCATIVE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                provocative_count += len(matches)
                markers_found.extend([f"provocative: {m}" for m in matches[:2]])

        # Check neutral patterns
        neutral_count = 0
        for pattern in self.NEUTRAL_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                neutral_count += len(matches)

        # Determine classification
        total_problematic = emotional_count + opinion_count + provocative_count

        if provocative_count > 0:
            classification = ToneClassification.PROVOCATIVE
            is_acceptable = False
            recommendations.append("Remove provocative language and framing")
        elif emotional_count > opinion_count and emotional_count > 2:
            classification = ToneClassification.EMOTIONAL
            is_acceptable = False
            recommendations.append("Replace emotional language with neutral descriptions")
        elif opinion_count > 2:
            classification = ToneClassification.OPINION
            is_acceptable = False
            recommendations.append("Remove personal opinion markers, stick to facts")
        elif neutral_count > total_problematic:
            classification = ToneClassification.NEUTRAL if neutral_count > 3 else ToneClassification.INFORMATIVE
            is_acceptable = True
        elif total_problematic == 0:
            classification = ToneClassification.NEUTRAL
            is_acceptable = True
        else:
            classification = ToneClassification.ANALYTICAL
            is_acceptable = total_problematic < 2

        # Calculate confidence based on marker density
        total_markers = emotional_count + opinion_count + provocative_count + neutral_count
        if total_markers == 0:
            confidence = 0.5  # Low confidence when no markers found
        else:
            confidence = min(0.95, 0.5 + (total_markers * 0.05))

        return ToneCheckResult(
            classification=classification,
            is_acceptable=is_acceptable,
            confidence=confidence,
            markers_found=markers_found,
            recommendations=recommendations
        )

    def check_source_fidelity(self,
                              thread_content: str,
                              source_content: Optional[str],
                              source_title: Optional[str],
                              key_points: Optional[List[str]] = None) -> SourceFidelityCheck:
        """
        Check how faithfully the thread represents its source.

        Args:
            thread_content: Combined text of all tweets
            source_content: Original long-form content
            source_title: Title of source article
            key_points: List of key points that should be preserved
        """
        if not source_content:
            return SourceFidelityCheck(
                fidelity_level=SourceFidelity.UNKNOWN,
                source_title=source_title,
                key_points_preserved=[],
                key_points_missing=[],
                distortions_detected=["Cannot assess - no source content provided"],
                compression_ratio=0.0
            )

        preserved = []
        missing = []
        distortions = []

        # Calculate compression ratio
        source_words = len(source_content.split())
        thread_words = len(thread_content.split())
        compression_ratio = thread_words / source_words if source_words > 0 else 0.0

        # Check key points if provided
        if key_points:
            thread_lower = thread_content.lower()
            for point in key_points:
                # Check if key concept appears in thread
                point_words = point.lower().split()
                key_term = max(point_words, key=len) if point_words else point

                if key_term.lower() in thread_lower or any(w in thread_lower for w in point_words if len(w) > 4):
                    preserved.append(point)
                else:
                    missing.append(point)

        # Check for potential distortions
        # Look for absolutizing language not in source
        absolutes_in_thread = re.findall(r'\b(always|never|all|none|every|no one)\b', thread_content, re.IGNORECASE)
        absolutes_in_source = re.findall(r'\b(always|never|all|none|every|no one)\b', source_content, re.IGNORECASE)

        if len(absolutes_in_thread) > len(absolutes_in_source):
            distortions.append("Thread contains more absolute language than source")

        # Determine fidelity level
        if key_points:
            preservation_rate = len(preserved) / len(key_points) if key_points else 1.0
            if preservation_rate >= 0.8 and not distortions:
                fidelity_level = SourceFidelity.HIGH
            elif preservation_rate >= 0.5:
                fidelity_level = SourceFidelity.MODERATE
            else:
                fidelity_level = SourceFidelity.LOW
        else:
            # Without key points, assess based on compression and distortions
            if compression_ratio > 0.3 and not distortions:
                fidelity_level = SourceFidelity.HIGH
            elif compression_ratio > 0.15:
                fidelity_level = SourceFidelity.MODERATE
            else:
                fidelity_level = SourceFidelity.LOW

        return SourceFidelityCheck(
            fidelity_level=fidelity_level,
            source_title=source_title,
            key_points_preserved=preserved,
            key_points_missing=missing,
            distortions_detected=distortions,
            compression_ratio=round(compression_ratio, 3)
        )

    def validate_tweets(self, tweets: List[Tweet]) -> tuple[bool, List[str]]:
        """Validate all tweets in a thread."""
        errors = []

        if len(tweets) < MIN_THREAD_SIZE:
            errors.append(f"Thread too short: {len(tweets)} tweets (minimum {MIN_THREAD_SIZE})")

        if len(tweets) > MAX_THREAD_SIZE:
            errors.append(f"Thread too long: {len(tweets)} tweets (maximum {MAX_THREAD_SIZE})")

        for tweet in tweets:
            if not tweet.is_valid:
                for error in tweet.validation_errors:
                    errors.append(f"Tweet {tweet.position}: {error}")

        return len(errors) == 0, errors

    def create_tweets_from_content(self, content_blocks: List[str]) -> List[Tweet]:
        """Create Tweet objects from content blocks."""
        tweets = []
        for i, content in enumerate(content_blocks, 1):
            tweet = Tweet(content=content.strip(), position=i)
            tweets.append(tweet)
        return tweets

    def split_long_content(self, content: str, max_chars: int = TWITTER_CHAR_LIMIT) -> List[str]:
        """
        Split long content into tweet-sized chunks.
        Tries to break at sentence boundaries.
        """
        if len(content) <= max_chars:
            return [content]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > max_chars:
                # Sentence itself is too long, need to break mid-sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Break long sentence at word boundaries
                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk += (" " if current_chunk else "") + word
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = word
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def create_thread(self,
                      tweet_contents: List[str],
                      source_content: Optional[str] = None,
                      source_title: Optional[str] = None,
                      source_url: Optional[str] = None,
                      key_points: Optional[List[str]] = None) -> AgentResponse:
        """
        Create a thread draft from tweet contents.

        Args:
            tweet_contents: List of tweet text contents
            source_content: Original long-form content for fidelity check
            source_title: Title of source for reference
            source_url: URL to source article
            key_points: Key points that should be preserved from source

        Returns:
            AgentResponse with THREAD_DRAFT or blocked status
        """
        # HARD BLOCK check first
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        errors = []
        warnings = []

        # Create tweet objects
        tweets = self.create_tweets_from_content(tweet_contents)

        # Validate tweets
        valid, validation_errors = self.validate_tweets(tweets)
        if not valid:
            errors.extend(validation_errors)

        # Combine content for tone check
        full_thread_content = " ".join([t.content for t in tweets])

        # Run tone check
        tone_result = self.analyze_tone(full_thread_content)
        if not tone_result.is_acceptable:
            errors.append(f"Tone check failed: {tone_result.classification.value}")
            errors.extend(tone_result.recommendations)

        # Check source fidelity
        fidelity_result = self.check_source_fidelity(
            full_thread_content,
            source_content,
            source_title,
            key_points
        )

        if fidelity_result.fidelity_level == SourceFidelity.LOW:
            errors.append("Source fidelity too low - thread distorts original content")
            if fidelity_result.key_points_missing:
                warnings.append(f"Missing key points: {fidelity_result.key_points_missing[:3]}")
        elif fidelity_result.distortions_detected:
            for distortion in fidelity_result.distortions_detected:
                warnings.append(f"Potential distortion: {distortion}")

        # Calculate totals
        total_chars = sum(t.character_count for t in tweets)
        all_valid = all(t.is_valid for t in tweets)

        # If there are errors, return failure
        if errors:
            return self.create_response(
                data={
                    "output_type": "THREAD_DRAFT",
                    "draft_status": "validation_failed",
                    "tweets": [t.to_dict() for t in tweets],
                    "tone_check": tone_result.to_dict(),
                    "source_fidelity": fidelity_result.to_dict()
                },
                success=False,
                errors=errors,
                warnings=warnings
            )

        # Create successful draft
        draft = ThreadDraft(
            tweets=tweets,
            total_character_count=total_chars,
            tone_check=tone_result,
            source_fidelity=fidelity_result,
            thread_length=len(tweets),
            all_tweets_valid=all_valid,
            source_reference=source_url,
            draft_status="ready_for_review"
        )

        return self.create_response(
            data=draft.to_dict(),
            success=True,
            warnings=warnings
        )

    def translate_longform(self,
                           longform_content: str,
                           longform_title: str,
                           key_points: List[str],
                           source_url: Optional[str] = None,
                           max_tweets: int = 10) -> AgentResponse:
        """
        Translate long-form content into a thread format.

        This is the primary translation method that takes approved
        long-form content and creates a thread draft.

        Args:
            longform_content: The full long-form article content
            longform_title: Title of the source article
            key_points: Key points to preserve in translation
            source_url: URL to the source article
            max_tweets: Maximum tweets in the thread

        Returns:
            AgentResponse with THREAD_DRAFT or blocked status
        """
        # HARD BLOCK check
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        warnings = []

        # Extract hook from title (first tweet)
        hook = f"Thread: {longform_title}"
        if len(hook) > TWITTER_CHAR_LIMIT:
            hook = hook[:TWITTER_CHAR_LIMIT - 3] + "..."
            warnings.append("Title was truncated for hook tweet")

        # Build tweets from key points
        tweet_contents = [hook]

        for i, point in enumerate(key_points[:max_tweets - 2], 1):  # Reserve space for hook and closer
            # Format point as tweet
            tweet_text = point.strip()
            if len(tweet_text) > TWITTER_CHAR_LIMIT:
                # Split long points
                chunks = self.split_long_content(tweet_text)
                tweet_contents.extend(chunks)
            else:
                tweet_contents.append(tweet_text)

        # Add closer with source link
        if source_url:
            closer = f"Full article with sources and nuance: {source_url}"
        else:
            closer = "Full article linked in bio with complete sources and nuance."

        tweet_contents.append(closer)

        # Trim to max length
        if len(tweet_contents) > max_tweets:
            tweet_contents = tweet_contents[:max_tweets - 1] + [tweet_contents[-1]]
            warnings.append(f"Thread trimmed to {max_tweets} tweets")

        # Create thread through standard method
        return self.create_thread(
            tweet_contents=tweet_contents,
            source_content=longform_content,
            source_title=longform_title,
            source_url=source_url,
            key_points=key_points
        )

    def validate_thread(self, tweets: List[str]) -> AgentResponse:
        """
        Validate a thread without creating a full draft.

        Useful for checking content before full processing.
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        tweet_objects = self.create_tweets_from_content(tweets)
        valid, errors = self.validate_tweets(tweet_objects)

        full_content = " ".join(tweets)
        tone_result = self.analyze_tone(full_content)

        return self.create_response(
            data={
                "tweets_valid": valid,
                "tweet_count": len(tweets),
                "total_characters": sum(len(t) for t in tweets),
                "tone_check": tone_result.to_dict(),
                "validation_errors": errors,
                "individual_tweets": [t.to_dict() for t in tweet_objects]
            },
            success=valid and tone_result.is_acceptable,
            errors=errors if not valid else (tone_result.recommendations if not tone_result.is_acceptable else [])
        )
