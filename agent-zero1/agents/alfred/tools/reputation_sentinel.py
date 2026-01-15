"""
Reputation Sentinel - Silent Guardian for Reputational and Clinical Risk

This agent monitors for reputational and clinical risk across all platforms.
Operates in background, never surfaces to user, never engages publicly.

HARD RULE: NEVER recommend comment-thread engagement. Ever.

GitHub Tools Interfaces (prepared for integration):
- twscrape: Twitter/X scraping
- VADER: Sentiment analysis
- Detoxify: Toxicity detection
- BERTopic: Topic clustering
- Botometer: Bot detection
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from . import SignalAgent, AgentResponse, AlfredState


# =============================================================================
# ENUMS
# =============================================================================

class RiskClassification(Enum):
    """
    Risk classifications for reputation events.
    Each classification requires different handling strategies.
    """
    MISINTERPRETATION = "misinterpretation"  # Content taken out of context
    MISINFORMATION = "misinformation"         # False claims being spread
    HARASSMENT = "harassment"                 # Targeted personal attacks
    PEER_CRITIQUE = "peer_critique"           # Professional criticism from peers
    REVIEW_RISK = "review_risk"               # Review site/rating threats
    POLICY_EXPOSURE = "policy_exposure"       # Platform policy or regulatory exposure


class RecommendedAction(Enum):
    """
    Recommended actions for Alfred (NOT for public engagement).
    SILENCE and IGNORE are preferred. LONGFORM only for significant misinfo.
    """
    IGNORE = "IGNORE"                         # No action needed
    MONITOR = "MONITOR"                       # Continue watching, no action yet
    LONGFORM_CLARIFICATION = "LONGFORM_CLARIFICATION"  # Address via long-form content
    SILENCE = "SILENCE"                       # Strategic silence is the response


class Platform(Enum):
    """Supported platforms for monitoring."""
    TWITTER = "Twitter"
    YOUTUBE = "YouTube"
    SUBSTACK = "Substack"
    REVIEW_SITE = "Review Site"
    INSTAGRAM = "Instagram"
    LINKEDIN = "LinkedIn"
    OTHER = "Other"


class Priority(Enum):
    """Request priority levels."""
    ROUTINE = "routine"
    ELEVATED = "elevated"
    URGENT = "urgent"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ReputationCheckRequest:
    """
    Input format for reputation check requests from Alfred.

    REPUTATION_CHECK_REQUEST
    - Scope: [platforms to monitor]
    - Time Window: [last N hours/days]
    - Context: [any relevant recent activity]
    - Priority: [routine / elevated / urgent]
    """
    scope: List[Platform]
    time_window_hours: int
    context: str = ""
    priority: Priority = Priority.ROUTINE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReputationCheckRequest:
        """Parse a request from dictionary format."""
        scope = [Platform(p) if isinstance(p, str) else p for p in data.get("scope", [])]
        return cls(
            scope=scope or [Platform.TWITTER, Platform.YOUTUBE, Platform.SUBSTACK],
            time_window_hours=data.get("time_window_hours", 24),
            context=data.get("context", ""),
            priority=Priority(data.get("priority", "routine"))
        )


@dataclass
class ReputationEvent:
    """
    Raw signal detected from a platform.
    Internal use only - never exposed to Alfred or user.
    """
    event_id: str
    platform: Platform
    timestamp: datetime
    content_hash: str  # Hash of content for deduplication

    # Signal characteristics (not exposed raw)
    sentiment_score: float = 0.0        # -1 to 1 (VADER)
    toxicity_score: float = 0.0         # 0 to 1 (Detoxify)
    bot_probability: float = 0.0        # 0 to 1 (Botometer)
    reach_estimate: int = 0             # Estimated audience reach
    engagement_velocity: float = 0.0    # Rate of spread

    # Classification metadata
    topic_cluster: Optional[str] = None  # BERTopic cluster
    author_type: str = "unknown"         # peer, public, bot, troll
    is_coordinated: bool = False         # Part of coordinated campaign

    # Content indicators (never expose raw content)
    contains_clinical_claims: bool = False
    contains_personal_attack: bool = False
    references_specific_content: bool = False

    def __post_init__(self):
        """Generate event ID if not provided."""
        if not self.event_id:
            self.event_id = hashlib.sha256(
                f"{self.platform.value}:{self.content_hash}:{self.timestamp}".encode()
            ).hexdigest()[:16]


@dataclass
class PatternRecord:
    """
    Tracks patterns across multiple signals over time.
    Used for pattern detection and trend analysis.
    """
    pattern_id: str
    pattern_type: str  # escalating, recurring, coordinated, seasonal
    first_seen: datetime
    last_seen: datetime
    event_count: int = 1
    platforms_involved: Set[Platform] = field(default_factory=set)
    classification: Optional[RiskClassification] = None
    trend_direction: str = "stable"  # rising, stable, declining
    notes: List[str] = field(default_factory=list)

    def update(self, event: ReputationEvent, note: str = ""):
        """Update pattern with new event."""
        self.last_seen = event.timestamp
        self.event_count += 1
        self.platforms_involved.add(event.platform)
        if note:
            self.notes.append(f"[{event.timestamp.isoformat()}] {note}")


@dataclass
class ReputationPacket:
    """
    Output format for Alfred (to Alfred).

    REPUTATION_PACKET
    - Event: (1-2 line neutral summary)
    - Platform: [Twitter / YouTube / Substack / Review Site / Other]
    - Classification: misinterpretation | misinformation | harassment | peer_critique | review_risk | policy_exposure
    - Risk Score: 0-100
    - Recommended State: GREEN | YELLOW | RED
    - Recommended Action: IGNORE | MONITOR | LONGFORM_CLARIFICATION | SILENCE
    - Rationale: (1 line explaining recommendation)
    - Pattern Note: [if this connects to ongoing pattern, note here]
    """
    event_summary: str  # 1-2 line neutral summary (NO raw content)
    platform: Platform
    classification: RiskClassification
    risk_score: int  # 0-100
    recommended_state: AlfredState
    recommended_action: RecommendedAction
    rationale: str
    pattern_note: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate packet data."""
        # Ensure risk score is bounded
        self.risk_score = max(0, min(100, self.risk_score))

        # Ensure summary doesn't contain raw content
        self._sanitize_summary()

    def _sanitize_summary(self):
        """
        Ensure summary doesn't expose raw data, usernames, or inflammatory content.
        This is critical - we NEVER expose raw social media content.
        """
        # Remove potential @mentions
        self.event_summary = re.sub(r'@[\w]+', '[user]', self.event_summary)
        # Remove potential URLs
        self.event_summary = re.sub(r'https?://\S+', '[link]', self.event_summary)
        # Truncate if too long
        if len(self.event_summary) > 200:
            self.event_summary = self.event_summary[:197] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "event": self.event_summary,
            "platform": self.platform.value,
            "classification": self.classification.value,
            "risk_score": self.risk_score,
            "recommended_state": self.recommended_state.value,
            "recommended_action": self.recommended_action.value,
            "rationale": self.rationale,
            "pattern_note": self.pattern_note,
            "timestamp": self.timestamp
        }

    def to_formatted_string(self) -> str:
        """Format as specified output format."""
        lines = [
            "REPUTATION_PACKET",
            f"- Event: {self.event_summary}",
            f"- Platform: {self.platform.value}",
            f"- Classification: {self.classification.value}",
            f"- Risk Score: {self.risk_score}",
            f"- Recommended State: {self.recommended_state.value}",
            f"- Recommended Action: {self.recommended_action.value}",
            f"- Rationale: {self.rationale}",
        ]
        if self.pattern_note:
            lines.append(f"- Pattern Note: {self.pattern_note}")
        return "\n".join(lines)


@dataclass
class SilencePacket:
    """
    Output when no actionable signals detected.
    Normal operations confirmed.
    """
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    platforms_scanned: List[Platform] = field(default_factory=list)
    time_window_hours: int = 24
    signals_processed: int = 0
    signals_filtered: int = 0  # Noise that was filtered out
    recommended_state: AlfredState = AlfredState.GREEN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "CLEAR",
            "timestamp": self.timestamp,
            "platforms_scanned": [p.value for p in self.platforms_scanned],
            "time_window_hours": self.time_window_hours,
            "signals_processed": self.signals_processed,
            "signals_filtered": self.signals_filtered,
            "recommended_state": self.recommended_state.value,
            "message": "No actionable reputation signals detected. Normal operations."
        }


# =============================================================================
# PLATFORM ADAPTERS (Interface definitions for external tools)
# =============================================================================

class PlatformAdapter(ABC):
    """
    Abstract base class for platform-specific adapters.
    Each adapter interfaces with external tools for that platform.
    """

    def __init__(self, platform: Platform):
        self.platform = platform
        self._last_scan: Optional[datetime] = None
        self._rate_limit_remaining: int = 100

    @abstractmethod
    async def fetch_signals(
        self,
        time_window_hours: int,
        context: str = ""
    ) -> List[ReputationEvent]:
        """
        Fetch raw signals from the platform.
        Returns internal ReputationEvent objects, never raw content.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the adapter is properly configured and available."""
        pass

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Return current rate limit status."""
        return {
            "platform": self.platform.value,
            "remaining": self._rate_limit_remaining,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None
        }


class TwitterAdapter(PlatformAdapter):
    """
    Twitter/X adapter using twscrape interface.

    Integration points:
    - twscrape: For fetching tweets and mentions
    - Botometer: For bot detection
    - VADER: For sentiment analysis
    """

    def __init__(self):
        super().__init__(Platform.TWITTER)
        self._twscrape_client = None
        self._botometer_client = None
        self._search_queries: List[str] = []

    def configure(
        self,
        search_queries: List[str],
        twscrape_accounts: Optional[List[Dict]] = None,
        botometer_api_key: Optional[str] = None
    ):
        """
        Configure the Twitter adapter.

        Args:
            search_queries: List of search terms (username, keywords, etc.)
            twscrape_accounts: Account credentials for twscrape
            botometer_api_key: API key for Botometer
        """
        self._search_queries = search_queries
        # twscrape initialization would go here
        # self._twscrape_client = twscrape.API()
        # Botometer initialization would go here
        # self._botometer_client = botometer.Botometer(...)

    def is_available(self) -> bool:
        """Check if Twitter adapter is configured."""
        return len(self._search_queries) > 0

    async def fetch_signals(
        self,
        time_window_hours: int,
        context: str = ""
    ) -> List[ReputationEvent]:
        """
        Fetch signals from Twitter.

        In production, this would:
        1. Use twscrape to fetch mentions and search results
        2. Run VADER sentiment analysis on each tweet
        3. Check Botometer for bot probability
        4. Return ReputationEvent objects (never raw tweets)
        """
        events: List[ReputationEvent] = []
        self._last_scan = datetime.now()

        # Placeholder for actual implementation
        # In production:
        # async for tweet in self._twscrape_client.search(query, limit=100):
        #     sentiment = vader.polarity_scores(tweet.text)
        #     bot_score = await self._botometer_client.check_account(tweet.user.id)
        #     event = ReputationEvent(...)
        #     events.append(event)

        return events

    async def check_bot_probability(self, user_id: str) -> float:
        """
        Check bot probability for a user using Botometer.
        Returns 0.0-1.0 probability.
        """
        # Placeholder for Botometer integration
        # return await self._botometer_client.check_account(user_id)
        return 0.0


class YouTubeAdapter(PlatformAdapter):
    """
    YouTube adapter for comment monitoring.

    Integration points:
    - YouTube Data API: For fetching comments
    - Detoxify: For toxicity detection
    - VADER: For sentiment analysis
    """

    def __init__(self):
        super().__init__(Platform.YOUTUBE)
        self._api_key: Optional[str] = None
        self._channel_id: Optional[str] = None
        self._video_ids: List[str] = []

    def configure(
        self,
        api_key: str,
        channel_id: str,
        video_ids: Optional[List[str]] = None
    ):
        """Configure YouTube adapter."""
        self._api_key = api_key
        self._channel_id = channel_id
        self._video_ids = video_ids or []

    def is_available(self) -> bool:
        return self._api_key is not None and self._channel_id is not None

    async def fetch_signals(
        self,
        time_window_hours: int,
        context: str = ""
    ) -> List[ReputationEvent]:
        """
        Fetch signals from YouTube comments.

        In production, this would:
        1. Fetch recent comments via YouTube Data API
        2. Run Detoxify toxicity analysis
        3. Run VADER sentiment analysis
        4. Return ReputationEvent objects
        """
        events: List[ReputationEvent] = []
        self._last_scan = datetime.now()

        # Placeholder for actual implementation
        return events


class SubstackAdapter(PlatformAdapter):
    """
    Substack adapter for comment and mention monitoring.
    """

    def __init__(self):
        super().__init__(Platform.SUBSTACK)
        self._publication_url: Optional[str] = None

    def configure(self, publication_url: str):
        """Configure Substack adapter."""
        self._publication_url = publication_url

    def is_available(self) -> bool:
        return self._publication_url is not None

    async def fetch_signals(
        self,
        time_window_hours: int,
        context: str = ""
    ) -> List[ReputationEvent]:
        """Fetch signals from Substack."""
        events: List[ReputationEvent] = []
        self._last_scan = datetime.now()
        return events


class ReviewSiteAdapter(PlatformAdapter):
    """
    Review site adapter (Google Reviews, Healthgrades, etc.).
    Critical for clinical reputation monitoring.
    """

    def __init__(self):
        super().__init__(Platform.REVIEW_SITE)
        self._review_sources: List[Dict[str, str]] = []

    def configure(self, review_sources: List[Dict[str, str]]):
        """
        Configure review site monitoring.

        Args:
            review_sources: List of {"name": "Google", "url": "..."} dicts
        """
        self._review_sources = review_sources

    def is_available(self) -> bool:
        return len(self._review_sources) > 0

    async def fetch_signals(
        self,
        time_window_hours: int,
        context: str = ""
    ) -> List[ReputationEvent]:
        """Fetch signals from review sites."""
        events: List[ReputationEvent] = []
        self._last_scan = datetime.now()
        return events


# =============================================================================
# ANALYSIS TOOLS (Interfaces for NLP/ML tools)
# =============================================================================

class SentimentAnalyzer:
    """
    Interface for VADER sentiment analysis.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    tuned for social media sentiment.
    """

    def __init__(self):
        self._analyzer = None
        # In production: self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.

        Returns:
            Dictionary with 'neg', 'neu', 'pos', 'compound' scores.
            compound is the overall score from -1 (negative) to 1 (positive).
        """
        # Placeholder for VADER integration
        # return self._analyzer.polarity_scores(text)
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    def get_compound_score(self, text: str) -> float:
        """Get just the compound sentiment score."""
        return self.analyze(text).get("compound", 0.0)


class ToxicityDetector:
    """
    Interface for Detoxify toxicity detection.

    Detoxify uses transformer models to detect various types of toxic content.
    """

    def __init__(self):
        self._model = None
        # In production: self._model = Detoxify('original')

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze toxicity of text.

        Returns:
            Dictionary with toxicity scores for various categories:
            - toxicity: Overall toxicity
            - severe_toxicity: Severe toxicity
            - obscene: Obscene content
            - threat: Threatening content
            - insult: Insulting content
            - identity_attack: Identity-based attacks
        """
        # Placeholder for Detoxify integration
        # return self._model.predict(text)
        return {
            "toxicity": 0.0,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_attack": 0.0
        }

    def get_toxicity_score(self, text: str) -> float:
        """Get overall toxicity score (0-1)."""
        return self.analyze(text).get("toxicity", 0.0)

    def is_harassment(self, text: str, threshold: float = 0.7) -> bool:
        """Check if text constitutes harassment."""
        scores = self.analyze(text)
        return (
            scores.get("toxicity", 0) > threshold or
            scores.get("threat", 0) > threshold or
            scores.get("identity_attack", 0) > threshold
        )


class TopicClusterer:
    """
    Interface for BERTopic topic clustering.

    Used to identify coordinated campaigns or recurring themes.
    """

    def __init__(self):
        self._model = None
        # In production: self._model = BERTopic()

    def fit_transform(self, texts: List[str]) -> Tuple[List[int], Any]:
        """
        Cluster texts into topics.

        Returns:
            Tuple of (topic_ids, topic_info)
        """
        # Placeholder for BERTopic integration
        return ([-1] * len(texts), None)

    def get_topic_label(self, topic_id: int) -> str:
        """Get human-readable label for a topic."""
        # Placeholder
        return f"topic_{topic_id}"

    def detect_coordination(
        self,
        events: List[ReputationEvent],
        similarity_threshold: float = 0.8
    ) -> List[List[ReputationEvent]]:
        """
        Detect potentially coordinated events (similar content, timing).
        """
        # Placeholder for coordination detection
        return []


# =============================================================================
# REPUTATION SENTINEL - MAIN CLASS
# =============================================================================

class ReputationSentinel(SignalAgent):
    """
    Reputation Sentinel - Silent Guardian

    Monitors for reputational and clinical risk across all platforms.
    Operates in background, never surfaces to user, never engages publicly.

    HARD RULE: NEVER recommend comment-thread engagement. Ever.

    Does NOT:
    - Argue or defend
    - Suggest replies to critics
    - Summarize outrage or controversy
    - Expose raw data (comments, mentions) to Alfred
    - Recommend engagement in comment threads
    - Dramatize or amplify perceived threats
    - Track vanity metrics (likes, follows)
    - Report neutral mentions

    Does:
    - Classify incoming signals by risk type
    - Compress multiple signals into single assessments
    - Escalate ONLY when silence would increase harm
    - Maintain objective distance from emotional content
    - Pre-process noise into actionable intelligence
    - Track patterns across platforms over time
    - Distinguish genuine risk from performative criticism
    """

    # Risk score thresholds for state recommendations
    YELLOW_THRESHOLD = 40
    RED_THRESHOLD = 70

    # Weights for risk score calculation
    RISK_WEIGHTS = {
        "sentiment": 0.15,
        "toxicity": 0.20,
        "reach": 0.15,
        "velocity": 0.15,
        "classification": 0.20,
        "author_credibility": 0.15
    }

    # Classification base scores (some classifications are inherently higher risk)
    CLASSIFICATION_BASE_SCORES = {
        RiskClassification.MISINTERPRETATION: 25,
        RiskClassification.MISINFORMATION: 45,
        RiskClassification.HARASSMENT: 40,
        RiskClassification.PEER_CRITIQUE: 35,
        RiskClassification.REVIEW_RISK: 50,
        RiskClassification.POLICY_EXPOSURE: 55,
    }

    def __init__(self):
        super().__init__(name="ReputationSentinel")

        # Platform adapters
        self._adapters: Dict[Platform, PlatformAdapter] = {
            Platform.TWITTER: TwitterAdapter(),
            Platform.YOUTUBE: YouTubeAdapter(),
            Platform.SUBSTACK: SubstackAdapter(),
            Platform.REVIEW_SITE: ReviewSiteAdapter(),
        }

        # Analysis tools
        self._sentiment_analyzer = SentimentAnalyzer()
        self._toxicity_detector = ToxicityDetector()
        self._topic_clusterer = TopicClusterer()

        # Pattern tracking
        self._active_patterns: Dict[str, PatternRecord] = {}
        self._event_history: List[ReputationEvent] = []
        self._history_retention_days: int = 90

        # State tracking
        self._last_scan: Optional[datetime] = None
        self._consecutive_clear_scans: int = 0

        # Configuration
        self._noise_threshold: float = 0.3  # Below this, signal is noise
        self._compression_window_hours: int = 4  # Compress signals within this window

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def configure_twitter(
        self,
        search_queries: List[str],
        twscrape_accounts: Optional[List[Dict]] = None,
        botometer_api_key: Optional[str] = None
    ):
        """Configure Twitter monitoring."""
        adapter = self._adapters[Platform.TWITTER]
        if isinstance(adapter, TwitterAdapter):
            adapter.configure(search_queries, twscrape_accounts, botometer_api_key)

    def configure_youtube(
        self,
        api_key: str,
        channel_id: str,
        video_ids: Optional[List[str]] = None
    ):
        """Configure YouTube monitoring."""
        adapter = self._adapters[Platform.YOUTUBE]
        if isinstance(adapter, YouTubeAdapter):
            adapter.configure(api_key, channel_id, video_ids)

    def configure_substack(self, publication_url: str):
        """Configure Substack monitoring."""
        adapter = self._adapters[Platform.SUBSTACK]
        if isinstance(adapter, SubstackAdapter):
            adapter.configure(publication_url)

    def configure_review_sites(self, review_sources: List[Dict[str, str]]):
        """Configure review site monitoring."""
        adapter = self._adapters[Platform.REVIEW_SITE]
        if isinstance(adapter, ReviewSiteAdapter):
            adapter.configure(review_sources)

    # =========================================================================
    # MAIN SCANNING INTERFACE
    # =========================================================================

    async def process_request(
        self,
        request: ReputationCheckRequest
    ) -> AgentResponse:
        """
        Main entry point for processing reputation check requests.

        Args:
            request: ReputationCheckRequest from Alfred

        Returns:
            AgentResponse containing ReputationPacket(s) or SilencePacket
        """
        # Adjust monitoring level based on current state
        monitoring_level = self.get_monitoring_level()

        # Scan all requested platforms
        all_events = await self.scan_platforms(
            platforms=request.scope,
            time_window_hours=request.time_window_hours,
            context=request.context
        )

        # Filter noise
        significant_events = self._filter_noise(all_events)

        # If no significant signals, return silence packet
        if not significant_events:
            return self._create_silence_response(
                request.scope,
                request.time_window_hours,
                len(all_events)
            )

        # Compress multiple signals into assessments
        compressed_events = self._compress_signals(significant_events)

        # Generate packets for each significant signal group
        packets: List[ReputationPacket] = []
        for event_group in compressed_events:
            classification = self.classify_risk(event_group)
            risk_score = self.calculate_risk_score(event_group, classification)
            recommended_state = self.recommend_state(risk_score, classification)
            recommended_action = self._recommend_action(
                classification, risk_score, recommended_state
            )
            pattern_note = self.track_pattern(event_group, classification)

            packet = self.generate_packet(
                events=event_group,
                classification=classification,
                risk_score=risk_score,
                recommended_state=recommended_state,
                recommended_action=recommended_action,
                pattern_note=pattern_note
            )
            packets.append(packet)

        # Determine overall recommended state (highest risk wins)
        overall_state = max(
            [p.recommended_state for p in packets],
            key=lambda s: [AlfredState.GREEN, AlfredState.YELLOW, AlfredState.RED].index(s)
        )

        return self.create_response(
            data={
                "packets": [p.to_dict() for p in packets],
                "overall_recommended_state": overall_state.value,
                "total_signals_processed": len(all_events),
                "significant_signals": len(significant_events),
                "monitoring_level": monitoring_level
            },
            success=True
        )

    async def scan_platforms(
        self,
        platforms: List[Platform],
        time_window_hours: int,
        context: str = ""
    ) -> List[ReputationEvent]:
        """
        Scan specified platforms for reputation signals.

        Args:
            platforms: List of platforms to scan
            time_window_hours: How far back to look
            context: Any relevant context for the scan

        Returns:
            List of ReputationEvent objects (internal format, never raw content)
        """
        all_events: List[ReputationEvent] = []

        for platform in platforms:
            adapter = self._adapters.get(platform)
            if adapter and adapter.is_available():
                try:
                    events = await adapter.fetch_signals(time_window_hours, context)
                    all_events.extend(events)
                except Exception as e:
                    # Log error but continue with other platforms
                    # Never expose raw error details
                    pass

        self._last_scan = datetime.now()
        return all_events

    # =========================================================================
    # RISK CLASSIFICATION
    # =========================================================================

    def classify_risk(
        self,
        events: List[ReputationEvent]
    ) -> RiskClassification:
        """
        Classify a group of events by risk type.

        Classification priority (highest to lowest):
        1. POLICY_EXPOSURE - Regulatory/platform risk
        2. REVIEW_RISK - Direct reputation impact
        3. MISINFORMATION - False claims spreading
        4. HARASSMENT - Personal attacks
        5. PEER_CRITIQUE - Professional criticism
        6. MISINTERPRETATION - Content taken out of context

        Args:
            events: List of related events to classify

        Returns:
            RiskClassification enum value
        """
        if not events:
            return RiskClassification.MISINTERPRETATION

        # Aggregate signals from all events
        has_clinical_claims = any(e.contains_clinical_claims for e in events)
        has_personal_attacks = any(e.contains_personal_attack for e in events)
        avg_toxicity = sum(e.toxicity_score for e in events) / len(events)
        has_peer_authors = any(e.author_type == "peer" for e in events)
        has_review_platform = any(e.platform == Platform.REVIEW_SITE for e in events)

        # Classification logic

        # Policy exposure: clinical claims that could trigger regulatory attention
        if has_clinical_claims and any(
            e.reach_estimate > 10000 or e.engagement_velocity > 0.5
            for e in events
        ):
            return RiskClassification.POLICY_EXPOSURE

        # Review risk: negative signals on review platforms
        if has_review_platform:
            return RiskClassification.REVIEW_RISK

        # Misinformation: false clinical claims spreading
        if has_clinical_claims and not has_peer_authors:
            return RiskClassification.MISINFORMATION

        # Harassment: high toxicity personal attacks
        if has_personal_attacks and avg_toxicity > 0.6:
            return RiskClassification.HARASSMENT

        # Peer critique: criticism from professional peers
        if has_peer_authors:
            return RiskClassification.PEER_CRITIQUE

        # Default: misinterpretation
        return RiskClassification.MISINTERPRETATION

    # =========================================================================
    # RISK SCORING
    # =========================================================================

    def calculate_risk_score(
        self,
        events: List[ReputationEvent],
        classification: RiskClassification
    ) -> int:
        """
        Calculate risk score (0-100) for a group of events.

        Factors:
        - Sentiment severity
        - Toxicity level
        - Reach/audience size
        - Spread velocity
        - Classification base score
        - Author credibility (peer vs public vs bot)

        Args:
            events: List of events to score
            classification: The risk classification

        Returns:
            Risk score from 0 to 100
        """
        if not events:
            return 0

        # Start with classification base score
        base_score = self.CLASSIFICATION_BASE_SCORES.get(classification, 25)

        # Calculate component scores

        # Sentiment score (negative sentiment increases risk)
        avg_sentiment = sum(e.sentiment_score for e in events) / len(events)
        sentiment_score = max(0, -avg_sentiment) * 100  # Convert -1,0 to 0,100

        # Toxicity score
        max_toxicity = max(e.toxicity_score for e in events)
        toxicity_score = max_toxicity * 100

        # Reach score (logarithmic scale)
        total_reach = sum(e.reach_estimate for e in events)
        if total_reach > 0:
            import math
            reach_score = min(100, math.log10(total_reach) * 20)
        else:
            reach_score = 0

        # Velocity score
        max_velocity = max(e.engagement_velocity for e in events)
        velocity_score = min(100, max_velocity * 100)

        # Author credibility (peer criticism is higher risk)
        peer_ratio = sum(1 for e in events if e.author_type == "peer") / len(events)
        bot_ratio = sum(1 for e in events if e.bot_probability > 0.7) / len(events)
        # Bots reduce risk (not genuine), peers increase risk
        credibility_score = (peer_ratio * 80) - (bot_ratio * 30) + 20
        credibility_score = max(0, min(100, credibility_score))

        # Weighted combination
        weighted_score = (
            self.RISK_WEIGHTS["sentiment"] * sentiment_score +
            self.RISK_WEIGHTS["toxicity"] * toxicity_score +
            self.RISK_WEIGHTS["reach"] * reach_score +
            self.RISK_WEIGHTS["velocity"] * velocity_score +
            self.RISK_WEIGHTS["classification"] * base_score +
            self.RISK_WEIGHTS["author_credibility"] * credibility_score
        )

        # Apply coordination multiplier (coordinated attacks are higher risk)
        if any(e.is_coordinated for e in events):
            weighted_score *= 1.3

        return int(min(100, max(0, weighted_score)))

    # =========================================================================
    # STATE RECOMMENDATION
    # =========================================================================

    def recommend_state(
        self,
        risk_score: int,
        classification: RiskClassification
    ) -> AlfredState:
        """
        Recommend Alfred state based on risk assessment.

        State Definitions:
        - GREEN: No action required, normal operations
        - YELLOW: Elevated monitoring, restrict reactive content
        - RED: Active threat, all public-facing output paused

        Args:
            risk_score: Calculated risk score (0-100)
            classification: Risk classification

        Returns:
            Recommended AlfredState
        """
        # RED threshold (active threat)
        if risk_score >= self.RED_THRESHOLD:
            return AlfredState.RED

        # YELLOW threshold (elevated monitoring)
        if risk_score >= self.YELLOW_THRESHOLD:
            return AlfredState.YELLOW

        # Special cases that warrant YELLOW regardless of score
        if classification in [
            RiskClassification.POLICY_EXPOSURE,
            RiskClassification.REVIEW_RISK
        ] and risk_score >= 30:
            return AlfredState.YELLOW

        # GREEN (normal operations)
        return AlfredState.GREEN

    def _recommend_action(
        self,
        classification: RiskClassification,
        risk_score: int,
        state: AlfredState
    ) -> RecommendedAction:
        """
        Recommend action based on classification and risk.

        CRITICAL: NEVER recommend engagement. Options are:
        - IGNORE: No action needed
        - MONITOR: Keep watching
        - LONGFORM_CLARIFICATION: Address via long-form content (not replies)
        - SILENCE: Strategic silence is the response

        Args:
            classification: Risk classification
            risk_score: Calculated risk score
            state: Recommended state

        Returns:
            RecommendedAction (NEVER engagement)
        """
        # Low risk: ignore
        if risk_score < 25:
            return RecommendedAction.IGNORE

        # Harassment or trolling: always silence
        if classification == RiskClassification.HARASSMENT:
            return RecommendedAction.SILENCE

        # Misinformation with significant reach: may warrant clarification
        if classification == RiskClassification.MISINFORMATION and risk_score >= 50:
            return RecommendedAction.LONGFORM_CLARIFICATION

        # Peer critique: often best addressed via quality content over time
        if classification == RiskClassification.PEER_CRITIQUE:
            if risk_score >= 50:
                return RecommendedAction.LONGFORM_CLARIFICATION
            return RecommendedAction.MONITOR

        # Policy exposure: monitor closely, silence publicly
        if classification == RiskClassification.POLICY_EXPOSURE:
            return RecommendedAction.SILENCE

        # Review risk: cannot be addressed via engagement
        if classification == RiskClassification.REVIEW_RISK:
            return RecommendedAction.MONITOR

        # Misinterpretation: usually resolves or can be addressed in future content
        if classification == RiskClassification.MISINTERPRETATION:
            if risk_score >= 40:
                return RecommendedAction.LONGFORM_CLARIFICATION
            return RecommendedAction.MONITOR

        # Default: monitor
        return RecommendedAction.MONITOR

    # =========================================================================
    # PATTERN TRACKING
    # =========================================================================

    def track_pattern(
        self,
        events: List[ReputationEvent],
        classification: RiskClassification
    ) -> Optional[str]:
        """
        Track and detect patterns across signals over time.

        Pattern types:
        - escalating: Increasing frequency/severity
        - recurring: Same issue appearing repeatedly
        - coordinated: Multiple accounts, similar timing
        - seasonal: Tied to events/dates

        Args:
            events: Current events to analyze
            classification: Classification of these events

        Returns:
            Pattern note string if pattern detected, None otherwise
        """
        if not events:
            return None

        # Add events to history
        self._event_history.extend(events)
        self._prune_history()

        # Check for coordination
        coordinated_groups = self._topic_clusterer.detect_coordination(events)
        if coordinated_groups:
            pattern_id = f"coord_{datetime.now().strftime('%Y%m%d')}"
            if pattern_id not in self._active_patterns:
                self._active_patterns[pattern_id] = PatternRecord(
                    pattern_id=pattern_id,
                    pattern_type="coordinated",
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    classification=classification
                )
            return "Coordinated activity detected across multiple accounts"

        # Check for recurring patterns (same classification recently)
        recent_same_class = [
            e for e in self._event_history[-100:]
            if self._event_matches_classification(e, classification)
        ]
        if len(recent_same_class) >= 5:
            pattern_id = f"recurring_{classification.value}"
            if pattern_id in self._active_patterns:
                pattern = self._active_patterns[pattern_id]
                pattern.update(events[0], "Recurring signal detected")
                if pattern.event_count >= 10:
                    return f"Recurring pattern: {pattern.event_count} similar signals over {(datetime.now() - pattern.first_seen).days} days"
            else:
                self._active_patterns[pattern_id] = PatternRecord(
                    pattern_id=pattern_id,
                    pattern_type="recurring",
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    event_count=len(recent_same_class),
                    classification=classification
                )
                return "New recurring pattern emerging"

        # Check for escalation
        for pattern_id, pattern in self._active_patterns.items():
            if pattern.classification == classification:
                # Check if escalating
                recent_count = len([
                    e for e in self._event_history
                    if (datetime.now() - e.timestamp).days <= 7
                    and self._event_matches_classification(e, classification)
                ])
                older_count = len([
                    e for e in self._event_history
                    if 7 < (datetime.now() - e.timestamp).days <= 14
                    and self._event_matches_classification(e, classification)
                ])

                if recent_count > older_count * 1.5 and older_count > 0:
                    pattern.trend_direction = "rising"
                    return f"Escalating pattern: {classification.value} signals increasing"

        return None

    def _event_matches_classification(
        self,
        event: ReputationEvent,
        classification: RiskClassification
    ) -> bool:
        """Check if event would match the given classification."""
        # Simplified check based on event characteristics
        if classification == RiskClassification.HARASSMENT:
            return event.contains_personal_attack or event.toxicity_score > 0.6
        if classification == RiskClassification.MISINFORMATION:
            return event.contains_clinical_claims
        if classification == RiskClassification.REVIEW_RISK:
            return event.platform == Platform.REVIEW_SITE
        if classification == RiskClassification.PEER_CRITIQUE:
            return event.author_type == "peer"
        return True

    def _prune_history(self):
        """Remove events older than retention period."""
        cutoff = datetime.now() - timedelta(days=self._history_retention_days)
        self._event_history = [
            e for e in self._event_history
            if e.timestamp > cutoff
        ]

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked patterns."""
        return {
            pattern_id: {
                "type": pattern.pattern_type,
                "classification": pattern.classification.value if pattern.classification else None,
                "event_count": pattern.event_count,
                "trend": pattern.trend_direction,
                "first_seen": pattern.first_seen.isoformat(),
                "last_seen": pattern.last_seen.isoformat(),
                "platforms": [p.value for p in pattern.platforms_involved]
            }
            for pattern_id, pattern in self._active_patterns.items()
        }

    # =========================================================================
    # PACKET GENERATION
    # =========================================================================

    def generate_packet(
        self,
        events: List[ReputationEvent],
        classification: RiskClassification,
        risk_score: int,
        recommended_state: AlfredState,
        recommended_action: RecommendedAction,
        pattern_note: Optional[str] = None
    ) -> ReputationPacket:
        """
        Generate a ReputationPacket for Alfred.

        CRITICAL: This method generates neutral summaries WITHOUT exposing:
        - Raw content
        - Usernames
        - Specific quotes
        - Inflammatory language

        Args:
            events: Events being summarized
            classification: Risk classification
            risk_score: Calculated risk score
            recommended_state: Recommended state
            recommended_action: Recommended action
            pattern_note: Optional pattern note

        Returns:
            ReputationPacket with sanitized summary
        """
        # Generate neutral summary (never expose raw content)
        summary = self._generate_neutral_summary(events, classification)

        # Determine primary platform
        platform_counts: Dict[Platform, int] = defaultdict(int)
        for event in events:
            platform_counts[event.platform] += 1
        primary_platform = max(platform_counts, key=platform_counts.get) if platform_counts else Platform.OTHER

        # Generate rationale
        rationale = self._generate_rationale(
            classification, risk_score, recommended_action, len(events)
        )

        return ReputationPacket(
            event_summary=summary,
            platform=primary_platform,
            classification=classification,
            risk_score=risk_score,
            recommended_state=recommended_state,
            recommended_action=recommended_action,
            rationale=rationale,
            pattern_note=pattern_note
        )

    def _generate_neutral_summary(
        self,
        events: List[ReputationEvent],
        classification: RiskClassification
    ) -> str:
        """
        Generate a neutral, compressed summary of events.

        Rules:
        - No raw content
        - No usernames
        - No inflammatory language
        - Maximum 2 sentences
        - Focus on pattern, not details
        """
        if not events:
            return "No significant signals detected."

        count = len(events)
        platforms = set(e.platform.value for e in events)
        platform_str = ", ".join(platforms) if len(platforms) <= 2 else f"{len(platforms)} platforms"

        summaries = {
            RiskClassification.MISINTERPRETATION:
                f"{count} signal(s) on {platform_str} indicating content may be misinterpreted or taken out of context.",
            RiskClassification.MISINFORMATION:
                f"{count} signal(s) on {platform_str} involving inaccurate claims that may require clarification.",
            RiskClassification.HARASSMENT:
                f"{count} signal(s) on {platform_str} involving targeted negative attention. Silence recommended.",
            RiskClassification.PEER_CRITIQUE:
                f"{count} signal(s) on {platform_str} from professional peers expressing criticism or disagreement.",
            RiskClassification.REVIEW_RISK:
                f"{count} signal(s) on {platform_str} affecting public reviews or ratings.",
            RiskClassification.POLICY_EXPOSURE:
                f"{count} signal(s) on {platform_str} with potential regulatory or policy implications.",
        }

        return summaries.get(
            classification,
            f"{count} signal(s) detected on {platform_str} requiring attention."
        )

    def _generate_rationale(
        self,
        classification: RiskClassification,
        risk_score: int,
        action: RecommendedAction,
        signal_count: int
    ) -> str:
        """Generate a one-line rationale for the recommendation."""

        if action == RecommendedAction.IGNORE:
            return f"Low risk score ({risk_score}) and limited reach; normal operations continue."

        if action == RecommendedAction.SILENCE:
            return f"Engagement would amplify; strategic silence prevents escalation."

        if action == RecommendedAction.LONGFORM_CLARIFICATION:
            return f"Risk score {risk_score} with {signal_count} signals suggests proactive clarification via owned content."

        if action == RecommendedAction.MONITOR:
            return f"Moderate risk ({risk_score}); continue monitoring for escalation before action."

        return f"Risk assessment: {risk_score}/100 based on {signal_count} signals."

    # =========================================================================
    # SIGNAL PROCESSING HELPERS
    # =========================================================================

    def _filter_noise(
        self,
        events: List[ReputationEvent]
    ) -> List[ReputationEvent]:
        """
        Filter out noise from signal events.

        Noise includes:
        - Low engagement/reach
        - Bot-generated content
        - Neutral sentiment
        - Isolated incidents (no pattern)
        """
        significant = []

        for event in events:
            # Skip if likely bot
            if event.bot_probability > 0.8:
                continue

            # Skip if neutral sentiment and low toxicity
            if abs(event.sentiment_score) < 0.2 and event.toxicity_score < 0.3:
                continue

            # Skip if very low reach
            if event.reach_estimate < 100 and event.engagement_velocity < 0.1:
                continue

            significant.append(event)

        return significant

    def _compress_signals(
        self,
        events: List[ReputationEvent]
    ) -> List[List[ReputationEvent]]:
        """
        Compress multiple signals into grouped assessments.

        Groups events by:
        - Platform
        - Time window
        - Topic cluster
        """
        if not events:
            return []

        # Group by platform and time window
        groups: Dict[str, List[ReputationEvent]] = defaultdict(list)

        for event in events:
            # Create group key
            time_bucket = event.timestamp.replace(
                minute=0, second=0, microsecond=0
            )
            time_bucket = time_bucket.replace(
                hour=(time_bucket.hour // self._compression_window_hours) * self._compression_window_hours
            )
            group_key = f"{event.platform.value}_{time_bucket.isoformat()}"

            # Add topic cluster if available
            if event.topic_cluster:
                group_key += f"_{event.topic_cluster}"

            groups[group_key].append(event)

        return list(groups.values())

    def _create_silence_response(
        self,
        platforms_scanned: List[Platform],
        time_window_hours: int,
        total_signals: int
    ) -> AgentResponse:
        """Create response when no actionable signals found."""
        self._consecutive_clear_scans += 1

        silence_packet = SilencePacket(
            platforms_scanned=platforms_scanned,
            time_window_hours=time_window_hours,
            signals_processed=total_signals,
            signals_filtered=total_signals,  # All filtered as noise
            recommended_state=AlfredState.GREEN
        )

        return self.create_response(
            data=silence_packet.to_dict(),
            success=True
        )

    # =========================================================================
    # STATE AND STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the Reputation Sentinel."""
        return {
            "agent": self.name,
            "monitoring_level": self.get_monitoring_level(),
            "alfred_state": self.alfred_state.value,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "consecutive_clear_scans": self._consecutive_clear_scans,
            "active_patterns": len(self._active_patterns),
            "history_size": len(self._event_history),
            "adapters": {
                platform.value: adapter.is_available()
                for platform, adapter in self._adapters.items()
            }
        }

    def reset_clear_scan_counter(self):
        """Reset the consecutive clear scans counter (e.g., after state change)."""
        self._consecutive_clear_scans = 0

    # =========================================================================
    # ENFORCEMENT OF HARD RULES
    # =========================================================================

    def _validate_action(self, action: RecommendedAction) -> RecommendedAction:
        """
        HARD RULE ENFORCEMENT: Ensure we NEVER recommend engagement.

        This is a safety check that should never trigger, but exists as
        a final guard against any code path that might violate the hard rule.
        """
        # These would be engagement - we don't have them as options, but guard anyway
        forbidden_actions = {"REPLY", "COMMENT", "ENGAGE", "RESPOND", "QUOTE_TWEET"}

        if action.value.upper() in forbidden_actions:
            # This should never happen, but if it does, default to SILENCE
            return RecommendedAction.SILENCE

        return action


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_reputation_check_request(
    scope: List[str],
    time_window_hours: int = 24,
    context: str = "",
    priority: str = "routine"
) -> ReputationCheckRequest:
    """
    Convenience function to create a ReputationCheckRequest.

    Args:
        scope: List of platform names (e.g., ["Twitter", "YouTube"])
        time_window_hours: How many hours back to check
        context: Any relevant context
        priority: "routine", "elevated", or "urgent"

    Returns:
        ReputationCheckRequest object
    """
    platforms = []
    for p in scope:
        try:
            platforms.append(Platform(p))
        except ValueError:
            platforms.append(Platform.OTHER)

    return ReputationCheckRequest(
        scope=platforms,
        time_window_hours=time_window_hours,
        context=context,
        priority=Priority(priority)
    )


async def quick_scan(
    sentinel: ReputationSentinel,
    platforms: Optional[List[str]] = None,
    hours: int = 24
) -> AgentResponse:
    """
    Convenience function for quick reputation scan.

    Args:
        sentinel: Configured ReputationSentinel instance
        platforms: List of platform names (defaults to all)
        hours: Hours to look back

    Returns:
        AgentResponse with scan results
    """
    if platforms is None:
        platforms = ["Twitter", "YouTube", "Substack"]

    request = create_reputation_check_request(
        scope=platforms,
        time_window_hours=hours,
        priority="routine"
    )

    return await sentinel.process_request(request)
