# Social Triage Agent
# Processes comments and mentions for content opportunities
# Distinct from Reputation Sentinel (risk monitoring) - this extracts value from interactions

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import re
from collections import Counter, defaultdict

from . import SignalAgent, AgentResponse, AlfredState


class CommentType(Enum):
    """Classification types for social comments."""
    QUESTION = "QUESTION"       # Genuine query seeking information
    CONFUSION = "CONFUSION"     # Misunderstanding that content could clarify
    OBJECTION = "OBJECTION"     # Disagreement that could inform future content
    PRAISE = "PRAISE"           # Positive feedback indicating resonance
    CRITIQUE = "CRITIQUE"       # Valid criticism worth considering
    NOISE = "NOISE"             # Low-signal, ignore
    TROLL = "TROLL"             # Bad-faith, ignore
    PEER = "PEER"               # From recognized professional


class Platform(Enum):
    """Supported social platforms."""
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    SUBSTACK = "substack"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"


class TriageDepth(Enum):
    """Analysis depth levels."""
    QUICK = "quick"             # Fast scan, surface-level analysis
    STANDARD = "standard"       # Normal processing depth
    COMPREHENSIVE = "comprehensive"  # Deep analysis with full theme extraction


@dataclass
class Comment:
    """Represents a single social media comment or mention."""
    id: str
    platform: Platform
    content: str
    author: str
    author_is_peer: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    parent_content_id: Optional[str] = None  # The post/video this comment is on
    parent_content_title: Optional[str] = None
    engagement_count: int = 0  # Likes/upvotes on this comment
    reply_count: int = 0
    classification: Optional[CommentType] = None
    signal_score: float = 0.0  # 0-1, higher = more signal
    themes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure platform is Platform enum."""
        if isinstance(self.platform, str):
            self.platform = Platform(self.platform.lower())


@dataclass
class ContentOpportunity:
    """A content opportunity identified from comment analysis."""
    theme: str
    opportunity_type: str  # "article", "video", "thread", "faq_entry"
    frequency: int  # How often this theme appears
    example_comments: List[str] = field(default_factory=list)
    suggestion: str = ""
    priority_score: float = 0.0  # 0-1, based on frequency and signal strength
    platforms_seen: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Convert platforms_seen to set if needed."""
        if isinstance(self.platforms_seen, list):
            self.platforms_seen = set(self.platforms_seen)


@dataclass
class RecurringConfusion:
    """A misconception appearing repeatedly in comments."""
    misconception: str
    frequency: int
    example_comment: str
    suggested_clarification: str
    content_that_triggered: List[str] = field(default_factory=list)


@dataclass
class PraisePattern:
    """Pattern of positive feedback."""
    topic_or_format: str
    frequency: int
    example_quotes: List[str] = field(default_factory=list)
    what_resonates: str = ""


@dataclass
class PeerEngagement:
    """Notable engagement from recognized professionals."""
    peer_name: str
    platform: Platform
    comment_summary: str
    engagement_type: CommentType
    timestamp: datetime
    potential_value: str  # Why this engagement matters


@dataclass
class TriageReport:
    """Complete social triage report output."""
    report_date: datetime
    period_start: datetime
    period_end: datetime
    platforms_analyzed: List[Platform]
    comments_processed: int
    signal_rate: float  # Percentage of comments worth surfacing

    # Main sections
    content_opportunities: List[ContentOpportunity] = field(default_factory=list)
    recurring_confusions: List[RecurringConfusion] = field(default_factory=list)
    praise_patterns: List[PraisePattern] = field(default_factory=list)
    objections_to_address: List[Dict[str, Any]] = field(default_factory=list)
    peer_engagements: List[PeerEngagement] = field(default_factory=list)
    high_signal_comments: List[Comment] = field(default_factory=list)
    themes_summary: str = ""

    # Metadata
    depth: TriageDepth = TriageDepth.STANDARD
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "report_date": self.report_date.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            },
            "platforms": [p.value for p in self.platforms_analyzed],
            "comments_processed": self.comments_processed,
            "signal_rate": f"{self.signal_rate:.1%}",
            "content_opportunities": [
                {
                    "theme": opp.theme,
                    "frequency": opp.frequency,
                    "example": opp.example_comments[0] if opp.example_comments else "",
                    "suggestion": opp.suggestion,
                    "priority": opp.priority_score
                }
                for opp in self.content_opportunities
            ],
            "recurring_confusions": [
                {
                    "misconception": conf.misconception,
                    "frequency": conf.frequency,
                    "example": conf.example_comment,
                    "clarification_needed": conf.suggested_clarification
                }
                for conf in self.recurring_confusions
            ],
            "praise_patterns": [
                {
                    "topic": pattern.topic_or_format,
                    "frequency": pattern.frequency,
                    "what_resonates": pattern.what_resonates,
                    "examples": pattern.example_quotes[:3]
                }
                for pattern in self.praise_patterns
            ],
            "objections_to_address": self.objections_to_address,
            "peer_engagements": [
                {
                    "peer": pe.peer_name,
                    "platform": pe.platform.value,
                    "summary": pe.comment_summary,
                    "type": pe.engagement_type.value,
                    "value": pe.potential_value
                }
                for pe in self.peer_engagements
            ],
            "high_signal_comments": [
                {
                    "platform": c.platform.value,
                    "author": c.author,
                    "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    "type": c.classification.value if c.classification else "UNCLASSIFIED",
                    "signal_score": c.signal_score
                }
                for c in self.high_signal_comments[:10]  # Limit to top 10
            ],
            "themes_summary": self.themes_summary,
            "depth": self.depth.value,
            "warnings": self.warnings
        }


class SocialTriage(SignalAgent):
    """
    Social Triage Agent - Extracts value from audience interactions.

    Processes comments and mentions for content opportunities.
    Distinct from Reputation Sentinel (which monitors for risk).

    HARD RULE: Never recommend comment-thread engagement.
    All responses via dedicated content.
    """

    # Known peer identifiers (would be configurable in production)
    KNOWN_PEERS: Set[str] = set()

    # Signal threshold for comments to be surfaced
    SIGNAL_THRESHOLD = 0.3

    # Question indicators
    QUESTION_PATTERNS = [
        r'\?$', r'\?["\']', r'^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does)',
        r'wondering', r'curious', r'asking', r'question', r'anyone know', r'help me understand'
    ]

    # Confusion indicators
    CONFUSION_PATTERNS = [
        r"i (don'?t|dont) (understand|get)", r"confused", r"wait,?\s*(so|what)",
        r"but (i thought|isn'?t)", r"doesn'?t (this|that) (mean|contradict)",
        r"i'?m lost", r"can you clarify", r"what do you mean"
    ]

    # Praise indicators
    PRAISE_PATTERNS = [
        r'(great|excellent|amazing|awesome|fantastic|wonderful|brilliant|best)\s+(video|article|post|content|explanation)',
        r'thank(s| you)', r'helped me', r'finally understand', r'this (is|was) (so )?helpful',
        r'love (this|your)', r'subscribed', r'following', r'game.?changer', r'eye.?opening'
    ]

    # Troll/bad-faith indicators
    TROLL_PATTERNS = [
        r'(you\'?re|ur) (an? )?(idiot|moron|stupid|dumb)', r'shill', r'paid by',
        r'wake up', r'sheep', r'fake (news|doctor)', r'(big )?pharma', r'conspiracy',
        r'!!+', r'CAPS.{20,}CAPS', r'clown', r'fraud'
    ]

    # Objection indicators (substantive disagreement)
    OBJECTION_PATTERNS = [
        r"i disagree", r"but (what about|isn'?t)", r"counterpoint",
        r"study (shows|says)", r"evidence (suggests|shows)", r"actually,",
        r"this (ignores|misses|overlooks)", r"have you considered"
    ]

    # Critique indicators (valid criticism)
    CRITIQUE_PATTERNS = [
        r"could (be )?better", r"missing", r"you (didn'?t|forgot to) (mention|address|cover)",
        r"oversimplif", r"nuance", r"more (detail|depth)", r"bias(ed)?",
        r"one.?sided", r"fair point but"
    ]

    def __init__(self, known_peers: Optional[Set[str]] = None):
        """
        Initialize Social Triage agent.

        Args:
            known_peers: Set of known peer/professional identifiers (usernames, etc.)
        """
        super().__init__("Social Triage")
        if known_peers:
            self.KNOWN_PEERS = known_peers

        # Compiled regex patterns for efficiency
        self._question_re = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS]
        self._confusion_re = [re.compile(p, re.IGNORECASE) for p in self.CONFUSION_PATTERNS]
        self._praise_re = [re.compile(p, re.IGNORECASE) for p in self.PRAISE_PATTERNS]
        self._troll_re = [re.compile(p, re.IGNORECASE) for p in self.TROLL_PATTERNS]
        self._objection_re = [re.compile(p, re.IGNORECASE) for p in self.OBJECTION_PATTERNS]
        self._critique_re = [re.compile(p, re.IGNORECASE) for p in self.CRITIQUE_PATTERNS]

    def check_state_permission(self) -> Tuple[bool, str]:
        """
        Check if recommendations are permitted in current state.

        Signal agents continue operating in all states, but recommendations
        for engagement are blocked in YELLOW/RED states.
        """
        # Analysis always permitted, but recommendations restricted
        if self.alfred_state in (AlfredState.YELLOW, AlfredState.RED):
            return True, "Analysis permitted, engagement recommendations blocked"
        return True, "Full operation permitted"

    def _can_recommend_engagement(self) -> bool:
        """Check if engagement recommendations are allowed."""
        return self.alfred_state == AlfredState.GREEN

    def classify_comment(self, comment: Comment) -> Comment:
        """
        Classify a single comment by type and calculate signal score.

        Args:
            comment: Comment to classify

        Returns:
            Comment with classification and signal_score populated
        """
        content_lower = comment.content.lower()

        # Check for peer status first
        if comment.author_is_peer or comment.author in self.KNOWN_PEERS:
            comment.classification = CommentType.PEER
            comment.signal_score = 0.9  # Peer comments are high signal
            return comment

        # Check for troll/bad-faith (filter these out early)
        for pattern in self._troll_re:
            if pattern.search(content_lower):
                comment.classification = CommentType.TROLL
                comment.signal_score = 0.0  # Zero signal, ignore
                return comment

        # Score different classification types
        scores = {
            CommentType.QUESTION: 0.0,
            CommentType.CONFUSION: 0.0,
            CommentType.OBJECTION: 0.0,
            CommentType.PRAISE: 0.0,
            CommentType.CRITIQUE: 0.0,
            CommentType.NOISE: 0.0
        }

        # Check question patterns
        for pattern in self._question_re:
            if pattern.search(comment.content):
                scores[CommentType.QUESTION] += 0.3

        # Check confusion patterns
        for pattern in self._confusion_re:
            if pattern.search(content_lower):
                scores[CommentType.CONFUSION] += 0.35

        # Check praise patterns
        for pattern in self._praise_re:
            if pattern.search(content_lower):
                scores[CommentType.PRAISE] += 0.25

        # Check objection patterns
        for pattern in self._objection_re:
            if pattern.search(content_lower):
                scores[CommentType.OBJECTION] += 0.35

        # Check critique patterns
        for pattern in self._critique_re:
            if pattern.search(content_lower):
                scores[CommentType.CRITIQUE] += 0.3

        # Length and engagement bonuses
        word_count = len(comment.content.split())
        if word_count > 30:  # Longer comments often more substantive
            for key in scores:
                if key != CommentType.NOISE:
                    scores[key] *= 1.2

        if comment.engagement_count > 10:  # Others found it valuable
            for key in scores:
                if key != CommentType.NOISE:
                    scores[key] *= 1.1

        # Find highest scoring classification
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        # If no strong signal, classify as noise
        if max_score < 0.2:
            comment.classification = CommentType.NOISE
            comment.signal_score = 0.1
        else:
            comment.classification = max_type
            comment.signal_score = min(max_score, 1.0)

        return comment

    def filter_signal(self, comments: List[Comment]) -> Tuple[List[Comment], List[Comment]]:
        """
        Separate high-signal comments from noise.

        Args:
            comments: List of classified comments

        Returns:
            Tuple of (signal_comments, noise_comments)
        """
        signal = []
        noise = []

        for comment in comments:
            if comment.classification in (CommentType.NOISE, CommentType.TROLL):
                noise.append(comment)
            elif comment.signal_score >= self.SIGNAL_THRESHOLD:
                signal.append(comment)
            else:
                noise.append(comment)

        # Sort signal comments by score descending
        signal.sort(key=lambda c: c.signal_score, reverse=True)

        return signal, noise

    def extract_themes(self, comments: List[Comment], depth: TriageDepth) -> Dict[str, List[Comment]]:
        """
        Extract recurring themes from comments.

        Args:
            comments: List of signal comments to analyze
            depth: Analysis depth level

        Returns:
            Dictionary mapping theme keywords to relevant comments
        """
        # Word frequency analysis (excluding common stopwords)
        STOPWORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'your',
            'my', 'me', 'and', 'or', 'but', 'if', 'for', 'on', 'with', 'as', 'by',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'to', 'from', 'up', 'down', 'in', 'out', 'of', 'off', 'over',
            'under', 'again', 'then', 'once', 'here', 'there', 'any', 'many'
        }

        # Extract words and bigrams
        word_counts = Counter()
        bigram_counts = Counter()
        word_to_comments = defaultdict(list)

        for comment in comments:
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]{3,}\b', comment.content.lower())
            filtered_words = [w for w in words if w not in STOPWORDS]

            # Count words
            for word in set(filtered_words):  # Use set to count once per comment
                word_counts[word] += 1
                word_to_comments[word].append(comment)

            # Count bigrams for comprehensive depth
            if depth == TriageDepth.COMPREHENSIVE:
                for i in range(len(filtered_words) - 1):
                    bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
                    bigram_counts[bigram] += 1

        # Build theme dictionary
        themes = {}

        # Get top themes based on depth
        if depth == TriageDepth.QUICK:
            top_n = 5
        elif depth == TriageDepth.STANDARD:
            top_n = 10
        else:  # COMPREHENSIVE
            top_n = 20

        # Add word-based themes (minimum 2 occurrences)
        for word, count in word_counts.most_common(top_n):
            if count >= 2:
                themes[word] = word_to_comments[word]

        # Add bigram themes for comprehensive analysis
        if depth == TriageDepth.COMPREHENSIVE:
            for bigram, count in bigram_counts.most_common(top_n // 2):
                if count >= 2:
                    # Find comments containing this bigram
                    relevant = [c for c in comments if bigram in c.content.lower()]
                    if relevant:
                        themes[bigram] = relevant

        return themes

    def identify_opportunities(
        self,
        comments: List[Comment],
        themes: Dict[str, List[Comment]]
    ) -> List[ContentOpportunity]:
        """
        Identify content opportunities from classified comments and themes.

        Args:
            comments: List of signal comments
            themes: Extracted themes mapping

        Returns:
            List of content opportunities sorted by priority
        """
        opportunities = []

        # Group comments by classification for opportunity detection
        by_type = defaultdict(list)
        for comment in comments:
            if comment.classification:
                by_type[comment.classification].append(comment)

        # Questions -> FAQ entries or explainer content
        questions = by_type[CommentType.QUESTION]
        if len(questions) >= 2:
            # Cluster similar questions
            question_texts = [q.content for q in questions]
            opportunity = ContentOpportunity(
                theme="Frequently Asked Questions",
                opportunity_type="faq_entry",
                frequency=len(questions),
                example_comments=question_texts[:3],
                suggestion="Create FAQ content or explainer addressing these common questions",
                priority_score=min(len(questions) / 10, 1.0),
                platforms_seen={q.platform.value for q in questions}
            )
            opportunities.append(opportunity)

        # Confusions -> Clarification content
        confusions = by_type[CommentType.CONFUSION]
        if len(confusions) >= 2:
            opportunity = ContentOpportunity(
                theme="Audience Misconceptions",
                opportunity_type="article",
                frequency=len(confusions),
                example_comments=[c.content for c in confusions[:3]],
                suggestion="Write clarification piece addressing common misunderstandings",
                priority_score=min(len(confusions) / 8, 1.0),  # Weight confusions higher
                platforms_seen={c.platform.value for c in confusions}
            )
            opportunities.append(opportunity)

        # Objections -> Address in future content
        objections = by_type[CommentType.OBJECTION]
        if objections:
            opportunity = ContentOpportunity(
                theme="Valid Objections to Address",
                opportunity_type="article",
                frequency=len(objections),
                example_comments=[o.content for o in objections[:3]],
                suggestion="Consider these objections when creating future content on this topic",
                priority_score=min(len(objections) / 5, 0.8),
                platforms_seen={o.platform.value for o in objections}
            )
            opportunities.append(opportunity)

        # Theme-based opportunities
        for theme, theme_comments in themes.items():
            if len(theme_comments) >= 3:
                # Determine best content type based on comment types in theme
                theme_types = [c.classification for c in theme_comments if c.classification]
                most_common_type = Counter(theme_types).most_common(1)

                if most_common_type:
                    dominant_type = most_common_type[0][0]
                    if dominant_type == CommentType.QUESTION:
                        content_type = "video"
                        suggestion = f"Create educational video explaining '{theme}'"
                    elif dominant_type == CommentType.CONFUSION:
                        content_type = "article"
                        suggestion = f"Write clarification piece on '{theme}'"
                    elif dominant_type == CommentType.PRAISE:
                        content_type = "thread"
                        suggestion = f"Topic '{theme}' resonates - consider more content"
                    else:
                        content_type = "article"
                        suggestion = f"Address audience interest in '{theme}'"

                    opportunity = ContentOpportunity(
                        theme=theme,
                        opportunity_type=content_type,
                        frequency=len(theme_comments),
                        example_comments=[c.content for c in theme_comments[:2]],
                        suggestion=suggestion,
                        priority_score=min(len(theme_comments) / 10, 0.9),
                        platforms_seen={c.platform.value for c in theme_comments}
                    )
                    opportunities.append(opportunity)

        # Sort by priority
        opportunities.sort(key=lambda o: o.priority_score, reverse=True)

        # Deduplicate similar opportunities
        seen_themes = set()
        unique_opportunities = []
        for opp in opportunities:
            theme_key = opp.theme.lower()[:20]
            if theme_key not in seen_themes:
                seen_themes.add(theme_key)
                unique_opportunities.append(opp)

        return unique_opportunities[:10]  # Return top 10 opportunities

    def _extract_recurring_confusions(self, comments: List[Comment]) -> List[RecurringConfusion]:
        """Extract recurring misconceptions from confusion-type comments."""
        confusions = [c for c in comments if c.classification == CommentType.CONFUSION]

        if len(confusions) < 2:
            return []

        # Group by similarity (simplified - production would use embeddings)
        recurring = []

        # Look for common confusion patterns
        confusion_themes = self.extract_themes(confusions, TriageDepth.QUICK)

        for theme, theme_comments in confusion_themes.items():
            if len(theme_comments) >= 2:
                recurring.append(RecurringConfusion(
                    misconception=f"Confusion about: {theme}",
                    frequency=len(theme_comments),
                    example_comment=theme_comments[0].content,
                    suggested_clarification=f"Clarify the concept of '{theme}' in dedicated content",
                    content_that_triggered=[c.parent_content_title or "Unknown"
                                           for c in theme_comments if c.parent_content_title]
                ))

        return recurring[:5]  # Top 5 confusions

    def _extract_praise_patterns(self, comments: List[Comment]) -> List[PraisePattern]:
        """Identify what content/topics generate positive responses."""
        praise = [c for c in comments if c.classification == CommentType.PRAISE]

        if not praise:
            return []

        # Group by parent content
        by_content = defaultdict(list)
        for p in praise:
            key = p.parent_content_title or "General"
            by_content[key].append(p)

        patterns = []
        for topic, topic_praise in by_content.items():
            if len(topic_praise) >= 2:
                patterns.append(PraisePattern(
                    topic_or_format=topic,
                    frequency=len(topic_praise),
                    example_quotes=[p.content for p in topic_praise[:3]],
                    what_resonates=f"Content on '{topic}' resonates with audience"
                ))

        # Also extract common praise themes
        praise_themes = self.extract_themes(praise, TriageDepth.QUICK)
        for theme, theme_praise in praise_themes.items():
            if len(theme_praise) >= 3:
                patterns.append(PraisePattern(
                    topic_or_format=f"Theme: {theme}",
                    frequency=len(theme_praise),
                    example_quotes=[p.content for p in theme_praise[:2]],
                    what_resonates=f"Audience appreciates content involving '{theme}'"
                ))

        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns[:5]

    def _extract_peer_engagements(self, comments: List[Comment]) -> List[PeerEngagement]:
        """Extract notable engagements from recognized professionals."""
        peers = [c for c in comments if c.classification == CommentType.PEER]

        engagements = []
        for peer_comment in peers:
            # Determine engagement type by re-analyzing without peer status
            temp_comment = Comment(
                id=peer_comment.id,
                platform=peer_comment.platform,
                content=peer_comment.content,
                author=peer_comment.author,
                author_is_peer=False,
                timestamp=peer_comment.timestamp
            )
            self.classify_comment(temp_comment)

            # Determine value
            if temp_comment.classification == CommentType.QUESTION:
                value = "Peer has question - potential collaboration or clarification opportunity"
            elif temp_comment.classification == CommentType.PRAISE:
                value = "Professional endorsement - authority building"
            elif temp_comment.classification == CommentType.OBJECTION:
                value = "Professional disagreement - worth careful consideration"
            elif temp_comment.classification == CommentType.CRITIQUE:
                value = "Professional feedback - valuable for improvement"
            else:
                value = "Peer visibility - professional network engagement"

            engagements.append(PeerEngagement(
                peer_name=peer_comment.author,
                platform=peer_comment.platform,
                comment_summary=peer_comment.content[:150] + "..." if len(peer_comment.content) > 150 else peer_comment.content,
                engagement_type=temp_comment.classification or CommentType.PEER,
                timestamp=peer_comment.timestamp,
                potential_value=value
            ))

        return engagements

    def _generate_themes_summary(
        self,
        themes: Dict[str, List[Comment]],
        comments: List[Comment]
    ) -> str:
        """Generate narrative summary of overall themes."""
        if not themes and not comments:
            return "Insufficient data for theme summary."

        # Analyze comment type distribution
        type_counts = Counter(c.classification for c in comments if c.classification)
        total = len(comments)

        summary_parts = []

        # Overall sentiment
        positive = type_counts.get(CommentType.PRAISE, 0)
        negative = type_counts.get(CommentType.OBJECTION, 0) + type_counts.get(CommentType.CRITIQUE, 0)

        if total > 0:
            if positive > negative * 2:
                summary_parts.append("Overall audience sentiment is strongly positive.")
            elif positive > negative:
                summary_parts.append("Audience sentiment is generally positive with some constructive criticism.")
            elif negative > positive:
                summary_parts.append("Notable amount of objections and critiques - worth reviewing.")
            else:
                summary_parts.append("Mixed audience response with balanced positive and critical feedback.")

        # Questions indicate knowledge gaps
        questions = type_counts.get(CommentType.QUESTION, 0)
        if questions > total * 0.3:
            summary_parts.append(f"High volume of questions ({questions}) suggests audience knowledge gaps to address.")

        # Confusions indicate need for clarification
        confusions = type_counts.get(CommentType.CONFUSION, 0)
        if confusions >= 3:
            summary_parts.append(f"Multiple instances of confusion ({confusions}) indicate topics needing clarification.")

        # Top themes
        if themes:
            top_themes = list(themes.keys())[:5]
            summary_parts.append(f"Top recurring themes: {', '.join(top_themes)}.")

        return " ".join(summary_parts) if summary_parts else "No significant patterns detected."

    def process_comments(
        self,
        comments: List[Comment],
        depth: TriageDepth = TriageDepth.STANDARD
    ) -> Tuple[List[Comment], List[Comment]]:
        """
        Process and classify a batch of comments.

        Args:
            comments: Raw comments to process
            depth: Analysis depth level

        Returns:
            Tuple of (classified_signal_comments, noise_comments)
        """
        # Classify all comments
        for comment in comments:
            self.classify_comment(comment)

        # Separate signal from noise
        signal, noise = self.filter_signal(comments)

        return signal, noise

    def generate_report(
        self,
        comments: List[Comment],
        platforms: List[Platform],
        time_window_start: datetime,
        time_window_end: datetime,
        depth: TriageDepth = TriageDepth.STANDARD
    ) -> TriageReport:
        """
        Generate complete social triage report.

        Args:
            comments: Raw comments to analyze
            platforms: Platforms being analyzed
            time_window_start: Start of analysis period
            time_window_end: End of analysis period
            depth: Analysis depth level

        Returns:
            Complete TriageReport
        """
        # Process comments
        signal_comments, noise_comments = self.process_comments(comments, depth)

        # Calculate signal rate
        total = len(comments)
        signal_rate = len(signal_comments) / total if total > 0 else 0.0

        # Extract themes
        themes = self.extract_themes(signal_comments, depth)

        # Identify opportunities
        opportunities = self.identify_opportunities(signal_comments, themes)

        # Extract specific patterns
        confusions = self._extract_recurring_confusions(signal_comments)
        praise_patterns = self._extract_praise_patterns(signal_comments)
        peer_engagements = self._extract_peer_engagements(signal_comments)

        # Extract objections for addressing
        objections = [
            {
                "objection": c.content,
                "platform": c.platform.value,
                "engagement": c.engagement_count,
                "parent_content": c.parent_content_title
            }
            for c in signal_comments
            if c.classification == CommentType.OBJECTION
        ][:5]

        # Select high-signal comments (top by score)
        high_signal = sorted(signal_comments, key=lambda c: c.signal_score, reverse=True)[:10]

        # Generate summary
        themes_summary = self._generate_themes_summary(themes, signal_comments)

        # Build warnings
        warnings = []
        if not self._can_recommend_engagement():
            warnings.append(
                f"Alfred state is {self.alfred_state.value} - engagement recommendations blocked"
            )
        if signal_rate < 0.1:
            warnings.append("Very low signal rate - comments may be predominantly noise")
        if len(comments) < 10:
            warnings.append("Small sample size - patterns may not be representative")

        return TriageReport(
            report_date=datetime.now(),
            period_start=time_window_start,
            period_end=time_window_end,
            platforms_analyzed=platforms,
            comments_processed=total,
            signal_rate=signal_rate,
            content_opportunities=opportunities,
            recurring_confusions=confusions,
            praise_patterns=praise_patterns,
            objections_to_address=objections,
            peer_engagements=peer_engagements,
            high_signal_comments=high_signal,
            themes_summary=themes_summary,
            depth=depth,
            warnings=warnings
        )

    def execute(
        self,
        platforms: List[str],
        time_window_days: int = 7,
        content_focus: Optional[List[str]] = None,
        depth: str = "standard",
        comments_data: Optional[List[Dict[str, Any]]] = None
    ) -> AgentResponse:
        """
        Execute social triage analysis.

        This is the main entry point when commissioned by Alfred.

        Args:
            platforms: List of platform names to analyze
            time_window_days: Number of days to analyze
            content_focus: Specific content IDs to focus on (or all)
            depth: Analysis depth ("quick", "standard", "comprehensive")
            comments_data: Raw comment data (would come from platform APIs)

        Returns:
            AgentResponse with triage report
        """
        # Check permissions
        permitted, reason = self.check_state_permission()

        # Convert inputs
        platform_enums = []
        for p in platforms:
            try:
                platform_enums.append(Platform(p.lower()))
            except ValueError:
                pass  # Skip unknown platforms

        if not platform_enums:
            return self.blocked_response("No valid platforms specified")

        try:
            depth_enum = TriageDepth(depth.lower())
        except ValueError:
            depth_enum = TriageDepth.STANDARD

        # Calculate time window (add small buffer to time_end for edge cases)
        time_end = datetime.now() + timedelta(seconds=1)
        time_start = time_end - timedelta(days=time_window_days)

        # Default timestamp for comments without one (within the window)
        default_timestamp = datetime.now()

        # Convert raw comment data to Comment objects
        comments = []
        if comments_data:
            for cd in comments_data:
                try:
                    comment = Comment(
                        id=cd.get("id", str(len(comments))),
                        platform=cd.get("platform", "twitter"),
                        content=cd.get("content", ""),
                        author=cd.get("author", "unknown"),
                        author_is_peer=cd.get("author_is_peer", False),
                        timestamp=datetime.fromisoformat(cd["timestamp"]) if "timestamp" in cd else default_timestamp,
                        parent_content_id=cd.get("parent_content_id"),
                        parent_content_title=cd.get("parent_content_title"),
                        engagement_count=cd.get("engagement_count", 0),
                        reply_count=cd.get("reply_count", 0)
                    )

                    # Filter by content focus if specified
                    if content_focus is None or comment.parent_content_id in content_focus:
                        # Filter by time window
                        if time_start <= comment.timestamp <= time_end:
                            comments.append(comment)
                except (KeyError, ValueError) as e:
                    continue  # Skip malformed comment data

        # Generate report
        report = self.generate_report(
            comments=comments,
            platforms=platform_enums,
            time_window_start=time_start,
            time_window_end=time_end,
            depth=depth_enum
        )

        # Build response data
        response_data = {
            "report": report.to_dict(),
            "monitoring_level": self.get_monitoring_level(),
            "engagement_recommendations_available": self._can_recommend_engagement()
        }

        # Add warnings about state if not GREEN
        warnings = report.warnings.copy()
        if not self._can_recommend_engagement():
            warnings.append(
                "HARD RULE REMINDER: Never recommend comment-thread engagement. "
                "All responses via dedicated content (articles, videos, threads)."
            )

        return self.create_response(
            data=response_data,
            success=True,
            warnings=warnings
        )

    def format_output(self, report: TriageReport) -> str:
        """
        Format report as structured text output for Alfred.

        Args:
            report: Completed triage report

        Returns:
            Formatted string in specification format
        """
        lines = [
            "SOCIAL_TRIAGE_REPORT",
            f"- Report Date: {report.report_date.isoformat()}",
            f"- Period: {report.period_start.date()} to {report.period_end.date()}",
            f"- Platforms: {', '.join(p.value for p in report.platforms_analyzed)}",
            f"- Comments Processed: {report.comments_processed}",
            f"- Signal Rate: {report.signal_rate:.1%}",
            ""
        ]

        # Content Opportunities
        lines.append("- Content Opportunities:")
        if report.content_opportunities:
            for i, opp in enumerate(report.content_opportunities[:5], 1):
                lines.append(f"  {i}. {opp.theme}")
                lines.append(f"     - Frequency: {opp.frequency}")
                if opp.example_comments:
                    example = opp.example_comments[0][:100] + "..." if len(opp.example_comments[0]) > 100 else opp.example_comments[0]
                    lines.append(f'     - Example: "{example}"')
                lines.append(f"     - Content Suggestion: {opp.suggestion}")
        else:
            lines.append("  (No opportunities identified)")
        lines.append("")

        # Recurring Confusions
        lines.append("- Recurring Confusions:")
        if report.recurring_confusions:
            for conf in report.recurring_confusions[:3]:
                lines.append(f"  - {conf.misconception} (appeared {conf.frequency}x)")
        else:
            lines.append("  (None identified)")
        lines.append("")

        # Praise Patterns
        lines.append("- Praise Patterns:")
        if report.praise_patterns:
            for pattern in report.praise_patterns[:3]:
                lines.append(f"  - {pattern.topic_or_format}: {pattern.what_resonates}")
        else:
            lines.append("  (Insufficient data)")
        lines.append("")

        # Objections to Address
        lines.append("- Objections to Address:")
        if report.objections_to_address:
            for obj in report.objections_to_address[:3]:
                objection_text = obj["objection"][:100] + "..." if len(obj["objection"]) > 100 else obj["objection"]
                lines.append(f'  - "{objection_text}"')
        else:
            lines.append("  (None requiring attention)")
        lines.append("")

        # Peer Engagement
        lines.append("- Peer Engagement:")
        if report.peer_engagements:
            for pe in report.peer_engagements[:3]:
                lines.append(f"  - {pe.peer_name} ({pe.platform.value}): {pe.potential_value}")
        else:
            lines.append("  (No peer interactions)")
        lines.append("")

        # High-Signal Comments
        lines.append("- High-Signal Comments:")
        if report.high_signal_comments:
            lines.append(f"  [Top {len(report.high_signal_comments)} comments by signal score]")
            for hsc in report.high_signal_comments[:3]:
                content_preview = hsc.content[:80] + "..." if len(hsc.content) > 80 else hsc.content
                lines.append(f"  - [{hsc.classification.value if hsc.classification else 'UNKNOWN'}] {content_preview}")
        else:
            lines.append("  (None meeting threshold)")
        lines.append("")

        # Themes Summary
        lines.append("- Themes Summary:")
        lines.append(f"  {report.themes_summary}")

        # Warnings
        if report.warnings:
            lines.append("")
            lines.append("- Warnings:")
            for warning in report.warnings:
                lines.append(f"  ! {warning}")

        return "\n".join(lines)


# Convenience function for creating requests
def create_triage_request(
    platforms: List[str],
    time_window_days: int = 7,
    content_focus: Optional[List[str]] = None,
    depth: str = "standard"
) -> Dict[str, Any]:
    """
    Create a properly formatted SOCIAL_TRIAGE_REQUEST.

    Args:
        platforms: Platforms to analyze
        time_window_days: Days to cover
        content_focus: Specific content to focus on
        depth: Analysis depth

    Returns:
        Formatted request dictionary
    """
    return {
        "request_type": "SOCIAL_TRIAGE_REQUEST",
        "platforms": platforms,
        "time_window": f"last {time_window_days} days",
        "content_focus": content_focus or "all",
        "depth": depth
    }
