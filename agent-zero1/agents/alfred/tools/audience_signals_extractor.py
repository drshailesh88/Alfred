"""
Audience Signals Extractor for Alfred

Qualitative analysis engine that clusters audience feedback into actionable themes.
Identifies what the audience cares about, what confuses them, what builds trust,
and what damages it.

Does NOT:
- Create content
- Recommend specific actions
- Engage with audience directly
- Judge audience quality
- Prioritize vocal minorities
- Surface individual trolls
- Conflate volume with importance

Does:
- Cluster comments/feedback by theme
- Identify recurring questions
- Surface common confusions/misconceptions
- Note trust-building content patterns
- Flag trust-damaging patterns
- Extract audience language for future content
- Track theme evolution over time
- Distinguish signal from noise
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re
import json
import hashlib

from . import StrategyAgent, AgentResponse, AlfredState


class SignalCategory(Enum):
    """Categories of audience signals."""
    QUESTIONS = "QUESTIONS"           # What audience wants to know
    CONFUSIONS = "CONFUSIONS"         # Where audience misunderstands
    OBJECTIONS = "OBJECTIONS"         # What audience pushes back on
    PRAISE = "PRAISE"                 # What audience values
    TRUST_BUILDERS = "TRUST_BUILDERS" # What increases credibility
    TRUST_KILLERS = "TRUST_KILLERS"   # What decreases credibility
    OPPORTUNITIES = "OPPORTUNITIES"   # Gaps content could fill
    LANGUAGE = "LANGUAGE"             # How audience talks about topics


class AnalysisDepth(Enum):
    """Depth levels for audience signal analysis."""
    QUICK = "quick"       # Fast scan, top themes only
    STANDARD = "standard" # Balanced analysis
    DEEP = "deep"         # Comprehensive analysis with full clustering


class FeedbackType(Enum):
    """Types of feedback items."""
    COMMENT = "comment"
    REPLY = "reply"
    MENTION = "mention"
    DM = "dm"
    REVIEW = "review"
    EMAIL = "email"


class Platform(Enum):
    """Supported platforms for analysis."""
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    SUBSTACK = "substack"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    EMAIL = "email"
    ALL = "all"


@dataclass
class FeedbackItem:
    """Single piece of audience feedback."""
    id: str
    platform: Platform
    content: str
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    feedback_type: FeedbackType = FeedbackType.COMMENT
    parent_content_id: Optional[str] = None
    parent_content_title: Optional[str] = None
    engagement_count: int = 0  # likes, upvotes, etc.
    reply_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform.value,
            "content": self.content,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "feedback_type": self.feedback_type.value,
            "parent_content_id": self.parent_content_id,
            "parent_content_title": self.parent_content_title,
            "engagement_count": self.engagement_count,
            "reply_count": self.reply_count,
            "metadata": self.metadata
        }


@dataclass
class AudienceSignal:
    """An extracted signal from audience feedback."""
    id: str
    category: SignalCategory
    theme: str
    description: str
    frequency: int = 1
    representative_quotes: List[str] = field(default_factory=list)
    source_feedback_ids: List[str] = field(default_factory=list)
    platforms: Set[Platform] = field(default_factory=set)
    confidence: float = 0.0  # 0.0 to 1.0
    content_opportunity: Optional[str] = None
    clarification_needed: Optional[str] = None
    evidence: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    trend: str = "stable"  # increasing, decreasing, stable, new
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "theme": self.theme,
            "description": self.description,
            "frequency": self.frequency,
            "representative_quotes": self.representative_quotes[:3],  # Limit quotes
            "source_count": len(self.source_feedback_ids),
            "platforms": [p.value for p in self.platforms],
            "confidence": round(self.confidence, 2),
            "content_opportunity": self.content_opportunity,
            "clarification_needed": self.clarification_needed,
            "evidence": self.evidence,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "trend": self.trend,
            "metadata": self.metadata
        }


@dataclass
class ThemeCluster:
    """A cluster of related feedback grouped by theme."""
    id: str
    theme: str
    keywords: List[str]
    feedback_ids: List[str] = field(default_factory=list)
    representative_samples: List[str] = field(default_factory=list)
    size: int = 0
    coherence_score: float = 0.0
    category_distribution: Dict[SignalCategory, int] = field(default_factory=dict)
    centroid_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "theme": self.theme,
            "keywords": self.keywords[:10],
            "size": self.size,
            "coherence_score": round(self.coherence_score, 2),
            "category_distribution": {k.value: v for k, v in self.category_distribution.items()},
            "representative_samples": self.representative_samples[:3]
        }


@dataclass
class LanguagePattern:
    """Pattern in how audience talks about topics."""
    phrase: str
    frequency: int
    context: str  # How they use it
    examples: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase,
            "frequency": self.frequency,
            "context": self.context,
            "examples": self.examples[:2],
            "related_topics": self.related_topics[:3]
        }


@dataclass
class AudienceSignalsReport:
    """Complete report of audience signals analysis."""
    report_id: str
    report_date: datetime
    period_start: datetime
    period_end: datetime
    platforms_analyzed: List[Platform]
    total_feedback_analyzed: int

    # Core signal outputs
    top_questions: List[AudienceSignal] = field(default_factory=list)
    top_confusions: List[AudienceSignal] = field(default_factory=list)
    trust_builders: List[AudienceSignal] = field(default_factory=list)
    trust_killers: List[AudienceSignal] = field(default_factory=list)
    objections_worth_addressing: List[AudienceSignal] = field(default_factory=list)

    # Language and theme analysis
    audience_language: List[LanguagePattern] = field(default_factory=list)
    emerging_themes: List[str] = field(default_factory=list)
    content_opportunities: List[str] = field(default_factory=list)

    # Pattern tracking
    pattern_changes: Dict[str, str] = field(default_factory=dict)
    theme_clusters: List[ThemeCluster] = field(default_factory=list)

    # Metadata
    analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD
    focus_areas: List[str] = field(default_factory=list)
    signal_to_noise_ratio: float = 0.0
    processing_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_date": self.report_date.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            },
            "platforms_analyzed": [p.value for p in self.platforms_analyzed],
            "total_feedback_analyzed": self.total_feedback_analyzed,
            "signal_to_noise_ratio": round(self.signal_to_noise_ratio, 2),

            "top_5_questions": [q.to_dict() for q in self.top_questions[:5]],
            "top_5_confusions": [c.to_dict() for c in self.top_confusions[:5]],
            "top_3_trust_builders": [t.to_dict() for t in self.trust_builders[:3]],
            "top_3_trust_killers": [t.to_dict() for t in self.trust_killers[:3]],
            "objections_worth_addressing": [o.to_dict() for o in self.objections_worth_addressing[:5]],

            "audience_language": [l.to_dict() for l in self.audience_language[:10]],
            "emerging_themes": self.emerging_themes[:5],
            "content_opportunities": self.content_opportunities[:5],

            "pattern_changes": self.pattern_changes,
            "theme_clusters": [c.to_dict() for c in self.theme_clusters[:10]],

            "analysis_depth": self.analysis_depth.value,
            "focus_areas": self.focus_areas,
            "processing_notes": self.processing_notes
        }


# =============================================================================
# NLP Tool Interfaces (prepared for external integrations)
# =============================================================================

class TopicModelingInterface:
    """
    Interface for BERTopic integration.
    BERTopic is used for neural topic modeling to discover themes in feedback.

    GitHub: https://github.com/MaartenGr/BERTopic
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._is_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if BERTopic is available."""
        try:
            from bertopic import BERTopic
            self._is_available = True
        except ImportError:
            self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    def fit_transform(self, documents: List[str]) -> Tuple[List[int], Optional[Any]]:
        """
        Fit BERTopic model and transform documents to topics.

        Args:
            documents: List of text documents to cluster

        Returns:
            Tuple of (topic_assignments, topic_info)
        """
        if not self._is_available:
            # Fallback: return all documents as single topic
            return [0] * len(documents), None

        try:
            from bertopic import BERTopic
            if self._model is None:
                self._model = BERTopic(embedding_model=self.model_name)

            topics, probs = self._model.fit_transform(documents)
            topic_info = self._model.get_topic_info()
            return topics, topic_info
        except Exception as e:
            # Fallback on error
            return [0] * len(documents), None

    def get_topic_keywords(self, topic_id: int, n_words: int = 5) -> List[str]:
        """Get keywords for a specific topic."""
        if not self._is_available or self._model is None:
            return []

        try:
            topic = self._model.get_topic(topic_id)
            if topic:
                return [word for word, _ in topic[:n_words]]
            return []
        except Exception:
            return []


class KeywordExtractionInterface:
    """
    Interface for KeyBERT integration.
    KeyBERT extracts keywords and key phrases from text.

    GitHub: https://github.com/MaartenGr/KeyBERT
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._is_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if KeyBERT is available."""
        try:
            from keybert import KeyBERT
            self._is_available = True
        except ImportError:
            self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    def extract_keywords(
        self,
        text: str,
        top_n: int = 5,
        keyphrase_ngram_range: Tuple[int, int] = (1, 2),
        use_mmr: bool = True,
        diversity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords from text.

        Args:
            text: Input text
            top_n: Number of keywords to extract
            keyphrase_ngram_range: N-gram range for keyphrases
            use_mmr: Use Maximal Marginal Relevance for diversity
            diversity: Diversity parameter for MMR

        Returns:
            List of (keyword, score) tuples
        """
        if not self._is_available:
            # Fallback: simple word frequency
            return self._fallback_extraction(text, top_n)

        try:
            from keybert import KeyBERT
            if self._model is None:
                self._model = KeyBERT(model=self.model_name)

            keywords = self._model.extract_keywords(
                text,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words='english',
                use_mmr=use_mmr,
                diversity=diversity,
                top_n=top_n
            )
            return keywords
        except Exception:
            return self._fallback_extraction(text, top_n)

    def _fallback_extraction(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Simple fallback keyword extraction using word frequency."""
        # Simple stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don',
            'now', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'this', 'that', 'these', 'those', 'am', 'and', 'but', 'if',
            'or', 'because', 'until', 'while', 'about', 'against', 'your', 'my'
        }

        # Tokenize and count
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = defaultdict(int)
        for word in words:
            if word not in stopwords:
                word_freq[word] += 1

        # Sort by frequency and return top_n
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        total = sum(word_freq.values()) or 1
        return [(word, count/total) for word, count in sorted_words[:top_n]]


class SentimentAnalysisInterface:
    """
    Interface for PyABSA (Aspect-Based Sentiment Analysis) integration.
    Used for understanding sentiment toward specific aspects of content.

    GitHub: https://github.com/yangheng95/PyABSA
    """

    def __init__(self):
        self._model = None
        self._is_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if PyABSA is available."""
        try:
            import pyabsa
            self._is_available = True
        except ImportError:
            self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    def analyze_aspects(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for aspects in text.

        Args:
            text: Input text

        Returns:
            List of aspect sentiment results
        """
        if not self._is_available:
            return self._fallback_sentiment(text)

        try:
            from pyabsa import AspectTermExtraction as ATEPC

            if self._model is None:
                self._model = ATEPC.AspectExtractor(
                    'multilingual',
                    auto_device=True
                )

            result = self._model.predict(text)
            return self._format_pyabsa_result(result)
        except Exception:
            return self._fallback_sentiment(text)

    def _format_pyabsa_result(self, result) -> List[Dict[str, Any]]:
        """Format PyABSA result to standard format."""
        aspects = []
        if hasattr(result, 'aspect') and hasattr(result, 'sentiment'):
            for aspect, sentiment in zip(result.aspect, result.sentiment):
                aspects.append({
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "confidence": 0.8  # PyABSA confidence if available
                })
        return aspects

    def _fallback_sentiment(self, text: str) -> List[Dict[str, Any]]:
        """Fallback sentiment analysis using keyword matching."""
        positive_words = {
            'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
            'helpful', 'useful', 'clear', 'informative', 'thanks', 'thank',
            'appreciate', 'insightful', 'valuable', 'brilliant', 'perfect'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'wrong', 'incorrect', 'misleading',
            'confusing', 'confused', 'unclear', 'useless', 'waste', 'disagree',
            'disappointed', 'frustrating', 'annoying', 'hate', 'worst'
        }

        words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count:
            sentiment = "positive"
            confidence = min(0.5 + (pos_count - neg_count) * 0.1, 0.9)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = min(0.5 + (neg_count - pos_count) * 0.1, 0.9)
        else:
            sentiment = "neutral"
            confidence = 0.5

        return [{
            "aspect": "overall",
            "sentiment": sentiment,
            "confidence": confidence
        }]


class EmbeddingInterface:
    """
    Interface for sentence-transformers integration.
    Used for generating embeddings for semantic similarity and clustering.

    GitHub: https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._is_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if sentence-transformers is available."""
        try:
            from sentence_transformers import SentenceTransformer
            self._is_available = True
        except ImportError:
            self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        if not self._is_available:
            # Fallback: return simple hash-based pseudo-embeddings
            return self._fallback_embeddings(texts)

        try:
            from sentence_transformers import SentenceTransformer

            if self._model is None:
                self._model = SentenceTransformer(self.model_name)

            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception:
            return self._fallback_embeddings(texts)

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self.encode([text1, text2])
        return self._cosine_similarity(embeddings[0], embeddings[1])

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simple bag-of-words based pseudo-embeddings."""
        # Create a simple vocabulary from all texts
        vocab = set()
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            vocab.update(words)

        vocab = sorted(list(vocab))[:100]  # Limit vocabulary size
        vocab_idx = {word: i for i, word in enumerate(vocab)}

        embeddings = []
        for text in texts:
            vec = [0.0] * len(vocab)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            for word in words:
                if word in vocab_idx:
                    vec[vocab_idx[word]] += 1.0

            # Normalize
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]

            embeddings.append(vec)

        return embeddings


# =============================================================================
# Main Audience Signals Extractor Class
# =============================================================================

class AudienceSignalsExtractor(StrategyAgent):
    """
    Qualitative analysis engine that clusters audience feedback into actionable themes.

    This agent analyzes audience feedback to identify:
    - What questions the audience is asking
    - Where confusion exists
    - What builds or damages trust
    - How the audience talks about topics
    - Emerging themes and content opportunities
    """

    def __init__(self):
        super().__init__(name="AudienceSignalsExtractor")

        # Initialize NLP tool interfaces
        self.topic_modeler = TopicModelingInterface()
        self.keyword_extractor = KeywordExtractionInterface()
        self.sentiment_analyzer = SentimentAnalysisInterface()
        self.embedding_model = EmbeddingInterface()

        # Historical data storage
        self._previous_signals: Dict[str, AudienceSignal] = {}
        self._historical_themes: List[ThemeCluster] = []
        self._language_patterns: Dict[str, LanguagePattern] = {}

        # Configuration
        self._min_cluster_size = 3
        self._noise_threshold = 0.2
        self._question_patterns = self._compile_question_patterns()
        self._confusion_patterns = self._compile_confusion_patterns()
        self._trust_positive_patterns = self._compile_trust_positive_patterns()
        self._trust_negative_patterns = self._compile_trust_negative_patterns()

    def _compile_question_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for detecting questions."""
        patterns = [
            r'\?$',  # Ends with question mark
            r'^(what|how|why|when|where|who|which|can|could|would|should|do|does|is|are|will)\b',
            r'\b(wondering|curious|anyone know|does anyone|has anyone)\b',
            r'\b(can you|could you|would you|please explain|help me understand)\b',
            r'\b(what is|what are|what does|what do)\b',
            r'\b(how do|how does|how can|how to)\b',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _compile_confusion_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for detecting confusion."""
        patterns = [
            r'\b(confused|confusing|unclear|don\'t understand|doesn\'t make sense)\b',
            r'\b(thought|assumed|expected)\b.+\b(but|however|instead)\b',
            r'\b(wait|huh|what do you mean)\b',
            r'\b(i thought|wasn\'t it|isn\'t it)\b',
            r'\b(contradicts|contradiction|inconsistent)\b',
            r'\b(lost|lost me|you lost me)\b',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _compile_trust_positive_patterns(self) -> List[re.Pattern]:
        """Compile patterns indicating trust-building signals."""
        patterns = [
            r'\b(trust|reliable|credible|honest|transparent)\b',
            r'\b(finally someone|refreshing to see|appreciate your)\b',
            r'\b(evidence|research|studies|data|source)\b.+\b(good|great|helpful)\b',
            r'\b(nuanced|balanced|fair|objective)\b',
            r'\b(changed my mind|opened my eyes|learned)\b',
            r'\b(recommend|share|follow|subscribe)\b',
            r'\b(clear|clarified|makes sense|understand now)\b',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _compile_trust_negative_patterns(self) -> List[re.Pattern]:
        """Compile patterns indicating trust-damaging signals."""
        patterns = [
            r'\b(misleading|misinformation|wrong|incorrect|false)\b',
            r'\b(biased|agenda|shill|paid|sponsored)\b',
            r'\b(unsubscribe|unfollow|lost me|done with)\b',
            r'\b(cherry.?pick|one.?sided|ignoring|dismissing)\b',
            r'\b(clickbait|sensational|exaggerat)\b',
            r'\b(disappointed|expected better|used to trust)\b',
            r'\b(no source|no evidence|where\'s the proof)\b',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _generate_id(self, *args) -> str:
        """Generate a deterministic ID from arguments."""
        content = "|".join(str(arg) for arg in args)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    # =========================================================================
    # Core Analysis Methods
    # =========================================================================

    def analyze_feedback(
        self,
        feedback_items: List[FeedbackItem],
        platforms: Optional[List[Platform]] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
        focus_areas: Optional[List[str]] = None,
        content_focus: Optional[str] = None
    ) -> AgentResponse:
        """
        Main entry point for audience signal analysis.

        Args:
            feedback_items: List of feedback items to analyze
            platforms: Platforms to include (None = all)
            period_start: Start of analysis period
            period_end: End of analysis period
            depth: Analysis depth level
            focus_areas: Specific themes to track
            content_focus: Specific content to focus on

        Returns:
            AgentResponse containing AudienceSignalsReport
        """
        # Check permission based on Alfred state
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Filter feedback by platform and time if specified
        filtered_feedback = self._filter_feedback(
            feedback_items, platforms, period_start, period_end, content_focus
        )

        if not filtered_feedback:
            return self.create_response(
                data={"status": "NO_DATA", "message": "No feedback items to analyze"},
                warnings=["No feedback items matched the specified criteria"]
            )

        # Determine analysis parameters based on depth
        params = self._get_analysis_parameters(depth)

        try:
            # Run analysis pipeline
            # 1. Cluster themes
            theme_clusters = self.cluster_themes(filtered_feedback, params)

            # 2. Extract questions
            questions = self.extract_questions(filtered_feedback, theme_clusters)

            # 3. Identify confusions
            confusions = self._identify_confusions(filtered_feedback, theme_clusters)

            # 4. Identify trust signals
            trust_builders, trust_killers = self.identify_trust_signals(
                filtered_feedback, theme_clusters
            )

            # 5. Extract objections
            objections = self._identify_objections(filtered_feedback, theme_clusters)

            # 6. Extract language patterns
            language_patterns = self.extract_language(filtered_feedback, focus_areas)

            # 7. Identify emerging themes
            emerging_themes = self._identify_emerging_themes(theme_clusters)

            # 8. Generate content opportunities
            content_opportunities = self._generate_content_opportunities(
                questions, confusions, objections, theme_clusters
            )

            # 9. Compare with previous period
            pattern_changes = self._compare_with_previous(
                questions, confusions, trust_builders, trust_killers
            )

            # 10. Generate report
            report = self.generate_report(
                feedback_items=filtered_feedback,
                platforms=platforms or [Platform.ALL],
                period_start=period_start or self._get_earliest_timestamp(filtered_feedback),
                period_end=period_end or datetime.now(),
                depth=depth,
                focus_areas=focus_areas or [],
                theme_clusters=theme_clusters,
                questions=questions,
                confusions=confusions,
                trust_builders=trust_builders,
                trust_killers=trust_killers,
                objections=objections,
                language_patterns=language_patterns,
                emerging_themes=emerging_themes,
                content_opportunities=content_opportunities,
                pattern_changes=pattern_changes
            )

            # Store for future comparison
            self._store_signals_for_comparison(questions, confusions, trust_builders, trust_killers)

            return self.create_response(
                data={
                    "status": "SUCCESS",
                    "report": report.to_dict()
                }
            )

        except Exception as e:
            return self.create_response(
                data={"status": "ERROR"},
                success=False,
                errors=[f"Analysis failed: {str(e)}"]
            )

    def cluster_themes(
        self,
        feedback_items: List[FeedbackItem],
        params: Dict[str, Any]
    ) -> List[ThemeCluster]:
        """
        Cluster feedback items by theme using topic modeling.

        Args:
            feedback_items: List of feedback to cluster
            params: Analysis parameters

        Returns:
            List of ThemeCluster objects
        """
        if len(feedback_items) < self._min_cluster_size:
            # Not enough items to cluster meaningfully
            return self._single_cluster_fallback(feedback_items)

        # Extract texts
        texts = [item.content for item in feedback_items]

        # Try to use BERTopic for clustering
        if self.topic_modeler.is_available and params.get("use_neural_clustering", True):
            topics, topic_info = self.topic_modeler.fit_transform(texts)
            clusters = self._build_clusters_from_topics(
                feedback_items, texts, topics, topic_info
            )
        else:
            # Fallback to keyword-based clustering
            clusters = self._keyword_based_clustering(feedback_items, texts)

        # Enrich clusters with category distribution
        for cluster in clusters:
            cluster.category_distribution = self._compute_category_distribution(
                [fi for fi in feedback_items if fi.id in cluster.feedback_ids]
            )

        return sorted(clusters, key=lambda c: c.size, reverse=True)

    def extract_questions(
        self,
        feedback_items: List[FeedbackItem],
        theme_clusters: List[ThemeCluster]
    ) -> List[AudienceSignal]:
        """
        Extract recurring questions from feedback.

        Args:
            feedback_items: List of feedback items
            theme_clusters: Pre-computed theme clusters

        Returns:
            List of AudienceSignal objects for questions
        """
        questions = []
        question_items = []

        # Identify question-type feedback
        for item in feedback_items:
            is_question = any(
                pattern.search(item.content)
                for pattern in self._question_patterns
            )
            if is_question:
                question_items.append(item)

        if not question_items:
            return []

        # Group questions by similarity/theme
        question_groups = self._group_similar_items(question_items)

        for group_theme, items in question_groups.items():
            if len(items) >= 2:  # Only surface recurring questions
                signal = AudienceSignal(
                    id=self._generate_id("question", group_theme),
                    category=SignalCategory.QUESTIONS,
                    theme=group_theme,
                    description=f"Audience is asking about: {group_theme}",
                    frequency=len(items),
                    representative_quotes=[item.content[:200] for item in items[:3]],
                    source_feedback_ids=[item.id for item in items],
                    platforms={item.platform for item in items},
                    confidence=min(0.5 + len(items) * 0.1, 0.95),
                    content_opportunity=self._suggest_content_for_question(group_theme, items),
                    first_seen=min((i.timestamp for i in items if i.timestamp), default=None),
                    last_seen=max((i.timestamp for i in items if i.timestamp), default=None)
                )
                questions.append(signal)

        return sorted(questions, key=lambda q: q.frequency, reverse=True)

    def identify_trust_signals(
        self,
        feedback_items: List[FeedbackItem],
        theme_clusters: List[ThemeCluster]
    ) -> Tuple[List[AudienceSignal], List[AudienceSignal]]:
        """
        Identify signals that build or damage trust.

        Args:
            feedback_items: List of feedback items
            theme_clusters: Pre-computed theme clusters

        Returns:
            Tuple of (trust_builders, trust_killers)
        """
        trust_builders = []
        trust_killers = []

        builder_items = []
        killer_items = []

        for item in feedback_items:
            # Check for trust-building signals
            is_builder = any(
                pattern.search(item.content)
                for pattern in self._trust_positive_patterns
            )

            # Check for trust-damaging signals
            is_killer = any(
                pattern.search(item.content)
                for pattern in self._trust_negative_patterns
            )

            if is_builder and not is_killer:
                builder_items.append(item)
            elif is_killer and not is_builder:
                killer_items.append(item)

        # Use sentiment analysis for additional context if available
        if self.sentiment_analyzer.is_available:
            for item in feedback_items:
                if item not in builder_items and item not in killer_items:
                    aspects = self.sentiment_analyzer.analyze_aspects(item.content)
                    for aspect in aspects:
                        if aspect.get("sentiment") == "positive" and aspect.get("confidence", 0) > 0.7:
                            builder_items.append(item)
                        elif aspect.get("sentiment") == "negative" and aspect.get("confidence", 0) > 0.7:
                            killer_items.append(item)

        # Group and create signals
        builder_groups = self._group_by_trust_theme(builder_items)
        for theme, items in builder_groups.items():
            signal = AudienceSignal(
                id=self._generate_id("trust_builder", theme),
                category=SignalCategory.TRUST_BUILDERS,
                theme=theme,
                description=f"Audience trust increased by: {theme}",
                frequency=len(items),
                representative_quotes=[item.content[:200] for item in items[:3]],
                source_feedback_ids=[item.id for item in items],
                platforms={item.platform for item in items},
                confidence=min(0.5 + len(items) * 0.1, 0.95),
                evidence=f"{len(items)} positive mentions about {theme}"
            )
            trust_builders.append(signal)

        killer_groups = self._group_by_trust_theme(killer_items)
        for theme, items in killer_groups.items():
            signal = AudienceSignal(
                id=self._generate_id("trust_killer", theme),
                category=SignalCategory.TRUST_KILLERS,
                theme=theme,
                description=f"Audience trust damaged by: {theme}",
                frequency=len(items),
                representative_quotes=[item.content[:200] for item in items[:3]],
                source_feedback_ids=[item.id for item in items],
                platforms={item.platform for item in items},
                confidence=min(0.5 + len(items) * 0.1, 0.95),
                evidence=f"{len(items)} negative mentions about {theme}"
            )
            trust_killers.append(signal)

        return (
            sorted(trust_builders, key=lambda t: t.frequency, reverse=True),
            sorted(trust_killers, key=lambda t: t.frequency, reverse=True)
        )

    def extract_language(
        self,
        feedback_items: List[FeedbackItem],
        focus_areas: Optional[List[str]] = None
    ) -> List[LanguagePattern]:
        """
        Extract key phrases and language patterns from audience feedback.

        Args:
            feedback_items: List of feedback items
            focus_areas: Specific topics to focus on

        Returns:
            List of LanguagePattern objects
        """
        # Combine all feedback text
        combined_text = " ".join(item.content for item in feedback_items)

        # Extract keywords using KeyBERT or fallback
        keywords = self.keyword_extractor.extract_keywords(
            combined_text,
            top_n=20,
            keyphrase_ngram_range=(1, 3),
            use_mmr=True,
            diversity=0.7
        )

        patterns = []
        for phrase, score in keywords:
            # Find examples of this phrase in context
            examples = self._find_phrase_examples(phrase, feedback_items, max_examples=3)

            # Determine the context of how the phrase is used
            context = self._analyze_phrase_context(phrase, feedback_items)

            # Find related topics
            related = self._find_related_topics(phrase, keywords)

            pattern = LanguagePattern(
                phrase=phrase,
                frequency=self._count_phrase_frequency(phrase, feedback_items),
                context=context,
                examples=examples,
                related_topics=related
            )
            patterns.append(pattern)

        # Filter by focus areas if specified
        if focus_areas:
            patterns = [
                p for p in patterns
                if any(area.lower() in p.phrase.lower() or
                       any(area.lower() in topic.lower() for topic in p.related_topics)
                       for area in focus_areas)
            ]

        return sorted(patterns, key=lambda p: p.frequency, reverse=True)

    def generate_report(
        self,
        feedback_items: List[FeedbackItem],
        platforms: List[Platform],
        period_start: datetime,
        period_end: datetime,
        depth: AnalysisDepth,
        focus_areas: List[str],
        theme_clusters: List[ThemeCluster],
        questions: List[AudienceSignal],
        confusions: List[AudienceSignal],
        trust_builders: List[AudienceSignal],
        trust_killers: List[AudienceSignal],
        objections: List[AudienceSignal],
        language_patterns: List[LanguagePattern],
        emerging_themes: List[str],
        content_opportunities: List[str],
        pattern_changes: Dict[str, str]
    ) -> AudienceSignalsReport:
        """
        Generate the final audience signals report.

        Returns:
            Complete AudienceSignalsReport object
        """
        # Calculate signal to noise ratio
        total_items = len(feedback_items)
        signal_items = (
            sum(len(q.source_feedback_ids) for q in questions) +
            sum(len(c.source_feedback_ids) for c in confusions) +
            sum(len(t.source_feedback_ids) for t in trust_builders) +
            sum(len(t.source_feedback_ids) for t in trust_killers) +
            sum(len(o.source_feedback_ids) for o in objections)
        )
        # Deduplicate signal items
        signal_ratio = min(signal_items / max(total_items, 1), 1.0)

        # Generate processing notes
        processing_notes = []
        if not self.topic_modeler.is_available:
            processing_notes.append("BERTopic not available - used fallback clustering")
        if not self.sentiment_analyzer.is_available:
            processing_notes.append("PyABSA not available - used pattern-based sentiment")
        if not self.embedding_model.is_available:
            processing_notes.append("sentence-transformers not available - used simple embeddings")

        return AudienceSignalsReport(
            report_id=self._generate_id(datetime.now().isoformat(), str(platforms)),
            report_date=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            platforms_analyzed=platforms,
            total_feedback_analyzed=total_items,

            top_questions=questions[:5],
            top_confusions=confusions[:5],
            trust_builders=trust_builders[:3],
            trust_killers=trust_killers[:3],
            objections_worth_addressing=objections[:5],

            audience_language=language_patterns[:10],
            emerging_themes=emerging_themes[:5],
            content_opportunities=content_opportunities[:5],

            pattern_changes=pattern_changes,
            theme_clusters=theme_clusters[:10],

            analysis_depth=depth,
            focus_areas=focus_areas,
            signal_to_noise_ratio=signal_ratio,
            processing_notes=processing_notes
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _filter_feedback(
        self,
        items: List[FeedbackItem],
        platforms: Optional[List[Platform]],
        period_start: Optional[datetime],
        period_end: Optional[datetime],
        content_focus: Optional[str]
    ) -> List[FeedbackItem]:
        """Filter feedback items by criteria."""
        filtered = items

        if platforms and Platform.ALL not in platforms:
            filtered = [i for i in filtered if i.platform in platforms]

        if period_start:
            filtered = [i for i in filtered if not i.timestamp or i.timestamp >= period_start]

        if period_end:
            filtered = [i for i in filtered if not i.timestamp or i.timestamp <= period_end]

        if content_focus:
            filtered = [
                i for i in filtered
                if content_focus.lower() in (i.parent_content_title or "").lower() or
                   content_focus.lower() in i.content.lower()
            ]

        return filtered

    def _get_analysis_parameters(self, depth: AnalysisDepth) -> Dict[str, Any]:
        """Get analysis parameters based on depth level."""
        params = {
            AnalysisDepth.QUICK: {
                "use_neural_clustering": False,
                "max_clusters": 5,
                "min_examples_per_signal": 2,
                "detailed_sentiment": False
            },
            AnalysisDepth.STANDARD: {
                "use_neural_clustering": True,
                "max_clusters": 10,
                "min_examples_per_signal": 2,
                "detailed_sentiment": True
            },
            AnalysisDepth.DEEP: {
                "use_neural_clustering": True,
                "max_clusters": 20,
                "min_examples_per_signal": 1,
                "detailed_sentiment": True
            }
        }
        return params.get(depth, params[AnalysisDepth.STANDARD])

    def _get_earliest_timestamp(self, items: List[FeedbackItem]) -> datetime:
        """Get the earliest timestamp from feedback items."""
        timestamps = [i.timestamp for i in items if i.timestamp]
        if timestamps:
            return min(timestamps)
        return datetime.now() - timedelta(days=7)  # Default to 7 days ago

    def _single_cluster_fallback(self, items: List[FeedbackItem]) -> List[ThemeCluster]:
        """Create a single cluster when there aren't enough items to cluster."""
        if not items:
            return []

        # Extract keywords from all items
        combined_text = " ".join(i.content for i in items)
        keywords = self.keyword_extractor.extract_keywords(combined_text, top_n=5)
        keyword_list = [kw for kw, _ in keywords]

        cluster = ThemeCluster(
            id=self._generate_id("single_cluster"),
            theme=keyword_list[0] if keyword_list else "general",
            keywords=keyword_list,
            feedback_ids=[i.id for i in items],
            representative_samples=[i.content[:200] for i in items[:3]],
            size=len(items),
            coherence_score=0.5
        )
        return [cluster]

    def _build_clusters_from_topics(
        self,
        feedback_items: List[FeedbackItem],
        texts: List[str],
        topics: List[int],
        topic_info
    ) -> List[ThemeCluster]:
        """Build ThemeCluster objects from BERTopic results."""
        clusters = []
        topic_to_items = defaultdict(list)

        for item, topic in zip(feedback_items, topics):
            if topic != -1:  # -1 is noise in BERTopic
                topic_to_items[topic].append(item)

        for topic_id, items in topic_to_items.items():
            keywords = self.topic_modeler.get_topic_keywords(topic_id, n_words=5)
            theme = keywords[0] if keywords else f"topic_{topic_id}"

            cluster = ThemeCluster(
                id=self._generate_id("topic", str(topic_id)),
                theme=theme,
                keywords=keywords,
                feedback_ids=[i.id for i in items],
                representative_samples=[i.content[:200] for i in items[:3]],
                size=len(items),
                coherence_score=0.8  # BERTopic generally produces coherent topics
            )
            clusters.append(cluster)

        return clusters

    def _keyword_based_clustering(
        self,
        feedback_items: List[FeedbackItem],
        texts: List[str]
    ) -> List[ThemeCluster]:
        """Fallback clustering using keyword similarity."""
        # Extract keywords for each item
        item_keywords = {}
        for item in feedback_items:
            keywords = self.keyword_extractor.extract_keywords(item.content, top_n=3)
            item_keywords[item.id] = set(kw for kw, _ in keywords)

        # Group by overlapping keywords
        clusters_dict = defaultdict(list)
        used_items = set()

        for item in feedback_items:
            if item.id in used_items:
                continue

            # Find items with overlapping keywords
            item_kws = item_keywords.get(item.id, set())
            cluster_key = tuple(sorted(item_kws))[:2]  # Use first 2 keywords as key

            if cluster_key:
                for other in feedback_items:
                    if other.id not in used_items:
                        other_kws = item_keywords.get(other.id, set())
                        if item_kws & other_kws:  # Has overlap
                            clusters_dict[cluster_key].append(other)
                            used_items.add(other.id)

        # Convert to ThemeCluster objects
        clusters = []
        for keywords, items in clusters_dict.items():
            if len(items) >= self._min_cluster_size:
                cluster = ThemeCluster(
                    id=self._generate_id("keyword_cluster", str(keywords)),
                    theme=keywords[0] if keywords else "misc",
                    keywords=list(keywords),
                    feedback_ids=[i.id for i in items],
                    representative_samples=[i.content[:200] for i in items[:3]],
                    size=len(items),
                    coherence_score=0.6
                )
                clusters.append(cluster)

        return clusters

    def _compute_category_distribution(
        self,
        items: List[FeedbackItem]
    ) -> Dict[SignalCategory, int]:
        """Compute distribution of signal categories in items."""
        distribution = defaultdict(int)

        for item in items:
            # Check each category
            if any(p.search(item.content) for p in self._question_patterns):
                distribution[SignalCategory.QUESTIONS] += 1
            if any(p.search(item.content) for p in self._confusion_patterns):
                distribution[SignalCategory.CONFUSIONS] += 1
            if any(p.search(item.content) for p in self._trust_positive_patterns):
                distribution[SignalCategory.TRUST_BUILDERS] += 1
            if any(p.search(item.content) for p in self._trust_negative_patterns):
                distribution[SignalCategory.TRUST_KILLERS] += 1

        return dict(distribution)

    def _group_similar_items(
        self,
        items: List[FeedbackItem]
    ) -> Dict[str, List[FeedbackItem]]:
        """Group items by semantic similarity."""
        if not items:
            return {}

        # Use embeddings if available
        if self.embedding_model.is_available and len(items) > 1:
            texts = [i.content for i in items]
            embeddings = self.embedding_model.encode(texts)

            # Simple clustering by similarity threshold
            groups = defaultdict(list)
            used = set()

            for i, item in enumerate(items):
                if i in used:
                    continue

                # Find similar items
                group_items = [item]
                used.add(i)

                for j, other in enumerate(items):
                    if j not in used:
                        similarity = self.embedding_model._cosine_similarity(
                            embeddings[i], embeddings[j]
                        )
                        if similarity > 0.7:  # Similarity threshold
                            group_items.append(other)
                            used.add(j)

                # Extract theme from group
                keywords = self.keyword_extractor.extract_keywords(
                    " ".join(i.content for i in group_items), top_n=2
                )
                theme = keywords[0][0] if keywords else f"group_{len(groups)}"
                groups[theme] = group_items
        else:
            # Fallback: group by shared keywords
            groups = self._group_by_keywords(items)

        return dict(groups)

    def _group_by_keywords(
        self,
        items: List[FeedbackItem]
    ) -> Dict[str, List[FeedbackItem]]:
        """Group items by shared keywords."""
        groups = defaultdict(list)

        for item in items:
            keywords = self.keyword_extractor.extract_keywords(item.content, top_n=2)
            if keywords:
                theme = keywords[0][0]
                groups[theme].append(item)
            else:
                groups["general"].append(item)

        return groups

    def _group_by_trust_theme(
        self,
        items: List[FeedbackItem]
    ) -> Dict[str, List[FeedbackItem]]:
        """Group trust-related items by their theme."""
        trust_themes = {
            "transparency": ["honest", "transparent", "open", "clear"],
            "evidence": ["research", "studies", "data", "source", "evidence", "proof"],
            "balance": ["nuanced", "balanced", "fair", "objective", "both sides"],
            "consistency": ["consistent", "reliable", "always", "every time"],
            "bias": ["biased", "agenda", "shill", "paid", "sponsored", "one-sided"],
            "accuracy": ["wrong", "incorrect", "misleading", "false", "misinformation"],
            "sensationalism": ["clickbait", "sensational", "exaggerat", "overhyp"],
            "responsiveness": ["respond", "reply", "engage", "listen"]
        }

        groups = defaultdict(list)
        for item in items:
            content_lower = item.content.lower()
            matched = False

            for theme, keywords in trust_themes.items():
                if any(kw in content_lower for kw in keywords):
                    groups[theme].append(item)
                    matched = True
                    break

            if not matched:
                groups["general_trust"].append(item)

        return dict(groups)

    def _suggest_content_for_question(
        self,
        theme: str,
        items: List[FeedbackItem]
    ) -> str:
        """Suggest content that could address a recurring question."""
        # Analyze the nature of questions
        question_types = {
            "how": "tutorial or guide",
            "why": "explainer article or video",
            "what": "educational content defining concepts",
            "when": "timeline or scheduling guide",
            "can": "capability overview or FAQ",
            "should": "recommendation or best practices guide"
        }

        combined = " ".join(i.content.lower() for i in items)

        for qword, content_type in question_types.items():
            if combined.startswith(qword) or f" {qword} " in combined:
                return f"Consider creating a {content_type} about {theme}"

        return f"Address this topic in upcoming content: {theme}"

    def _identify_confusions(
        self,
        feedback_items: List[FeedbackItem],
        theme_clusters: List[ThemeCluster]
    ) -> List[AudienceSignal]:
        """Identify recurring confusions and misconceptions."""
        confusions = []
        confusion_items = []

        for item in feedback_items:
            is_confusion = any(
                pattern.search(item.content)
                for pattern in self._confusion_patterns
            )
            if is_confusion:
                confusion_items.append(item)

        if not confusion_items:
            return []

        # Group confusions by theme
        confusion_groups = self._group_similar_items(confusion_items)

        for theme, items in confusion_groups.items():
            if len(items) >= 2:
                # Try to identify the source of confusion
                source = self._identify_confusion_source(items)

                signal = AudienceSignal(
                    id=self._generate_id("confusion", theme),
                    category=SignalCategory.CONFUSIONS,
                    theme=theme,
                    description=f"Audience is confused about: {theme}",
                    frequency=len(items),
                    representative_quotes=[item.content[:200] for item in items[:3]],
                    source_feedback_ids=[item.id for item in items],
                    platforms={item.platform for item in items},
                    confidence=min(0.5 + len(items) * 0.1, 0.95),
                    clarification_needed=f"Clarify {theme} - {source}",
                    first_seen=min((i.timestamp for i in items if i.timestamp), default=None),
                    last_seen=max((i.timestamp for i in items if i.timestamp), default=None)
                )
                confusions.append(signal)

        return sorted(confusions, key=lambda c: c.frequency, reverse=True)

    def _identify_confusion_source(self, items: List[FeedbackItem]) -> str:
        """Try to identify what's causing the confusion."""
        combined = " ".join(i.content.lower() for i in items)

        # Look for patterns that indicate the source
        if "thought" in combined and "but" in combined:
            return "expectation mismatch"
        if "contradicts" in combined or "inconsistent" in combined:
            return "perceived inconsistency"
        if "unclear" in combined or "vague" in combined:
            return "insufficient explanation"
        if "assumed" in combined:
            return "incorrect assumption"

        return "unclear messaging"

    def _identify_objections(
        self,
        feedback_items: List[FeedbackItem],
        theme_clusters: List[ThemeCluster]
    ) -> List[AudienceSignal]:
        """Identify objections and pushback worth addressing."""
        objection_patterns = [
            r'\b(disagree|but|however|although|actually)\b',
            r'\b(that\'s not|that isn\'t|you\'re wrong|incorrect)\b',
            r'\b(what about|but what if|doesn\'t account for)\b',
            r'\b(fails to|ignores|overlooks|misses)\b',
            r'\b(counterpoint|counter.?argument|alternative)\b'
        ]
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in objection_patterns]

        objection_items = []
        for item in feedback_items:
            # Must have objection pattern but not be a troll/low-quality
            has_objection = any(p.search(item.content) for p in compiled_patterns)
            is_substantive = len(item.content) > 50  # Minimum length for substance
            has_engagement = item.engagement_count >= 1  # Others agree

            if has_objection and is_substantive:
                objection_items.append(item)

        # Group by theme
        objection_groups = self._group_similar_items(objection_items)
        objections = []

        for theme, items in objection_groups.items():
            if len(items) >= 2:  # Must be recurring
                signal = AudienceSignal(
                    id=self._generate_id("objection", theme),
                    category=SignalCategory.OBJECTIONS,
                    theme=theme,
                    description=f"Audience pushes back on: {theme}",
                    frequency=len(items),
                    representative_quotes=[item.content[:200] for item in items[:3]],
                    source_feedback_ids=[item.id for item in items],
                    platforms={item.platform for item in items},
                    confidence=min(0.5 + len(items) * 0.1, 0.95),
                    content_opportunity=f"Address objection about {theme} in future content"
                )
                objections.append(signal)

        return sorted(objections, key=lambda o: o.frequency, reverse=True)

    def _find_phrase_examples(
        self,
        phrase: str,
        items: List[FeedbackItem],
        max_examples: int = 3
    ) -> List[str]:
        """Find examples of a phrase used in context."""
        examples = []
        phrase_lower = phrase.lower()

        for item in items:
            if phrase_lower in item.content.lower():
                # Extract context around the phrase
                start = max(0, item.content.lower().find(phrase_lower) - 50)
                end = min(len(item.content), start + len(phrase) + 100)
                context = item.content[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(item.content):
                    context = context + "..."
                examples.append(context)

                if len(examples) >= max_examples:
                    break

        return examples

    def _analyze_phrase_context(
        self,
        phrase: str,
        items: List[FeedbackItem]
    ) -> str:
        """Analyze how a phrase is typically used."""
        contexts = []
        phrase_lower = phrase.lower()

        for item in items:
            if phrase_lower in item.content.lower():
                # Categorize by signal type
                if any(p.search(item.content) for p in self._question_patterns):
                    contexts.append("question")
                elif any(p.search(item.content) for p in self._confusion_patterns):
                    contexts.append("confusion")
                elif any(p.search(item.content) for p in self._trust_positive_patterns):
                    contexts.append("positive")
                elif any(p.search(item.content) for p in self._trust_negative_patterns):
                    contexts.append("concern")
                else:
                    contexts.append("neutral")

        if not contexts:
            return "general discussion"

        # Return the most common context
        from collections import Counter
        most_common = Counter(contexts).most_common(1)[0][0]

        context_descriptions = {
            "question": "often used when asking questions",
            "confusion": "associated with confusion or uncertainty",
            "positive": "used in positive feedback",
            "concern": "mentioned in critical feedback",
            "neutral": "used in general discussion"
        }

        return context_descriptions.get(most_common, "general discussion")

    def _find_related_topics(
        self,
        phrase: str,
        all_keywords: List[Tuple[str, float]]
    ) -> List[str]:
        """Find topics related to a given phrase."""
        related = []
        phrase_words = set(phrase.lower().split())

        for kw, _ in all_keywords:
            if kw.lower() != phrase.lower():
                kw_words = set(kw.lower().split())
                # Check for word overlap or semantic similarity
                if phrase_words & kw_words:  # Word overlap
                    related.append(kw)

        return related[:3]

    def _count_phrase_frequency(
        self,
        phrase: str,
        items: List[FeedbackItem]
    ) -> int:
        """Count how many items contain a phrase."""
        phrase_lower = phrase.lower()
        return sum(1 for item in items if phrase_lower in item.content.lower())

    def _identify_emerging_themes(
        self,
        theme_clusters: List[ThemeCluster]
    ) -> List[str]:
        """Identify themes that are new or growing."""
        emerging = []

        for cluster in theme_clusters:
            # Check if this is a new theme (not in historical data)
            is_new = not any(
                cluster.theme == hist.theme
                for hist in self._historical_themes
            )

            if is_new and cluster.size >= self._min_cluster_size:
                emerging.append(cluster.theme)

        # Update historical themes
        self._historical_themes = theme_clusters.copy()

        return emerging[:5]

    def _generate_content_opportunities(
        self,
        questions: List[AudienceSignal],
        confusions: List[AudienceSignal],
        objections: List[AudienceSignal],
        theme_clusters: List[ThemeCluster]
    ) -> List[str]:
        """Generate specific content opportunities from analysis."""
        opportunities = []

        # From questions
        for q in questions[:3]:
            if q.content_opportunity:
                opportunities.append(q.content_opportunity)

        # From confusions
        for c in confusions[:2]:
            opportunities.append(
                f"Clarification content needed: {c.theme}"
            )

        # From objections
        for o in objections[:2]:
            opportunities.append(
                f"Address common objection: {o.theme}"
            )

        # From underserved clusters
        large_clusters = [c for c in theme_clusters if c.size >= 5]
        for cluster in large_clusters[:2]:
            if cluster.theme not in [o.split(": ")[-1] for o in opportunities]:
                opportunities.append(
                    f"High-interest topic: {cluster.theme} ({cluster.size} mentions)"
                )

        return opportunities[:5]

    def _compare_with_previous(
        self,
        questions: List[AudienceSignal],
        confusions: List[AudienceSignal],
        trust_builders: List[AudienceSignal],
        trust_killers: List[AudienceSignal]
    ) -> Dict[str, str]:
        """Compare current signals with previous period."""
        changes = {}

        # Check for new questions
        current_q_themes = {q.theme for q in questions}
        prev_q_themes = {
            s.theme for s in self._previous_signals.values()
            if s.category == SignalCategory.QUESTIONS
        }

        new_questions = current_q_themes - prev_q_themes
        if new_questions:
            changes["new_questions"] = f"New topics: {', '.join(list(new_questions)[:3])}"

        # Check for resolved confusions
        current_c_themes = {c.theme for c in confusions}
        prev_c_themes = {
            s.theme for s in self._previous_signals.values()
            if s.category == SignalCategory.CONFUSIONS
        }

        resolved = prev_c_themes - current_c_themes
        if resolved:
            changes["resolved_confusions"] = f"No longer seeing: {', '.join(list(resolved)[:3])}"

        # Check trust signal changes
        current_builders = {t.theme for t in trust_builders}
        current_killers = {t.theme for t in trust_killers}

        prev_builders = {
            s.theme for s in self._previous_signals.values()
            if s.category == SignalCategory.TRUST_BUILDERS
        }

        new_trust = current_builders - prev_builders
        if new_trust:
            changes["new_trust_signals"] = f"New positive signals: {', '.join(list(new_trust)[:3])}"

        if not changes:
            changes["status"] = "No significant pattern changes from previous period"

        return changes

    def _store_signals_for_comparison(
        self,
        questions: List[AudienceSignal],
        confusions: List[AudienceSignal],
        trust_builders: List[AudienceSignal],
        trust_killers: List[AudienceSignal]
    ):
        """Store current signals for future comparison."""
        self._previous_signals.clear()

        for signal in questions + confusions + trust_builders + trust_killers:
            self._previous_signals[signal.id] = signal

    # =========================================================================
    # State-aware behavior
    # =========================================================================

    def check_state_permission(self) -> Tuple[bool, str]:
        """
        Check if the agent is permitted to operate in current state.
        Strategy agents pause in RED state to focus on recovery.
        """
        if self.alfred_state == AlfredState.RED:
            return False, "Strategy work paused in RED state - focus on recovery"
        return True, "Operation permitted"


# =============================================================================
# Request/Response Formatting
# =============================================================================

def parse_audience_signals_request(request_text: str) -> Dict[str, Any]:
    """
    Parse an AUDIENCE_SIGNALS_REQUEST formatted input.

    Expected format:
    AUDIENCE_SIGNALS_REQUEST
    - Platforms: [which to analyze]
    - Period: [date range]
    - Content Focus: [specific content, or all]
    - Depth: quick | standard | deep
    - Focus Areas: [specific themes to track]
    """
    result = {
        "platforms": [Platform.ALL],
        "period_start": None,
        "period_end": None,
        "content_focus": None,
        "depth": AnalysisDepth.STANDARD,
        "focus_areas": []
    }

    lines = request_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("- Platforms:"):
            platforms_str = line.replace("- Platforms:", "").strip()
            platforms = []
            for p in platforms_str.split(","):
                p = p.strip().lower()
                try:
                    platforms.append(Platform(p))
                except ValueError:
                    if p == "all":
                        platforms.append(Platform.ALL)
            if platforms:
                result["platforms"] = platforms

        elif line.startswith("- Period:"):
            period_str = line.replace("- Period:", "").strip()
            # Parse various date formats
            if "to" in period_str.lower() or "-" in period_str:
                # Date range
                parts = re.split(r'\s+to\s+|--+|\s+-\s+', period_str, flags=re.IGNORECASE)
                if len(parts) == 2:
                    try:
                        result["period_start"] = datetime.fromisoformat(parts[0].strip())
                        result["period_end"] = datetime.fromisoformat(parts[1].strip())
                    except ValueError:
                        pass
            elif "days" in period_str.lower():
                # "last N days" format
                match = re.search(r'(\d+)\s*days?', period_str, re.IGNORECASE)
                if match:
                    days = int(match.group(1))
                    result["period_end"] = datetime.now()
                    result["period_start"] = datetime.now() - timedelta(days=days)
            elif "week" in period_str.lower():
                result["period_end"] = datetime.now()
                result["period_start"] = datetime.now() - timedelta(weeks=1)
            elif "month" in period_str.lower():
                result["period_end"] = datetime.now()
                result["period_start"] = datetime.now() - timedelta(days=30)

        elif line.startswith("- Content Focus:"):
            focus = line.replace("- Content Focus:", "").strip()
            if focus.lower() != "all":
                result["content_focus"] = focus

        elif line.startswith("- Depth:"):
            depth_str = line.replace("- Depth:", "").strip().lower()
            try:
                result["depth"] = AnalysisDepth(depth_str)
            except ValueError:
                pass

        elif line.startswith("- Focus Areas:"):
            areas_str = line.replace("- Focus Areas:", "").strip()
            areas = [a.strip() for a in areas_str.split(",") if a.strip()]
            result["focus_areas"] = areas

    return result


def format_audience_signals_output(report: AudienceSignalsReport) -> str:
    """
    Format an AudienceSignalsReport to the standard output format.

    Output format:
    AUDIENCE_SIGNALS
    - Report Date, Period, Feedback Analyzed, Platforms
    - Top 5 Questions...
    - Top 5 Confusions...
    etc.
    """
    lines = [
        "AUDIENCE_SIGNALS",
        f"- Report Date: {report.report_date.strftime('%Y-%m-%d %H:%M')}",
        f"- Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}",
        f"- Feedback Analyzed: {report.total_feedback_analyzed}",
        f"- Platforms: {', '.join(p.value for p in report.platforms_analyzed)}",
        f"- Signal to Noise Ratio: {report.signal_to_noise_ratio:.0%}",
        "",
        "Top 5 Questions:"
    ]

    for i, q in enumerate(report.top_questions[:5], 1):
        lines.append(f"  {i}. {q.theme}")
        lines.append(f"     - Frequency: {q.frequency}")
        if q.representative_quotes:
            lines.append(f"     - Representative: \"{q.representative_quotes[0][:100]}...\"")
        if q.content_opportunity:
            lines.append(f"     - Content Opportunity: {q.content_opportunity}")

    lines.extend(["", "Top 5 Confusions/Misconceptions:"])
    for i, c in enumerate(report.top_confusions[:5], 1):
        lines.append(f"  {i}. {c.theme}")
        lines.append(f"     - Frequency: {c.frequency}")
        if c.clarification_needed:
            lines.append(f"     - Clarification Needed: {c.clarification_needed}")

    lines.extend(["", "Top 3 Trust Builders:"])
    for i, t in enumerate(report.trust_builders[:3], 1):
        lines.append(f"  {i}. {t.theme}")
        if t.evidence:
            lines.append(f"     - Evidence: {t.evidence}")

    lines.extend(["", "Top 3 Trust Killers:"])
    for i, t in enumerate(report.trust_killers[:3], 1):
        lines.append(f"  {i}. {t.theme}")
        if t.evidence:
            lines.append(f"     - Evidence: {t.evidence}")

    lines.extend(["", "Objections Worth Addressing:"])
    for o in report.objections_worth_addressing[:5]:
        lines.append(f"  - {o.theme}: {o.description}")

    lines.extend(["", "Audience Language:"])
    for lang in report.audience_language[:5]:
        lines.append(f"  - \"{lang.phrase}\" ({lang.frequency}x) - {lang.context}")

    lines.extend(["", "Emerging Themes:"])
    for theme in report.emerging_themes[:5]:
        lines.append(f"  - {theme}")

    lines.extend(["", "Content Opportunities:"])
    for opp in report.content_opportunities[:5]:
        lines.append(f"  - {opp}")

    lines.extend(["", "Pattern Changes:"])
    for key, value in report.pattern_changes.items():
        lines.append(f"  - {key}: {value}")

    if report.processing_notes:
        lines.extend(["", "Processing Notes:"])
        for note in report.processing_notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)
