"""
Learning Scout Agent - Resource Discovery Pipeline

Role: Search and discover learning resource candidates.

DOES:
- Search for learning resources across multiple sources
- Extract metadata from resources (duration, author, date)
- Assess source credibility
- Identify specific timestamps/sections for efficient consumption
- Provide relevance scoring

DOES NOT:
- Curate the final list (that's the Curator's job)
- Decide what to consume (that's the user's choice via Curator)
- Include gear reviews or product recommendations
- Make learning recommendations
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from . import LearningAgent, AgentResponse, AlfredState


class SourceType(Enum):
    """Types of learning resource sources."""
    YOUTUBE = "youtube"
    PODCAST = "podcast"
    ARTICLE = "article"
    PAPER = "paper"                   # Academic papers
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    COURSE = "course"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"
    NEWSLETTER = "newsletter"
    TWITTER_THREAD = "twitter_thread"
    CONFERENCE_TALK = "conference_talk"
    INTERVIEW = "interview"


class CredibilityLevel(Enum):
    """Credibility assessment levels."""
    AUTHORITATIVE = "authoritative"   # Recognized expert, peer-reviewed, official
    HIGH = "high"                      # Established creator, good track record
    MEDIUM = "medium"                  # Unknown but reasonable quality signals
    LOW = "low"                        # Red flags present
    UNVERIFIED = "unverified"          # Cannot assess


class RelevanceScore(Enum):
    """Relevance to the learning question."""
    DIRECT = "direct"                  # Directly answers the question
    RELATED = "related"                # Covers related concepts
    TANGENTIAL = "tangential"          # Loosely connected
    BACKGROUND = "background"          # General context


@dataclass
class AuthorProfile:
    """Profile of the resource author/creator."""
    name: str
    credentials: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    follower_count: Optional[int] = None
    verified: bool = False
    known_for: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "credentials": self.credentials,
            "affiliations": self.affiliations,
            "expertise_areas": self.expertise_areas,
            "follower_count": self.follower_count,
            "verified": self.verified,
            "known_for": self.known_for
        }


@dataclass
class TimestampSection:
    """A specific section/timestamp in a resource."""
    start_time: str                    # Format: "HH:MM:SS" or "MM:SS"
    end_time: Optional[str] = None
    title: str = ""
    relevance: RelevanceScore = RelevanceScore.RELATED
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "title": self.title,
            "relevance": self.relevance.value,
            "summary": self.summary
        }

    @property
    def duration_seconds(self) -> Optional[int]:
        """Calculate duration if both times available."""
        if not self.end_time:
            return None
        try:
            start = self._parse_time(self.start_time)
            end = self._parse_time(self.end_time)
            return end - start
        except (ValueError, TypeError):
            return None

    def _parse_time(self, time_str: str) -> int:
        """Parse time string to seconds."""
        parts = time_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(parts[0])


@dataclass
class ResourceMetadata:
    """Metadata extracted from a learning resource."""
    title: str
    url: str
    source_type: SourceType
    author: AuthorProfile
    duration_minutes: Optional[int] = None
    word_count: Optional[int] = None
    publish_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    language: str = "en"
    topics: List[str] = field(default_factory=list)
    description: str = ""
    thumbnail_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type.value,
            "author": self.author.to_dict(),
            "duration_minutes": self.duration_minutes,
            "word_count": self.word_count,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "language": self.language,
            "topics": self.topics,
            "description": self.description,
            "thumbnail_url": self.thumbnail_url
        }


@dataclass
class CredibilityAssessment:
    """Credibility assessment for a resource."""
    level: CredibilityLevel
    score: float                       # 0.0 to 1.0
    factors: Dict[str, Any] = field(default_factory=dict)
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "score": self.score,
            "factors": self.factors,
            "red_flags": self.red_flags,
            "green_flags": self.green_flags,
            "notes": self.notes
        }


@dataclass
class LearningCandidate:
    """A candidate learning resource discovered by the Scout."""
    metadata: ResourceMetadata
    credibility: CredibilityAssessment
    relevance: RelevanceScore
    relevance_score: float             # 0.0 to 1.0
    key_sections: List[TimestampSection] = field(default_factory=list)
    audio_friendly: bool = False
    requires_subscription: bool = False
    estimated_consumption_time: Optional[int] = None  # Actual time needed (may differ from duration)
    discovered_at: datetime = field(default_factory=datetime.now)
    search_query: str = ""             # Query that found this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.metadata.title,
            "source": self.metadata.url,
            "author": self.metadata.author.name,
            "type": self.metadata.source_type.value,
            "duration": self.metadata.duration_minutes,
            "credibility": self.credibility.level.value,
            "credibility_score": self.credibility.score,
            "relevance": self.relevance.value,
            "relevance_score": self.relevance_score,
            "key_sections": [s.to_dict() for s in self.key_sections],
            "audio_friendly": self.audio_friendly,
            "requires_subscription": self.requires_subscription,
            "estimated_consumption_time": self.estimated_consumption_time,
            "topics": self.metadata.topics,
            "description": self.metadata.description,
            "discovered_at": self.discovered_at.isoformat(),
            "search_query": self.search_query,
            "metadata": self.metadata.to_dict(),
            "credibility_details": self.credibility.to_dict()
        }


@dataclass
class SearchResults:
    """Results from a learning resource search."""
    query: str
    candidates: List[LearningCandidate] = field(default_factory=list)
    total_found: int = 0
    sources_searched: List[str] = field(default_factory=list)
    search_time_ms: int = 0
    filters_applied: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "candidates": [c.to_dict() for c in self.candidates],
            "total_found": self.total_found,
            "sources_searched": self.sources_searched,
            "search_time_ms": self.search_time_ms,
            "filters_applied": self.filters_applied
        }


class LearningScout(LearningAgent):
    """
    Learning Scout - Resource Discovery Agent

    Searches for and evaluates learning resources without making consumption
    decisions. Provides the Curator with high-quality candidates.
    """

    # Known authoritative sources by domain
    AUTHORITATIVE_DOMAINS = {
        "medical": [
            "pubmed.ncbi.nlm.nih.gov",
            "nejm.org",
            "jamanetwork.com",
            "thelancet.com",
            "bmj.com",
            "uptodate.com"
        ],
        "tech": [
            "arxiv.org",
            "github.com",
            "stackoverflow.com",
            "developer.mozilla.org",
            "docs.python.org"
        ],
        "business": [
            "hbr.org",
            "mckinsey.com",
            "stratechery.com"
        ]
    }

    # Red flag patterns for credibility
    CREDIBILITY_RED_FLAGS = [
        "sponsored content",
        "affiliate link",
        "gear review",
        "product review",
        "unboxing",
        "paid promotion",
        "buy now",
        "limited time offer"
    ]

    def __init__(self):
        super().__init__("LearningScout")
        self._search_history: List[SearchResults] = []

    def check_state_permission(self) -> tuple[bool, str]:
        """Learning scout is paused in RED state."""
        if self.alfred_state == AlfredState.RED:
            return False, "Learning scouting paused in RED state"
        return True, "Operation permitted"

    def search_resources(
        self,
        query: str,
        source_types: Optional[List[SourceType]] = None,
        max_duration_minutes: Optional[int] = None,
        min_credibility: CredibilityLevel = CredibilityLevel.MEDIUM,
        max_results: int = 20,
        exclude_subscriptions: bool = False
    ) -> AgentResponse:
        """
        Search for learning resources matching a query.

        Args:
            query: The learning question or topic to search
            source_types: Filter by source types (None = all)
            max_duration_minutes: Maximum duration filter
            min_credibility: Minimum credibility level
            max_results: Maximum number of results
            exclude_subscriptions: Exclude paid/subscription content

        Returns:
            AgentResponse with LEARNING_CANDIDATES
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Build search filters
        filters = {
            "source_types": [st.value for st in source_types] if source_types else "all",
            "max_duration_minutes": max_duration_minutes,
            "min_credibility": min_credibility.value,
            "exclude_subscriptions": exclude_subscriptions
        }

        # Perform search (simulated - would integrate with actual search APIs)
        candidates = self._execute_search(
            query=query,
            filters=filters,
            max_results=max_results
        )

        # Filter by credibility
        credibility_order = list(CredibilityLevel)
        min_index = credibility_order.index(min_credibility)
        candidates = [
            c for c in candidates
            if credibility_order.index(c.credibility.level) <= min_index
        ]

        # Filter by duration
        if max_duration_minutes:
            candidates = [
                c for c in candidates
                if c.metadata.duration_minutes is None or
                   c.metadata.duration_minutes <= max_duration_minutes
            ]

        # Filter subscriptions
        if exclude_subscriptions:
            candidates = [c for c in candidates if not c.requires_subscription]

        # Sort by relevance * credibility
        candidates.sort(
            key=lambda c: c.relevance_score * c.credibility.score,
            reverse=True
        )

        # Limit results
        candidates = candidates[:max_results]

        # Build results
        results = SearchResults(
            query=query,
            candidates=candidates,
            total_found=len(candidates),
            sources_searched=self._get_searched_sources(source_types),
            filters_applied=filters
        )

        # Store in history
        self._search_history.append(results)

        return self.create_response(
            data={
                "LEARNING_CANDIDATES": [c.to_dict() for c in candidates],
                "search_summary": {
                    "query": query,
                    "total_candidates": len(candidates),
                    "by_source_type": self._count_by_source(candidates),
                    "by_credibility": self._count_by_credibility(candidates),
                    "avg_duration_minutes": self._avg_duration(candidates),
                    "filters_applied": filters
                }
            }
        )

    def _execute_search(
        self,
        query: str,
        filters: Dict[str, Any],
        max_results: int
    ) -> List[LearningCandidate]:
        """
        Execute the actual search across sources.

        In production, this would integrate with:
        - YouTube Data API
        - Podcast indexes (Apple, Spotify)
        - Academic search (Google Scholar, PubMed)
        - Newsletter archives
        - Course platforms (Coursera, etc.)

        Returns placeholder structure for now.
        """
        # This is a placeholder - actual implementation would call APIs
        return []

    def _get_searched_sources(
        self,
        source_types: Optional[List[SourceType]]
    ) -> List[str]:
        """Get list of sources that were searched."""
        if source_types:
            return [st.value for st in source_types]
        return [st.value for st in SourceType]

    def _count_by_source(
        self,
        candidates: List[LearningCandidate]
    ) -> Dict[str, int]:
        """Count candidates by source type."""
        counts = {}
        for c in candidates:
            key = c.metadata.source_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_by_credibility(
        self,
        candidates: List[LearningCandidate]
    ) -> Dict[str, int]:
        """Count candidates by credibility level."""
        counts = {}
        for c in candidates:
            key = c.credibility.level.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _avg_duration(self, candidates: List[LearningCandidate]) -> Optional[float]:
        """Calculate average duration of candidates."""
        durations = [
            c.metadata.duration_minutes for c in candidates
            if c.metadata.duration_minutes is not None
        ]
        if not durations:
            return None
        return sum(durations) / len(durations)

    def extract_metadata(self, url: str) -> AgentResponse:
        """
        Extract metadata from a specific URL.

        Args:
            url: URL of the resource to analyze

        Returns:
            AgentResponse with extracted metadata
        """
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Determine source type from URL
        source_type = self._infer_source_type(url)

        # Extract metadata (would use actual APIs/scraping)
        metadata = self._extract_metadata_from_url(url, source_type)

        if metadata:
            return self.create_response(
                data={
                    "metadata": metadata.to_dict(),
                    "extraction_successful": True
                }
            )
        else:
            return self.create_response(
                data={
                    "metadata": None,
                    "extraction_successful": False,
                    "error": "Could not extract metadata from URL"
                },
                success=False,
                errors=["Metadata extraction failed"]
            )

    def _infer_source_type(self, url: str) -> SourceType:
        """Infer source type from URL."""
        url_lower = url.lower()

        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            return SourceType.YOUTUBE
        elif any(pod in url_lower for pod in ["podcast", "spotify.com/episode", "apple.com/podcast"]):
            return SourceType.PODCAST
        elif "arxiv.org" in url_lower or "pubmed" in url_lower:
            return SourceType.PAPER
        elif any(course in url_lower for course in ["coursera", "udemy", "edx", "skillshare"]):
            return SourceType.COURSE
        elif "twitter.com" in url_lower or "x.com" in url_lower:
            return SourceType.TWITTER_THREAD
        elif "substack.com" in url_lower:
            return SourceType.NEWSLETTER
        elif "docs." in url_lower or "documentation" in url_lower:
            return SourceType.DOCUMENTATION
        else:
            return SourceType.ARTICLE

    def _extract_metadata_from_url(
        self,
        url: str,
        source_type: SourceType
    ) -> Optional[ResourceMetadata]:
        """Extract metadata from URL. Placeholder for actual implementation."""
        # Would use appropriate APIs based on source type
        return None

    def assess_credibility(
        self,
        metadata: ResourceMetadata,
        content_sample: Optional[str] = None
    ) -> CredibilityAssessment:
        """
        Assess the credibility of a resource.

        Args:
            metadata: Resource metadata
            content_sample: Optional sample of content for analysis

        Returns:
            CredibilityAssessment with score and factors
        """
        factors = {}
        red_flags = []
        green_flags = []
        score = 0.5  # Start neutral

        # Check author credentials
        if metadata.author.credentials:
            green_flags.append(f"Author has credentials: {', '.join(metadata.author.credentials)}")
            score += 0.1
            factors["author_credentials"] = True

        if metadata.author.verified:
            green_flags.append("Author is verified")
            score += 0.05
            factors["author_verified"] = True

        # Check domain authority
        for domain, authoritative_urls in self.AUTHORITATIVE_DOMAINS.items():
            if any(auth_url in metadata.url for auth_url in authoritative_urls):
                green_flags.append(f"From authoritative {domain} source")
                score += 0.2
                factors["authoritative_domain"] = domain
                break

        # Check for red flags in title/description
        content_to_check = f"{metadata.title} {metadata.description}".lower()
        for flag in self.CREDIBILITY_RED_FLAGS:
            if flag in content_to_check:
                red_flags.append(f"Contains '{flag}'")
                score -= 0.15
                factors["red_flag_content"] = True

        # Check content sample if provided
        if content_sample:
            sample_lower = content_sample.lower()
            for flag in self.CREDIBILITY_RED_FLAGS:
                if flag in sample_lower:
                    red_flags.append(f"Content contains '{flag}'")
                    score -= 0.1

        # Check recency
        if metadata.publish_date:
            age_days = (datetime.now() - metadata.publish_date).days
            if age_days < 365:
                green_flags.append("Published within last year")
                factors["recent"] = True
            elif age_days > 1825:  # 5 years
                red_flags.append("Content is over 5 years old")
                score -= 0.1
                factors["outdated"] = True

        # Clamp score
        score = max(0.0, min(1.0, score))

        # Determine level
        if score >= 0.8:
            level = CredibilityLevel.AUTHORITATIVE
        elif score >= 0.6:
            level = CredibilityLevel.HIGH
        elif score >= 0.4:
            level = CredibilityLevel.MEDIUM
        elif score >= 0.2:
            level = CredibilityLevel.LOW
        else:
            level = CredibilityLevel.UNVERIFIED

        return CredibilityAssessment(
            level=level,
            score=score,
            factors=factors,
            red_flags=red_flags,
            green_flags=green_flags
        )

    def identify_key_sections(
        self,
        metadata: ResourceMetadata,
        learning_question: str,
        chapters: Optional[List[Dict[str, Any]]] = None
    ) -> List[TimestampSection]:
        """
        Identify key sections/timestamps relevant to the learning question.

        Args:
            metadata: Resource metadata
            learning_question: The question we're trying to answer
            chapters: Optional chapter/timestamp data from the resource

        Returns:
            List of relevant TimestampSection objects
        """
        sections = []

        if not chapters:
            return sections

        question_lower = learning_question.lower()
        question_words = set(question_lower.split())

        for chapter in chapters:
            title = chapter.get("title", "").lower()
            title_words = set(title.split())

            # Calculate relevance based on word overlap
            overlap = question_words & title_words
            overlap_ratio = len(overlap) / max(len(question_words), 1)

            if overlap_ratio > 0.2:  # Some relevance
                if overlap_ratio > 0.5:
                    relevance = RelevanceScore.DIRECT
                elif overlap_ratio > 0.3:
                    relevance = RelevanceScore.RELATED
                else:
                    relevance = RelevanceScore.TANGENTIAL

                section = TimestampSection(
                    start_time=chapter.get("start_time", "0:00"),
                    end_time=chapter.get("end_time"),
                    title=chapter.get("title", ""),
                    relevance=relevance,
                    summary=chapter.get("summary", "")
                )
                sections.append(section)

        # Sort by relevance
        relevance_order = [
            RelevanceScore.DIRECT,
            RelevanceScore.RELATED,
            RelevanceScore.TANGENTIAL,
            RelevanceScore.BACKGROUND
        ]
        sections.sort(key=lambda s: relevance_order.index(s.relevance))

        return sections

    def calculate_relevance(
        self,
        metadata: ResourceMetadata,
        learning_question: str
    ) -> tuple[RelevanceScore, float]:
        """
        Calculate relevance of a resource to a learning question.

        Returns:
            Tuple of (RelevanceScore enum, numeric score 0-1)
        """
        question_lower = learning_question.lower()
        question_words = set(question_lower.split())

        # Check title
        title_lower = metadata.title.lower()
        title_words = set(title_lower.split())
        title_overlap = len(question_words & title_words) / max(len(question_words), 1)

        # Check description
        desc_lower = metadata.description.lower()
        desc_words = set(desc_lower.split())
        desc_overlap = len(question_words & desc_words) / max(len(question_words), 1)

        # Check topics
        topic_text = " ".join(metadata.topics).lower()
        topic_overlap = sum(1 for w in question_words if w in topic_text) / max(len(question_words), 1)

        # Weighted combination
        score = (title_overlap * 0.4) + (desc_overlap * 0.3) + (topic_overlap * 0.3)

        # Determine level
        if score >= 0.6:
            level = RelevanceScore.DIRECT
        elif score >= 0.4:
            level = RelevanceScore.RELATED
        elif score >= 0.2:
            level = RelevanceScore.TANGENTIAL
        else:
            level = RelevanceScore.BACKGROUND

        return level, score

    def filter_gear_reviews(
        self,
        candidates: List[LearningCandidate]
    ) -> List[LearningCandidate]:
        """
        Filter out gear reviews and product recommendations.

        Args:
            candidates: List of candidates to filter

        Returns:
            Filtered list without gear/product content
        """
        gear_keywords = [
            "gear review", "product review", "unboxing",
            "best gear", "top 10 products", "buying guide",
            "amazon finds", "wish list", "gear list",
            "what's in my bag", "equipment review"
        ]

        filtered = []
        for candidate in candidates:
            content = f"{candidate.metadata.title} {candidate.metadata.description}".lower()
            is_gear_review = any(keyword in content for keyword in gear_keywords)

            if not is_gear_review:
                filtered.append(candidate)

        return filtered

    def get_search_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent search history."""
        recent = self._search_history[-limit:] if self._search_history else []
        return [
            {
                "query": r.query,
                "total_found": r.total_found,
                "sources_searched": r.sources_searched
            }
            for r in recent
        ]
