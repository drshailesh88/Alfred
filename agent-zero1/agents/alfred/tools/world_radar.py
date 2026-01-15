# World Radar - Strategic Signal Detection Agent
# Detects global developments that could change constraints on clinical practice,
# AI tooling, regulatory environment, or reputational exposure.

from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import re
from abc import ABC, abstractmethod

from . import SignalAgent, AgentResponse, AlfredState


# =============================================================================
# ENUMS
# =============================================================================

class Domain(Enum):
    """Domains that World Radar monitors for constraint-changing events."""
    CLINICAL = "clinical"
    AI = "AI"
    REGULATION = "regulation"
    REPUTATION = "reputation"
    PLATFORM_POLICY = "platform_policy"

    @classmethod
    def from_string(cls, value: str) -> Optional['Domain']:
        """Parse domain from string input."""
        value_lower = value.lower().strip()
        mapping = {
            "clinical": cls.CLINICAL,
            "ai": cls.AI,
            "regulation": cls.REGULATION,
            "reputation": cls.REPUTATION,
            "platform_policy": cls.PLATFORM_POLICY,
            "platform": cls.PLATFORM_POLICY,
            "policy": cls.PLATFORM_POLICY,
        }
        return mapping.get(value_lower)

    @classmethod
    def parse_domains(cls, domains_input: str) -> List['Domain']:
        """Parse multiple domains from input string."""
        if domains_input.lower().strip() == "all":
            return list(cls)

        domains = []
        for part in domains_input.split("/"):
            domain = cls.from_string(part.strip())
            if domain:
                domains.append(domain)
        return domains if domains else list(cls)


class TimeHorizon(Enum):
    """Time horizon for required action on detected signal."""
    IMMEDIATE = "immediate"      # Action required now
    NEAR = "near"               # Action required within 30 days
    LONG = "long"               # Action timeline > 30 days

    @property
    def description(self) -> str:
        descriptions = {
            TimeHorizon.IMMEDIATE: "Action required immediately",
            TimeHorizon.NEAR: "Action required within 30 days",
            TimeHorizon.LONG: "Action timeline exceeds 30 days"
        }
        return descriptions[self]

    @property
    def days_threshold(self) -> int:
        """Maximum days for this horizon category."""
        thresholds = {
            TimeHorizon.IMMEDIATE: 0,
            TimeHorizon.NEAR: 30,
            TimeHorizon.LONG: 365  # Arbitrary upper bound
        }
        return thresholds[self]


class ActionRequired(Enum):
    """Level of action required in response to a detected signal."""
    NONE = "none"
    MONITOR = "monitor"
    REVIEW = "review"
    URGENT_REVIEW = "urgent_review"

    @property
    def priority_level(self) -> int:
        """Numeric priority for sorting (higher = more urgent)."""
        levels = {
            ActionRequired.NONE: 0,
            ActionRequired.MONITOR: 1,
            ActionRequired.REVIEW: 2,
            ActionRequired.URGENT_REVIEW: 3
        }
        return levels[self]


class Confidence(Enum):
    """Confidence level in the detected signal."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def weight(self) -> float:
        """Numeric weight for confidence scoring."""
        weights = {
            Confidence.HIGH: 1.0,
            Confidence.MEDIUM: 0.6,
            Confidence.LOW: 0.3
        }
        return weights[self]


class SourceCredibility(Enum):
    """Credibility assessment of information source."""
    AUTHORITATIVE = "authoritative"    # Government, major institutions
    CREDIBLE = "credible"              # Established media, peer-reviewed
    EMERGING = "emerging"              # Newer but verifiable sources
    UNVERIFIED = "unverified"          # Requires additional confirmation

    @property
    def trust_score(self) -> float:
        """Trust score for source weighting."""
        scores = {
            SourceCredibility.AUTHORITATIVE: 1.0,
            SourceCredibility.CREDIBLE: 0.8,
            SourceCredibility.EMERGING: 0.5,
            SourceCredibility.UNVERIFIED: 0.2
        }
        return scores[self]


class EscalationCriterion(Enum):
    """The five escalation criteria for World Radar."""
    CLINICAL_GUIDELINES_CHANGE = "clinical_practice_guidelines_may_change"
    REGULATION_AFFECTS_SPEECH_TOOLS = "regulation_affects_speech_or_tool_usage"
    REPUTATIONAL_EXPOSURE_INCREASE = "reputational_exposure_increases_materially"
    PLATFORM_POLICY_AFFECTS_STRATEGY = "platform_policy_changes_affect_content_strategy"
    AI_TOOL_AVAILABILITY_SHIFTS = "ai_tool_availability_or_legality_shifts"

    @property
    def description(self) -> str:
        descriptions = {
            EscalationCriterion.CLINICAL_GUIDELINES_CHANGE:
                "Clinical practice guidelines may change",
            EscalationCriterion.REGULATION_AFFECTS_SPEECH_TOOLS:
                "Regulation affects speech or tool usage",
            EscalationCriterion.REPUTATIONAL_EXPOSURE_INCREASE:
                "Reputational exposure increases materially",
            EscalationCriterion.PLATFORM_POLICY_AFFECTS_STRATEGY:
                "Platform policy changes affect content strategy",
            EscalationCriterion.AI_TOOL_AVAILABILITY_SHIFTS:
                "AI tool availability or legality shifts"
        }
        return descriptions[self]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WorldScanRequest:
    """
    Input format for World Radar scan requests.

    WORLD_SCAN_REQUEST
    - Domains: [clinical / AI / regulation / reputation / all]
    - Geographic Focus: [regions of relevance]
    - Time Since Last Scan: [duration]
    - Active Concerns: [any specific topics to weight]
    """
    domains: List[Domain] = field(default_factory=lambda: list(Domain))
    geographic_focus: List[str] = field(default_factory=list)
    time_since_last_scan: Optional[timedelta] = None
    active_concerns: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldScanRequest':
        """Parse a WorldScanRequest from dictionary input."""
        domains_input = data.get("domains", "all")
        if isinstance(domains_input, str):
            domains = Domain.parse_domains(domains_input)
        elif isinstance(domains_input, list):
            domains = []
            for d in domains_input:
                if isinstance(d, Domain):
                    domains.append(d)
                elif isinstance(d, str):
                    parsed = Domain.from_string(d)
                    if parsed:
                        domains.append(parsed)
        else:
            domains = list(Domain)

        geo_focus = data.get("geographic_focus", [])
        if isinstance(geo_focus, str):
            geo_focus = [g.strip() for g in geo_focus.split(",") if g.strip()]

        time_since = data.get("time_since_last_scan")
        if isinstance(time_since, str):
            # Parse duration strings like "24h", "7d", "1w"
            time_since = cls._parse_duration(time_since)
        elif isinstance(time_since, (int, float)):
            time_since = timedelta(hours=time_since)
        elif isinstance(time_since, timedelta):
            pass
        else:
            time_since = None

        concerns = data.get("active_concerns", [])
        if isinstance(concerns, str):
            concerns = [c.strip() for c in concerns.split(",") if c.strip()]

        return cls(
            domains=domains,
            geographic_focus=geo_focus,
            time_since_last_scan=time_since,
            active_concerns=concerns
        )

    @staticmethod
    def _parse_duration(duration_str: str) -> Optional[timedelta]:
        """Parse duration string to timedelta."""
        duration_str = duration_str.lower().strip()
        match = re.match(r'^(\d+(?:\.\d+)?)\s*(h|d|w|m|hours?|days?|weeks?|months?)$', duration_str)
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)

        if unit.startswith('h'):
            return timedelta(hours=value)
        elif unit.startswith('d'):
            return timedelta(days=value)
        elif unit.startswith('w'):
            return timedelta(weeks=value)
        elif unit.startswith('m'):
            return timedelta(days=value * 30)  # Approximate month
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "domains": [d.value for d in self.domains],
            "geographic_focus": self.geographic_focus,
            "time_since_last_scan": str(self.time_since_last_scan) if self.time_since_last_scan else None,
            "active_concerns": self.active_concerns
        }


@dataclass
class ConstraintImpact:
    """Analysis of how a signal impacts operational constraints."""
    constraint_type: str
    current_state: str
    projected_state: str
    impact_severity: str  # minor, moderate, major, critical
    affected_operations: List[str]
    mitigation_options: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "current_state": self.current_state,
            "projected_state": self.projected_state,
            "impact_severity": self.impact_severity,
            "affected_operations": self.affected_operations,
            "mitigation_options": self.mitigation_options
        }


@dataclass
class WorldSignal:
    """
    Output format for detected world signals.

    WORLD_SIGNAL
    - Event: (concise description)
    - Domain: clinical | AI | regulation | reputation | platform_policy
    - Source: [credible source reference]
    - Time Horizon: immediate | near (<30 days) | long (>30 days)
    - Constraint Impact: [what operational constraint changes]
    - Action Required: none | monitor | review | urgent_review
    - Confidence: high | medium | low
    """
    event: str
    domain: Domain
    source: str
    source_credibility: SourceCredibility
    time_horizon: TimeHorizon
    constraint_impact: ConstraintImpact
    action_required: ActionRequired
    confidence: Confidence
    escalation_criteria_met: List[EscalationCriterion] = field(default_factory=list)
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    geographic_relevance: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event": self.event,
            "domain": self.domain.value,
            "source": self.source,
            "source_credibility": self.source_credibility.value,
            "time_horizon": self.time_horizon.value,
            "constraint_impact": self.constraint_impact.to_dict(),
            "action_required": self.action_required.value,
            "confidence": self.confidence.value,
            "escalation_criteria_met": [c.value for c in self.escalation_criteria_met],
            "detected_at": self.detected_at,
            "geographic_relevance": self.geographic_relevance
        }

    def to_formatted_output(self) -> str:
        """Generate formatted WORLD_SIGNAL output."""
        lines = [
            "WORLD_SIGNAL",
            f"- Event: {self.event}",
            f"- Domain: {self.domain.value}",
            f"- Source: {self.source}",
            f"- Time Horizon: {self.time_horizon.value}",
            f"- Constraint Impact: {self.constraint_impact.projected_state}",
            f"- Action Required: {self.action_required.value}",
            f"- Confidence: {self.confidence.value}"
        ]
        return "\n".join(lines)

    @classmethod
    def silence_signal(cls) -> 'WorldSignal':
        """Create silence protocol response when no events detected."""
        return cls(
            event="None detected",
            domain=Domain.CLINICAL,  # Placeholder
            source="World Radar internal",
            source_credibility=SourceCredibility.AUTHORITATIVE,
            time_horizon=TimeHorizon.LONG,
            constraint_impact=ConstraintImpact(
                constraint_type="none",
                current_state="nominal",
                projected_state="no change",
                impact_severity="none",
                affected_operations=[],
                mitigation_options=[]
            ),
            action_required=ActionRequired.NONE,
            confidence=Confidence.HIGH,
            escalation_criteria_met=[]
        )


# =============================================================================
# EXTERNAL SERVICE INTERFACES
# =============================================================================

class ExternalServiceInterface(ABC):
    """Abstract interface for external data services."""

    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute a search against the external service."""
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """Return the name of this service."""
        pass


class PubMedInterface(ExternalServiceInterface):
    """
    Interface for mcp-simple-pubmed MCP tool.
    Handles medical literature searches for clinical domain signals.
    """

    def __init__(self, mcp_client: Optional[Any] = None):
        """
        Initialize PubMed interface.

        Args:
            mcp_client: MCP client instance for tool communication.
                        If None, operates in stub mode for testing.
        """
        self._client = mcp_client
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(hours=1)

    def get_service_name(self) -> str:
        return "mcp-simple-pubmed"

    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search PubMed for medical literature.

        Args:
            query: Search query string
            **kwargs: Additional search parameters
                - max_results: Maximum number of results (default 20)
                - date_range: Date range for filtering
                - article_types: Types of articles to include

        Returns:
            List of article metadata dictionaries
        """
        max_results = kwargs.get("max_results", 20)
        date_range = kwargs.get("date_range")

        cache_key = f"{query}:{max_results}:{date_range}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if self._client is None:
            # Stub mode - return empty results
            return []

        try:
            # MCP tool call format
            result = await self._client.call_tool(
                "pubmed_search",
                {
                    "query": query,
                    "max_results": max_results,
                    "date_range": date_range
                }
            )

            articles = self._parse_pubmed_response(result)
            self._set_cached(cache_key, articles)
            return articles

        except Exception as e:
            # Log error but don't fail the scan
            return []

    def _parse_pubmed_response(self, response: Any) -> List[Dict[str, Any]]:
        """Parse PubMed API response into standardized format."""
        if not response:
            return []

        articles = []
        for item in response.get("articles", []):
            articles.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "authors": item.get("authors", []),
                "journal": item.get("journal", ""),
                "publication_date": item.get("pub_date", ""),
                "pmid": item.get("pmid", ""),
                "doi": item.get("doi", ""),
                "mesh_terms": item.get("mesh_terms", []),
                "source": "pubmed"
            })
        return articles

    def _get_cached(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached result if not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - entry["timestamp"] < self._cache_ttl:
                return entry["data"]
            del self._cache[key]
        return None

    def _set_cached(self, key: str, data: List[Dict[str, Any]]):
        """Cache search results."""
        self._cache[key] = {
            "data": data,
            "timestamp": datetime.now()
        }

    def build_clinical_query(self, topics: List[str],
                            geographic_focus: List[str] = None) -> str:
        """
        Build optimized PubMed query for clinical guideline changes.

        Args:
            topics: Clinical topics to search
            geographic_focus: Geographic regions to focus on

        Returns:
            Formatted PubMed query string
        """
        # Focus on guideline and practice changes
        base_terms = [
            "practice guideline[pt]",
            "guideline[pt]",
            "consensus development conference[pt]",
            "clinical protocol[mh]"
        ]

        topic_query = " OR ".join(f'"{t}"' for t in topics)
        base_query = " OR ".join(base_terms)

        query = f"({topic_query}) AND ({base_query})"

        # Add geographic filtering if specified
        if geographic_focus:
            geo_terms = " OR ".join(f'"{g}"[ad]' for g in geographic_focus)
            query = f"{query} AND ({geo_terms})"

        return query


class WebSearchInterface(ExternalServiceInterface):
    """
    Interface for WebSearch tool.
    Handles news and general web searches for regulatory and policy signals.
    """

    def __init__(self, search_client: Optional[Any] = None):
        """
        Initialize WebSearch interface.

        Args:
            search_client: WebSearch client instance.
                          If None, operates in stub mode for testing.
        """
        self._client = search_client
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=30)

    def get_service_name(self) -> str:
        return "WebSearch"

    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute web search.

        Args:
            query: Search query string
            **kwargs: Additional search parameters
                - domains: List of domains to search
                - exclude_domains: Domains to exclude
                - time_range: Time range filter (e.g., "past_week")
                - result_count: Number of results

        Returns:
            List of search result dictionaries
        """
        result_count = kwargs.get("result_count", 20)
        time_range = kwargs.get("time_range", "past_week")
        domains = kwargs.get("domains", [])
        exclude_domains = kwargs.get("exclude_domains", [])

        cache_key = f"{query}:{result_count}:{time_range}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if self._client is None:
            return []

        try:
            result = await self._client.search(
                query=query,
                allowed_domains=domains if domains else None,
                blocked_domains=exclude_domains if exclude_domains else None
            )

            parsed = self._parse_search_response(result)
            self._set_cached(cache_key, parsed)
            return parsed

        except Exception as e:
            return []

    def _parse_search_response(self, response: Any) -> List[Dict[str, Any]]:
        """Parse WebSearch response into standardized format."""
        if not response:
            return []

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "domain": item.get("domain", ""),
                "published_date": item.get("date", ""),
                "source": "web_search"
            })
        return results

    def _get_cached(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached result if not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - entry["timestamp"] < self._cache_ttl:
                return entry["data"]
            del self._cache[key]
        return None

    def _set_cached(self, key: str, data: List[Dict[str, Any]]):
        """Cache search results."""
        self._cache[key] = {
            "data": data,
            "timestamp": datetime.now()
        }

    def get_authoritative_domains(self, domain: Domain) -> List[str]:
        """
        Get list of authoritative domains for a given signal domain.

        Args:
            domain: The Domain to get authoritative sources for

        Returns:
            List of domain strings for focused searching
        """
        domain_sources = {
            Domain.CLINICAL: [
                "nih.gov", "cdc.gov", "who.int", "fda.gov",
                "cms.gov", "ncbi.nlm.nih.gov", "cochranelibrary.com",
                "nice.org.uk", "ahrq.gov"
            ],
            Domain.AI: [
                "whitehouse.gov", "congress.gov", "ftc.gov",
                "nist.gov", "europa.eu", "openai.com",
                "anthropic.com", "ai.google", "arxiv.org"
            ],
            Domain.REGULATION: [
                "federalregister.gov", "regulations.gov",
                "congress.gov", "whitehouse.gov", "ftc.gov",
                "sec.gov", "hhs.gov", "state.gov",
                "europa.eu", "gov.uk"
            ],
            Domain.REPUTATION: [
                "reuters.com", "apnews.com", "bbc.com",
                "nytimes.com", "washingtonpost.com",
                "wsj.com", "bloomberg.com"
            ],
            Domain.PLATFORM_POLICY: [
                "blog.twitter.com", "about.fb.com",
                "blog.youtube", "help.instagram.com",
                "newsroom.tiktok.com", "transparency.google"
            ]
        }
        return domain_sources.get(domain, [])

    def build_domain_query(self, domain: Domain,
                          concerns: List[str] = None,
                          geographic_focus: List[str] = None) -> str:
        """
        Build search query optimized for a specific domain.

        Args:
            domain: The Domain to build query for
            concerns: Specific topics to include
            geographic_focus: Geographic regions to focus on

        Returns:
            Formatted search query string
        """
        base_queries = {
            Domain.CLINICAL: "medical practice guidelines change OR clinical protocol update OR healthcare regulation",
            Domain.AI: "AI regulation OR artificial intelligence policy OR AI governance OR AI legislation",
            Domain.REGULATION: "new regulation OR regulatory change OR policy announcement OR legal framework",
            Domain.REPUTATION: "medical professional controversy OR healthcare provider news",
            Domain.PLATFORM_POLICY: "platform policy change OR content policy update OR terms of service change"
        }

        query = base_queries.get(domain, "")

        if concerns:
            concern_terms = " OR ".join(f'"{c}"' for c in concerns)
            query = f"({query}) AND ({concern_terms})"

        if geographic_focus:
            geo_terms = " OR ".join(geographic_focus)
            query = f"({query}) AND ({geo_terms})"

        return query


# =============================================================================
# ANALYSIS COMPONENTS
# =============================================================================

class EscalationAnalyzer:
    """
    Analyzes raw signals against escalation criteria.
    Determines if a signal warrants escalation to Alfred.
    """

    # Keywords that suggest each escalation criterion might be met
    CRITERION_KEYWORDS = {
        EscalationCriterion.CLINICAL_GUIDELINES_CHANGE: [
            "guideline", "practice change", "protocol update",
            "standard of care", "clinical recommendation",
            "medical board", "accreditation", "certification requirement"
        ],
        EscalationCriterion.REGULATION_AFFECTS_SPEECH_TOOLS: [
            "regulation", "legislation", "law", "ban", "restrict",
            "prohibit", "mandate", "require", "compliance",
            "speech", "communication", "disclosure"
        ],
        EscalationCriterion.REPUTATIONAL_EXPOSURE_INCREASE: [
            "investigation", "lawsuit", "allegation", "controversy",
            "scandal", "criticism", "complaint", "review",
            "malpractice", "disciplinary"
        ],
        EscalationCriterion.PLATFORM_POLICY_AFFECTS_STRATEGY: [
            "policy change", "terms of service", "content policy",
            "algorithm change", "moderation", "demonetization",
            "reach", "shadowban", "visibility"
        ],
        EscalationCriterion.AI_TOOL_AVAILABILITY_SHIFTS: [
            "AI ban", "AI regulation", "model restriction",
            "API change", "deprecation", "availability",
            "AI liability", "AI compliance", "AI certification"
        ]
    }

    def analyze(self, raw_signal: Dict[str, Any]) -> List[EscalationCriterion]:
        """
        Analyze a raw signal to determine which escalation criteria are met.

        Args:
            raw_signal: Raw signal data from external sources

        Returns:
            List of EscalationCriterion that the signal meets
        """
        met_criteria = []

        # Combine relevant text fields for analysis
        text = self._extract_text(raw_signal).lower()

        for criterion, keywords in self.CRITERION_KEYWORDS.items():
            if self._matches_criterion(text, keywords):
                met_criteria.append(criterion)

        return met_criteria

    def _extract_text(self, signal: Dict[str, Any]) -> str:
        """Extract searchable text from signal data."""
        text_fields = ["title", "abstract", "snippet", "content", "description"]
        texts = []
        for field in text_fields:
            if field in signal and signal[field]:
                texts.append(str(signal[field]))
        return " ".join(texts)

    def _matches_criterion(self, text: str, keywords: List[str]) -> bool:
        """Check if text matches any keywords for a criterion."""
        # Require at least 2 keyword matches for higher confidence
        matches = sum(1 for kw in keywords if kw.lower() in text)
        return matches >= 2


class SourceCredibilityAssessor:
    """
    Assesses the credibility of information sources.
    Uses domain reputation and source characteristics.
    """

    # Authoritative source domains by category
    AUTHORITATIVE_SOURCES = {
        # Government and official bodies
        "gov": ["nih.gov", "cdc.gov", "fda.gov", "cms.gov", "hhs.gov",
                "whitehouse.gov", "congress.gov", "ftc.gov", "sec.gov",
                "federalregister.gov", "who.int", "europa.eu", "gov.uk"],
        # Medical and scientific
        "medical": ["pubmed.ncbi.nlm.nih.gov", "cochranelibrary.com",
                   "nejm.org", "jamanetwork.com", "thelancet.com",
                   "bmj.com", "nature.com", "science.org"],
        # Major news (credible, not authoritative)
        "news": ["reuters.com", "apnews.com", "bbc.com",
                "nytimes.com", "wsj.com", "bloomberg.com"],
        # Platform official
        "platform": ["blog.twitter.com", "about.fb.com", "blog.youtube",
                    "newsroom.tiktok.com", "transparency.google"]
    }

    def assess(self, source_url: str, signal_domain: Domain) -> SourceCredibility:
        """
        Assess credibility of a source.

        Args:
            source_url: URL or source identifier
            signal_domain: The domain this signal relates to

        Returns:
            SourceCredibility assessment
        """
        source_lower = source_url.lower()

        # Check authoritative sources
        for source in self.AUTHORITATIVE_SOURCES["gov"]:
            if source in source_lower:
                return SourceCredibility.AUTHORITATIVE

        for source in self.AUTHORITATIVE_SOURCES["medical"]:
            if source in source_lower:
                return SourceCredibility.AUTHORITATIVE

        # Check credible sources
        for source in self.AUTHORITATIVE_SOURCES["news"]:
            if source in source_lower:
                return SourceCredibility.CREDIBLE

        for source in self.AUTHORITATIVE_SOURCES["platform"]:
            if source in source_lower:
                return SourceCredibility.CREDIBLE

        # Check for signs of emerging but verifiable sources
        if self._has_verification_indicators(source_lower):
            return SourceCredibility.EMERGING

        return SourceCredibility.UNVERIFIED

    def _has_verification_indicators(self, source: str) -> bool:
        """Check for indicators that a source may be verifiable."""
        # Domain patterns suggesting some credibility
        credible_patterns = [
            ".edu", ".org", "research", "institute",
            "university", "journal", "academic"
        ]
        return any(pattern in source for pattern in credible_patterns)


class TimeHorizonAssessor:
    """
    Assesses the time horizon for action based on signal content.
    """

    IMMEDIATE_INDICATORS = [
        "effective immediately", "now in effect", "emergency",
        "urgent", "breaking", "today", "immediate",
        "with immediate effect", "as of today"
    ]

    NEAR_INDICATORS = [
        "next month", "coming weeks", "within 30 days",
        "effective [date within 30 days]", "soon",
        "this quarter", "upcoming", "imminent"
    ]

    LONG_INDICATORS = [
        "proposed", "under consideration", "public comment",
        "draft", "preliminary", "future", "long-term",
        "next year", "planning stages"
    ]

    def assess(self, signal_text: str,
              effective_date: Optional[datetime] = None) -> TimeHorizon:
        """
        Assess time horizon for a signal.

        Args:
            signal_text: Text content of the signal
            effective_date: Known effective date if available

        Returns:
            TimeHorizon assessment
        """
        text_lower = signal_text.lower()

        # If we have an effective date, use it
        if effective_date:
            days_until = (effective_date - datetime.now()).days
            if days_until <= 0:
                return TimeHorizon.IMMEDIATE
            elif days_until <= 30:
                return TimeHorizon.NEAR
            else:
                return TimeHorizon.LONG

        # Otherwise analyze text
        if any(ind in text_lower for ind in self.IMMEDIATE_INDICATORS):
            return TimeHorizon.IMMEDIATE

        if any(ind in text_lower for ind in self.NEAR_INDICATORS):
            return TimeHorizon.NEAR

        if any(ind in text_lower for ind in self.LONG_INDICATORS):
            return TimeHorizon.LONG

        # Default to NEAR if uncertain but signal exists
        return TimeHorizon.NEAR


class ConstraintImpactAnalyzer:
    """
    Analyzes how detected signals impact operational constraints.
    """

    # Constraint types by domain
    CONSTRAINT_TYPES = {
        Domain.CLINICAL: [
            "clinical_practice_scope",
            "treatment_protocols",
            "documentation_requirements",
            "licensure_requirements",
            "continuing_education"
        ],
        Domain.AI: [
            "tool_availability",
            "usage_restrictions",
            "liability_requirements",
            "disclosure_requirements",
            "data_handling"
        ],
        Domain.REGULATION: [
            "speech_limitations",
            "advertising_restrictions",
            "privacy_requirements",
            "reporting_obligations",
            "compliance_requirements"
        ],
        Domain.REPUTATION: [
            "public_communication",
            "content_strategy",
            "platform_presence",
            "professional_positioning"
        ],
        Domain.PLATFORM_POLICY: [
            "content_distribution",
            "monetization",
            "audience_reach",
            "content_format",
            "engagement_rules"
        ]
    }

    def analyze(self, signal_data: Dict[str, Any],
               domain: Domain,
               escalation_criteria: List[EscalationCriterion]) -> ConstraintImpact:
        """
        Analyze constraint impact of a signal.

        Args:
            signal_data: Raw signal data
            domain: Domain of the signal
            escalation_criteria: Met escalation criteria

        Returns:
            ConstraintImpact analysis
        """
        text = self._extract_text(signal_data).lower()

        # Determine affected constraint type
        constraint_type = self._identify_constraint_type(domain, text)

        # Assess severity
        severity = self._assess_severity(escalation_criteria, text)

        # Identify affected operations
        affected_ops = self._identify_affected_operations(domain, escalation_criteria)

        # Generate mitigation options
        mitigations = self._generate_mitigations(domain, severity)

        return ConstraintImpact(
            constraint_type=constraint_type,
            current_state="nominal",
            projected_state=self._describe_projected_state(domain, severity),
            impact_severity=severity,
            affected_operations=affected_ops,
            mitigation_options=mitigations
        )

    def _extract_text(self, data: Dict[str, Any]) -> str:
        """Extract text from signal data."""
        fields = ["title", "snippet", "abstract", "content"]
        return " ".join(str(data.get(f, "")) for f in fields)

    def _identify_constraint_type(self, domain: Domain, text: str) -> str:
        """Identify the primary constraint type affected."""
        constraints = self.CONSTRAINT_TYPES.get(domain, [])
        # Return first constraint as primary (could be enhanced with text analysis)
        return constraints[0] if constraints else "general_operations"

    def _assess_severity(self, criteria: List[EscalationCriterion],
                        text: str) -> str:
        """Assess impact severity."""
        # More criteria met = higher severity
        if len(criteria) >= 3:
            return "critical"
        elif len(criteria) == 2:
            return "major"
        elif len(criteria) == 1:
            return "moderate"
        return "minor"

    def _identify_affected_operations(self, domain: Domain,
                                     criteria: List[EscalationCriterion]) -> List[str]:
        """Identify operations affected by the signal."""
        ops_map = {
            Domain.CLINICAL: ["patient_care", "documentation", "billing"],
            Domain.AI: ["content_generation", "research_assistance", "automation"],
            Domain.REGULATION: ["public_communication", "marketing", "compliance"],
            Domain.REPUTATION: ["social_media", "content_strategy", "public_relations"],
            Domain.PLATFORM_POLICY: ["content_distribution", "engagement", "monetization"]
        }
        return ops_map.get(domain, ["general_operations"])

    def _generate_mitigations(self, domain: Domain, severity: str) -> List[str]:
        """Generate potential mitigation options."""
        base_mitigations = [
            "Monitor for further developments",
            "Document current practices for comparison"
        ]

        if severity in ["critical", "major"]:
            base_mitigations.extend([
                "Review current workflows for compliance",
                "Consult with relevant advisors",
                "Prepare contingency plans"
            ])

        return base_mitigations

    def _describe_projected_state(self, domain: Domain, severity: str) -> str:
        """Describe projected constraint state."""
        projections = {
            "critical": f"{domain.value} operations may require significant adjustment",
            "major": f"{domain.value} operations may need review and modification",
            "moderate": f"{domain.value} practices may need minor updates",
            "minor": f"No immediate change to {domain.value} operations expected"
        }
        return projections.get(severity, "Impact assessment pending")


# =============================================================================
# SIGNAL FILTER
# =============================================================================

class SignalFilter:
    """
    Ruthlessly filters noise from signal.
    Only passes through constraint-changing events.
    """

    # Content types that are NOT constraint-changing (noise)
    NOISE_PATTERNS = [
        # News reporting without action items
        r"reported that",
        r"according to sources",
        r"rumors suggest",
        # Trend and interest pieces
        r"trending",
        r"viral",
        r"popular",
        r"influencer",
        r"celebrity",
        # Opinion and speculation
        r"may eventually",
        r"could potentially",
        r"experts speculate",
        r"some believe",
        # General updates without specifics
        r"continues to grow",
        r"remains stable",
        r"no change expected"
    ]

    # Minimum criteria for signal to pass
    SIGNAL_REQUIREMENTS = {
        "min_criteria_met": 1,
        "min_confidence": Confidence.LOW,
        "exclude_unverified": False  # Include unverified with lower weight
    }

    def __init__(self, requirements: Dict[str, Any] = None):
        """Initialize filter with optional custom requirements."""
        self._requirements = requirements or self.SIGNAL_REQUIREMENTS
        self._noise_patterns = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]

    def is_signal(self, raw_data: Dict[str, Any],
                 escalation_criteria: List[EscalationCriterion],
                 source_credibility: SourceCredibility,
                 confidence: Confidence) -> bool:
        """
        Determine if raw data constitutes a true signal.

        Args:
            raw_data: Raw source data
            escalation_criteria: Criteria met by this signal
            source_credibility: Assessed source credibility
            confidence: Confidence in the signal

        Returns:
            True if this is a signal, False if noise
        """
        # Must meet minimum escalation criteria
        if len(escalation_criteria) < self._requirements["min_criteria_met"]:
            return False

        # Check confidence threshold
        min_conf = self._requirements["min_confidence"]
        if confidence.weight < min_conf.weight:
            return False

        # Exclude unverified if required
        if (self._requirements["exclude_unverified"] and
            source_credibility == SourceCredibility.UNVERIFIED):
            return False

        # Check for noise patterns
        text = self._extract_text(raw_data).lower()
        noise_matches = sum(1 for p in self._noise_patterns if p.search(text))

        # If more than 2 noise patterns match, likely not a signal
        if noise_matches > 2:
            return False

        return True

    def _extract_text(self, data: Dict[str, Any]) -> str:
        """Extract text for noise pattern matching."""
        fields = ["title", "snippet", "abstract", "content"]
        return " ".join(str(data.get(f, "")) for f in fields)


# =============================================================================
# MAIN WORLD RADAR CLASS
# =============================================================================

class WorldRadar(SignalAgent):
    """
    World Radar - Strategic Signal Detection Agent

    Detects global developments that could change constraints on:
    - Clinical practice
    - AI tooling
    - Regulatory environment
    - Reputational exposure

    Does NOT:
    - Report news
    - Summarize headlines
    - Track trends for interest
    - Provide general updates
    - Curate reading material
    - Speculate on implications
    - Follow celebrity/influencer activity

    Does:
    - Detect constraint-changing events only
    - Filter signal from noise ruthlessly
    - Assess time horizon for action
    - Connect developments to specific operational impacts
    - Monitor regulatory bodies, medical boards, platform policies
    - Track AI governance developments
    """

    def __init__(self,
                 pubmed_interface: Optional[PubMedInterface] = None,
                 web_search_interface: Optional[WebSearchInterface] = None):
        """
        Initialize World Radar.

        Args:
            pubmed_interface: Interface for PubMed searches
            web_search_interface: Interface for web searches
        """
        super().__init__(name="World Radar")

        # External service interfaces
        self._pubmed = pubmed_interface or PubMedInterface()
        self._web_search = web_search_interface or WebSearchInterface()

        # Analysis components
        self._escalation_analyzer = EscalationAnalyzer()
        self._credibility_assessor = SourceCredibilityAssessor()
        self._time_horizon_assessor = TimeHorizonAssessor()
        self._constraint_analyzer = ConstraintImpactAnalyzer()
        self._signal_filter = SignalFilter()

        # Scan state
        self._last_scan: Optional[datetime] = None
        self._active_signals: List[WorldSignal] = []

    async def scan(self, request: WorldScanRequest) -> AgentResponse:
        """
        Execute a world scan based on request parameters.

        This is the main entry point for World Radar operations.

        Args:
            request: WorldScanRequest specifying scan parameters

        Returns:
            AgentResponse containing detected signals
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        signals = []
        errors = []
        warnings = []

        # Scan each requested domain
        for domain in request.domains:
            try:
                domain_signals = await self.scan_domain(
                    domain=domain,
                    geographic_focus=request.geographic_focus,
                    concerns=request.active_concerns
                )
                signals.extend(domain_signals)
            except Exception as e:
                errors.append(f"Error scanning {domain.value}: {str(e)}")

        # Update scan state
        self._last_scan = datetime.now()
        self._active_signals = signals

        # Apply heightened monitoring in elevated states
        monitoring_level = self.get_monitoring_level()
        if monitoring_level == "HEIGHTENED":
            warnings.append("Operating in heightened monitoring mode (YELLOW state)")
        elif monitoring_level == "CRITICAL":
            warnings.append("Operating in critical monitoring mode (RED state)")
            # In RED state, prioritize reputation-related signals
            signals = self._prioritize_reputation_signals(signals)

        # Generate response
        if not signals:
            # Silence protocol
            silence_signal = WorldSignal.silence_signal()
            return self.create_response(
                data={
                    "signals": [silence_signal.to_dict()],
                    "scan_summary": {
                        "domains_scanned": [d.value for d in request.domains],
                        "signals_detected": 0,
                        "monitoring_level": monitoring_level
                    },
                    "formatted_output": silence_signal.to_formatted_output()
                },
                errors=errors if errors else None,
                warnings=warnings if warnings else None
            )

        # Sort signals by action priority
        signals.sort(key=lambda s: s.action_required.priority_level, reverse=True)

        return self.create_response(
            data={
                "signals": [s.to_dict() for s in signals],
                "scan_summary": {
                    "domains_scanned": [d.value for d in request.domains],
                    "signals_detected": len(signals),
                    "urgent_signals": sum(1 for s in signals
                                         if s.action_required == ActionRequired.URGENT_REVIEW),
                    "monitoring_level": monitoring_level
                },
                "formatted_output": "\n\n".join(s.to_formatted_output() for s in signals)
            },
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )

    async def scan_domain(self, domain: Domain,
                         geographic_focus: List[str] = None,
                         concerns: List[str] = None) -> List[WorldSignal]:
        """
        Scan a specific domain for constraint-changing signals.

        Args:
            domain: Domain to scan
            geographic_focus: Geographic regions to focus on
            concerns: Specific topics to weight

        Returns:
            List of detected WorldSignal objects
        """
        raw_signals = await self._gather_raw_signals(domain, geographic_focus, concerns)

        processed_signals = []
        for raw in raw_signals:
            signal = self._process_raw_signal(raw, domain)
            if signal:
                processed_signals.append(signal)

        return processed_signals

    async def _gather_raw_signals(self, domain: Domain,
                                  geographic_focus: List[str],
                                  concerns: List[str]) -> List[Dict[str, Any]]:
        """Gather raw signals from external sources."""
        raw_signals = []

        # Use PubMed for clinical domain
        if domain == Domain.CLINICAL:
            query = self._pubmed.build_clinical_query(
                topics=concerns or ["medical practice", "clinical guidelines"],
                geographic_focus=geographic_focus
            )
            pubmed_results = await self._pubmed.search(
                query=query,
                max_results=20,
                date_range="last_30_days"
            )
            raw_signals.extend(pubmed_results)

        # Use WebSearch for all domains
        web_query = self._web_search.build_domain_query(
            domain=domain,
            concerns=concerns,
            geographic_focus=geographic_focus
        )

        authoritative_domains = self._web_search.get_authoritative_domains(domain)
        web_results = await self._web_search.search(
            query=web_query,
            domains=authoritative_domains,
            time_range="past_week",
            result_count=20
        )
        raw_signals.extend(web_results)

        return raw_signals

    def _process_raw_signal(self, raw: Dict[str, Any],
                           domain: Domain) -> Optional[WorldSignal]:
        """
        Process a raw signal through analysis pipeline.

        Args:
            raw: Raw signal data
            domain: Domain this signal belongs to

        Returns:
            WorldSignal if passes filter, None otherwise
        """
        # Analyze escalation criteria
        escalation_criteria = self._escalation_analyzer.analyze(raw)

        # If no criteria met, not a signal
        if not escalation_criteria:
            return None

        # Assess source credibility
        source_url = raw.get("url", raw.get("source", "unknown"))
        source_credibility = self._credibility_assessor.assess(source_url, domain)

        # Determine confidence
        confidence = self._calculate_confidence(escalation_criteria, source_credibility)

        # Apply signal filter
        if not self._signal_filter.is_signal(
            raw, escalation_criteria, source_credibility, confidence
        ):
            return None

        # Assess time horizon
        text = raw.get("title", "") + " " + raw.get("snippet", raw.get("abstract", ""))
        time_horizon = self._time_horizon_assessor.assess(text)

        # Analyze constraint impact
        constraint_impact = self._constraint_analyzer.analyze(
            raw, domain, escalation_criteria
        )

        # Determine action required
        action = self._determine_action(time_horizon, confidence, escalation_criteria)

        # Build event description
        event = raw.get("title", "")[:200]  # Concise description

        return WorldSignal(
            event=event,
            domain=domain,
            source=source_url,
            source_credibility=source_credibility,
            time_horizon=time_horizon,
            constraint_impact=constraint_impact,
            action_required=action,
            confidence=confidence,
            escalation_criteria_met=escalation_criteria,
            raw_data=raw
        )

    def _calculate_confidence(self, criteria: List[EscalationCriterion],
                             credibility: SourceCredibility) -> Confidence:
        """Calculate confidence level for a signal."""
        # Base confidence on source credibility
        if credibility == SourceCredibility.AUTHORITATIVE:
            base_confidence = Confidence.HIGH
        elif credibility == SourceCredibility.CREDIBLE:
            base_confidence = Confidence.MEDIUM
        else:
            base_confidence = Confidence.LOW

        # Boost if multiple criteria met
        if len(criteria) >= 3 and base_confidence != Confidence.HIGH:
            return Confidence.HIGH if base_confidence == Confidence.MEDIUM else Confidence.MEDIUM

        return base_confidence

    def _determine_action(self, horizon: TimeHorizon,
                         confidence: Confidence,
                         criteria: List[EscalationCriterion]) -> ActionRequired:
        """Determine required action level."""
        # Urgent if immediate horizon with high confidence
        if horizon == TimeHorizon.IMMEDIATE and confidence == Confidence.HIGH:
            return ActionRequired.URGENT_REVIEW

        # Review if near term or multiple criteria
        if horizon in [TimeHorizon.IMMEDIATE, TimeHorizon.NEAR]:
            if len(criteria) >= 2 or confidence == Confidence.HIGH:
                return ActionRequired.REVIEW

        # Monitor for everything else that passed filter
        if len(criteria) >= 1:
            return ActionRequired.MONITOR

        return ActionRequired.NONE

    def _prioritize_reputation_signals(self,
                                       signals: List[WorldSignal]) -> List[WorldSignal]:
        """Prioritize reputation-related signals for RED state."""
        reputation_signals = []
        other_signals = []

        for signal in signals:
            is_reputation = (
                signal.domain == Domain.REPUTATION or
                EscalationCriterion.REPUTATIONAL_EXPOSURE_INCREASE in signal.escalation_criteria_met
            )
            if is_reputation:
                reputation_signals.append(signal)
            else:
                other_signals.append(signal)

        # Return reputation signals first
        return reputation_signals + other_signals

    # =========================================================================
    # PUBLIC METHODS FOR SPECIFIC ANALYSES
    # =========================================================================

    def check_escalation_criteria(self,
                                  raw_signal: Dict[str, Any]) -> List[EscalationCriterion]:
        """
        Check which escalation criteria a raw signal meets.

        Args:
            raw_signal: Raw signal data

        Returns:
            List of met EscalationCriterion
        """
        return self._escalation_analyzer.analyze(raw_signal)

    def assess_source_credibility(self, source: str,
                                  domain: Domain) -> SourceCredibility:
        """
        Assess credibility of an information source.

        Args:
            source: Source URL or identifier
            domain: Domain context for assessment

        Returns:
            SourceCredibility assessment
        """
        return self._credibility_assessor.assess(source, domain)

    def assess_time_horizon(self, signal_text: str,
                           effective_date: Optional[datetime] = None) -> TimeHorizon:
        """
        Assess time horizon for action on a signal.

        Args:
            signal_text: Text content of the signal
            effective_date: Known effective date if available

        Returns:
            TimeHorizon assessment
        """
        return self._time_horizon_assessor.assess(signal_text, effective_date)

    def analyze_constraint_impact(self, signal_data: Dict[str, Any],
                                  domain: Domain,
                                  criteria: List[EscalationCriterion]) -> ConstraintImpact:
        """
        Analyze how a signal impacts operational constraints.

        Args:
            signal_data: Signal data
            domain: Domain of the signal
            criteria: Met escalation criteria

        Returns:
            ConstraintImpact analysis
        """
        return self._constraint_analyzer.analyze(signal_data, domain, criteria)

    def generate_signal(self, event: str,
                       domain: Domain,
                       source: str,
                       constraint_impact: str,
                       time_horizon: TimeHorizon = TimeHorizon.NEAR,
                       action: ActionRequired = ActionRequired.MONITOR,
                       confidence: Confidence = Confidence.MEDIUM) -> WorldSignal:
        """
        Generate a WorldSignal from provided parameters.

        Utility method for creating signals from external analysis.

        Args:
            event: Event description
            domain: Signal domain
            source: Source reference
            constraint_impact: Description of constraint impact
            time_horizon: Time horizon for action
            action: Required action level
            confidence: Confidence level

        Returns:
            Constructed WorldSignal
        """
        source_credibility = self._credibility_assessor.assess(source, domain)

        impact = ConstraintImpact(
            constraint_type=domain.value,
            current_state="nominal",
            projected_state=constraint_impact,
            impact_severity=self._severity_from_action(action),
            affected_operations=[],
            mitigation_options=[]
        )

        return WorldSignal(
            event=event,
            domain=domain,
            source=source,
            source_credibility=source_credibility,
            time_horizon=time_horizon,
            constraint_impact=impact,
            action_required=action,
            confidence=confidence
        )

    def _severity_from_action(self, action: ActionRequired) -> str:
        """Convert action required to severity string."""
        mapping = {
            ActionRequired.URGENT_REVIEW: "critical",
            ActionRequired.REVIEW: "major",
            ActionRequired.MONITOR: "moderate",
            ActionRequired.NONE: "minor"
        }
        return mapping.get(action, "minor")

    # =========================================================================
    # SYNCHRONOUS INTERFACE
    # =========================================================================

    def scan_sync(self, request: WorldScanRequest) -> AgentResponse:
        """
        Synchronous wrapper for scan operation.

        For use in non-async contexts.

        Args:
            request: WorldScanRequest specifying scan parameters

        Returns:
            AgentResponse containing detected signals
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.scan(request))

    def process_request(self, request_data: Dict[str, Any]) -> AgentResponse:
        """
        Process a request in standard Alfred format.

        Args:
            request_data: Dictionary with request parameters

        Returns:
            AgentResponse with scan results
        """
        request = WorldScanRequest.from_dict(request_data)
        return self.scan_sync(request)


# =============================================================================
# FACTORY AND CONVENIENCE FUNCTIONS
# =============================================================================

def create_world_radar(pubmed_client: Any = None,
                       web_search_client: Any = None) -> WorldRadar:
    """
    Factory function to create a configured WorldRadar instance.

    Args:
        pubmed_client: MCP client for PubMed tool
        web_search_client: Client for web search

    Returns:
        Configured WorldRadar instance
    """
    pubmed = PubMedInterface(pubmed_client)
    web_search = WebSearchInterface(web_search_client)

    return WorldRadar(
        pubmed_interface=pubmed,
        web_search_interface=web_search
    )


def parse_world_scan_request(request_text: str) -> WorldScanRequest:
    """
    Parse a WORLD_SCAN_REQUEST from text format.

    Args:
        request_text: Text in WORLD_SCAN_REQUEST format

    Returns:
        Parsed WorldScanRequest
    """
    lines = request_text.strip().split("\n")
    data = {}

    for line in lines:
        if line.startswith("- Domains:"):
            data["domains"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Geographic Focus:"):
            data["geographic_focus"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Time Since Last Scan:"):
            data["time_since_last_scan"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Active Concerns:"):
            data["active_concerns"] = line.split(":", 1)[1].strip()

    return WorldScanRequest.from_dict(data)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "WorldRadar",

    # Enums
    "Domain",
    "TimeHorizon",
    "ActionRequired",
    "Confidence",
    "SourceCredibility",
    "EscalationCriterion",

    # Data classes
    "WorldScanRequest",
    "WorldSignal",
    "ConstraintImpact",

    # Interfaces
    "PubMedInterface",
    "WebSearchInterface",
    "ExternalServiceInterface",

    # Analyzers
    "EscalationAnalyzer",
    "SourceCredibilityAssessor",
    "TimeHorizonAssessor",
    "ConstraintImpactAnalyzer",
    "SignalFilter",

    # Factory functions
    "create_world_radar",
    "parse_world_scan_request"
]
