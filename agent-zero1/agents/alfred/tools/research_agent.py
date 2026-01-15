# Research Agent (Evidence Scout) for Alfred
# Evidence retrieval and synthesis engine - provides facts, not framing

"""
RESEARCH AGENT (EVIDENCE SCOUT)

Role: Evidence retrieval and synthesis engine. Provides facts, not framing.
Summarizes what is known, flags what is uncertain, never suggests what to do
with the information.

INTEGRITY RULE: If evidence is weak, you MUST say so. Never present weak
evidence as strong.

Does NOT:
- Frame narratives
- Suggest posting angles
- Infer intent or motivation
- Provide opinion on controversies
- Recommend positions
- Cherry-pick supporting evidence
- Minimize contradictory findings
- Extrapolate beyond data

Does:
- Summarize clinical guidelines accurately
- Extract evidence with strength ratings
- Flag uncertainty and knowledge gaps explicitly
- Provide citations for all claims
- Note study quality and limitations
- Distinguish consensus from emerging findings
- Identify conflicting evidence

NOTE: Research Agent inherits from ContentAgent but has special permission
to operate in YELLOW state (evidence gathering continues).
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import re
from abc import ABC, abstractmethod

from . import ContentAgent, AgentResponse, AlfredState


# =============================================================================
# ENUMS
# =============================================================================

class EvidenceStrength(Enum):
    """Classification of evidence strength following modified GRADE approach."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_score(cls, score: float) -> "EvidenceStrength":
        """Convert a numeric score (0-100) to evidence strength."""
        if score >= 75:
            return cls.STRONG
        elif score >= 50:
            return cls.MODERATE
        elif score >= 25:
            return cls.WEAK
        else:
            return cls.INSUFFICIENT


class ResearchDepth(Enum):
    """Depth of research to conduct."""
    LIGHT = "light"       # Quick scan, key sources only
    MEDIUM = "medium"     # Standard review, balanced coverage
    DEEP = "deep"         # Comprehensive review, exhaustive search

    def __str__(self) -> str:
        return self.value

    @property
    def max_sources(self) -> int:
        """Maximum number of sources to include at each depth."""
        return {
            ResearchDepth.LIGHT: 5,
            ResearchDepth.MEDIUM: 15,
            ResearchDepth.DEEP: 50
        }[self]

    @property
    def time_budget_minutes(self) -> int:
        """Estimated time budget in minutes for each depth."""
        return {
            ResearchDepth.LIGHT: 5,
            ResearchDepth.MEDIUM: 15,
            ResearchDepth.DEEP: 60
        }[self]


class StudyType(Enum):
    """Classification of study types for evidence hierarchy."""
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "randomized_controlled_trial"
    COHORT = "cohort_study"
    CASE_CONTROL = "case_control_study"
    CROSS_SECTIONAL = "cross_sectional_study"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    EXPERT_OPINION = "expert_opinion"
    ANIMAL_STUDY = "animal_study"
    IN_VITRO = "in_vitro"
    GUIDELINE = "clinical_guideline"
    CONSENSUS = "consensus_statement"
    REVIEW = "narrative_review"
    UNKNOWN = "unknown"

    @property
    def evidence_rank(self) -> int:
        """
        Evidence hierarchy ranking (higher = stronger).
        Based on traditional evidence pyramids.
        """
        hierarchy = {
            StudyType.META_ANALYSIS: 10,
            StudyType.SYSTEMATIC_REVIEW: 9,
            StudyType.RCT: 8,
            StudyType.COHORT: 7,
            StudyType.CASE_CONTROL: 6,
            StudyType.CROSS_SECTIONAL: 5,
            StudyType.CASE_SERIES: 4,
            StudyType.CASE_REPORT: 3,
            StudyType.ANIMAL_STUDY: 2,
            StudyType.IN_VITRO: 1,
            StudyType.EXPERT_OPINION: 2,
            StudyType.GUIDELINE: 9,
            StudyType.CONSENSUS: 7,
            StudyType.REVIEW: 3,
            StudyType.UNKNOWN: 0
        }
        return hierarchy.get(self, 0)


class RecencyCategory(Enum):
    """Classification of how current the evidence is."""
    CURRENT = "current"           # Within 2 years
    RECENT = "recent"             # 2-5 years
    DATED = "dated"               # 5-10 years
    HISTORICAL = "historical"     # >10 years

    @classmethod
    def from_date(cls, pub_date: datetime) -> "RecencyCategory":
        """Classify recency based on publication date."""
        age_years = (datetime.now() - pub_date).days / 365
        if age_years <= 2:
            return cls.CURRENT
        elif age_years <= 5:
            return cls.RECENT
        elif age_years <= 10:
            return cls.DATED
        else:
            return cls.HISTORICAL


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Citation:
    """
    Full citation for a source.
    Follows academic citation standards.
    """
    id: str = ""
    authors: List[str] = field(default_factory=list)
    title: str = ""
    journal: str = ""
    year: int = 0
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    pmid: str = ""
    url: str = ""
    study_type: StudyType = StudyType.UNKNOWN
    publication_date: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            content = f"{self.title}{self.doi}{self.pmid}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]

    def format_ama(self) -> str:
        """Format citation in AMA (American Medical Association) style."""
        author_str = ""
        if self.authors:
            if len(self.authors) <= 6:
                author_str = ", ".join(self.authors)
            else:
                author_str = ", ".join(self.authors[:3]) + ", et al"

        parts = []
        if author_str:
            parts.append(author_str)
        if self.title:
            parts.append(self.title)
        if self.journal:
            journal_part = f"{self.journal}."
            if self.year:
                journal_part += f" {self.year}"
            if self.volume:
                journal_part += f";{self.volume}"
                if self.issue:
                    journal_part += f"({self.issue})"
                if self.pages:
                    journal_part += f":{self.pages}"
            parts.append(journal_part)
        if self.doi:
            parts.append(f"doi:{self.doi}")

        return " ".join(parts)

    def format_vancouver(self) -> str:
        """Format citation in Vancouver style."""
        parts = []

        # Authors
        if self.authors:
            author_list = self.authors[:6]
            if len(self.authors) > 6:
                author_list.append("et al")
            parts.append(", ".join(author_list) + ".")

        # Title
        if self.title:
            parts.append(f"{self.title}.")

        # Journal info
        if self.journal:
            journal_info = self.journal
            if self.year:
                journal_info += f" {self.year}"
            if self.volume:
                journal_info += f";{self.volume}"
            if self.issue:
                journal_info += f"({self.issue})"
            if self.pages:
                journal_info += f":{self.pages}"
            parts.append(journal_info + ".")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "authors": self.authors,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "pmid": self.pmid,
            "url": self.url,
            "study_type": self.study_type.value,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None
        }

    @property
    def recency(self) -> RecencyCategory:
        if self.publication_date:
            return RecencyCategory.from_date(self.publication_date)
        elif self.year:
            pub_date = datetime(self.year, 6, 1)  # Assume mid-year
            return RecencyCategory.from_date(pub_date)
        return RecencyCategory.HISTORICAL


@dataclass
class StudyLimitation:
    """Documented limitation of a study."""
    description: str
    severity: str = "moderate"  # minor, moderate, major
    impact_on_conclusions: str = ""


@dataclass
class KeyFinding:
    """
    A key finding extracted from evidence.
    Must include citation and strength assessment.
    """
    id: str = ""
    statement: str = ""
    citation_ids: List[str] = field(default_factory=list)
    supporting_count: int = 0
    contradicting_count: int = 0
    strength: EvidenceStrength = EvidenceStrength.INSUFFICIENT
    confidence_score: float = 0.0
    study_types_supporting: List[StudyType] = field(default_factory=list)
    limitations: List[StudyLimitation] = field(default_factory=list)
    is_consensus: bool = False
    is_emerging: bool = False
    caveats: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            content = f"{self.statement}{datetime.now().isoformat()}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "citation_ids": self.citation_ids,
            "supporting_count": self.supporting_count,
            "contradicting_count": self.contradicting_count,
            "strength": self.strength.value,
            "confidence_score": self.confidence_score,
            "study_types_supporting": [st.value for st in self.study_types_supporting],
            "limitations": [
                {"description": lim.description, "severity": lim.severity,
                 "impact": lim.impact_on_conclusions}
                for lim in self.limitations
            ],
            "is_consensus": self.is_consensus,
            "is_emerging": self.is_emerging,
            "caveats": self.caveats
        }


@dataclass
class Contradiction:
    """
    Represents conflicting evidence between findings.
    """
    id: str = ""
    finding_a: str = ""
    finding_b: str = ""
    nature_of_conflict: str = ""
    potential_explanations: List[str] = field(default_factory=list)
    citation_ids_a: List[str] = field(default_factory=list)
    citation_ids_b: List[str] = field(default_factory=list)
    resolution_status: str = "unresolved"  # unresolved, likely_resolved, resolved

    def __post_init__(self):
        if not self.id:
            content = f"{self.finding_a}{self.finding_b}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "finding_a": self.finding_a,
            "finding_b": self.finding_b,
            "nature_of_conflict": self.nature_of_conflict,
            "potential_explanations": self.potential_explanations,
            "citation_ids_a": self.citation_ids_a,
            "citation_ids_b": self.citation_ids_b,
            "resolution_status": self.resolution_status
        }


@dataclass
class UncertaintyGap:
    """
    Represents a gap or uncertainty in the evidence.
    """
    id: str = ""
    description: str = ""
    gap_type: str = ""  # data_gap, methodology_gap, population_gap, outcome_gap
    importance: str = "moderate"  # low, moderate, high, critical
    studies_needed: str = ""
    why_uncertain: str = ""

    def __post_init__(self):
        if not self.id:
            content = f"{self.description}{datetime.now().isoformat()}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "gap_type": self.gap_type,
            "importance": self.importance,
            "studies_needed": self.studies_needed,
            "why_uncertain": self.why_uncertain
        }


@dataclass
class EvidenceRequest:
    """
    Structured request for evidence research.
    Matches INPUT FORMAT from specification.
    """
    topic: str
    depth: ResearchDepth = ResearchDepth.MEDIUM
    intended_output: str = ""
    time_constraint: Optional[str] = None
    specific_questions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRequest":
        depth = data.get("depth", "medium")
        if isinstance(depth, str):
            depth = ResearchDepth(depth.lower())

        return cls(
            topic=data.get("topic", ""),
            depth=depth,
            intended_output=data.get("intended_output", ""),
            time_constraint=data.get("time_constraint"),
            specific_questions=data.get("specific_questions", [])
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "depth": self.depth.value,
            "intended_output": self.intended_output,
            "time_constraint": self.time_constraint,
            "specific_questions": self.specific_questions
        }


@dataclass
class EvidenceBrief:
    """
    Complete evidence brief output.
    Matches OUTPUT FORMAT from specification.
    """
    topic: str
    depth_completed: ResearchDepth
    key_findings: List[KeyFinding]
    evidence_strength: EvidenceStrength
    strength_rationale: str
    uncertainties: List[UncertaintyGap]
    contradictions: List[Contradiction]
    citations: List[Citation]
    recency_note: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    search_terms_used: List[str] = field(default_factory=list)
    sources_reviewed: int = 0
    sources_included: int = 0
    time_taken_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "depth_completed": self.depth_completed.value,
            "key_findings": [f.to_dict() for f in self.key_findings],
            "evidence_strength": self.evidence_strength.value,
            "strength_rationale": self.strength_rationale,
            "uncertainties": [u.to_dict() for u in self.uncertainties],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "citations": [c.to_dict() for c in self.citations],
            "recency_note": self.recency_note,
            "generated_at": self.generated_at,
            "search_terms_used": self.search_terms_used,
            "sources_reviewed": self.sources_reviewed,
            "sources_included": self.sources_included,
            "time_taken_seconds": self.time_taken_seconds,
            "warnings": self.warnings
        }

    def format_output(self) -> str:
        """Format as structured text matching specification."""
        lines = [
            "EVIDENCE_BRIEF",
            f"- Topic: {self.topic}",
            f"- Depth Completed: {self.depth_completed.value}",
            "",
            "- Key Findings:"
        ]

        for i, finding in enumerate(self.key_findings, 1):
            lines.append(f"  {i}. {finding.statement} [{', '.join(finding.citation_ids)}]")
            if finding.caveats:
                for caveat in finding.caveats:
                    lines.append(f"     Caveat: {caveat}")

        lines.extend([
            "",
            f"- Evidence Strength: {self.evidence_strength.value}",
            f"- Strength Rationale: {self.strength_rationale}",
            "",
            "- Uncertainty / Gaps:"
        ])

        for gap in self.uncertainties:
            lines.append(f"  - {gap.description}")
            if gap.studies_needed:
                lines.append(f"    Studies needed: {gap.studies_needed}")

        lines.extend([
            "",
            "- Contradictory Evidence:"
        ])

        if self.contradictions:
            for contradiction in self.contradictions:
                lines.append(f"  - {contradiction.finding_a} vs {contradiction.finding_b}")
                lines.append(f"    Nature: {contradiction.nature_of_conflict}")
        else:
            lines.append("  - None identified")

        lines.extend([
            "",
            "- Citations:"
        ])

        for i, citation in enumerate(self.citations, 1):
            lines.append(f"  {i}. {citation.format_vancouver()}")

        lines.extend([
            "",
            f"- Recency Note: {self.recency_note}"
        ])

        if self.warnings:
            lines.extend([
                "",
                "- Warnings:"
            ])
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


# =============================================================================
# RESEARCH PROVIDER INTERFACES
# =============================================================================

class ResearchProvider(ABC):
    """
    Abstract base class for research data providers.
    Enables integration with various APIs and tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Type of sources this provider returns."""
        pass

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Execute search and return raw results."""
        pass

    @abstractmethod
    def parse_citation(self, raw_result: Dict[str, Any]) -> Citation:
        """Parse raw result into a Citation object."""
        pass

    @abstractmethod
    def extract_finding(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """Extract key finding from result if available."""
        pass


class PubMedProvider(ResearchProvider):
    """
    Interface for mcp-simple-pubmed integration.
    Provides access to PubMed/MEDLINE database.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self._tool = None  # Will hold MCP tool reference

    @property
    def name(self) -> str:
        return "PubMed"

    @property
    def source_type(self) -> str:
        return "biomedical_literature"

    def configure_mcp(self, mcp_tool: Any) -> None:
        """Configure with mcp-simple-pubmed tool instance."""
        self._tool = mcp_tool

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search PubMed using mcp-simple-pubmed.
        Falls back to placeholder if MCP not configured.
        """
        if self._tool:
            # Use MCP tool if available
            try:
                results = await self._tool.search(
                    query=query,
                    max_results=max_results
                )
                return results
            except Exception as e:
                return [{
                    "error": str(e),
                    "query": query,
                    "provider": "pubmed"
                }]

        # Placeholder for when MCP is not configured
        return [{
            "pmid": "placeholder",
            "title": f"[PubMed results for: {query}]",
            "abstract": "MCP PubMed provider not configured",
            "authors": [],
            "journal": "",
            "year": datetime.now().year,
            "status": "mcp_not_configured"
        }]

    def parse_citation(self, raw_result: Dict[str, Any]) -> Citation:
        """Parse PubMed result into Citation object."""
        pub_date = None
        if raw_result.get("pubdate"):
            try:
                pub_date = datetime.strptime(raw_result["pubdate"], "%Y %b %d")
            except:
                try:
                    pub_date = datetime.strptime(raw_result["pubdate"], "%Y")
                except:
                    pass

        return Citation(
            authors=raw_result.get("authors", []),
            title=raw_result.get("title", ""),
            journal=raw_result.get("journal", ""),
            year=raw_result.get("year", 0),
            volume=raw_result.get("volume", ""),
            issue=raw_result.get("issue", ""),
            pages=raw_result.get("pages", ""),
            doi=raw_result.get("doi", ""),
            pmid=raw_result.get("pmid", ""),
            study_type=self._infer_study_type(raw_result),
            publication_date=pub_date
        )

    def _infer_study_type(self, result: Dict[str, Any]) -> StudyType:
        """Infer study type from PubMed metadata."""
        pub_types = result.get("publication_types", [])
        title_lower = result.get("title", "").lower()

        # Check publication types
        for pt in pub_types:
            pt_lower = pt.lower()
            if "meta-analysis" in pt_lower:
                return StudyType.META_ANALYSIS
            elif "systematic review" in pt_lower:
                return StudyType.SYSTEMATIC_REVIEW
            elif "randomized controlled trial" in pt_lower:
                return StudyType.RCT
            elif "guideline" in pt_lower:
                return StudyType.GUIDELINE
            elif "review" in pt_lower:
                return StudyType.REVIEW

        # Infer from title
        if "meta-analysis" in title_lower:
            return StudyType.META_ANALYSIS
        elif "systematic review" in title_lower:
            return StudyType.SYSTEMATIC_REVIEW
        elif "randomized" in title_lower or "rct" in title_lower:
            return StudyType.RCT
        elif "cohort" in title_lower:
            return StudyType.COHORT
        elif "case-control" in title_lower:
            return StudyType.CASE_CONTROL
        elif "cross-sectional" in title_lower:
            return StudyType.CROSS_SECTIONAL
        elif "case report" in title_lower:
            return StudyType.CASE_REPORT
        elif "case series" in title_lower:
            return StudyType.CASE_SERIES
        elif "guideline" in title_lower:
            return StudyType.GUIDELINE
        elif "consensus" in title_lower:
            return StudyType.CONSENSUS

        return StudyType.UNKNOWN

    def extract_finding(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """Extract conclusion/finding from abstract if available."""
        abstract = raw_result.get("abstract", "")
        if not abstract:
            return None

        # Look for conclusion section
        conclusion_markers = [
            "conclusion:", "conclusions:", "in conclusion",
            "we conclude", "our findings suggest", "results indicate"
        ]
        abstract_lower = abstract.lower()

        for marker in conclusion_markers:
            if marker in abstract_lower:
                idx = abstract_lower.find(marker)
                conclusion = abstract[idx:].split(".")[0:2]
                return ". ".join(conclusion) + "."

        # Return last sentence if no explicit conclusion
        sentences = abstract.split(".")
        if len(sentences) >= 2:
            return sentences[-2].strip() + "."

        return None


class PerplexityProvider(ResearchProvider):
    """
    Interface for Perplexity API integration.
    Provides AI-powered research synthesis.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._base_url = "https://api.perplexity.ai"
        self._configured = api_key is not None

    @property
    def name(self) -> str:
        return "Perplexity"

    @property
    def source_type(self) -> str:
        return "ai_synthesis"

    def configure(self, api_key: str) -> None:
        """Configure with API key."""
        self._api_key = api_key
        self._configured = True

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query Perplexity API for research synthesis.
        Returns structured results with citations.
        """
        if not self._configured:
            return [{
                "query": query,
                "response": "[Perplexity API not configured]",
                "citations": [],
                "status": "api_not_configured"
            }]

        # Placeholder for actual API implementation
        # In production, this would make actual API calls
        return [{
            "query": query,
            "response": f"[Perplexity results for: {query}]",
            "citations": [],
            "status": "placeholder"
        }]

    def parse_citation(self, raw_result: Dict[str, Any]) -> Citation:
        """Parse Perplexity citation into Citation object."""
        return Citation(
            title=raw_result.get("title", ""),
            url=raw_result.get("url", ""),
            year=raw_result.get("year", datetime.now().year)
        )

    def extract_finding(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """Extract key finding from Perplexity response."""
        return raw_result.get("response")


class LlamaIndexProvider(ResearchProvider):
    """
    Interface for LlamaIndex document processing.
    Handles local document retrieval and synthesis.
    """

    def __init__(self, index_path: Optional[str] = None):
        self._index_path = index_path
        self._index = None  # Will hold LlamaIndex instance
        self._configured = False

    @property
    def name(self) -> str:
        return "LlamaIndex"

    @property
    def source_type(self) -> str:
        return "document_store"

    def configure(self, index: Any) -> None:
        """Configure with LlamaIndex instance."""
        self._index = index
        self._configured = True

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query LlamaIndex for relevant documents.
        """
        if not self._configured or self._index is None:
            return [{
                "query": query,
                "response": "[LlamaIndex not configured]",
                "source_nodes": [],
                "status": "index_not_configured"
            }]

        # Placeholder for actual LlamaIndex query
        try:
            # In production:
            # response = self._index.query(query)
            # return self._parse_response(response, max_results)
            return [{
                "query": query,
                "response": f"[LlamaIndex results for: {query}]",
                "source_nodes": [],
                "status": "placeholder"
            }]
        except Exception as e:
            return [{
                "error": str(e),
                "query": query,
                "status": "error"
            }]

    def parse_citation(self, raw_result: Dict[str, Any]) -> Citation:
        """Parse LlamaIndex source node into Citation object."""
        metadata = raw_result.get("metadata", {})
        return Citation(
            title=metadata.get("title", raw_result.get("filename", "")),
            authors=metadata.get("authors", []),
            url=metadata.get("source", ""),
            year=metadata.get("year", 0)
        )

    def extract_finding(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """Extract text from source node."""
        return raw_result.get("text", raw_result.get("response"))


# =============================================================================
# MAIN RESEARCH AGENT CLASS
# =============================================================================

class ResearchAgent(ContentAgent):
    """
    Research Agent (Evidence Scout) for Alfred.

    Evidence retrieval and synthesis engine that provides facts, not framing.
    Summarizes what is known, flags what is uncertain, never suggests what
    to do with the information.

    SPECIAL PERMISSION: Unlike other ContentAgents, the Research Agent can
    operate in YELLOW state because evidence gathering supports crisis
    assessment and response planning.

    INTEGRITY RULE: If evidence is weak, you MUST say so. Never present
    weak evidence as strong.
    """

    def __init__(self):
        super().__init__("Research Agent")

        # Research providers
        self._pubmed = PubMedProvider()
        self._perplexity = PerplexityProvider()
        self._llamaindex = LlamaIndexProvider()

        # Internal state
        self._current_request: Optional[EvidenceRequest] = None
        self._findings_cache: Dict[str, List[KeyFinding]] = {}
        self._citations_index: Dict[str, Citation] = {}

        # Configuration
        self._min_sources_for_strong = 5
        self._min_sources_for_moderate = 2
        self._contradiction_threshold = 0.3  # 30% contradicting = flag

    # -------------------------------------------------------------------------
    # STATE PERMISSION OVERRIDE
    # -------------------------------------------------------------------------

    def check_state_permission(self) -> Tuple[bool, str]:
        """
        Override base ContentAgent state check.
        Research Agent CAN operate in YELLOW state (evidence gathering continues).
        Only blocked in RED state.
        """
        if self.alfred_state == AlfredState.RED:
            return False, "Research paused in RED state - focus on crisis response"

        if self.alfred_state == AlfredState.YELLOW:
            return True, "Research permitted in YELLOW state (evidence gathering continues)"

        return True, "Operation permitted"

    # -------------------------------------------------------------------------
    # PROVIDER CONFIGURATION
    # -------------------------------------------------------------------------

    def configure_pubmed(self, mcp_tool: Any = None, api_key: str = None) -> None:
        """Configure PubMed provider with MCP tool or API key."""
        if mcp_tool:
            self._pubmed.configure_mcp(mcp_tool)
        if api_key:
            self._pubmed._api_key = api_key

    def configure_perplexity(self, api_key: str) -> None:
        """Configure Perplexity provider with API key."""
        self._perplexity.configure(api_key)

    def configure_llamaindex(self, index: Any) -> None:
        """Configure LlamaIndex provider with index instance."""
        self._llamaindex.configure(index)

    # -------------------------------------------------------------------------
    # CORE RESEARCH METHODS
    # -------------------------------------------------------------------------

    async def search_evidence(
        self,
        request: EvidenceRequest,
        providers: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for evidence across configured providers.

        Args:
            request: The evidence request parameters
            providers: List of provider names to use (default: all configured)

        Returns:
            Dictionary mapping provider names to their results
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return {"error": reason}

        self._current_request = request
        results = {}

        # Generate search terms
        search_terms = self._generate_search_terms(request)

        # Determine max results per provider based on depth
        max_per_provider = request.depth.max_sources // 3 + 1

        # Select providers
        provider_map = {
            "pubmed": self._pubmed,
            "perplexity": self._perplexity,
            "llamaindex": self._llamaindex
        }

        if providers:
            active_providers = {k: v for k, v in provider_map.items()
                             if k in [p.lower() for p in providers]}
        else:
            active_providers = provider_map

        # Search each provider
        for name, provider in active_providers.items():
            provider_results = []
            for term in search_terms:
                try:
                    term_results = await provider.search(term, max_per_provider)
                    provider_results.extend(term_results)
                except Exception as e:
                    provider_results.append({
                        "error": str(e),
                        "search_term": term,
                        "provider": name
                    })
            results[name] = provider_results

        return results

    def _generate_search_terms(self, request: EvidenceRequest) -> List[str]:
        """Generate search terms from request."""
        terms = [request.topic]

        # Add specific questions as search terms
        terms.extend(request.specific_questions)

        # Generate variants based on depth
        if request.depth in [ResearchDepth.MEDIUM, ResearchDepth.DEEP]:
            # Add synonym expansions
            words = request.topic.split()
            if len(words) > 1:
                terms.append(" AND ".join(words))

        if request.depth == ResearchDepth.DEEP:
            # Add clinical focus terms
            clinical_suffixes = [
                "systematic review",
                "meta-analysis",
                "clinical guidelines",
                "randomized controlled trial"
            ]
            for suffix in clinical_suffixes:
                terms.append(f"{request.topic} {suffix}")

        return list(set(terms))  # Deduplicate

    def assess_strength(
        self,
        findings: List[KeyFinding],
        citations: List[Citation]
    ) -> Tuple[EvidenceStrength, str]:
        """
        Assess overall evidence strength with rationale.

        INTEGRITY RULE: If evidence is weak, we MUST say so.
        Never present weak evidence as strong.

        Args:
            findings: List of key findings extracted
            citations: List of citations supporting the findings

        Returns:
            Tuple of (strength rating, rationale explaining the rating)
        """
        if not findings or not citations:
            return (
                EvidenceStrength.INSUFFICIENT,
                "No evidence or citations available to assess"
            )

        # Calculate component scores
        scores = {}
        rationale_parts = []

        # 1. Source count score
        source_count = len(citations)
        if source_count >= self._min_sources_for_strong:
            scores["source_count"] = 25
            rationale_parts.append(f"{source_count} sources reviewed (adequate)")
        elif source_count >= self._min_sources_for_moderate:
            scores["source_count"] = 15
            rationale_parts.append(f"Only {source_count} sources (limited)")
        else:
            scores["source_count"] = 5
            rationale_parts.append(f"Only {source_count} source(s) (very limited)")

        # 2. Study quality score
        study_types = [c.study_type for c in citations]
        avg_rank = sum(st.evidence_rank for st in study_types) / len(study_types)

        if avg_rank >= 7:
            scores["study_quality"] = 25
            rationale_parts.append("High-quality study designs (RCTs, systematic reviews)")
        elif avg_rank >= 5:
            scores["study_quality"] = 15
            rationale_parts.append("Moderate study quality (observational studies)")
        else:
            scores["study_quality"] = 5
            rationale_parts.append("Lower-quality evidence (case reports, expert opinion)")

        # 3. Consistency score
        total_supporting = sum(f.supporting_count for f in findings)
        total_contradicting = sum(f.contradicting_count for f in findings)

        if total_supporting > 0:
            contradiction_rate = total_contradicting / (total_supporting + total_contradicting)
        else:
            contradiction_rate = 0.5  # Assume moderate inconsistency if no data

        if contradiction_rate < 0.1:
            scores["consistency"] = 25
            rationale_parts.append("Highly consistent findings")
        elif contradiction_rate < self._contradiction_threshold:
            scores["consistency"] = 15
            rationale_parts.append("Generally consistent with some variation")
        else:
            scores["consistency"] = 5
            rationale_parts.append(f"Significant contradictions ({int(contradiction_rate*100)}% conflicting)")

        # 4. Recency score
        recent_count = sum(1 for c in citations
                         if c.recency in [RecencyCategory.CURRENT, RecencyCategory.RECENT])
        recency_rate = recent_count / len(citations) if citations else 0

        if recency_rate >= 0.7:
            scores["recency"] = 25
            rationale_parts.append("Evidence is current (mostly <5 years)")
        elif recency_rate >= 0.4:
            scores["recency"] = 15
            rationale_parts.append("Mixed recency (some dated sources)")
        else:
            scores["recency"] = 5
            rationale_parts.append("Evidence may be outdated (mostly >5 years)")

        # Calculate total score
        total_score = sum(scores.values())
        strength = EvidenceStrength.from_score(total_score)

        # Build rationale
        rationale = f"Evidence rated as {strength.value.upper()}. " + "; ".join(rationale_parts)

        # Add explicit weakness statement if applicable (INTEGRITY RULE)
        if strength in [EvidenceStrength.WEAK, EvidenceStrength.INSUFFICIENT]:
            rationale += (
                f". IMPORTANT: This evidence is {strength.value} and should be "
                "interpreted with significant caution. Additional research needed."
            )

        return strength, rationale

    def identify_gaps(
        self,
        request: EvidenceRequest,
        findings: List[KeyFinding],
        citations: List[Citation]
    ) -> List[UncertaintyGap]:
        """
        Identify uncertainties and knowledge gaps in the evidence.

        Args:
            request: Original evidence request
            findings: Extracted findings
            citations: Supporting citations

        Returns:
            List of identified uncertainty gaps
        """
        gaps = []

        # Check for questions not adequately answered
        for question in request.specific_questions:
            question_answered = False
            for finding in findings:
                if self._question_matches_finding(question, finding):
                    question_answered = True
                    break

            if not question_answered:
                gaps.append(UncertaintyGap(
                    description=f"Specific question not fully addressed: {question}",
                    gap_type="data_gap",
                    importance="high",
                    studies_needed="Research directly addressing this question"
                ))

        # Check study population limitations
        populations_covered = set()
        for citation in citations:
            # Would normally extract from abstract/metadata
            populations_covered.add("general")  # Placeholder

        if len(populations_covered) == 1:
            gaps.append(UncertaintyGap(
                description="Limited population diversity in available studies",
                gap_type="population_gap",
                importance="moderate",
                studies_needed="Studies in diverse populations"
            ))

        # Check for missing outcome types
        outcome_types_seen = set()
        for finding in findings:
            # Categorize findings by outcome type
            if any(word in finding.statement.lower()
                   for word in ["mortality", "death", "survival"]):
                outcome_types_seen.add("mortality")
            if any(word in finding.statement.lower()
                   for word in ["quality of life", "qol", "well-being"]):
                outcome_types_seen.add("quality_of_life")
            if any(word in finding.statement.lower()
                   for word in ["side effect", "adverse", "safety"]):
                outcome_types_seen.add("safety")

        missing_outcomes = {"mortality", "quality_of_life", "safety"} - outcome_types_seen
        if missing_outcomes:
            gaps.append(UncertaintyGap(
                description=f"Missing outcome data: {', '.join(missing_outcomes)}",
                gap_type="outcome_gap",
                importance="moderate",
                studies_needed=f"Studies reporting {', '.join(missing_outcomes)} outcomes"
            ))

        # Check recency of evidence
        dated_citations = sum(1 for c in citations
                            if c.recency in [RecencyCategory.DATED, RecencyCategory.HISTORICAL])
        if citations and dated_citations / len(citations) > 0.5:
            gaps.append(UncertaintyGap(
                description="Much of the evidence is dated and may not reflect current practice",
                gap_type="methodology_gap",
                importance="high",
                why_uncertain="Medical knowledge evolves; older studies may use outdated methods",
                studies_needed="Updated research with current methodologies"
            ))

        # Check for low-powered studies
        underpowered_count = sum(
            1 for f in findings
            if f.supporting_count < 3 and not f.is_consensus
        )
        if findings and underpowered_count / len(findings) > 0.5:
            gaps.append(UncertaintyGap(
                description="Many findings supported by limited number of studies",
                gap_type="data_gap",
                importance="high",
                why_uncertain="Single or few studies prone to random error and bias",
                studies_needed="Replication studies and larger trials"
            ))

        return gaps

    def _question_matches_finding(self, question: str, finding: KeyFinding) -> bool:
        """Check if a finding addresses a specific question."""
        # Simple keyword matching - could be enhanced with NLP
        question_words = set(question.lower().split())
        finding_words = set(finding.statement.lower().split())

        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                     "does", "do", "can", "will", "should", "for", "of", "in", "to"}
        question_words -= stop_words
        finding_words -= stop_words

        # Check overlap
        overlap = question_words & finding_words
        return len(overlap) >= 2 or len(overlap) / max(len(question_words), 1) > 0.5

    def detect_contradictions(
        self,
        findings: List[KeyFinding]
    ) -> List[Contradiction]:
        """
        Detect contradictory evidence between findings.

        Args:
            findings: List of extracted findings to compare

        Returns:
            List of identified contradictions
        """
        contradictions = []

        # Compare each pair of findings
        for i, finding_a in enumerate(findings):
            for finding_b in findings[i+1:]:
                # Skip if from same source
                if set(finding_a.citation_ids) == set(finding_b.citation_ids):
                    continue

                # Check for semantic contradiction
                contradiction = self._check_contradiction(finding_a, finding_b)
                if contradiction:
                    contradictions.append(contradiction)

        return contradictions

    def _check_contradiction(
        self,
        finding_a: KeyFinding,
        finding_b: KeyFinding
    ) -> Optional[Contradiction]:
        """
        Check if two findings contradict each other.
        Uses heuristic detection - could be enhanced with NLP.
        """
        text_a = finding_a.statement.lower()
        text_b = finding_b.statement.lower()

        # Look for explicit contradiction markers
        negation_pairs = [
            ("increased", "decreased"),
            ("higher", "lower"),
            ("more", "less"),
            ("positive", "negative"),
            ("beneficial", "harmful"),
            ("effective", "ineffective"),
            ("significant", "no significant"),
            ("associated with", "not associated with"),
            ("increases", "decreases"),
            ("improves", "worsens"),
        ]

        for pos, neg in negation_pairs:
            # Check if one finding has positive and other has negative
            a_has_pos = pos in text_a
            a_has_neg = neg in text_a
            b_has_pos = pos in text_b
            b_has_neg = neg in text_b

            # They contradict if one is positive and other is negative
            # about similar topics
            if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                # Check if they're about the same topic (share key nouns)
                if self._findings_share_topic(finding_a, finding_b):
                    return Contradiction(
                        finding_a=finding_a.statement,
                        finding_b=finding_b.statement,
                        nature_of_conflict=f"Opposing conclusions about direction of effect ({pos} vs {neg})",
                        potential_explanations=[
                            "Different study populations",
                            "Different outcome definitions",
                            "Different time periods",
                            "Statistical variation"
                        ],
                        citation_ids_a=finding_a.citation_ids,
                        citation_ids_b=finding_b.citation_ids
                    )

        return None

    def _findings_share_topic(self, finding_a: KeyFinding, finding_b: KeyFinding) -> bool:
        """Check if two findings are about the same topic."""
        # Extract key nouns (simplified - would use NLP in production)
        def extract_nouns(text: str) -> Set[str]:
            words = text.lower().split()
            # Simple heuristic: longer words more likely to be meaningful nouns
            return {w for w in words if len(w) > 5 and w.isalpha()}

        nouns_a = extract_nouns(finding_a.statement)
        nouns_b = extract_nouns(finding_b.statement)

        # Significant overlap suggests same topic
        if nouns_a and nouns_b:
            overlap = len(nouns_a & nouns_b)
            return overlap >= 2 or overlap / min(len(nouns_a), len(nouns_b)) > 0.4

        return False

    def format_citations(
        self,
        citations: List[Citation],
        style: str = "vancouver"
    ) -> List[str]:
        """
        Format citations in specified style.

        Args:
            citations: List of citations to format
            style: Citation style (vancouver, ama)

        Returns:
            List of formatted citation strings
        """
        formatted = []

        for i, citation in enumerate(citations, 1):
            if style.lower() == "ama":
                formatted.append(f"{i}. {citation.format_ama()}")
            else:  # Default to Vancouver
                formatted.append(f"{i}. {citation.format_vancouver()}")

        return formatted

    def _generate_recency_note(self, citations: List[Citation]) -> str:
        """Generate recency assessment note for evidence brief."""
        if not citations:
            return "No citations to assess recency"

        recency_counts = {}
        for citation in citations:
            cat = citation.recency.value
            recency_counts[cat] = recency_counts.get(cat, 0) + 1

        total = len(citations)
        parts = []

        if recency_counts.get("current", 0) > 0:
            pct = int(recency_counts["current"] / total * 100)
            parts.append(f"{pct}% current (<2 years)")

        if recency_counts.get("recent", 0) > 0:
            pct = int(recency_counts["recent"] / total * 100)
            parts.append(f"{pct}% recent (2-5 years)")

        if recency_counts.get("dated", 0) + recency_counts.get("historical", 0) > 0:
            dated_count = recency_counts.get("dated", 0) + recency_counts.get("historical", 0)
            pct = int(dated_count / total * 100)
            parts.append(f"{pct}% older (>5 years)")

        # Get year range
        years = [c.year for c in citations if c.year > 0]
        if years:
            year_range = f"Publication years: {min(years)}-{max(years)}"
            parts.append(year_range)

        # Add warning if mostly dated
        dated_pct = (recency_counts.get("dated", 0) + recency_counts.get("historical", 0)) / total
        if dated_pct > 0.5:
            parts.append("NOTE: Significant portion of evidence may be outdated")

        return ". ".join(parts)

    # -------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------------------

    async def generate_brief(
        self,
        request: EvidenceRequest
    ) -> AgentResponse:
        """
        Generate a complete evidence brief.
        Main entry point for Research Agent.

        Args:
            request: The evidence request to process

        Returns:
            AgentResponse containing EvidenceBrief or error
        """
        start_time = datetime.now()
        warnings = []

        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        # Add warning if in YELLOW state
        if self.alfred_state == AlfredState.YELLOW:
            warnings.append(
                "Operating in YELLOW state - evidence gathering continues but "
                "content generation from this evidence is restricted"
            )

        try:
            # Step 1: Search for evidence
            search_results = await self.search_evidence(request)

            if "error" in search_results:
                return self.create_response(
                    data={"error": search_results["error"]},
                    success=False,
                    errors=[search_results["error"]]
                )

            # Step 2: Parse citations from results
            citations = []
            for provider_name, results in search_results.items():
                provider = {
                    "pubmed": self._pubmed,
                    "perplexity": self._perplexity,
                    "llamaindex": self._llamaindex
                }.get(provider_name)

                if provider:
                    for result in results:
                        if "error" not in result and result.get("status") != "mcp_not_configured":
                            try:
                                citation = provider.parse_citation(result)
                                if citation.title:  # Only add if has content
                                    citations.append(citation)
                                    self._citations_index[citation.id] = citation
                            except Exception:
                                continue

            # Step 3: Extract findings
            findings = await self._extract_findings(search_results, citations)

            # Step 4: Assess evidence strength
            strength, strength_rationale = self.assess_strength(findings, citations)

            # Step 5: Identify gaps
            uncertainties = self.identify_gaps(request, findings, citations)

            # Step 6: Detect contradictions
            contradictions = self.detect_contradictions(findings)

            # Add warning if significant contradictions
            if contradictions:
                warnings.append(
                    f"Identified {len(contradictions)} contradictions in the evidence - "
                    "interpret conclusions with caution"
                )

            # Step 7: Generate recency note
            recency_note = self._generate_recency_note(citations)

            # Calculate time taken
            time_taken = (datetime.now() - start_time).total_seconds()

            # Build evidence brief
            brief = EvidenceBrief(
                topic=request.topic,
                depth_completed=request.depth,
                key_findings=findings,
                evidence_strength=strength,
                strength_rationale=strength_rationale,
                uncertainties=uncertainties,
                contradictions=contradictions,
                citations=citations,
                recency_note=recency_note,
                search_terms_used=self._generate_search_terms(request),
                sources_reviewed=sum(len(r) for r in search_results.values()),
                sources_included=len(citations),
                time_taken_seconds=time_taken,
                warnings=warnings
            )

            return self.create_response(
                data={
                    "brief": brief.to_dict(),
                    "formatted_output": brief.format_output()
                },
                success=True,
                warnings=warnings
            )

        except Exception as e:
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Evidence brief generation failed: {str(e)}"]
            )

    async def _extract_findings(
        self,
        search_results: Dict[str, List[Dict[str, Any]]],
        citations: List[Citation]
    ) -> List[KeyFinding]:
        """Extract key findings from search results."""
        findings = []
        finding_texts = set()  # Track to avoid duplicates

        for provider_name, results in search_results.items():
            provider = {
                "pubmed": self._pubmed,
                "perplexity": self._perplexity,
                "llamaindex": self._llamaindex
            }.get(provider_name)

            if not provider:
                continue

            for result in results:
                if "error" in result:
                    continue

                # Extract finding text
                finding_text = provider.extract_finding(result)
                if not finding_text or finding_text in finding_texts:
                    continue

                finding_texts.add(finding_text)

                # Find matching citation
                citation_id = None
                for citation in citations:
                    if (result.get("title", "") == citation.title or
                        result.get("pmid", "") == citation.pmid):
                        citation_id = citation.id
                        break

                # Create finding
                finding = KeyFinding(
                    statement=finding_text,
                    citation_ids=[citation_id] if citation_id else [],
                    supporting_count=1,
                    study_types_supporting=[
                        next((c.study_type for c in citations if c.id == citation_id),
                             StudyType.UNKNOWN)
                    ] if citation_id else []
                )

                findings.append(finding)

        # Deduplicate and merge similar findings
        findings = self._merge_similar_findings(findings)

        # Assess individual finding strength
        for finding in findings:
            finding.confidence_score = self._calculate_finding_confidence(finding)
            finding.strength = EvidenceStrength.from_score(finding.confidence_score * 100)

        return findings

    def _merge_similar_findings(self, findings: List[KeyFinding]) -> List[KeyFinding]:
        """Merge findings that express similar conclusions."""
        if len(findings) <= 1:
            return findings

        merged = []
        merged_indices = set()

        for i, finding_a in enumerate(findings):
            if i in merged_indices:
                continue

            # Start with this finding
            merged_finding = KeyFinding(
                statement=finding_a.statement,
                citation_ids=list(finding_a.citation_ids),
                supporting_count=finding_a.supporting_count,
                study_types_supporting=list(finding_a.study_types_supporting)
            )

            # Look for similar findings to merge
            for j, finding_b in enumerate(findings[i+1:], i+1):
                if j in merged_indices:
                    continue

                if self._findings_share_topic(finding_a, finding_b):
                    # Check if same direction (not contradicting)
                    contradiction = self._check_contradiction(finding_a, finding_b)
                    if contradiction is None:
                        # Merge
                        merged_finding.citation_ids.extend(finding_b.citation_ids)
                        merged_finding.supporting_count += finding_b.supporting_count
                        merged_finding.study_types_supporting.extend(
                            finding_b.study_types_supporting
                        )
                        merged_indices.add(j)
                    else:
                        # Track contradiction
                        merged_finding.contradicting_count += 1

            merged_indices.add(i)
            merged.append(merged_finding)

        return merged

    def _calculate_finding_confidence(self, finding: KeyFinding) -> float:
        """Calculate confidence score for a finding (0-1)."""
        score = 0.0

        # Base on number of supporting sources
        if finding.supporting_count >= 5:
            score += 0.4
        elif finding.supporting_count >= 2:
            score += 0.25
        elif finding.supporting_count >= 1:
            score += 0.1

        # Adjust for study quality
        if finding.study_types_supporting:
            avg_rank = sum(st.evidence_rank for st in finding.study_types_supporting) / len(finding.study_types_supporting)
            score += (avg_rank / 10) * 0.4  # Max 0.4 for highest quality

        # Penalize for contradictions
        if finding.supporting_count > 0:
            contradiction_rate = finding.contradicting_count / (finding.supporting_count + finding.contradicting_count)
            score -= contradiction_rate * 0.3

        # Bonus for consensus designation
        if finding.is_consensus:
            score += 0.2

        return max(0.0, min(1.0, score))

    # -------------------------------------------------------------------------
    # CONVENIENCE METHODS
    # -------------------------------------------------------------------------

    async def quick_search(
        self,
        topic: str,
        questions: Optional[List[str]] = None
    ) -> AgentResponse:
        """
        Convenience method for light-depth search.

        Args:
            topic: Research topic
            questions: Optional specific questions

        Returns:
            AgentResponse with evidence brief
        """
        request = EvidenceRequest(
            topic=topic,
            depth=ResearchDepth.LIGHT,
            specific_questions=questions or []
        )
        return await self.generate_brief(request)

    async def deep_research(
        self,
        topic: str,
        questions: List[str],
        intended_output: str = ""
    ) -> AgentResponse:
        """
        Convenience method for deep research.

        Args:
            topic: Research topic
            questions: Specific questions to answer
            intended_output: What this research will inform

        Returns:
            AgentResponse with comprehensive evidence brief
        """
        request = EvidenceRequest(
            topic=topic,
            depth=ResearchDepth.DEEP,
            intended_output=intended_output,
            specific_questions=questions
        )
        return await self.generate_brief(request)

    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Retrieve a citation by ID from the index."""
        return self._citations_index.get(citation_id)

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._findings_cache.clear()
        self._citations_index.clear()
        self._current_request = None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "ResearchAgent",

    # Enums
    "EvidenceStrength",
    "ResearchDepth",
    "StudyType",
    "RecencyCategory",

    # Data classes
    "Citation",
    "KeyFinding",
    "Contradiction",
    "UncertaintyGap",
    "StudyLimitation",
    "EvidenceRequest",
    "EvidenceBrief",

    # Provider interfaces
    "ResearchProvider",
    "PubMedProvider",
    "PerplexityProvider",
    "LlamaIndexProvider",
]
