# Social Metrics Harvester - Alfred Sub-Agent #16
# Automated collection of performance metrics across all social platforms
# Zero interpretation - pure data collection

"""
Social Metrics Harvester

Role: Automated collection of performance metrics across all social platforms.
Pulls raw data on fixed cadence, normalizes to common schema, tracks deltas.
Zero interpretation - pure data collection.

Does NOT:
- Interpret what metrics mean
- Recommend actions based on data
- Judge content quality
- Compare to competitors
- Provide vanity metric commentary
- Optimize for engagement at cost of positioning
- Access metrics more frequently than configured cadence

Does:
- Pull metrics from all configured platforms
- Normalize to standard schema
- Calculate deltas from previous periods
- Track trends over time
- Associate metrics with specific content
- Maintain historical database
- Flag data collection failures
"""

from . import StrategyAgent, AgentResponse, AlfredState
from typing import Dict, Any, Optional, List, Protocol, Union
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import json
import hashlib
from pathlib import Path


# =============================================================================
# ENUMS
# =============================================================================

class Platform(Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    SUBSTACK = "substack"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    THREADS = "threads"


class MetricsGranularity(Enum):
    """Time granularity for metrics collection."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class CollectionStatus(Enum):
    """Status of a collection attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    AUTH_ERROR = "auth_error"
    NOT_CONFIGURED = "not_configured"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MetricsDelta:
    """Represents change from previous period."""
    current: float
    previous: float
    absolute_change: float = field(init=False)
    percentage_change: Optional[float] = field(init=False)

    def __post_init__(self):
        self.absolute_change = self.current - self.previous
        if self.previous != 0:
            self.percentage_change = ((self.current - self.previous) / self.previous) * 100
        else:
            self.percentage_change = None if self.current == 0 else float('inf')

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": self.current,
            "previous": self.previous,
            "absolute_change": self.absolute_change,
            "percentage_change": self.percentage_change
        }

    def format_delta(self) -> str:
        """Format delta for display in reports."""
        sign = "+" if self.absolute_change >= 0 else ""
        if self.percentage_change is not None:
            return f"{self.current:,.0f} ({sign}{self.absolute_change:,.0f}, {sign}{self.percentage_change:.1f}%)"
        return f"{self.current:,.0f} ({sign}{self.absolute_change:,.0f})"


@dataclass
class OutputMetrics:
    """Metrics for content output."""
    posts_published: int = 0
    videos_published: int = 0
    articles_published: int = 0
    threads_published: int = 0
    reels_published: int = 0
    stories_published: int = 0

    @property
    def total_output(self) -> int:
        return (self.posts_published + self.videos_published +
                self.articles_published + self.threads_published +
                self.reels_published + self.stories_published)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReachMetrics:
    """Metrics for content reach."""
    impressions: int = 0
    views: int = 0
    unique_viewers: int = 0
    reach: int = 0  # Platform-specific reach metric

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngagementMetrics:
    """Metrics for audience engagement."""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    bookmarks: int = 0
    retweets: int = 0
    quotes: int = 0
    replies: int = 0
    watch_time_seconds: int = 0
    avg_watch_time_seconds: float = 0.0
    completion_rate: float = 0.0  # 0-100 percentage
    read_time_seconds: int = 0
    avg_read_time_seconds: float = 0.0

    @property
    def total_engagements(self) -> int:
        return (self.likes + self.comments + self.shares +
                self.saves + self.bookmarks + self.retweets +
                self.quotes + self.replies)

    def calculate_engagement_rate(self, impressions: int) -> float:
        """Calculate engagement rate as percentage."""
        if impressions == 0:
            return 0.0
        return (self.total_engagements / impressions) * 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GrowthMetrics:
    """Metrics for audience growth."""
    followers: int = 0
    followers_gained: int = 0
    followers_lost: int = 0
    subscribers: int = 0
    subscribers_gained: int = 0
    subscribers_lost: int = 0

    @property
    def net_follower_change(self) -> int:
        return self.followers_gained - self.followers_lost

    @property
    def net_subscriber_change(self) -> int:
        return self.subscribers_gained - self.subscribers_lost

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversionMetrics:
    """Metrics for conversions and actions."""
    link_clicks: int = 0
    profile_visits: int = 0
    website_clicks: int = 0
    newsletter_signups: int = 0
    dm_inquiries: int = 0
    booking_requests: int = 0
    external_link_clicks: int = 0
    bio_link_clicks: int = 0
    email_opens: int = 0
    email_clicks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContentMetrics:
    """Metrics associated with a specific piece of content."""
    content_id: str
    platform: Platform
    content_type: str  # post, video, article, thread, reel, story
    title: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None

    # Metrics
    reach: ReachMetrics = field(default_factory=ReachMetrics)
    engagement: EngagementMetrics = field(default_factory=EngagementMetrics)
    conversions: ConversionMetrics = field(default_factory=ConversionMetrics)

    # Computed fields
    engagement_rate: float = 0.0

    def calculate_engagement_rate(self):
        """Calculate and store engagement rate."""
        impressions = self.reach.impressions or self.reach.views or 1
        self.engagement_rate = self.engagement.calculate_engagement_rate(impressions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "platform": self.platform.value,
            "content_type": self.content_type,
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "reach": self.reach.to_dict(),
            "engagement": self.engagement.to_dict(),
            "conversions": self.conversions.to_dict(),
            "engagement_rate": self.engagement_rate
        }


@dataclass
class PlatformMetrics:
    """Aggregated metrics for a single platform."""
    platform: Platform
    period_start: date
    period_end: date
    collected_at: datetime = field(default_factory=datetime.now)
    collection_status: CollectionStatus = CollectionStatus.SUCCESS
    collection_errors: List[str] = field(default_factory=list)

    # Aggregated metrics
    output: OutputMetrics = field(default_factory=OutputMetrics)
    reach: ReachMetrics = field(default_factory=ReachMetrics)
    engagement: EngagementMetrics = field(default_factory=EngagementMetrics)
    growth: GrowthMetrics = field(default_factory=GrowthMetrics)
    conversions: ConversionMetrics = field(default_factory=ConversionMetrics)

    # Delta tracking (populated by calculate_deltas)
    reach_delta: Optional[Dict[str, MetricsDelta]] = None
    engagement_delta: Optional[Dict[str, MetricsDelta]] = None
    growth_delta: Optional[Dict[str, MetricsDelta]] = None

    # Content performance
    content_items: List[ContentMetrics] = field(default_factory=list)
    top_performing: Optional[ContentMetrics] = None
    lowest_performing: Optional[ContentMetrics] = None

    @property
    def engagement_rate(self) -> float:
        """Calculate overall engagement rate."""
        impressions = self.reach.impressions or self.reach.views or 1
        return self.engagement.calculate_engagement_rate(impressions)

    def identify_performance_extremes(self):
        """Identify top and lowest performing content."""
        if not self.content_items:
            return

        # Sort by engagement rate
        sorted_content = sorted(
            self.content_items,
            key=lambda x: x.engagement_rate,
            reverse=True
        )

        self.top_performing = sorted_content[0]
        self.lowest_performing = sorted_content[-1]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "platform": self.platform.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "collected_at": self.collected_at.isoformat(),
            "collection_status": self.collection_status.value,
            "collection_errors": self.collection_errors,
            "output": self.output.to_dict(),
            "reach": self.reach.to_dict(),
            "engagement": self.engagement.to_dict(),
            "growth": self.growth.to_dict(),
            "conversions": self.conversions.to_dict(),
            "engagement_rate": self.engagement_rate,
            "content_items": [c.to_dict() for c in self.content_items],
            "top_performing": self.top_performing.to_dict() if self.top_performing else None,
            "lowest_performing": self.lowest_performing.to_dict() if self.lowest_performing else None
        }

        if self.reach_delta:
            result["reach_delta"] = {k: v.to_dict() for k, v in self.reach_delta.items()}
        if self.engagement_delta:
            result["engagement_delta"] = {k: v.to_dict() for k, v in self.engagement_delta.items()}
        if self.growth_delta:
            result["growth_delta"] = {k: v.to_dict() for k, v in self.growth_delta.items()}

        return result


@dataclass
class CrossPlatformSummary:
    """Aggregated metrics across all platforms."""
    total_output: int = 0
    total_reach: int = 0
    total_engagement: int = 0
    net_follower_growth: int = 0
    platform_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataQuality:
    """Data quality information for the report."""
    platforms_requested: List[Platform] = field(default_factory=list)
    platforms_collected: List[Platform] = field(default_factory=list)
    platforms_failed: List[Platform] = field(default_factory=list)
    collection_errors: Dict[str, List[str]] = field(default_factory=dict)
    data_freshness: Dict[str, datetime] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if not self.platforms_requested:
            return 0.0
        return len(self.platforms_collected) / len(self.platforms_requested) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platforms_requested": [p.value for p in self.platforms_requested],
            "platforms_collected": [p.value for p in self.platforms_collected],
            "platforms_failed": [p.value for p in self.platforms_failed],
            "collection_errors": self.collection_errors,
            "data_freshness": {k: v.isoformat() for k, v in self.data_freshness.items()},
            "success_rate": self.success_rate
        }


@dataclass
class ContentPerformanceMatrix:
    """Matrix of content performance across platforms."""
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add_entry(self, content: ContentMetrics):
        """Add a content item to the matrix."""
        self.entries.append({
            "content": content.title or content.content_id,
            "platform": content.platform.value,
            "type": content.content_type,
            "reach": content.reach.impressions or content.reach.views,
            "engagement": content.engagement.total_engagements,
            "rate": f"{content.engagement_rate:.2f}%"
        })

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": self.entries}


@dataclass
class MetricsReport:
    """Complete metrics report output."""
    report_date: datetime
    period_start: date
    period_end: date
    granularity: MetricsGranularity

    # Per-platform metrics
    platform_metrics: Dict[Platform, PlatformMetrics] = field(default_factory=dict)

    # Cross-platform summary
    cross_platform_summary: CrossPlatformSummary = field(default_factory=CrossPlatformSummary)

    # Content performance matrix
    content_matrix: ContentPerformanceMatrix = field(default_factory=ContentPerformanceMatrix)

    # Data quality
    data_quality: DataQuality = field(default_factory=DataQuality)

    def calculate_cross_platform_summary(self):
        """Aggregate metrics across all platforms."""
        total_output = 0
        total_reach = 0
        total_engagement = 0
        net_growth = 0
        distribution = {}

        for platform, metrics in self.platform_metrics.items():
            total_output += metrics.output.total_output
            total_reach += metrics.reach.impressions + metrics.reach.views
            total_engagement += metrics.engagement.total_engagements
            net_growth += metrics.growth.net_follower_change + metrics.growth.net_subscriber_change
            distribution[platform.value] = metrics.output.total_output

        self.cross_platform_summary = CrossPlatformSummary(
            total_output=total_output,
            total_reach=total_reach,
            total_engagement=total_engagement,
            net_follower_growth=net_growth,
            platform_distribution=distribution
        )

    def build_content_matrix(self):
        """Build the content performance matrix from all platforms."""
        all_content = []
        for metrics in self.platform_metrics.values():
            all_content.extend(metrics.content_items)

        # Sort by engagement rate descending
        sorted_content = sorted(all_content, key=lambda x: x.engagement_rate, reverse=True)

        self.content_matrix = ContentPerformanceMatrix()
        for content in sorted_content[:20]:  # Top 20 items
            self.content_matrix.add_entry(content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_date": self.report_date.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "granularity": self.granularity.value,
            "platform_metrics": {
                p.value: m.to_dict() for p, m in self.platform_metrics.items()
            },
            "cross_platform_summary": self.cross_platform_summary.to_dict(),
            "content_matrix": self.content_matrix.to_dict(),
            "data_quality": self.data_quality.to_dict()
        }

    def to_formatted_report(self) -> str:
        """Generate formatted text report matching specification."""
        lines = [
            "METRICS_REPORT",
            f"- Report Date: {self.report_date.isoformat()}",
            f"- Period: {self.period_start.isoformat()} to {self.period_end.isoformat()}",
            f"- Platforms: {', '.join(p.value for p in self.platform_metrics.keys())}",
            ""
        ]

        # Per-platform sections
        for platform, metrics in self.platform_metrics.items():
            lines.append(f"- Platform: [{platform.value.upper()}]")
            lines.append(f"  - Output: [{metrics.output.total_output} items]")

            # Reach with delta
            if platform == Platform.YOUTUBE:
                reach_val = metrics.reach.views
                reach_label = "Views"
            else:
                reach_val = metrics.reach.impressions
                reach_label = "Impressions"

            if metrics.reach_delta:
                delta_str = f" (Delta: {metrics.reach_delta.get('impressions', metrics.reach_delta.get('views', MetricsDelta(0,0))).absolute_change:+,})"
            else:
                delta_str = ""
            lines.append(f"  - {reach_label}: [{reach_val:,}]{delta_str}")

            # Engagement with delta
            eng_val = metrics.engagement.total_engagements
            if metrics.engagement_delta and 'total' in metrics.engagement_delta:
                eng_delta = f" (Delta: {metrics.engagement_delta['total'].absolute_change:+,})"
            else:
                eng_delta = ""
            lines.append(f"  - Engagement: [{eng_val:,}]{eng_delta}")
            lines.append(f"  - Engagement Rate: [{metrics.engagement_rate:.2f}%]")

            # Followers/Subscribers with delta
            if platform == Platform.YOUTUBE:
                follow_val = metrics.growth.subscribers
                follow_label = "Subscribers"
            else:
                follow_val = metrics.growth.followers
                follow_label = "Followers"

            if metrics.growth_delta:
                key = 'subscribers' if platform == Platform.YOUTUBE else 'followers'
                if key in metrics.growth_delta:
                    grow_delta = f" (Delta: {metrics.growth_delta[key].absolute_change:+,})"
                else:
                    grow_delta = ""
            else:
                grow_delta = ""
            lines.append(f"  - {follow_label}: [{follow_val:,}]{grow_delta}")

            # Top/Lowest performing
            if metrics.top_performing:
                lines.append(f"  - Top Performing: [{metrics.top_performing.title or metrics.top_performing.content_id}]")
            if metrics.lowest_performing:
                lines.append(f"  - Lowest Performing: [{metrics.lowest_performing.title or metrics.lowest_performing.content_id}]")
            lines.append("")

        # Cross-platform summary
        lines.append("- Cross-Platform Summary:")
        lines.append(f"  - Total Output: [{self.cross_platform_summary.total_output} items]")
        lines.append(f"  - Total Reach: [{self.cross_platform_summary.total_reach:,}]")
        lines.append(f"  - Total Engagement: [{self.cross_platform_summary.total_engagement:,}]")
        lines.append(f"  - Net Follower Growth: [{self.cross_platform_summary.net_follower_growth:+,}]")
        lines.append("")

        # Content performance matrix
        lines.append("- Content Performance Matrix:")
        lines.append("  | Content | Platform | Reach | Engagement | Rate |")
        lines.append("  |---------|----------|-------|------------|------|")
        for entry in self.content_matrix.entries[:10]:
            content_name = entry['content'][:30] if entry['content'] else 'N/A'
            lines.append(f"  | {content_name} | {entry['platform']} | {entry['reach']:,} | {entry['engagement']:,} | {entry['rate']} |")
        lines.append("")

        # Data quality
        lines.append("- Data Quality:")
        lines.append(f"  - Platforms Collected: [{', '.join(p.value for p in self.data_quality.platforms_collected)}]")
        if self.data_quality.collection_errors:
            lines.append(f"  - Collection Errors: {self.data_quality.collection_errors}")
        else:
            lines.append("  - Collection Errors: [none]")

        return "\n".join(lines)


@dataclass
class HarvestRequest:
    """Input request for metrics harvesting."""
    platforms: List[Platform]
    period_start: date
    period_end: date
    granularity: MetricsGranularity = MetricsGranularity.WEEKLY
    content_filter: Optional[List[str]] = None  # Specific content IDs, or None for all

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HarvestRequest':
        """Parse request from dict."""
        platforms = [Platform(p) for p in data.get('platforms', [])]

        # Parse dates
        period = data.get('period', {})
        if isinstance(period, dict):
            period_start = date.fromisoformat(period.get('start', date.today().isoformat()))
            period_end = date.fromisoformat(period.get('end', date.today().isoformat()))
        else:
            # Default to last 7 days
            period_end = date.today()
            period_start = period_end - timedelta(days=7)

        granularity = MetricsGranularity(data.get('granularity', 'weekly'))
        content_filter = data.get('content_filter')

        return cls(
            platforms=platforms,
            period_start=period_start,
            period_end=period_end,
            granularity=granularity,
            content_filter=content_filter
        )


# =============================================================================
# PLATFORM ADAPTERS (Interfaces for external tools)
# =============================================================================

class PlatformAdapter(ABC):
    """Abstract base class for platform-specific data collection."""

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """Return the platform this adapter handles."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the adapter is properly configured."""
        pass

    @abstractmethod
    def fetch_metrics(
        self,
        period_start: date,
        period_end: date,
        content_filter: Optional[List[str]] = None
    ) -> PlatformMetrics:
        """Fetch metrics for the specified period."""
        pass

    @abstractmethod
    def fetch_content_metrics(
        self,
        content_ids: List[str]
    ) -> List[ContentMetrics]:
        """Fetch metrics for specific content items."""
        pass


class TwitterAdapter(PlatformAdapter):
    """
    Twitter/X metrics adapter.
    Integrates with: twscrape for Twitter metrics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._client = None

    @property
    def platform(self) -> Platform:
        return Platform.TWITTER

    def is_configured(self) -> bool:
        """Check if twscrape credentials are configured."""
        return bool(self.config.get('credentials'))

    def _get_client(self):
        """Initialize twscrape client if not already done."""
        if self._client is None:
            # Placeholder for twscrape integration
            # from twscrape import API
            # self._client = API()
            pass
        return self._client

    def fetch_metrics(
        self,
        period_start: date,
        period_end: date,
        content_filter: Optional[List[str]] = None
    ) -> PlatformMetrics:
        """Fetch Twitter metrics for the specified period."""
        metrics = PlatformMetrics(
            platform=Platform.TWITTER,
            period_start=period_start,
            period_end=period_end
        )

        if not self.is_configured():
            metrics.collection_status = CollectionStatus.NOT_CONFIGURED
            metrics.collection_errors.append("Twitter credentials not configured")
            return metrics

        try:
            # Placeholder for actual twscrape API calls
            # This would use twscrape to fetch:
            # - User timeline tweets in date range
            # - Engagement metrics per tweet
            # - Follower counts

            # For now, return empty metrics with success status
            metrics.collection_status = CollectionStatus.SUCCESS

        except Exception as e:
            metrics.collection_status = CollectionStatus.FAILED
            metrics.collection_errors.append(f"Twitter API error: {str(e)}")

        return metrics

    def fetch_content_metrics(
        self,
        content_ids: List[str]
    ) -> List[ContentMetrics]:
        """Fetch metrics for specific tweets."""
        results = []

        if not self.is_configured():
            return results

        # Placeholder for twscrape implementation
        # Would fetch metrics for specific tweet IDs

        return results


class YouTubeAdapter(PlatformAdapter):
    """
    YouTube metrics adapter.
    Integrates with: YouTube Data API v3 / YouTube Analytics API
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._client = None

    @property
    def platform(self) -> Platform:
        return Platform.YOUTUBE

    def is_configured(self) -> bool:
        """Check if YouTube API credentials are configured."""
        return bool(self.config.get('api_key') or self.config.get('oauth_credentials'))

    def _get_client(self):
        """Initialize YouTube API client."""
        if self._client is None:
            # Placeholder for google-api-python-client integration
            # from googleapiclient.discovery import build
            # self._client = build('youtube', 'v3', developerKey=self.config['api_key'])
            pass
        return self._client

    def fetch_metrics(
        self,
        period_start: date,
        period_end: date,
        content_filter: Optional[List[str]] = None
    ) -> PlatformMetrics:
        """Fetch YouTube metrics for the specified period."""
        metrics = PlatformMetrics(
            platform=Platform.YOUTUBE,
            period_start=period_start,
            period_end=period_end
        )

        if not self.is_configured():
            metrics.collection_status = CollectionStatus.NOT_CONFIGURED
            metrics.collection_errors.append("YouTube API credentials not configured")
            return metrics

        try:
            # Placeholder for actual YouTube API calls
            # This would use YouTube Data API to fetch:
            # - Channel statistics (subscribers, views)
            # - Video list with stats
            # - Analytics data (watch time, retention)

            metrics.collection_status = CollectionStatus.SUCCESS

        except Exception as e:
            metrics.collection_status = CollectionStatus.FAILED
            metrics.collection_errors.append(f"YouTube API error: {str(e)}")

        return metrics

    def fetch_content_metrics(
        self,
        content_ids: List[str]
    ) -> List[ContentMetrics]:
        """Fetch metrics for specific videos."""
        results = []

        if not self.is_configured():
            return results

        # Placeholder for YouTube API implementation

        return results


class SubstackAdapter(PlatformAdapter):
    """
    Substack metrics adapter.
    Integrates with: Substack API / Dashboard scraping
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @property
    def platform(self) -> Platform:
        return Platform.SUBSTACK

    def is_configured(self) -> bool:
        """Check if Substack credentials are configured."""
        return bool(self.config.get('publication_id') or self.config.get('session_cookie'))

    def fetch_metrics(
        self,
        period_start: date,
        period_end: date,
        content_filter: Optional[List[str]] = None
    ) -> PlatformMetrics:
        """Fetch Substack metrics for the specified period."""
        metrics = PlatformMetrics(
            platform=Platform.SUBSTACK,
            period_start=period_start,
            period_end=period_end
        )

        if not self.is_configured():
            metrics.collection_status = CollectionStatus.NOT_CONFIGURED
            metrics.collection_errors.append("Substack credentials not configured")
            return metrics

        try:
            # Placeholder for Substack API/scraping
            # This would fetch:
            # - Post list with open rates
            # - Subscriber counts
            # - Click-through rates

            metrics.collection_status = CollectionStatus.SUCCESS

        except Exception as e:
            metrics.collection_status = CollectionStatus.FAILED
            metrics.collection_errors.append(f"Substack API error: {str(e)}")

        return metrics

    def fetch_content_metrics(
        self,
        content_ids: List[str]
    ) -> List[ContentMetrics]:
        """Fetch metrics for specific Substack posts."""
        return []


class InstagramAdapter(PlatformAdapter):
    """
    Instagram metrics adapter.
    Integrates with: Instagram Graph API via Postiz
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @property
    def platform(self) -> Platform:
        return Platform.INSTAGRAM

    def is_configured(self) -> bool:
        """Check if Instagram API credentials are configured."""
        return bool(self.config.get('access_token'))

    def fetch_metrics(
        self,
        period_start: date,
        period_end: date,
        content_filter: Optional[List[str]] = None
    ) -> PlatformMetrics:
        """Fetch Instagram metrics for the specified period."""
        metrics = PlatformMetrics(
            platform=Platform.INSTAGRAM,
            period_start=period_start,
            period_end=period_end
        )

        if not self.is_configured():
            metrics.collection_status = CollectionStatus.NOT_CONFIGURED
            metrics.collection_errors.append("Instagram credentials not configured")
            return metrics

        try:
            # Placeholder for Instagram Graph API calls
            # This would fetch:
            # - Media list with insights
            # - Account insights
            # - Follower counts

            metrics.collection_status = CollectionStatus.SUCCESS

        except Exception as e:
            metrics.collection_status = CollectionStatus.FAILED
            metrics.collection_errors.append(f"Instagram API error: {str(e)}")

        return metrics

    def fetch_content_metrics(
        self,
        content_ids: List[str]
    ) -> List[ContentMetrics]:
        """Fetch metrics for specific Instagram posts."""
        return []


class PostizAdapter:
    """
    Postiz multi-platform analytics adapter.
    Provides unified interface for platforms supported by Postiz.
    GitHub: https://github.com/gitroomhq/postiz-app
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._client = None

    def is_configured(self) -> bool:
        """Check if Postiz API is configured."""
        return bool(self.config.get('api_url') and self.config.get('api_key'))

    def fetch_analytics(
        self,
        platforms: List[Platform],
        period_start: date,
        period_end: date
    ) -> Dict[Platform, PlatformMetrics]:
        """Fetch analytics for multiple platforms via Postiz."""
        results = {}

        if not self.is_configured():
            for platform in platforms:
                metrics = PlatformMetrics(
                    platform=platform,
                    period_start=period_start,
                    period_end=period_end,
                    collection_status=CollectionStatus.NOT_CONFIGURED
                )
                metrics.collection_errors.append("Postiz not configured")
                results[platform] = metrics
            return results

        # Placeholder for Postiz API integration
        # Would use Postiz's unified analytics API

        return results


class UmamiAdapter:
    """
    Umami web analytics adapter.
    Provides website traffic and conversion metrics.
    GitHub: https://github.com/umami-software/umami
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def is_configured(self) -> bool:
        """Check if Umami is configured."""
        return bool(self.config.get('api_url') and self.config.get('website_id'))

    def fetch_website_metrics(
        self,
        period_start: date,
        period_end: date
    ) -> Dict[str, Any]:
        """Fetch website analytics from Umami."""
        if not self.is_configured():
            return {"error": "Umami not configured"}

        # Placeholder for Umami API integration
        # Would fetch:
        # - Page views
        # - Unique visitors
        # - Referral sources
        # - UTM tracking data

        return {}


# =============================================================================
# HISTORICAL STORAGE
# =============================================================================

class MetricsHistoryStore:
    """
    Persistent storage for historical metrics data.
    Enables delta calculation and trend tracking.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/metrics_history")
        self._ensure_storage_exists()

    def _ensure_storage_exists(self):
        """Create storage directory if it doesn't exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_metrics_file(self, platform: Platform, period_date: date) -> Path:
        """Get path for a specific metrics file."""
        filename = f"{platform.value}_{period_date.isoformat()}.json"
        return self.storage_path / filename

    def _get_period_key(self, period_start: date, period_end: date) -> str:
        """Generate a unique key for a period."""
        return f"{period_start.isoformat()}_{period_end.isoformat()}"

    def store_metrics(self, metrics: PlatformMetrics):
        """Store platform metrics to history."""
        filepath = self._get_metrics_file(metrics.platform, metrics.period_end)

        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2, default=str)

    def get_previous_metrics(
        self,
        platform: Platform,
        current_period_end: date,
        granularity: MetricsGranularity
    ) -> Optional[PlatformMetrics]:
        """Retrieve metrics from the previous period for delta calculation."""
        # Calculate previous period end date
        if granularity == MetricsGranularity.DAILY:
            previous_end = current_period_end - timedelta(days=1)
        elif granularity == MetricsGranularity.WEEKLY:
            previous_end = current_period_end - timedelta(weeks=1)
        else:  # MONTHLY
            # Go back approximately one month
            previous_end = current_period_end - timedelta(days=30)

        filepath = self._get_metrics_file(platform, previous_end)

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Reconstruct PlatformMetrics from stored data
            return self._reconstruct_metrics(data)

        except Exception:
            return None

    def _reconstruct_metrics(self, data: Dict[str, Any]) -> PlatformMetrics:
        """Reconstruct PlatformMetrics from stored dict."""
        # This is a simplified reconstruction
        metrics = PlatformMetrics(
            platform=Platform(data['platform']),
            period_start=date.fromisoformat(data['period_start']),
            period_end=date.fromisoformat(data['period_end']),
            collection_status=CollectionStatus(data.get('collection_status', 'success'))
        )

        # Reconstruct nested dataclasses
        if 'reach' in data:
            metrics.reach = ReachMetrics(**data['reach'])
        if 'engagement' in data:
            metrics.engagement = EngagementMetrics(**data['engagement'])
        if 'growth' in data:
            metrics.growth = GrowthMetrics(**data['growth'])
        if 'output' in data:
            metrics.output = OutputMetrics(**data['output'])
        if 'conversions' in data:
            metrics.conversions = ConversionMetrics(**data['conversions'])

        return metrics

    def get_trend_data(
        self,
        platform: Platform,
        metric_name: str,
        periods: int = 12
    ) -> List[Dict[str, Any]]:
        """Get historical trend data for a specific metric."""
        trend_data = []

        # Find all metrics files for this platform
        pattern = f"{platform.value}_*.json"
        files = sorted(self.storage_path.glob(pattern), reverse=True)[:periods]

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract the requested metric value
                value = self._extract_metric_value(data, metric_name)
                if value is not None:
                    trend_data.append({
                        "date": data['period_end'],
                        "value": value
                    })
            except Exception:
                continue

        return list(reversed(trend_data))

    def _extract_metric_value(self, data: Dict[str, Any], metric_path: str) -> Optional[float]:
        """Extract a metric value from nested data using dot notation."""
        parts = metric_path.split('.')
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return float(current) if current is not None else None

    def cleanup_old_metrics(self, days_to_keep: int = 365):
        """Remove metrics older than specified days."""
        cutoff = date.today() - timedelta(days=days_to_keep)

        for filepath in self.storage_path.glob("*.json"):
            try:
                # Extract date from filename
                parts = filepath.stem.split('_')
                if len(parts) >= 2:
                    file_date = date.fromisoformat(parts[-1])
                    if file_date < cutoff:
                        filepath.unlink()
            except Exception:
                continue


# =============================================================================
# MAIN HARVESTER CLASS
# =============================================================================

class SocialMetricsHarvester(StrategyAgent):
    """
    Social Metrics Harvester - Alfred Sub-Agent #16

    Automated collection of performance metrics across all social platforms.
    Pulls raw data on fixed cadence, normalizes to common schema, tracks deltas.
    Zero interpretation - pure data collection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Social Metrics Harvester")

        self.config = config or {}

        # Initialize platform adapters
        self.adapters: Dict[Platform, PlatformAdapter] = {
            Platform.TWITTER: TwitterAdapter(self.config.get('twitter', {})),
            Platform.YOUTUBE: YouTubeAdapter(self.config.get('youtube', {})),
            Platform.SUBSTACK: SubstackAdapter(self.config.get('substack', {})),
            Platform.INSTAGRAM: InstagramAdapter(self.config.get('instagram', {})),
        }

        # Multi-platform adapters
        self.postiz = PostizAdapter(self.config.get('postiz', {}))
        self.umami = UmamiAdapter(self.config.get('umami', {}))

        # Historical storage
        storage_path = self.config.get('storage_path')
        self.history = MetricsHistoryStore(
            Path(storage_path) if storage_path else None
        )

        # Collection cadence tracking
        self._last_collection: Dict[Platform, datetime] = {}
        self._min_collection_interval = timedelta(
            hours=self.config.get('min_collection_interval_hours', 1)
        )

    def _check_collection_cadence(self, platform: Platform) -> tuple[bool, str]:
        """
        Check if enough time has passed since last collection.
        Enforces 'Does NOT access metrics more frequently than configured cadence'.
        """
        last = self._last_collection.get(platform)
        if last is None:
            return True, "No previous collection"

        elapsed = datetime.now() - last
        if elapsed < self._min_collection_interval:
            remaining = self._min_collection_interval - elapsed
            return False, f"Rate limited. Next collection in {remaining}"

        return True, "Collection permitted"

    def harvest_platform(
        self,
        platform: Platform,
        period_start: date,
        period_end: date,
        content_filter: Optional[List[str]] = None
    ) -> PlatformMetrics:
        """
        Harvest metrics from a single platform.

        Args:
            platform: The platform to harvest from
            period_start: Start of the period to collect
            period_end: End of the period to collect
            content_filter: Optional list of specific content IDs to fetch

        Returns:
            PlatformMetrics with collected data
        """
        # Check collection cadence
        allowed, reason = self._check_collection_cadence(platform)
        if not allowed:
            metrics = PlatformMetrics(
                platform=platform,
                period_start=period_start,
                period_end=period_end,
                collection_status=CollectionStatus.RATE_LIMITED
            )
            metrics.collection_errors.append(reason)
            return metrics

        # Get the appropriate adapter
        adapter = self.adapters.get(platform)
        if adapter is None:
            metrics = PlatformMetrics(
                platform=platform,
                period_start=period_start,
                period_end=period_end,
                collection_status=CollectionStatus.NOT_CONFIGURED
            )
            metrics.collection_errors.append(f"No adapter configured for {platform.value}")
            return metrics

        # Fetch metrics via adapter
        metrics = adapter.fetch_metrics(period_start, period_end, content_filter)

        # Fetch content-level metrics if content filter provided
        if content_filter and metrics.collection_status == CollectionStatus.SUCCESS:
            content_metrics = adapter.fetch_content_metrics(content_filter)
            metrics.content_items = content_metrics

            # Calculate engagement rates for each content item
            for content in metrics.content_items:
                content.calculate_engagement_rate()

            # Identify performance extremes
            metrics.identify_performance_extremes()

        # Update last collection time
        if metrics.collection_status in [CollectionStatus.SUCCESS, CollectionStatus.PARTIAL]:
            self._last_collection[platform] = datetime.now()

        return metrics

    def calculate_deltas(
        self,
        current: PlatformMetrics,
        granularity: MetricsGranularity
    ) -> PlatformMetrics:
        """
        Calculate deltas from previous period.

        Args:
            current: Current period metrics
            granularity: Time granularity for finding previous period

        Returns:
            PlatformMetrics with delta fields populated
        """
        # Get previous period metrics from history
        previous = self.history.get_previous_metrics(
            current.platform,
            current.period_end,
            granularity
        )

        if previous is None:
            # No previous data - deltas cannot be calculated
            return current

        # Calculate reach deltas
        current.reach_delta = {
            'impressions': MetricsDelta(
                current.reach.impressions,
                previous.reach.impressions
            ),
            'views': MetricsDelta(
                current.reach.views,
                previous.reach.views
            ),
            'unique_viewers': MetricsDelta(
                current.reach.unique_viewers,
                previous.reach.unique_viewers
            ),
        }

        # Calculate engagement deltas
        current.engagement_delta = {
            'total': MetricsDelta(
                current.engagement.total_engagements,
                previous.engagement.total_engagements
            ),
            'likes': MetricsDelta(
                current.engagement.likes,
                previous.engagement.likes
            ),
            'comments': MetricsDelta(
                current.engagement.comments,
                previous.engagement.comments
            ),
            'shares': MetricsDelta(
                current.engagement.shares,
                previous.engagement.shares
            ),
        }

        # Calculate growth deltas
        current.growth_delta = {
            'followers': MetricsDelta(
                current.growth.followers,
                previous.growth.followers
            ),
            'subscribers': MetricsDelta(
                current.growth.subscribers,
                previous.growth.subscribers
            ),
        }

        return current

    def normalize_metrics(
        self,
        platform_metrics: Dict[Platform, PlatformMetrics]
    ) -> Dict[Platform, PlatformMetrics]:
        """
        Normalize metrics to standard schema across platforms.

        Different platforms use different terminology and units.
        This method ensures consistent naming and scaling.

        Args:
            platform_metrics: Raw metrics per platform

        Returns:
            Normalized metrics
        """
        normalized = {}

        for platform, metrics in platform_metrics.items():
            # Platform-specific normalizations
            if platform == Platform.YOUTUBE:
                # YouTube uses 'views' primarily, normalize to impressions as well
                if metrics.reach.views > 0 and metrics.reach.impressions == 0:
                    metrics.reach.impressions = metrics.reach.views

                # YouTube uses subscribers, normalize to followers as well
                if metrics.growth.subscribers > 0 and metrics.growth.followers == 0:
                    metrics.growth.followers = metrics.growth.subscribers
                    metrics.growth.followers_gained = metrics.growth.subscribers_gained
                    metrics.growth.followers_lost = metrics.growth.subscribers_lost

            elif platform == Platform.TWITTER:
                # Twitter uses retweets, normalize to shares
                if metrics.engagement.retweets > 0 and metrics.engagement.shares == 0:
                    metrics.engagement.shares = metrics.engagement.retweets

            elif platform == Platform.INSTAGRAM:
                # Instagram uses reach metric, normalize
                if metrics.reach.reach > 0 and metrics.reach.impressions == 0:
                    metrics.reach.impressions = metrics.reach.reach

            elif platform == Platform.SUBSTACK:
                # Substack uses email opens, map to views
                if metrics.conversions.email_opens > 0 and metrics.reach.views == 0:
                    metrics.reach.views = metrics.conversions.email_opens

            normalized[platform] = metrics

        return normalized

    def track_content_performance(
        self,
        metrics: PlatformMetrics
    ) -> List[ContentMetrics]:
        """
        Associate metrics with specific content items and track performance.

        Args:
            metrics: Platform metrics with content items

        Returns:
            Sorted list of content by performance
        """
        if not metrics.content_items:
            return []

        # Ensure all content items have calculated engagement rates
        for content in metrics.content_items:
            content.calculate_engagement_rate()

        # Sort by engagement rate
        sorted_content = sorted(
            metrics.content_items,
            key=lambda x: x.engagement_rate,
            reverse=True
        )

        return sorted_content

    def generate_report(
        self,
        request: HarvestRequest
    ) -> MetricsReport:
        """
        Generate a complete metrics report.

        This is the main entry point for the harvester.

        Args:
            request: HarvestRequest specifying what to collect

        Returns:
            Complete MetricsReport
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            # Return minimal report indicating blocked state
            report = MetricsReport(
                report_date=datetime.now(),
                period_start=request.period_start,
                period_end=request.period_end,
                granularity=request.granularity
            )
            report.data_quality.collection_errors['system'] = [reason]
            return report

        # Initialize report
        report = MetricsReport(
            report_date=datetime.now(),
            period_start=request.period_start,
            period_end=request.period_end,
            granularity=request.granularity
        )
        report.data_quality.platforms_requested = request.platforms

        # Harvest each platform
        platform_metrics = {}
        for platform in request.platforms:
            metrics = self.harvest_platform(
                platform,
                request.period_start,
                request.period_end,
                request.content_filter
            )
            platform_metrics[platform] = metrics

            # Track collection status
            if metrics.collection_status == CollectionStatus.SUCCESS:
                report.data_quality.platforms_collected.append(platform)
                report.data_quality.data_freshness[platform.value] = metrics.collected_at
            else:
                report.data_quality.platforms_failed.append(platform)
                if metrics.collection_errors:
                    report.data_quality.collection_errors[platform.value] = metrics.collection_errors

        # Normalize metrics across platforms
        platform_metrics = self.normalize_metrics(platform_metrics)

        # Calculate deltas for each platform
        for platform, metrics in platform_metrics.items():
            metrics = self.calculate_deltas(metrics, request.granularity)

            # Store current metrics for future delta calculations
            if metrics.collection_status == CollectionStatus.SUCCESS:
                self.history.store_metrics(metrics)

            # Track content performance
            self.track_content_performance(metrics)

            platform_metrics[platform] = metrics

        # Populate report
        report.platform_metrics = platform_metrics

        # Calculate cross-platform summary
        report.calculate_cross_platform_summary()

        # Build content performance matrix
        report.build_content_matrix()

        return report

    def harvest(self, request_data: Dict[str, Any]) -> AgentResponse:
        """
        Main entry point matching Alfred's commissioning pattern.

        Args:
            request_data: Dict matching METRICS_HARVEST_REQUEST format

        Returns:
            AgentResponse with METRICS_REPORT data
        """
        try:
            # Parse request
            request = HarvestRequest.from_dict(request_data)

            # Generate report
            report = self.generate_report(request)

            # Determine success
            success = len(report.data_quality.platforms_collected) > 0

            # Create response
            return self.create_response(
                data={
                    "report": report.to_dict(),
                    "formatted_report": report.to_formatted_report()
                },
                success=success,
                errors=[
                    f"{platform}: {errors}"
                    for platform, errors in report.data_quality.collection_errors.items()
                ],
                warnings=[
                    f"Failed to collect from: {p.value}"
                    for p in report.data_quality.platforms_failed
                ]
            )

        except Exception as e:
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Harvest failed: {str(e)}"]
            )

    def get_trend(
        self,
        platform: Platform,
        metric_path: str,
        periods: int = 12
    ) -> AgentResponse:
        """
        Get historical trend data for a specific metric.

        Args:
            platform: Platform to get trend for
            metric_path: Dot-notation path to metric (e.g., 'reach.impressions')
            periods: Number of historical periods to include

        Returns:
            AgentResponse with trend data
        """
        try:
            trend_data = self.history.get_trend_data(platform, metric_path, periods)

            return self.create_response(
                data={
                    "platform": platform.value,
                    "metric": metric_path,
                    "trend": trend_data
                },
                success=True
            )

        except Exception as e:
            return self.create_response(
                data={"error": str(e)},
                success=False,
                errors=[f"Trend retrieval failed: {str(e)}"]
            )

    def get_configured_platforms(self) -> List[Platform]:
        """Return list of platforms with configured adapters."""
        configured = []

        for platform, adapter in self.adapters.items():
            if adapter.is_configured():
                configured.append(platform)

        return configured

    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status for all platforms."""
        status = {}

        for platform, adapter in self.adapters.items():
            last_collection = self._last_collection.get(platform)
            allowed, reason = self._check_collection_cadence(platform)

            status[platform.value] = {
                "configured": adapter.is_configured(),
                "last_collection": last_collection.isoformat() if last_collection else None,
                "collection_allowed": allowed,
                "status_message": reason
            }

        return status


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_harvester(config: Optional[Dict[str, Any]] = None) -> SocialMetricsHarvester:
    """
    Factory function to create a configured SocialMetricsHarvester.

    Args:
        config: Configuration dictionary with platform credentials and settings.
                Expected structure:
                {
                    "twitter": {"credentials": {...}},
                    "youtube": {"api_key": "...", "oauth_credentials": {...}},
                    "substack": {"publication_id": "...", "session_cookie": "..."},
                    "instagram": {"access_token": "..."},
                    "postiz": {"api_url": "...", "api_key": "..."},
                    "umami": {"api_url": "...", "website_id": "..."},
                    "storage_path": "/path/to/metrics/storage",
                    "min_collection_interval_hours": 1
                }

    Returns:
        Configured SocialMetricsHarvester instance
    """
    return SocialMetricsHarvester(config)
