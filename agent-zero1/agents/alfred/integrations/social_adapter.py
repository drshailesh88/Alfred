"""
Social Media Adapters for Alfred

Integration adapters for Twitter/X, YouTube, and Instagram.
Provides unified interfaces for reading mentions, comments, metrics,
and managing rate limits across platforms.

MCP Servers Supported:
- twitter-mcp
- youtube-mcp
- instagram-mcp

Direct APIs:
- Twitter API v2
- YouTube Data API v3
- Instagram Graph API
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import asyncio
from abc import abstractmethod

from . import (
    BaseAdapter,
    AdapterError,
    RateLimitError,
    AuthenticationError,
    ConnectionMode,
    ConnectionStatus,
    MCPClient,
)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class Platform(Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"  # Future support


class EngagementType(Enum):
    """Types of social engagement."""
    LIKE = "like"
    COMMENT = "comment"
    REPLY = "reply"
    RETWEET = "retweet"
    QUOTE = "quote"
    SHARE = "share"
    MENTION = "mention"
    TAG = "tag"
    SAVE = "save"
    VIEW = "view"
    CLICK = "click"


class ContentType(Enum):
    """Types of social content."""
    POST = "post"
    TWEET = "tweet"
    THREAD = "thread"
    VIDEO = "video"
    SHORT = "short"          # YouTube Shorts
    STORY = "story"
    REEL = "reel"
    CAROUSEL = "carousel"
    POLL = "poll"
    LIVE = "live"


class SentimentType(Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SocialUser:
    """Represents a social media user/account."""
    user_id: str
    username: str
    platform: Platform
    display_name: Optional[str] = None
    profile_url: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None

    # Metrics
    followers_count: int = 0
    following_count: int = 0
    posts_count: int = 0

    # Status
    is_verified: bool = False
    is_business: bool = False
    is_private: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.user_id,
            "username": self.username,
            "platform": self.platform.value,
            "displayName": self.display_name,
            "profileUrl": self.profile_url,
            "avatarUrl": self.avatar_url,
            "followers": self.followers_count,
            "isVerified": self.is_verified
        }


@dataclass
class SocialPost:
    """Represents a social media post."""
    post_id: str
    platform: Platform
    content_type: ContentType
    author: SocialUser

    # Content
    text: str = ""
    url: Optional[str] = None
    media_urls: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)

    # Metadata
    created_at: Optional[datetime] = None
    language: Optional[str] = None

    # Engagement metrics
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    views_count: int = 0

    # Platform-specific
    retweets_count: int = 0     # Twitter
    quotes_count: int = 0       # Twitter
    replies_count: int = 0      # Twitter
    saves_count: int = 0        # Instagram

    # Thread/conversation context
    is_reply: bool = False
    reply_to_id: Optional[str] = None
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None

    @property
    def total_engagement(self) -> int:
        """Total engagement count."""
        return (
            self.likes_count +
            self.comments_count +
            self.shares_count +
            self.retweets_count +
            self.quotes_count +
            self.replies_count +
            self.saves_count
        )

    @property
    def engagement_rate(self) -> float:
        """Engagement rate as percentage of views."""
        if self.views_count == 0:
            return 0.0
        return (self.total_engagement / self.views_count) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.post_id,
            "platform": self.platform.value,
            "type": self.content_type.value,
            "author": self.author.to_dict(),
            "text": self.text,
            "url": self.url,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "metrics": {
                "likes": self.likes_count,
                "comments": self.comments_count,
                "shares": self.shares_count,
                "views": self.views_count,
                "totalEngagement": self.total_engagement,
                "engagementRate": round(self.engagement_rate, 2)
            },
            "hashtags": self.hashtags,
            "mentions": self.mentions
        }


@dataclass
class SocialComment:
    """Represents a comment on social media."""
    comment_id: str
    post_id: str
    platform: Platform
    author: SocialUser

    # Content
    text: str = ""
    created_at: Optional[datetime] = None

    # Engagement
    likes_count: int = 0
    replies_count: int = 0

    # Context
    is_reply: bool = False
    parent_comment_id: Optional[str] = None

    # Analysis (populated by Alfred)
    sentiment: SentimentType = SentimentType.NEUTRAL
    is_question: bool = False
    is_actionable: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.comment_id,
            "postId": self.post_id,
            "platform": self.platform.value,
            "author": self.author.to_dict(),
            "text": self.text,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "likes": self.likes_count,
            "replies": self.replies_count,
            "sentiment": self.sentiment.value,
            "isQuestion": self.is_question,
            "isActionable": self.is_actionable
        }


@dataclass
class SocialMention:
    """Represents a mention of the user."""
    mention_id: str
    platform: Platform
    post: SocialPost
    mention_type: EngagementType

    # Context
    snippet: str = ""
    mentioned_at: Optional[datetime] = None

    # Analysis
    sentiment: SentimentType = SentimentType.NEUTRAL
    requires_response: bool = False
    response_urgency: str = "low"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.mention_id,
            "platform": self.platform.value,
            "post": self.post.to_dict(),
            "type": self.mention_type.value,
            "snippet": self.snippet,
            "mentionedAt": self.mentioned_at.isoformat() if self.mentioned_at else None,
            "sentiment": self.sentiment.value,
            "requiresResponse": self.requires_response,
            "urgency": self.response_urgency
        }


@dataclass
class SocialMetrics:
    """Aggregated social media metrics."""
    platform: Platform
    period_start: datetime
    period_end: datetime

    # Account metrics
    followers_count: int = 0
    followers_gained: int = 0
    followers_lost: int = 0
    following_count: int = 0

    # Content metrics
    posts_count: int = 0
    total_impressions: int = 0
    total_reach: int = 0

    # Engagement metrics
    total_likes: int = 0
    total_comments: int = 0
    total_shares: int = 0
    total_saves: int = 0
    total_clicks: int = 0

    # Calculated metrics
    engagement_rate: float = 0.0
    avg_engagement_per_post: float = 0.0

    # Top performing content
    top_posts: List[SocialPost] = field(default_factory=list)

    # Audience insights
    top_hashtags: List[Tuple[str, int]] = field(default_factory=list)
    peak_hours: List[int] = field(default_factory=list)  # Hours in UTC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            },
            "account": {
                "followers": self.followers_count,
                "followersGained": self.followers_gained,
                "followersLost": self.followers_lost,
                "following": self.following_count
            },
            "content": {
                "posts": self.posts_count,
                "impressions": self.total_impressions,
                "reach": self.total_reach
            },
            "engagement": {
                "likes": self.total_likes,
                "comments": self.total_comments,
                "shares": self.total_shares,
                "saves": self.total_saves,
                "clicks": self.total_clicks,
                "rate": round(self.engagement_rate, 2),
                "avgPerPost": round(self.avg_engagement_per_post, 2)
            },
            "topPosts": [p.to_dict() for p in self.top_posts[:5]],
            "topHashtags": self.top_hashtags[:10],
            "peakHours": self.peak_hours
        }


@dataclass
class RateLimitInfo:
    """Rate limit information for API calls."""
    platform: Platform
    endpoint: str
    limit: int
    remaining: int
    reset_at: datetime

    @property
    def is_limited(self) -> bool:
        """Check if currently rate limited."""
        return self.remaining <= 0 and datetime.now() < self.reset_at

    @property
    def seconds_until_reset(self) -> int:
        """Seconds until rate limit resets."""
        if datetime.now() >= self.reset_at:
            return 0
        return int((self.reset_at - datetime.now()).total_seconds())


# =============================================================================
# BASE SOCIAL ADAPTER
# =============================================================================

class BaseSocialAdapter(BaseAdapter):
    """
    Base class for social media adapters.

    Provides common functionality for:
    - Rate limit tracking
    - Mention retrieval
    - Comment retrieval
    - Metrics aggregation
    """

    def __init__(
        self,
        platform: Platform,
        adapter_name: str,
        mcp_client: Optional[MCPClient] = None,
        mcp_server_name: Optional[str] = None,
        api_credentials: Optional[Dict[str, Any]] = None,
        enable_mock: bool = False
    ):
        super().__init__(
            adapter_name=adapter_name,
            mcp_client=mcp_client,
            mcp_server_name=mcp_server_name,
            api_credentials=api_credentials,
            enable_mock=enable_mock
        )
        self.platform = platform
        self._rate_limits: Dict[str, RateLimitInfo] = {}

    def track_rate_limit(
        self,
        endpoint: str,
        limit: int,
        remaining: int,
        reset_at: datetime
    ) -> None:
        """Track rate limit info for an endpoint."""
        self._rate_limits[endpoint] = RateLimitInfo(
            platform=self.platform,
            endpoint=endpoint,
            limit=limit,
            remaining=remaining,
            reset_at=reset_at
        )
        self._update_rate_limit(remaining, reset_at)

    def check_endpoint_limit(self, endpoint: str) -> None:
        """Check if an endpoint is rate limited."""
        if endpoint in self._rate_limits:
            info = self._rate_limits[endpoint]
            if info.is_limited:
                raise RateLimitError(
                    self.adapter_name,
                    retry_after=info.seconds_until_reset,
                    limit_type=endpoint
                )

    @abstractmethod
    async def get_mentions(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SocialMention]:
        """Get mentions of the authenticated user."""
        pass

    @abstractmethod
    async def get_comments(
        self,
        post_id: str,
        limit: int = 100
    ) -> List[SocialComment]:
        """Get comments on a specific post."""
        pass

    @abstractmethod
    async def get_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """Get aggregated metrics for a time period."""
        pass


# =============================================================================
# TWITTER ADAPTER
# =============================================================================

class TwitterAdapter(BaseSocialAdapter):
    """
    Twitter/X adapter with MCP and direct API support.

    Provides:
    - Mention retrieval
    - Comment/reply retrieval
    - Tweet metrics
    - User timeline access
    - Rate limit management
    """

    MCP_TWITTER = "twitter-mcp"

    # Rate limit defaults (per 15-minute window)
    RATE_LIMITS = {
        "mentions": {"limit": 180, "window": 15},
        "tweets": {"limit": 900, "window": 15},
        "users": {"limit": 900, "window": 15},
        "search": {"limit": 180, "window": 15},
    }

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        api_credentials: Optional[Dict[str, Any]] = None,
        enable_mock: bool = False
    ):
        """
        Initialize Twitter adapter.

        Args:
            mcp_client: MCP client for server communication
            api_credentials: Twitter API credentials (bearer_token, api_key, etc.)
            enable_mock: Enable mock mode for testing
        """
        mcp_server = None
        if mcp_client and mcp_client.is_server_available(self.MCP_TWITTER):
            mcp_server = self.MCP_TWITTER

        super().__init__(
            platform=Platform.TWITTER,
            adapter_name="TwitterAdapter",
            mcp_client=mcp_client,
            mcp_server_name=mcp_server,
            api_credentials=api_credentials,
            enable_mock=enable_mock
        )

        self._user_id: Optional[str] = None
        self._username: Optional[str] = None

    async def _connect_api(self) -> None:
        """Establish direct Twitter API connection."""
        if not self.api_credentials:
            raise AuthenticationError(
                self.adapter_name,
                "No API credentials provided"
            )

        bearer_token = self.api_credentials.get("bearer_token")
        if not bearer_token:
            raise AuthenticationError(
                self.adapter_name,
                "Missing bearer_token in credentials"
            )

    async def get_mentions(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SocialMention]:
        """
        Get mentions of the authenticated user.

        Args:
            since: Get mentions since this time
            limit: Maximum number of mentions

        Returns:
            List of SocialMention objects
        """
        self.check_endpoint_limit("mentions")

        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_mentions_mcp(since, limit)
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_mentions_api(since, limit)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_mentions_mock()
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _get_mentions_mcp(
        self,
        since: Optional[datetime],
        limit: int
    ) -> List[SocialMention]:
        """Get mentions via MCP server."""
        args: Dict[str, Any] = {"max_results": min(limit, 100)}
        if since:
            args["start_time"] = since.isoformat() + "Z"

        try:
            result = await self._call_mcp("get_mentions", args)

            # Track rate limits from response
            if "rate_limit" in result:
                self.track_rate_limit(
                    "mentions",
                    result["rate_limit"]["limit"],
                    result["rate_limit"]["remaining"],
                    datetime.fromisoformat(result["rate_limit"]["reset"])
                )

            mentions = []
            for data in result.get("data", []):
                author = SocialUser(
                    user_id=data.get("author_id", ""),
                    username=data.get("author", {}).get("username", "unknown"),
                    platform=Platform.TWITTER,
                    display_name=data.get("author", {}).get("name")
                )

                post = SocialPost(
                    post_id=data.get("id", ""),
                    platform=Platform.TWITTER,
                    content_type=ContentType.TWEET,
                    author=author,
                    text=data.get("text", ""),
                    created_at=datetime.fromisoformat(
                        data.get("created_at", "").replace("Z", "+00:00")
                    ) if data.get("created_at") else None,
                    likes_count=data.get("public_metrics", {}).get("like_count", 0),
                    retweets_count=data.get("public_metrics", {}).get("retweet_count", 0),
                    replies_count=data.get("public_metrics", {}).get("reply_count", 0)
                )

                mention = SocialMention(
                    mention_id=data.get("id", ""),
                    platform=Platform.TWITTER,
                    post=post,
                    mention_type=EngagementType.MENTION,
                    snippet=data.get("text", "")[:100],
                    mentioned_at=post.created_at
                )
                mentions.append(mention)

            return mentions
        except Exception as e:
            raise AdapterError(
                f"Failed to get mentions: {str(e)}",
                self.adapter_name,
                "GET_MENTIONS_FAILED",
                original_error=e
            )

    async def _get_mentions_api(
        self,
        since: Optional[datetime],
        limit: int
    ) -> List[SocialMention]:
        """Get mentions via direct API."""
        # Placeholder for Twitter API v2 implementation
        return []

    def _get_mentions_mock(self) -> List[SocialMention]:
        """Get mock mentions for testing."""
        now = datetime.now()
        mock_author = SocialUser(
            user_id="12345",
            username="mockuser",
            platform=Platform.TWITTER,
            display_name="Mock User",
            followers_count=1000
        )

        mock_post = SocialPost(
            post_id="tweet_123",
            platform=Platform.TWITTER,
            content_type=ContentType.TWEET,
            author=mock_author,
            text="@yourhandle Great article on AI governance!",
            created_at=now - timedelta(hours=2),
            likes_count=15,
            retweets_count=3,
            replies_count=1
        )

        return [
            SocialMention(
                mention_id="mention_1",
                platform=Platform.TWITTER,
                post=mock_post,
                mention_type=EngagementType.MENTION,
                snippet="@yourhandle Great article on AI governance!",
                mentioned_at=now - timedelta(hours=2),
                sentiment=SentimentType.POSITIVE
            )
        ]

    async def get_comments(
        self,
        post_id: str,
        limit: int = 100
    ) -> List[SocialComment]:
        """
        Get replies/comments on a tweet.

        Args:
            post_id: Tweet ID
            limit: Maximum number of comments

        Returns:
            List of SocialComment objects (replies)
        """
        self.check_endpoint_limit("search")

        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_comments_mcp(post_id, limit)
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_comments_api(post_id, limit)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_comments_mock(post_id)
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _get_comments_mcp(
        self,
        post_id: str,
        limit: int
    ) -> List[SocialComment]:
        """Get comments via MCP server."""
        try:
            # Search for replies to the tweet
            result = await self._call_mcp("search_tweets", {
                "query": f"conversation_id:{post_id}",
                "max_results": min(limit, 100)
            })

            comments = []
            for data in result.get("data", []):
                author = SocialUser(
                    user_id=data.get("author_id", ""),
                    username=data.get("author", {}).get("username", "unknown"),
                    platform=Platform.TWITTER
                )

                comment = SocialComment(
                    comment_id=data.get("id", ""),
                    post_id=post_id,
                    platform=Platform.TWITTER,
                    author=author,
                    text=data.get("text", ""),
                    created_at=datetime.fromisoformat(
                        data.get("created_at", "").replace("Z", "+00:00")
                    ) if data.get("created_at") else None,
                    likes_count=data.get("public_metrics", {}).get("like_count", 0),
                    replies_count=data.get("public_metrics", {}).get("reply_count", 0),
                    is_reply=True,
                    parent_comment_id=data.get("referenced_tweets", [{}])[0].get("id")
                )
                comments.append(comment)

            return comments
        except Exception as e:
            raise AdapterError(
                f"Failed to get comments: {str(e)}",
                self.adapter_name,
                "GET_COMMENTS_FAILED",
                original_error=e
            )

    async def _get_comments_api(
        self,
        post_id: str,
        limit: int
    ) -> List[SocialComment]:
        """Get comments via direct API."""
        return []

    def _get_comments_mock(self, post_id: str) -> List[SocialComment]:
        """Get mock comments for testing."""
        now = datetime.now()
        mock_author = SocialUser(
            user_id="67890",
            username="replier",
            platform=Platform.TWITTER
        )

        return [
            SocialComment(
                comment_id="reply_1",
                post_id=post_id,
                platform=Platform.TWITTER,
                author=mock_author,
                text="This is a really insightful point!",
                created_at=now - timedelta(hours=1),
                likes_count=5,
                replies_count=0,
                is_reply=True,
                sentiment=SentimentType.POSITIVE
            )
        ]

    async def get_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """
        Get aggregated Twitter metrics.

        Args:
            start_date: Start of metrics period
            end_date: End of metrics period

        Returns:
            SocialMetrics for the period
        """
        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_metrics_mcp(start_date, end_date)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_metrics_mock(start_date, end_date)
        else:
            return SocialMetrics(
                platform=Platform.TWITTER,
                period_start=start_date,
                period_end=end_date
            )

    async def _get_metrics_mcp(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """Get metrics via MCP server."""
        try:
            result = await self._call_mcp("get_user_metrics", {
                "start_time": start_date.isoformat() + "Z",
                "end_time": end_date.isoformat() + "Z"
            })

            metrics = result.get("data", {})

            return SocialMetrics(
                platform=Platform.TWITTER,
                period_start=start_date,
                period_end=end_date,
                followers_count=metrics.get("followers_count", 0),
                followers_gained=metrics.get("followers_gained", 0),
                posts_count=metrics.get("tweet_count", 0),
                total_impressions=metrics.get("impressions", 0),
                total_likes=metrics.get("likes", 0),
                total_comments=metrics.get("replies", 0),
                total_shares=metrics.get("retweets", 0),
                engagement_rate=metrics.get("engagement_rate", 0.0)
            )
        except Exception:
            return SocialMetrics(
                platform=Platform.TWITTER,
                period_start=start_date,
                period_end=end_date
            )

    def _get_metrics_mock(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """Get mock metrics for testing."""
        return SocialMetrics(
            platform=Platform.TWITTER,
            period_start=start_date,
            period_end=end_date,
            followers_count=15000,
            followers_gained=250,
            followers_lost=50,
            posts_count=45,
            total_impressions=125000,
            total_reach=45000,
            total_likes=3200,
            total_comments=180,
            total_shares=420,
            engagement_rate=3.2,
            avg_engagement_per_post=84.4,
            peak_hours=[9, 12, 17, 20]
        )

    async def search_tweets(
        self,
        query: str,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[SocialPost]:
        """
        Search for tweets.

        Args:
            query: Search query
            limit: Maximum results
            since: Search since this time

        Returns:
            List of matching tweets
        """
        self.check_endpoint_limit("search")

        if self.connection_mode == ConnectionMode.MCP:
            try:
                args: Dict[str, Any] = {
                    "query": query,
                    "max_results": min(limit, 100)
                }
                if since:
                    args["start_time"] = since.isoformat() + "Z"

                result = await self._call_mcp("search_tweets", args)

                posts = []
                for data in result.get("data", []):
                    author = SocialUser(
                        user_id=data.get("author_id", ""),
                        username=data.get("author", {}).get("username", "unknown"),
                        platform=Platform.TWITTER
                    )

                    post = SocialPost(
                        post_id=data.get("id", ""),
                        platform=Platform.TWITTER,
                        content_type=ContentType.TWEET,
                        author=author,
                        text=data.get("text", ""),
                        created_at=datetime.fromisoformat(
                            data.get("created_at", "").replace("Z", "+00:00")
                        ) if data.get("created_at") else None,
                        likes_count=data.get("public_metrics", {}).get("like_count", 0),
                        retweets_count=data.get("public_metrics", {}).get("retweet_count", 0)
                    )
                    posts.append(post)

                return posts
            except Exception as e:
                raise AdapterError(
                    f"Search failed: {str(e)}",
                    self.adapter_name,
                    "SEARCH_FAILED",
                    original_error=e
                )
        else:
            return []


# =============================================================================
# YOUTUBE ADAPTER
# =============================================================================

class YouTubeAdapter(BaseSocialAdapter):
    """
    YouTube adapter with MCP and direct API support.

    Provides:
    - Comment retrieval
    - Video metrics
    - Channel metrics
    - Mention detection in comments
    """

    MCP_YOUTUBE = "youtube-mcp"

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        api_credentials: Optional[Dict[str, Any]] = None,
        channel_id: Optional[str] = None,
        enable_mock: bool = False
    ):
        """
        Initialize YouTube adapter.

        Args:
            mcp_client: MCP client for server communication
            api_credentials: YouTube API credentials (api_key)
            channel_id: YouTube channel ID
            enable_mock: Enable mock mode for testing
        """
        mcp_server = None
        if mcp_client and mcp_client.is_server_available(self.MCP_YOUTUBE):
            mcp_server = self.MCP_YOUTUBE

        super().__init__(
            platform=Platform.YOUTUBE,
            adapter_name="YouTubeAdapter",
            mcp_client=mcp_client,
            mcp_server_name=mcp_server,
            api_credentials=api_credentials,
            enable_mock=enable_mock
        )

        self.channel_id = channel_id

    async def _connect_api(self) -> None:
        """Establish direct YouTube API connection."""
        if not self.api_credentials:
            raise AuthenticationError(
                self.adapter_name,
                "No API credentials provided"
            )

        api_key = self.api_credentials.get("api_key")
        if not api_key:
            raise AuthenticationError(
                self.adapter_name,
                "Missing api_key in credentials"
            )

    async def get_mentions(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SocialMention]:
        """
        Get mentions in video comments.

        Note: YouTube doesn't have native mentions like Twitter.
        This searches comments for channel name or configured keywords.

        Args:
            since: Get mentions since this time
            limit: Maximum number of mentions

        Returns:
            List of SocialMention objects
        """
        # Get recent videos and their comments
        if self.connection_mode == ConnectionMode.MOCK:
            return self._get_mentions_mock()

        # Would search comments across videos for mentions
        return []

    def _get_mentions_mock(self) -> List[SocialMention]:
        """Get mock mentions for testing."""
        now = datetime.now()
        mock_author = SocialUser(
            user_id="yt_user_123",
            username="YouTube Viewer",
            platform=Platform.YOUTUBE
        )

        mock_post = SocialPost(
            post_id="video_abc123",
            platform=Platform.YOUTUBE,
            content_type=ContentType.VIDEO,
            author=mock_author,
            text="Comment mentioning your channel",
            created_at=now - timedelta(hours=5),
            views_count=1500,
            likes_count=45,
            comments_count=12
        )

        return [
            SocialMention(
                mention_id="yt_mention_1",
                platform=Platform.YOUTUBE,
                post=mock_post,
                mention_type=EngagementType.COMMENT,
                snippet="I learned so much from @yourchannel!",
                mentioned_at=now - timedelta(hours=5),
                sentiment=SentimentType.POSITIVE
            )
        ]

    async def get_comments(
        self,
        post_id: str,
        limit: int = 100,
        include_replies: bool = True
    ) -> List[SocialComment]:
        """
        Get comments on a YouTube video.

        Args:
            post_id: Video ID
            limit: Maximum number of comments
            include_replies: Include reply threads

        Returns:
            List of SocialComment objects
        """
        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_comments_mcp(post_id, limit, include_replies)
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_comments_api(post_id, limit, include_replies)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_comments_mock(post_id)
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _get_comments_mcp(
        self,
        post_id: str,
        limit: int,
        include_replies: bool
    ) -> List[SocialComment]:
        """Get comments via MCP server."""
        try:
            result = await self._call_mcp("get_video_comments", {
                "videoId": post_id,
                "maxResults": min(limit, 100),
                "includeReplies": include_replies
            })

            comments = []
            for item in result.get("items", []):
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})

                author = SocialUser(
                    user_id=snippet.get("authorChannelId", {}).get("value", ""),
                    username=snippet.get("authorDisplayName", "Unknown"),
                    platform=Platform.YOUTUBE,
                    avatar_url=snippet.get("authorProfileImageUrl")
                )

                comment = SocialComment(
                    comment_id=item.get("id", ""),
                    post_id=post_id,
                    platform=Platform.YOUTUBE,
                    author=author,
                    text=snippet.get("textDisplay", ""),
                    created_at=datetime.fromisoformat(
                        snippet.get("publishedAt", "").replace("Z", "+00:00")
                    ) if snippet.get("publishedAt") else None,
                    likes_count=snippet.get("likeCount", 0),
                    replies_count=item.get("snippet", {}).get("totalReplyCount", 0)
                )
                comments.append(comment)

            return comments
        except Exception as e:
            raise AdapterError(
                f"Failed to get comments: {str(e)}",
                self.adapter_name,
                "GET_COMMENTS_FAILED",
                original_error=e
            )

    async def _get_comments_api(
        self,
        post_id: str,
        limit: int,
        include_replies: bool
    ) -> List[SocialComment]:
        """Get comments via direct API."""
        # Placeholder for YouTube Data API implementation
        return []

    def _get_comments_mock(self, post_id: str) -> List[SocialComment]:
        """Get mock comments for testing."""
        now = datetime.now()

        return [
            SocialComment(
                comment_id="yt_comment_1",
                post_id=post_id,
                platform=Platform.YOUTUBE,
                author=SocialUser(
                    user_id="viewer_1",
                    username="HealthEnthusiast",
                    platform=Platform.YOUTUBE
                ),
                text="This video changed my perspective on nutrition!",
                created_at=now - timedelta(hours=3),
                likes_count=25,
                replies_count=2,
                sentiment=SentimentType.POSITIVE
            ),
            SocialComment(
                comment_id="yt_comment_2",
                post_id=post_id,
                platform=Platform.YOUTUBE,
                author=SocialUser(
                    user_id="viewer_2",
                    username="SkepticalSam",
                    platform=Platform.YOUTUBE
                ),
                text="Can you share the studies you mentioned?",
                created_at=now - timedelta(hours=6),
                likes_count=12,
                replies_count=1,
                is_question=True
            ),
        ]

    async def get_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """
        Get aggregated YouTube metrics.

        Args:
            start_date: Start of metrics period
            end_date: End of metrics period

        Returns:
            SocialMetrics for the period
        """
        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_metrics_mcp(start_date, end_date)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_metrics_mock(start_date, end_date)
        else:
            return SocialMetrics(
                platform=Platform.YOUTUBE,
                period_start=start_date,
                period_end=end_date
            )

    async def _get_metrics_mcp(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """Get metrics via MCP server."""
        try:
            result = await self._call_mcp("get_channel_analytics", {
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d")
            })

            data = result.get("data", {})

            return SocialMetrics(
                platform=Platform.YOUTUBE,
                period_start=start_date,
                period_end=end_date,
                followers_count=data.get("subscriberCount", 0),
                followers_gained=data.get("subscribersGained", 0),
                followers_lost=data.get("subscribersLost", 0),
                posts_count=data.get("videoCount", 0),
                total_impressions=data.get("impressions", 0),
                total_reach=data.get("uniqueViewers", 0),
                total_likes=data.get("likes", 0),
                total_comments=data.get("comments", 0),
                total_shares=data.get("shares", 0),
                engagement_rate=data.get("engagementRate", 0.0)
            )
        except Exception:
            return SocialMetrics(
                platform=Platform.YOUTUBE,
                period_start=start_date,
                period_end=end_date
            )

    def _get_metrics_mock(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """Get mock metrics for testing."""
        return SocialMetrics(
            platform=Platform.YOUTUBE,
            period_start=start_date,
            period_end=end_date,
            followers_count=50000,
            followers_gained=1200,
            followers_lost=150,
            posts_count=8,
            total_impressions=450000,
            total_reach=180000,
            total_likes=15000,
            total_comments=850,
            total_shares=320,
            engagement_rate=3.5,
            avg_engagement_per_post=2021.3,
            peak_hours=[14, 18, 20, 21]
        )

    async def get_video_stats(self, video_id: str) -> Optional[SocialPost]:
        """
        Get statistics for a specific video.

        Args:
            video_id: YouTube video ID

        Returns:
            SocialPost with video statistics
        """
        if self.connection_mode == ConnectionMode.MCP:
            try:
                result = await self._call_mcp("get_video", {
                    "videoId": video_id,
                    "part": "snippet,statistics"
                })

                item = result.get("items", [{}])[0]
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})

                return SocialPost(
                    post_id=video_id,
                    platform=Platform.YOUTUBE,
                    content_type=ContentType.VIDEO,
                    author=SocialUser(
                        user_id=snippet.get("channelId", ""),
                        username=snippet.get("channelTitle", ""),
                        platform=Platform.YOUTUBE
                    ),
                    text=snippet.get("title", ""),
                    url=f"https://youtube.com/watch?v={video_id}",
                    created_at=datetime.fromisoformat(
                        snippet.get("publishedAt", "").replace("Z", "+00:00")
                    ) if snippet.get("publishedAt") else None,
                    views_count=int(stats.get("viewCount", 0)),
                    likes_count=int(stats.get("likeCount", 0)),
                    comments_count=int(stats.get("commentCount", 0))
                )
            except Exception:
                return None
        elif self.connection_mode == ConnectionMode.MOCK:
            return SocialPost(
                post_id=video_id,
                platform=Platform.YOUTUBE,
                content_type=ContentType.VIDEO,
                author=SocialUser(
                    user_id="channel_123",
                    username="Your Channel",
                    platform=Platform.YOUTUBE
                ),
                text="Sample Video Title",
                views_count=25000,
                likes_count=1200,
                comments_count=85
            )
        return None


# =============================================================================
# INSTAGRAM ADAPTER
# =============================================================================

class InstagramAdapter(BaseSocialAdapter):
    """
    Instagram adapter with MCP and direct API support.

    Provides:
    - Comment retrieval
    - Post metrics
    - Story/Reel metrics
    - Mention detection
    """

    MCP_INSTAGRAM = "instagram-mcp"

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        api_credentials: Optional[Dict[str, Any]] = None,
        instagram_account_id: Optional[str] = None,
        enable_mock: bool = False
    ):
        """
        Initialize Instagram adapter.

        Args:
            mcp_client: MCP client for server communication
            api_credentials: Instagram Graph API credentials
            instagram_account_id: Instagram Business Account ID
            enable_mock: Enable mock mode for testing
        """
        mcp_server = None
        if mcp_client and mcp_client.is_server_available(self.MCP_INSTAGRAM):
            mcp_server = self.MCP_INSTAGRAM

        super().__init__(
            platform=Platform.INSTAGRAM,
            adapter_name="InstagramAdapter",
            mcp_client=mcp_client,
            mcp_server_name=mcp_server,
            api_credentials=api_credentials,
            enable_mock=enable_mock
        )

        self.instagram_account_id = instagram_account_id

    async def _connect_api(self) -> None:
        """Establish direct Instagram API connection."""
        if not self.api_credentials:
            raise AuthenticationError(
                self.adapter_name,
                "No API credentials provided"
            )

        access_token = self.api_credentials.get("access_token")
        if not access_token:
            raise AuthenticationError(
                self.adapter_name,
                "Missing access_token in credentials"
            )

    async def get_mentions(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SocialMention]:
        """
        Get mentions (tags and @mentions in comments).

        Args:
            since: Get mentions since this time
            limit: Maximum number of mentions

        Returns:
            List of SocialMention objects
        """
        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_mentions_mcp(since, limit)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_mentions_mock()
        return []

    async def _get_mentions_mcp(
        self,
        since: Optional[datetime],
        limit: int
    ) -> List[SocialMention]:
        """Get mentions via MCP server."""
        try:
            result = await self._call_mcp("get_mentioned_media", {
                "limit": limit
            })

            mentions = []
            for item in result.get("data", []):
                author = SocialUser(
                    user_id=item.get("owner", {}).get("id", ""),
                    username=item.get("owner", {}).get("username", ""),
                    platform=Platform.INSTAGRAM
                )

                content_type = ContentType.POST
                if item.get("media_type") == "VIDEO":
                    content_type = ContentType.VIDEO
                elif item.get("media_type") == "CAROUSEL_ALBUM":
                    content_type = ContentType.CAROUSEL

                post = SocialPost(
                    post_id=item.get("id", ""),
                    platform=Platform.INSTAGRAM,
                    content_type=content_type,
                    author=author,
                    text=item.get("caption", ""),
                    url=item.get("permalink"),
                    likes_count=item.get("like_count", 0),
                    comments_count=item.get("comments_count", 0)
                )

                mention = SocialMention(
                    mention_id=item.get("id", ""),
                    platform=Platform.INSTAGRAM,
                    post=post,
                    mention_type=EngagementType.TAG,
                    snippet=item.get("caption", "")[:100],
                    mentioned_at=datetime.fromisoformat(
                        item.get("timestamp", "").replace("Z", "+00:00")
                    ) if item.get("timestamp") else None
                )
                mentions.append(mention)

            return mentions
        except Exception as e:
            raise AdapterError(
                f"Failed to get mentions: {str(e)}",
                self.adapter_name,
                "GET_MENTIONS_FAILED",
                original_error=e
            )

    def _get_mentions_mock(self) -> List[SocialMention]:
        """Get mock mentions for testing."""
        now = datetime.now()
        mock_author = SocialUser(
            user_id="ig_user_456",
            username="healthfan",
            platform=Platform.INSTAGRAM,
            followers_count=5000
        )

        mock_post = SocialPost(
            post_id="ig_post_123",
            platform=Platform.INSTAGRAM,
            content_type=ContentType.POST,
            author=mock_author,
            text="Thanks @yourhandle for the great health tips!",
            created_at=now - timedelta(hours=4),
            likes_count=320,
            comments_count=18
        )

        return [
            SocialMention(
                mention_id="ig_mention_1",
                platform=Platform.INSTAGRAM,
                post=mock_post,
                mention_type=EngagementType.TAG,
                snippet="Thanks @yourhandle for the great health tips!",
                mentioned_at=now - timedelta(hours=4),
                sentiment=SentimentType.POSITIVE
            )
        ]

    async def get_comments(
        self,
        post_id: str,
        limit: int = 100
    ) -> List[SocialComment]:
        """
        Get comments on an Instagram post.

        Args:
            post_id: Media ID
            limit: Maximum number of comments

        Returns:
            List of SocialComment objects
        """
        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_comments_mcp(post_id, limit)
        elif self.connection_mode == ConnectionMode.API:
            return await self._get_comments_api(post_id, limit)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_comments_mock(post_id)
        else:
            raise AdapterError("Not connected", self.adapter_name, "NOT_CONNECTED")

    async def _get_comments_mcp(
        self,
        post_id: str,
        limit: int
    ) -> List[SocialComment]:
        """Get comments via MCP server."""
        try:
            result = await self._call_mcp("get_media_comments", {
                "mediaId": post_id,
                "limit": limit
            })

            comments = []
            for item in result.get("data", []):
                author = SocialUser(
                    user_id=item.get("from", {}).get("id", ""),
                    username=item.get("from", {}).get("username", ""),
                    platform=Platform.INSTAGRAM
                )

                comment = SocialComment(
                    comment_id=item.get("id", ""),
                    post_id=post_id,
                    platform=Platform.INSTAGRAM,
                    author=author,
                    text=item.get("text", ""),
                    created_at=datetime.fromisoformat(
                        item.get("timestamp", "").replace("Z", "+00:00")
                    ) if item.get("timestamp") else None,
                    likes_count=item.get("like_count", 0),
                    replies_count=len(item.get("replies", {}).get("data", []))
                )
                comments.append(comment)

            return comments
        except Exception as e:
            raise AdapterError(
                f"Failed to get comments: {str(e)}",
                self.adapter_name,
                "GET_COMMENTS_FAILED",
                original_error=e
            )

    async def _get_comments_api(
        self,
        post_id: str,
        limit: int
    ) -> List[SocialComment]:
        """Get comments via direct API."""
        return []

    def _get_comments_mock(self, post_id: str) -> List[SocialComment]:
        """Get mock comments for testing."""
        now = datetime.now()

        return [
            SocialComment(
                comment_id="ig_comment_1",
                post_id=post_id,
                platform=Platform.INSTAGRAM,
                author=SocialUser(
                    user_id="follower_1",
                    username="wellness_warrior",
                    platform=Platform.INSTAGRAM
                ),
                text="Love this content! So helpful!",
                created_at=now - timedelta(hours=2),
                likes_count=8,
                replies_count=1,
                sentiment=SentimentType.POSITIVE
            ),
            SocialComment(
                comment_id="ig_comment_2",
                post_id=post_id,
                platform=Platform.INSTAGRAM,
                author=SocialUser(
                    user_id="follower_2",
                    username="curious_cat",
                    platform=Platform.INSTAGRAM
                ),
                text="What time do you recommend taking this supplement?",
                created_at=now - timedelta(hours=4),
                likes_count=3,
                replies_count=0,
                is_question=True
            ),
        ]

    async def get_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """
        Get aggregated Instagram metrics.

        Args:
            start_date: Start of metrics period
            end_date: End of metrics period

        Returns:
            SocialMetrics for the period
        """
        if self.connection_mode == ConnectionMode.MCP:
            return await self._get_metrics_mcp(start_date, end_date)
        elif self.connection_mode == ConnectionMode.MOCK:
            return self._get_metrics_mock(start_date, end_date)
        else:
            return SocialMetrics(
                platform=Platform.INSTAGRAM,
                period_start=start_date,
                period_end=end_date
            )

    async def _get_metrics_mcp(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """Get metrics via MCP server."""
        try:
            result = await self._call_mcp("get_account_insights", {
                "since": int(start_date.timestamp()),
                "until": int(end_date.timestamp()),
                "metric": "impressions,reach,follower_count,profile_views"
            })

            data = {}
            for item in result.get("data", []):
                name = item.get("name")
                values = item.get("values", [{}])
                if values:
                    data[name] = values[0].get("value", 0)

            return SocialMetrics(
                platform=Platform.INSTAGRAM,
                period_start=start_date,
                period_end=end_date,
                followers_count=data.get("follower_count", 0),
                total_impressions=data.get("impressions", 0),
                total_reach=data.get("reach", 0)
            )
        except Exception:
            return SocialMetrics(
                platform=Platform.INSTAGRAM,
                period_start=start_date,
                period_end=end_date
            )

    def _get_metrics_mock(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SocialMetrics:
        """Get mock metrics for testing."""
        return SocialMetrics(
            platform=Platform.INSTAGRAM,
            period_start=start_date,
            period_end=end_date,
            followers_count=25000,
            followers_gained=800,
            followers_lost=100,
            posts_count=15,
            total_impressions=320000,
            total_reach=95000,
            total_likes=18500,
            total_comments=650,
            total_saves=1200,
            engagement_rate=6.8,
            avg_engagement_per_post=1356.7,
            peak_hours=[8, 12, 19, 21]
        )

    async def get_post_insights(
        self,
        post_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get insights for a specific post.

        Args:
            post_id: Media ID

        Returns:
            Dictionary of insights metrics
        """
        if self.connection_mode == ConnectionMode.MCP:
            try:
                result = await self._call_mcp("get_media_insights", {
                    "mediaId": post_id,
                    "metric": "impressions,reach,engagement,saved"
                })

                insights = {}
                for item in result.get("data", []):
                    insights[item.get("name")] = item.get("values", [{}])[0].get("value", 0)

                return insights
            except Exception:
                return None
        elif self.connection_mode == ConnectionMode.MOCK:
            return {
                "impressions": 5200,
                "reach": 3800,
                "engagement": 420,
                "saved": 85
            }
        return None
