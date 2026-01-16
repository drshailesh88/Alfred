"""
YouTube Analytics Integration Adapter for ALFRED

Provides structured access to YouTube Data API v3 and YouTube Analytics API.
Connects ALFRED's Social Metrics Harvester and Content Strategy Analyst to YouTube.

Features:
- Channel statistics (subscribers, views, video count)
- Video performance metrics (views, likes, comments, watch time)
- Comment extraction and analysis support
- Rate limiting and caching
- Comprehensive error handling

YouTube API Quota Notes:
- Default quota: 10,000 units/day
- search.list: 100 units
- videos.list: 1 unit
- channels.list: 1 unit
- commentThreads.list: 1 unit
- Analytics API: 1 unit per request
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import hashlib
import json
import logging
import time


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================

class YouTubeError(Exception):
    """Base exception for YouTube adapter errors."""
    pass


class AuthenticationError(YouTubeError):
    """Authentication or authorization failed."""
    pass


class QuotaExceededError(YouTubeError):
    """YouTube API quota has been exceeded."""
    pass


class RateLimitError(YouTubeError):
    """Rate limit exceeded, retry after cooldown."""
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class InvalidVideoError(YouTubeError):
    """Invalid or non-existent video ID."""
    pass


class InvalidChannelError(YouTubeError):
    """Invalid or non-existent channel ID."""
    pass


class OfflineError(YouTubeError):
    """Cannot connect to YouTube API."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class YouTubeChannelStats:
    """Channel-level statistics."""
    channel_id: str
    channel_title: str = ""
    subscriber_count: int = 0
    total_views: int = 0
    video_count: int = 0
    custom_url: str = ""
    description: str = ""
    thumbnail_url: str = ""
    collected_at: str = ""

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now().isoformat()


@dataclass
class YouTubeVideoStats:
    """Individual video statistics."""
    video_id: str
    title: str = ""
    description: str = ""
    published_at: str = ""
    channel_id: str = ""
    channel_title: str = ""
    views: int = 0
    likes: int = 0
    dislikes: int = 0  # No longer public but kept for historical data
    comments_count: int = 0
    favorites: int = 0
    watch_time_hours: float = 0.0
    avg_view_duration_seconds: int = 0
    engagement_rate: float = 0.0
    thumbnail_url: str = ""
    tags: List[str] = field(default_factory=list)
    category_id: str = ""
    duration_seconds: int = 0
    is_live: bool = False
    collected_at: str = ""

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now().isoformat()
        # Calculate engagement rate if we have views
        if self.views > 0:
            total_engagement = self.likes + self.comments_count
            self.engagement_rate = total_engagement / self.views


@dataclass
class YouTubeComment:
    """Individual comment data."""
    comment_id: str
    video_id: str
    author: str = ""
    author_channel_id: str = ""
    text: str = ""
    text_display: str = ""
    like_count: int = 0
    reply_count: int = 0
    published_at: str = ""
    updated_at: str = ""
    is_public: bool = True
    parent_id: str = ""  # For replies
    sentiment: str = ""  # neutral/positive/negative - set by analysis
    collected_at: str = ""

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now().isoformat()


@dataclass
class YouTubeAnalyticsReport:
    """Analytics report for a date range."""
    channel_id: str
    start_date: str
    end_date: str
    total_views: int = 0
    total_watch_time_minutes: int = 0
    avg_view_duration_seconds: int = 0
    total_subscribers_gained: int = 0
    total_subscribers_lost: int = 0
    net_subscriber_change: int = 0
    total_likes: int = 0
    total_dislikes: int = 0
    total_comments: int = 0
    total_shares: int = 0
    estimated_revenue: float = 0.0  # If monetized
    top_videos: List[Dict[str, Any]] = field(default_factory=list)
    traffic_sources: Dict[str, int] = field(default_factory=dict)
    demographics: Dict[str, Any] = field(default_factory=dict)
    devices: Dict[str, int] = field(default_factory=dict)
    collected_at: str = ""

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now().isoformat()
        self.net_subscriber_change = self.total_subscribers_gained - self.total_subscribers_lost


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    data: Any
    expires_at: float

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for YouTube API quota management.

    YouTube API has daily quota limits (default 10,000 units).
    Different operations cost different amounts:
    - channels.list: 1 unit
    - videos.list: 1 unit
    - search.list: 100 units
    - commentThreads.list: 1 unit
    """

    def __init__(self, daily_quota: int = 10000,
                 requests_per_second: float = 10.0):
        self.daily_quota = daily_quota
        self.requests_per_second = requests_per_second
        self.quota_used = 0
        self.quota_reset_time = self._next_reset_time()
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    def _next_reset_time(self) -> float:
        """Calculate next quota reset time (midnight Pacific Time)."""
        # YouTube quota resets at midnight Pacific Time
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        reset_time = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
        return reset_time.timestamp()

    async def acquire(self, cost: int = 1) -> bool:
        """
        Acquire quota for an API call.

        Args:
            cost: Quota units required for the operation

        Returns:
            True if quota acquired, raises QuotaExceededError if not available
        """
        async with self._lock:
            # Check if quota has reset
            current_time = time.time()
            if current_time > self.quota_reset_time:
                self.quota_used = 0
                self.quota_reset_time = self._next_reset_time()

            # Check quota availability
            if self.quota_used + cost > self.daily_quota:
                raise QuotaExceededError(
                    f"YouTube API daily quota exceeded. "
                    f"Used: {self.quota_used}/{self.daily_quota}. "
                    f"Resets at: {datetime.fromtimestamp(self.quota_reset_time).isoformat()}"
                )

            # Enforce rate limiting (requests per second)
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.requests_per_second
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)

            self.quota_used += cost
            self.last_request_time = time.time()
            return True

    def get_remaining_quota(self) -> int:
        """Get remaining daily quota."""
        if time.time() > self.quota_reset_time:
            return self.daily_quota
        return self.daily_quota - self.quota_used

    def get_quota_status(self) -> Dict[str, Any]:
        """Get detailed quota status."""
        return {
            "daily_quota": self.daily_quota,
            "quota_used": self.quota_used,
            "quota_remaining": self.get_remaining_quota(),
            "reset_time": datetime.fromtimestamp(self.quota_reset_time).isoformat(),
            "requests_per_second": self.requests_per_second
        }


# =============================================================================
# Response Cache
# =============================================================================

class ResponseCache:
    """
    Simple in-memory cache for API responses.

    Helps reduce API calls for frequently accessed data.
    """

    def __init__(self, default_ttl_seconds: int = 300):
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    async def get(self, *args) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        key = self._make_key(*args)
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                logger.debug(f"Cache hit for key: {key}")
                return entry.data
            elif entry:
                # Clean up expired entry
                del self._cache[key]
        return None

    async def set(self, value: Any, ttl_seconds: Optional[int] = None, *args):
        """Cache a value with optional custom TTL."""
        key = self._make_key(*args)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        async with self._lock:
            self._cache[key] = CacheEntry(
                data=value,
                expires_at=time.time() + ttl
            )
            logger.debug(f"Cached value for key: {key}, TTL: {ttl}s")

    async def invalidate(self, *args):
        """Invalidate a specific cache entry."""
        key = self._make_key(*args)
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self):
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self):
        """Remove all expired entries."""
        async with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


# =============================================================================
# YouTube Adapter
# =============================================================================

class YouTubeAdapter:
    """
    YouTube Analytics Integration Adapter for ALFRED.

    Provides structured access to YouTube Data API v3 and YouTube Analytics API.
    Designed to work with ALFRED's Social Metrics Harvester and Content Strategy Analyst.

    Usage:
        adapter = YouTubeAdapter(api_key="YOUR_API_KEY")

        # Get channel stats
        stats = await adapter.get_channel_stats()

        # Get video stats
        video = await adapter.get_video_stats("video_id")

        # Get comments for analysis
        comments = await adapter.get_video_comments("video_id", limit=100)
    """

    # API quota costs
    QUOTA_COSTS = {
        "channels.list": 1,
        "videos.list": 1,
        "search.list": 100,
        "commentThreads.list": 1,
        "comments.list": 1,
        "playlistItems.list": 1,
        "analytics.query": 1,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        channel_id: Optional[str] = None,
        oauth_credentials: Optional[Dict[str, str]] = None,
        daily_quota: int = 10000,
        cache_ttl_seconds: int = 300,
        enable_caching: bool = True,
        requests_per_second: float = 10.0
    ):
        """
        Initialize YouTube adapter.

        Args:
            api_key: YouTube Data API key (for public data)
            channel_id: Default channel ID to use for queries
            oauth_credentials: OAuth2 credentials dict for Analytics API
                             {"access_token": "...", "refresh_token": "...",
                              "client_id": "...", "client_secret": "..."}
            daily_quota: Daily API quota limit
            cache_ttl_seconds: Default cache TTL
            enable_caching: Whether to enable response caching
            requests_per_second: Rate limit for API requests
        """
        self.api_key = api_key
        self.channel_id = channel_id
        self.oauth_credentials = oauth_credentials
        self.enable_caching = enable_caching

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            daily_quota=daily_quota,
            requests_per_second=requests_per_second
        )

        # Initialize cache
        self.cache = ResponseCache(default_ttl_seconds=cache_ttl_seconds)

        # HTTP client placeholder (will be initialized on first use)
        self._http_client = None

        # Base URLs
        self.data_api_base = "https://www.googleapis.com/youtube/v3"
        self.analytics_api_base = "https://youtubeanalytics.googleapis.com/v2"

        logger.info(f"YouTubeAdapter initialized for channel: {channel_id}")

    # =========================================================================
    # HTTP Client Management
    # =========================================================================

    async def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import aiohttp
                self._http_client = aiohttp.ClientSession()
            except ImportError:
                raise ImportError(
                    "aiohttp is required for YouTubeAdapter. "
                    "Install it with: pip install aiohttp"
                )
        return self._http_client

    async def close(self):
        """Close HTTP client and cleanup resources."""
        if self._http_client:
            await self._http_client.close()
            self._http_client = None
        await self.cache.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Internal API Methods
    # =========================================================================

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        quota_cost: int = 1,
        cache_ttl: Optional[int] = None,
        use_oauth: bool = False
    ) -> Dict[str, Any]:
        """
        Make an API request with rate limiting and caching.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            quota_cost: Quota units for this request
            cache_ttl: Optional custom cache TTL
            use_oauth: Whether to use OAuth instead of API key

        Returns:
            API response as dictionary
        """
        # Check cache first
        if self.enable_caching:
            cached = await self.cache.get(endpoint, params)
            if cached is not None:
                return cached

        # Acquire rate limit quota
        await self.rate_limiter.acquire(quota_cost)

        # Build request
        if use_oauth:
            if not self.oauth_credentials:
                raise AuthenticationError(
                    "OAuth credentials required for this operation"
                )
            url = f"{self.analytics_api_base}/{endpoint}"
            headers = {
                "Authorization": f"Bearer {self.oauth_credentials.get('access_token', '')}"
            }
        else:
            if not self.api_key:
                raise AuthenticationError(
                    "API key required for this operation"
                )
            url = f"{self.data_api_base}/{endpoint}"
            params["key"] = self.api_key
            headers = {}

        # Make request
        try:
            client = await self._get_http_client()
            async with client.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Cache successful response
                    if self.enable_caching:
                        await self.cache.set(data, cache_ttl, endpoint, params)
                    return data
                elif response.status == 401:
                    raise AuthenticationError("Invalid API key or expired OAuth token")
                elif response.status == 403:
                    error_data = await response.json()
                    error_reason = error_data.get("error", {}).get("errors", [{}])[0].get("reason", "")
                    if error_reason == "quotaExceeded":
                        raise QuotaExceededError("YouTube API daily quota exceeded")
                    raise AuthenticationError(f"Access forbidden: {error_reason}")
                elif response.status == 404:
                    raise InvalidVideoError(f"Resource not found: {endpoint}")
                elif response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitError("Rate limit exceeded", retry_after)
                else:
                    error_text = await response.text()
                    raise YouTubeError(f"API error {response.status}: {error_text}")
        except Exception as e:
            if isinstance(e, YouTubeError):
                raise
            raise OfflineError(f"Failed to connect to YouTube API: {str(e)}")

    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds."""
        # Format: PT#H#M#S
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    # =========================================================================
    # Public API Methods
    # =========================================================================

    async def get_channel_stats(
        self,
        channel_id: Optional[str] = None
    ) -> YouTubeChannelStats:
        """
        Get channel statistics.

        Args:
            channel_id: Channel ID (uses default if not provided)

        Returns:
            YouTubeChannelStats with subscriber count, total views, video count
        """
        channel_id = channel_id or self.channel_id
        if not channel_id:
            raise InvalidChannelError("Channel ID required")

        params = {
            "part": "statistics,snippet,brandingSettings",
            "id": channel_id
        }

        data = await self._make_request(
            "channels",
            params,
            quota_cost=self.QUOTA_COSTS["channels.list"],
            cache_ttl=3600  # Cache for 1 hour
        )

        items = data.get("items", [])
        if not items:
            raise InvalidChannelError(f"Channel not found: {channel_id}")

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        return YouTubeChannelStats(
            channel_id=channel_id,
            channel_title=snippet.get("title", ""),
            subscriber_count=int(stats.get("subscriberCount", 0)),
            total_views=int(stats.get("viewCount", 0)),
            video_count=int(stats.get("videoCount", 0)),
            custom_url=snippet.get("customUrl", ""),
            description=snippet.get("description", "")[:500],
            thumbnail_url=snippet.get("thumbnails", {}).get("default", {}).get("url", "")
        )

    async def get_video_stats(
        self,
        video_id: str
    ) -> YouTubeVideoStats:
        """
        Get statistics for a specific video.

        Args:
            video_id: YouTube video ID

        Returns:
            YouTubeVideoStats with views, likes, comments, watch time, etc.
        """
        if not video_id:
            raise InvalidVideoError("Video ID required")

        params = {
            "part": "statistics,snippet,contentDetails,liveStreamingDetails",
            "id": video_id
        }

        data = await self._make_request(
            "videos",
            params,
            quota_cost=self.QUOTA_COSTS["videos.list"],
            cache_ttl=300  # Cache for 5 minutes
        )

        items = data.get("items", [])
        if not items:
            raise InvalidVideoError(f"Video not found: {video_id}")

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})
        live = item.get("liveStreamingDetails", {})

        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))
        duration_seconds = self._parse_duration(content.get("duration", "PT0S"))

        return YouTubeVideoStats(
            video_id=video_id,
            title=snippet.get("title", ""),
            description=snippet.get("description", "")[:500],
            published_at=snippet.get("publishedAt", ""),
            channel_id=snippet.get("channelId", ""),
            channel_title=snippet.get("channelTitle", ""),
            views=views,
            likes=likes,
            comments_count=comments,
            duration_seconds=duration_seconds,
            thumbnail_url=snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            tags=snippet.get("tags", [])[:10],
            category_id=snippet.get("categoryId", ""),
            is_live=bool(live)
        )

    async def get_recent_videos(
        self,
        limit: int = 10,
        channel_id: Optional[str] = None,
        include_stats: bool = True
    ) -> List[YouTubeVideoStats]:
        """
        Get recent video uploads with statistics.

        Args:
            limit: Maximum number of videos to return (max 50)
            channel_id: Channel ID (uses default if not provided)
            include_stats: Whether to fetch full stats for each video

        Returns:
            List of YouTubeVideoStats for recent videos
        """
        channel_id = channel_id or self.channel_id
        if not channel_id:
            raise InvalidChannelError("Channel ID required")

        limit = min(limit, 50)  # YouTube API max

        # First, get the uploads playlist ID
        params = {
            "part": "contentDetails",
            "id": channel_id
        }

        channel_data = await self._make_request(
            "channels",
            params,
            quota_cost=self.QUOTA_COSTS["channels.list"]
        )

        items = channel_data.get("items", [])
        if not items:
            raise InvalidChannelError(f"Channel not found: {channel_id}")

        uploads_playlist_id = (
            items[0]
            .get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads", "")
        )

        if not uploads_playlist_id:
            return []

        # Get videos from uploads playlist
        params = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": limit
        }

        playlist_data = await self._make_request(
            "playlistItems",
            params,
            quota_cost=self.QUOTA_COSTS["playlistItems.list"]
        )

        videos = []
        video_ids = []

        for item in playlist_data.get("items", []):
            video_id = item.get("contentDetails", {}).get("videoId", "")
            if video_id:
                video_ids.append(video_id)

        if include_stats and video_ids:
            # Batch fetch video stats (up to 50 per request)
            for i in range(0, len(video_ids), 50):
                batch_ids = video_ids[i:i+50]
                params = {
                    "part": "statistics,snippet,contentDetails",
                    "id": ",".join(batch_ids)
                }

                videos_data = await self._make_request(
                    "videos",
                    params,
                    quota_cost=self.QUOTA_COSTS["videos.list"]
                )

                for item in videos_data.get("items", []):
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    content = item.get("contentDetails", {})

                    videos.append(YouTubeVideoStats(
                        video_id=item.get("id", ""),
                        title=snippet.get("title", ""),
                        description=snippet.get("description", "")[:200],
                        published_at=snippet.get("publishedAt", ""),
                        channel_id=snippet.get("channelId", ""),
                        channel_title=snippet.get("channelTitle", ""),
                        views=int(stats.get("viewCount", 0)),
                        likes=int(stats.get("likeCount", 0)),
                        comments_count=int(stats.get("commentCount", 0)),
                        duration_seconds=self._parse_duration(content.get("duration", "PT0S")),
                        thumbnail_url=snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                        tags=snippet.get("tags", [])[:5],
                        category_id=snippet.get("categoryId", "")
                    ))
        else:
            # Return basic info without stats
            for item in playlist_data.get("items", []):
                snippet = item.get("snippet", {})
                videos.append(YouTubeVideoStats(
                    video_id=item.get("contentDetails", {}).get("videoId", ""),
                    title=snippet.get("title", ""),
                    description=snippet.get("description", "")[:200],
                    published_at=snippet.get("publishedAt", ""),
                    channel_id=snippet.get("channelId", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    thumbnail_url=snippet.get("thumbnails", {}).get("medium", {}).get("url", "")
                ))

        return videos

    async def get_video_comments(
        self,
        video_id: str,
        limit: int = 100,
        order: str = "relevance",
        include_replies: bool = False
    ) -> List[YouTubeComment]:
        """
        Get comments for a video.

        Args:
            video_id: YouTube video ID
            limit: Maximum number of comments to return
            order: Sort order ("relevance" or "time")
            include_replies: Whether to fetch reply comments

        Returns:
            List of YouTubeComment objects
        """
        if not video_id:
            raise InvalidVideoError("Video ID required")

        comments = []
        page_token = None
        remaining = limit

        while remaining > 0:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(remaining, 100),
                "order": order,
                "textFormat": "plainText"
            }

            if page_token:
                params["pageToken"] = page_token

            try:
                data = await self._make_request(
                    "commentThreads",
                    params,
                    quota_cost=self.QUOTA_COSTS["commentThreads.list"],
                    cache_ttl=600  # Cache for 10 minutes
                )
            except InvalidVideoError:
                # Comments might be disabled
                logger.warning(f"Could not fetch comments for video {video_id}")
                break

            for item in data.get("items", []):
                top_comment = item.get("snippet", {}).get("topLevelComment", {})
                snippet = top_comment.get("snippet", {})

                comment = YouTubeComment(
                    comment_id=top_comment.get("id", ""),
                    video_id=video_id,
                    author=snippet.get("authorDisplayName", ""),
                    author_channel_id=snippet.get("authorChannelId", {}).get("value", ""),
                    text=snippet.get("textOriginal", ""),
                    text_display=snippet.get("textDisplay", ""),
                    like_count=int(snippet.get("likeCount", 0)),
                    reply_count=int(item.get("snippet", {}).get("totalReplyCount", 0)),
                    published_at=snippet.get("publishedAt", ""),
                    updated_at=snippet.get("updatedAt", "")
                )
                comments.append(comment)
                remaining -= 1

                if remaining <= 0:
                    break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        # Fetch replies if requested
        if include_replies:
            comments_with_replies = [c for c in comments if c.reply_count > 0]
            for comment in comments_with_replies[:10]:  # Limit reply fetching
                replies = await self._get_comment_replies(
                    comment.comment_id,
                    limit=5
                )
                comments.extend(replies)

        return comments

    async def _get_comment_replies(
        self,
        parent_id: str,
        limit: int = 5
    ) -> List[YouTubeComment]:
        """Get replies to a comment."""
        params = {
            "part": "snippet",
            "parentId": parent_id,
            "maxResults": min(limit, 100),
            "textFormat": "plainText"
        }

        try:
            data = await self._make_request(
                "comments",
                params,
                quota_cost=self.QUOTA_COSTS["comments.list"]
            )
        except Exception as e:
            logger.warning(f"Could not fetch replies for comment {parent_id}: {e}")
            return []

        replies = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            replies.append(YouTubeComment(
                comment_id=item.get("id", ""),
                video_id=snippet.get("videoId", ""),
                author=snippet.get("authorDisplayName", ""),
                author_channel_id=snippet.get("authorChannelId", {}).get("value", ""),
                text=snippet.get("textOriginal", ""),
                text_display=snippet.get("textDisplay", ""),
                like_count=int(snippet.get("likeCount", 0)),
                reply_count=0,
                published_at=snippet.get("publishedAt", ""),
                updated_at=snippet.get("updatedAt", ""),
                parent_id=parent_id
            ))

        return replies

    async def get_analytics_report(
        self,
        start_date: str,
        end_date: str,
        channel_id: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> YouTubeAnalyticsReport:
        """
        Get detailed analytics report for a date range.

        Requires OAuth credentials with YouTube Analytics API access.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            channel_id: Channel ID (uses default if not provided)
            metrics: List of metrics to include (defaults to standard set)

        Returns:
            YouTubeAnalyticsReport with detailed metrics
        """
        channel_id = channel_id or self.channel_id
        if not channel_id:
            raise InvalidChannelError("Channel ID required")

        if not self.oauth_credentials:
            # Return estimated data from Data API if no OAuth
            return await self._get_analytics_from_data_api(
                start_date, end_date, channel_id
            )

        metrics = metrics or [
            "views", "estimatedMinutesWatched", "averageViewDuration",
            "subscribersGained", "subscribersLost", "likes", "dislikes",
            "comments", "shares"
        ]

        params = {
            "ids": f"channel=={channel_id}",
            "startDate": start_date,
            "endDate": end_date,
            "metrics": ",".join(metrics),
            "dimensions": "day"
        }

        data = await self._make_request(
            "reports",
            params,
            quota_cost=self.QUOTA_COSTS["analytics.query"],
            use_oauth=True
        )

        # Aggregate data
        rows = data.get("rows", [])
        totals = {metric: 0 for metric in metrics}

        for row in rows:
            for i, metric in enumerate(metrics):
                if i + 1 < len(row):  # First column is date
                    totals[metric] += row[i + 1]

        return YouTubeAnalyticsReport(
            channel_id=channel_id,
            start_date=start_date,
            end_date=end_date,
            total_views=int(totals.get("views", 0)),
            total_watch_time_minutes=int(totals.get("estimatedMinutesWatched", 0)),
            avg_view_duration_seconds=int(totals.get("averageViewDuration", 0)),
            total_subscribers_gained=int(totals.get("subscribersGained", 0)),
            total_subscribers_lost=int(totals.get("subscribersLost", 0)),
            total_likes=int(totals.get("likes", 0)),
            total_dislikes=int(totals.get("dislikes", 0)),
            total_comments=int(totals.get("comments", 0)),
            total_shares=int(totals.get("shares", 0))
        )

    async def _get_analytics_from_data_api(
        self,
        start_date: str,
        end_date: str,
        channel_id: str
    ) -> YouTubeAnalyticsReport:
        """
        Estimate analytics from Data API when Analytics API unavailable.

        This provides a best-effort approximation using publicly available data.
        """
        # Get channel stats
        channel_stats = await self.get_channel_stats(channel_id)

        # Get recent videos for estimation
        recent_videos = await self.get_recent_videos(
            limit=50,
            channel_id=channel_id,
            include_stats=True
        )

        # Filter videos by date range
        filtered_videos = []
        for video in recent_videos:
            if video.published_at:
                pub_date = video.published_at[:10]
                if start_date <= pub_date <= end_date:
                    filtered_videos.append(video)

        # Aggregate stats from videos
        total_views = sum(v.views for v in filtered_videos)
        total_likes = sum(v.likes for v in filtered_videos)
        total_comments = sum(v.comments_count for v in filtered_videos)

        # Calculate average view duration estimate
        avg_duration = 0
        if filtered_videos:
            avg_duration = sum(v.avg_view_duration_seconds for v in filtered_videos) // len(filtered_videos)

        # Estimate watch time (views * avg duration / 60)
        estimated_watch_time = (total_views * avg_duration) // 60 if avg_duration else 0

        # Build top videos list
        top_videos = sorted(
            filtered_videos,
            key=lambda v: v.views,
            reverse=True
        )[:10]

        return YouTubeAnalyticsReport(
            channel_id=channel_id,
            start_date=start_date,
            end_date=end_date,
            total_views=total_views,
            total_watch_time_minutes=estimated_watch_time,
            avg_view_duration_seconds=avg_duration,
            total_likes=total_likes,
            total_comments=total_comments,
            top_videos=[
                {
                    "video_id": v.video_id,
                    "title": v.title,
                    "views": v.views,
                    "likes": v.likes,
                    "engagement_rate": v.engagement_rate
                }
                for v in top_videos
            ]
        )

    async def search_comments(
        self,
        query: str,
        video_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        limit: int = 50
    ) -> List[YouTubeComment]:
        """
        Search through comments for specific keywords.

        Note: YouTube API doesn't support comment search directly.
        This fetches comments and filters locally.

        Args:
            query: Search query (case-insensitive substring match)
            video_id: Specific video to search (optional)
            channel_id: Channel to search across videos (optional)
            limit: Maximum number of matching comments to return

        Returns:
            List of YouTubeComment objects matching the query
        """
        query_lower = query.lower()
        matching_comments = []

        if video_id:
            # Search single video
            all_comments = await self.get_video_comments(
                video_id,
                limit=min(limit * 3, 300)  # Fetch more to filter
            )

            for comment in all_comments:
                if query_lower in comment.text.lower():
                    matching_comments.append(comment)
                    if len(matching_comments) >= limit:
                        break

        elif channel_id or self.channel_id:
            # Search across recent videos
            channel = channel_id or self.channel_id
            recent_videos = await self.get_recent_videos(
                limit=10,
                channel_id=channel,
                include_stats=False
            )

            for video in recent_videos:
                if len(matching_comments) >= limit:
                    break

                try:
                    video_comments = await self.get_video_comments(
                        video.video_id,
                        limit=50
                    )

                    for comment in video_comments:
                        if query_lower in comment.text.lower():
                            matching_comments.append(comment)
                            if len(matching_comments) >= limit:
                                break
                except Exception as e:
                    logger.warning(f"Could not fetch comments for video {video.video_id}: {e}")
                    continue
        else:
            raise InvalidChannelError("Either video_id or channel_id required for comment search")

        return matching_comments[:limit]

    # =========================================================================
    # Integration Helpers (for ALFRED tools)
    # =========================================================================

    async def get_metrics_for_harvester(
        self,
        period_start: str,
        period_end: str
    ) -> Dict[str, Any]:
        """
        Get metrics formatted for Social Metrics Harvester.

        Returns data in the schema expected by SocialMetricsHarvester.
        """
        channel_stats = await self.get_channel_stats()
        analytics = await self.get_analytics_report(period_start, period_end)
        recent_videos = await self.get_recent_videos(limit=20, include_stats=True)

        # Sort videos by engagement for top/lowest performing
        sorted_videos = sorted(
            recent_videos,
            key=lambda v: v.engagement_rate,
            reverse=True
        )

        return {
            "platform": "youtube",
            "period": {
                "start": period_start,
                "end": period_end
            },
            "raw_metrics": {
                "output": {
                    "posts": 0,
                    "videos": len([v for v in recent_videos if v.published_at[:10] >= period_start]),
                    "stories": 0
                },
                "reach": {
                    "impressions": analytics.total_views,
                    "views": analytics.total_views,
                    "unique_viewers": 0  # Not available without Analytics API
                },
                "engagement": {
                    "likes": analytics.total_likes,
                    "comments": analytics.total_comments,
                    "shares": analytics.total_shares,
                    "saves": 0,
                    "watch_time_seconds": analytics.total_watch_time_minutes * 60,
                    "avg_view_duration_seconds": analytics.avg_view_duration_seconds
                },
                "growth": {
                    "followers_start": channel_stats.subscriber_count - analytics.net_subscriber_change,
                    "followers_end": channel_stats.subscriber_count,
                    "subscribers_gained": analytics.total_subscribers_gained,
                    "subscribers_lost": analytics.total_subscribers_lost
                },
                "conversion": {
                    "link_clicks": 0,  # Requires Analytics API
                    "profile_visits": 0  # Requires Analytics API
                }
            },
            "content_items": [
                {
                    "content_id": v.video_id,
                    "platform": "youtube",
                    "content_type": "video",
                    "title": v.title,
                    "published_at": v.published_at,
                    "url": f"https://youtube.com/watch?v={v.video_id}",
                    "impressions": v.views,
                    "likes": v.likes,
                    "comments": v.comments_count,
                    "shares": 0,
                    "saves": 0,
                    "engagement_rate": v.engagement_rate,
                    "watch_time_seconds": v.watch_time_hours * 3600,
                    "completion_rate": 0.0
                }
                for v in sorted_videos[:10]
            ],
            "top_performing": [
                {
                    "content_id": v.video_id,
                    "title": v.title,
                    "engagement_rate": v.engagement_rate,
                    "views": v.views,
                    "likes": v.likes
                }
                for v in sorted_videos[:5]
            ],
            "lowest_performing": [
                {
                    "content_id": v.video_id,
                    "title": v.title,
                    "engagement_rate": v.engagement_rate,
                    "views": v.views,
                    "likes": v.likes
                }
                for v in sorted_videos[-5:]
            ] if len(sorted_videos) >= 5 else [],
            "api_status": "connected"
        }

    async def get_comments_for_audience_extractor(
        self,
        video_ids: Optional[List[str]] = None,
        limit_per_video: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get comments formatted for Audience Signals Extractor.

        Returns comments in the schema expected by AudienceSignalsExtractor.
        """
        if not video_ids:
            # Get recent videos
            recent_videos = await self.get_recent_videos(limit=5)
            video_ids = [v.video_id for v in recent_videos]

        all_comments = []
        for video_id in video_ids:
            try:
                comments = await self.get_video_comments(
                    video_id,
                    limit=limit_per_video
                )

                for comment in comments:
                    all_comments.append({
                        "platform": "youtube",
                        "content_id": video_id,
                        "text": comment.text,
                        "author": comment.author,
                        "timestamp": comment.published_at,
                        "like_count": comment.like_count,
                        "reply_count": comment.reply_count,
                        "comment_id": comment.comment_id
                    })
            except Exception as e:
                logger.warning(f"Could not fetch comments for video {video_id}: {e}")

        return all_comments

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota status."""
        return self.rate_limiter.get_quota_status()

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection and credentials.

        Returns:
            Dict with connection status and any error details
        """
        result = {
            "status": "unknown",
            "api_key_valid": False,
            "oauth_valid": False,
            "channel_accessible": False,
            "quota_remaining": self.rate_limiter.get_remaining_quota(),
            "errors": []
        }

        try:
            # Test API key with a simple request
            if self.api_key:
                params = {
                    "part": "id",
                    "chart": "mostPopular",
                    "maxResults": 1
                }
                await self._make_request(
                    "videos",
                    params,
                    quota_cost=1
                )
                result["api_key_valid"] = True
        except AuthenticationError as e:
            result["errors"].append(f"API key error: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Connection error: {str(e)}")

        try:
            # Test channel access
            if self.channel_id:
                await self.get_channel_stats()
                result["channel_accessible"] = True
        except InvalidChannelError as e:
            result["errors"].append(f"Channel error: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Channel access error: {str(e)}")

        # Determine overall status
        if result["api_key_valid"] and result["channel_accessible"]:
            result["status"] = "connected"
        elif result["api_key_valid"]:
            result["status"] = "partial"
        else:
            result["status"] = "disconnected"

        return result


# =============================================================================
# Factory Function
# =============================================================================

def create_youtube_adapter(
    api_key: Optional[str] = None,
    channel_id: Optional[str] = None,
    **kwargs
) -> YouTubeAdapter:
    """
    Factory function to create YouTubeAdapter instance.

    Can read credentials from environment variables if not provided.

    Args:
        api_key: YouTube Data API key
        channel_id: Default channel ID
        **kwargs: Additional arguments passed to YouTubeAdapter

    Returns:
        Configured YouTubeAdapter instance
    """
    import os

    api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
    channel_id = channel_id or os.environ.get("YOUTUBE_CHANNEL_ID")

    if not api_key:
        logger.warning(
            "No YouTube API key provided. "
            "Set YOUTUBE_API_KEY environment variable or pass api_key parameter."
        )

    return YouTubeAdapter(
        api_key=api_key,
        channel_id=channel_id,
        **kwargs
    )
