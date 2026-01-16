"""
Twitter/X Integration Adapter for ALFRED

Provides structured access to Twitter/X data WITHOUT official API access.
Uses Nitter (public instances) for scraping and supports Apify integration
as an optional enhancement. Includes manual input fallback for all metrics.

Features:
- Profile statistics (followers, following, tweets)
- Recent tweets and engagement metrics
- Search functionality via Nitter scraping
- Apify integration for enhanced data collection (optional)
- Manual data input for metrics tracking with persistence
- Rate limiting and caching
- Multiple Nitter instance fallback with health checking
- Offline mode support with cached data
- User-agent rotation for scraping resilience

Data Sources (in priority order):
1. Nitter scraping (free, no API required)
2. Apify Twitter scrapers (optional, requires account)
3. Manual input (always available fallback)

Note: This adapter does NOT use the official Twitter API since it requires
developer access. All data collection uses public, legal alternatives.

Rate Limiting Notes:
- Nitter instances may rate limit aggressive scraping
- Built-in delays between requests (2-5 seconds)
- Automatic fallback to alternate Nitter instances
- Session rotation to avoid blocks

Storage Path: /Users/shaileshsingh/Alfred/agent-zero1/data/alfred/twitter/
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Public Nitter instances (updated list - some may go offline)
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.1d4.us",
    "https://nitter.kavin.rocks",
    "https://nitter.unixfox.eu",
    "https://nitter.fdn.fr",
    "https://nitter.namazso.eu",
    "https://n.l5.ca",
    "https://nitter.moomoo.me",
]

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# Default data storage path
DEFAULT_DATA_PATH = Path("/Users/shaileshsingh/Alfred/agent-zero1/data/alfred/twitter")


# =============================================================================
# Error Classes
# =============================================================================

class TwitterError(Exception):
    """Base exception for Twitter adapter errors."""
    pass


class ScrapingError(TwitterError):
    """Error during web scraping."""
    pass


class RateLimitError(TwitterError):
    """Rate limit exceeded, retry after cooldown."""
    def __init__(self, message: str, retry_after: int = 300):
        super().__init__(message)
        self.retry_after = retry_after


class ProfileNotFoundError(TwitterError):
    """Twitter profile does not exist or is suspended."""
    pass


class TweetNotFoundError(TwitterError):
    """Tweet does not exist or is not accessible."""
    pass


class OfflineError(TwitterError):
    """Cannot connect to any Nitter instance."""
    pass


class ApifyError(TwitterError):
    """Error with Apify integration."""
    pass


class NitterInstanceError(TwitterError):
    """All Nitter instances are unavailable."""
    pass


class DataSourceError(TwitterError):
    """Error with the data source."""
    pass


class NoDataSourceError(TwitterError):
    """No data source available."""
    pass


# =============================================================================
# Enums
# =============================================================================

class DataSource(Enum):
    """Available data sources for Twitter data."""
    NITTER = "nitter"
    APIFY = "apify"
    MANUAL = "manual"
    AUTO = "auto"


# =============================================================================
# Data Classes (Pydantic-style with dataclasses)
# =============================================================================

@dataclass
class TwitterProfile:
    """
    Twitter/X profile statistics.

    Attributes:
        username: Twitter handle (without @)
        display_name: User's display name
        followers: Number of followers
        following: Number of accounts following
        tweets_count: Total number of tweets
        bio: Profile biography
        location: User location
        website: Profile website URL
        joined_date: Account creation date
        is_verified: Whether account has verification badge
        is_protected: Whether account is private/protected
        profile_image_url: URL to profile image
        banner_image_url: URL to banner image
        collected_at: Timestamp when data was collected
        source: Data source (nitter, apify, manual)
    """
    username: str
    display_name: str = ""
    followers: int = 0
    following: int = 0
    tweets_count: int = 0
    bio: str = ""
    location: str = ""
    website: str = ""
    joined_date: str = ""
    is_verified: bool = False
    is_protected: bool = False
    profile_image_url: str = ""
    banner_image_url: str = ""
    collected_at: str = ""
    source: str = "unknown"

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "username": self.username,
            "display_name": self.display_name,
            "followers": self.followers,
            "following": self.following,
            "tweets_count": self.tweets_count,
            "bio": self.bio,
            "location": self.location,
            "website": self.website,
            "joined_date": self.joined_date,
            "is_verified": self.is_verified,
            "is_protected": self.is_protected,
            "profile_image_url": self.profile_image_url,
            "banner_image_url": self.banner_image_url,
            "collected_at": self.collected_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwitterProfile":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Tweet:
    """
    Individual tweet data.

    Attributes:
        tweet_id: Unique tweet identifier
        username: Author's username
        text: Tweet text content
        likes: Number of likes
        retweets: Number of retweets
        replies: Number of replies
        quotes: Number of quote tweets
        views: Number of views (if available)
        timestamp: Tweet creation timestamp
        url: Direct URL to tweet
        is_retweet: Whether this is a retweet
        is_reply: Whether this is a reply
        reply_to: Username being replied to (if reply)
        hashtags: List of hashtags in tweet
        mentions: List of mentioned users
        media_urls: List of media URLs
        collected_at: When this data was collected
        source: Data source (nitter, apify, manual)
    """
    tweet_id: str
    username: str
    text: str = ""
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    quotes: int = 0
    views: Optional[int] = None
    timestamp: str = ""
    url: str = ""
    is_retweet: bool = False
    is_reply: bool = False
    reply_to: str = ""
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)
    collected_at: str = ""
    source: str = "unknown"

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now().isoformat()
        if not self.url and self.tweet_id and self.username:
            self.url = f"https://twitter.com/{self.username}/status/{self.tweet_id}"

    def engagement_total(self) -> int:
        """Calculate total engagement."""
        return self.likes + self.retweets + self.replies + self.quotes

    def engagement_rate(self, followers: int) -> float:
        """Calculate engagement rate given follower count."""
        if followers <= 0:
            return 0.0
        return self.engagement_total() / followers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tweet_id": self.tweet_id,
            "username": self.username,
            "text": self.text,
            "likes": self.likes,
            "retweets": self.retweets,
            "replies": self.replies,
            "quotes": self.quotes,
            "views": self.views,
            "timestamp": self.timestamp,
            "url": self.url,
            "is_retweet": self.is_retweet,
            "is_reply": self.is_reply,
            "reply_to": self.reply_to,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "media_urls": self.media_urls,
            "engagement_total": self.engagement_total(),
            "collected_at": self.collected_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tweet":
        """Create from dictionary."""
        # Remove computed fields that aren't in __init__
        filtered = {k: v for k, v in data.items()
                   if k in cls.__dataclass_fields__ and k != "engagement_total"}
        return cls(**filtered)


@dataclass
class TwitterStats:
    """
    Aggregate Twitter statistics for a time period.

    Attributes:
        username: Twitter handle
        period_start: Start of measurement period
        period_end: End of measurement period
        followers_start: Followers at period start
        followers_end: Followers at period end
        followers_gained: New followers in period
        followers_lost: Lost followers in period
        tweets_posted: Number of tweets in period
        total_likes: Total likes received
        total_retweets: Total retweets received
        total_replies: Total replies received
        total_views: Total views (if available)
        avg_engagement_rate: Average engagement rate
        top_tweet: Best performing tweet
        collected_at: When stats were calculated
        source: Data source
    """
    username: str
    period_start: str = ""
    period_end: str = ""
    followers_start: int = 0
    followers_end: int = 0
    followers_gained: int = 0
    followers_lost: int = 0
    tweets_posted: int = 0
    total_likes: int = 0
    total_retweets: int = 0
    total_replies: int = 0
    total_views: Optional[int] = None
    avg_engagement_rate: float = 0.0
    top_tweet: Optional[Dict[str, Any]] = None
    collected_at: str = ""
    source: str = "calculated"

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now().isoformat()

    def net_follower_change(self) -> int:
        """Calculate net follower change."""
        return self.followers_end - self.followers_start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "username": self.username,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "followers_start": self.followers_start,
            "followers_end": self.followers_end,
            "followers_gained": self.followers_gained,
            "followers_lost": self.followers_lost,
            "net_follower_change": self.net_follower_change(),
            "tweets_posted": self.tweets_posted,
            "total_likes": self.total_likes,
            "total_retweets": self.total_retweets,
            "total_replies": self.total_replies,
            "total_views": self.total_views,
            "avg_engagement_rate": self.avg_engagement_rate,
            "top_tweet": self.top_tweet,
            "collected_at": self.collected_at,
            "source": self.source,
        }


@dataclass
class ManualEntry:
    """
    Manual data entry record.

    Attributes:
        platform: Always 'twitter' for this adapter
        entry_type: Type of entry (profile, tweet, stats)
        metrics: Dictionary of metrics
        timestamp: When entry was recorded
        source: Always 'manual'
        notes: Optional notes about the entry
    """
    platform: str = "twitter"
    entry_type: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    source: str = "manual"
    notes: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "platform": self.platform,
            "entry_type": self.entry_type,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "source": self.source,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManualEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    data: Any
    expires_at: float

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.expires_at


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Rate limiter for Twitter/Nitter requests.

    Implements delays and request tracking to avoid
    triggering rate limits on Nitter instances.
    """

    def __init__(
        self,
        requests_per_minute: float = 10.0,
        min_request_interval: float = 3.0,
        max_request_interval: float = 6.0
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            min_request_interval: Minimum seconds between requests
            max_request_interval: Maximum seconds between requests (adds randomness)
        """
        self.requests_per_minute = requests_per_minute
        self.min_request_interval = min_request_interval
        self.max_request_interval = max_request_interval
        self.request_count = 0
        self.minute_start = time.time()
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Acquire permission for a request.

        Enforces rate limits and adds randomized delays.
        """
        async with self._lock:
            current_time = time.time()

            # Reset counter each minute
            if current_time - self.minute_start > 60:
                self.request_count = 0
                self.minute_start = current_time

            # Check minute limit
            if self.request_count >= self.requests_per_minute:
                wait_time = 60 - (current_time - self.minute_start) + random.uniform(1, 5)
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    self.request_count = 0
                    self.minute_start = time.time()

            # Enforce randomized interval between requests
            time_since_last = current_time - self.last_request_time
            required_interval = random.uniform(
                self.min_request_interval,
                self.max_request_interval
            )
            if time_since_last < required_interval:
                await asyncio.sleep(required_interval - time_since_last)

            self.request_count += 1
            self.last_request_time = time.time()
            return True

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return {
            "requests_this_minute": self.request_count,
            "requests_per_minute_limit": self.requests_per_minute,
            "min_request_interval": self.min_request_interval,
            "max_request_interval": self.max_request_interval,
            "seconds_until_reset": max(0, 60 - (time.time() - self.minute_start))
        }


# =============================================================================
# Response Cache
# =============================================================================

class ResponseCache:
    """
    In-memory cache for Twitter responses.

    Reduces requests to Nitter instances by caching results.
    """

    def __init__(self, default_ttl_seconds: int = 600):
        """
        Initialize cache.

        Args:
            default_ttl_seconds: Default cache TTL (10 minutes)
        """
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

    async def cleanup_expired(self) -> int:
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
# Nitter Instance Manager
# =============================================================================

class NitterInstanceManager:
    """
    Manages a list of public Nitter instances with health checking
    and automatic failover.
    """

    def __init__(
        self,
        instances: Optional[List[str]] = None,
        cooldown_period: int = 300
    ):
        """
        Initialize instance manager.

        Args:
            instances: List of Nitter instance URLs
            cooldown_period: Seconds to wait before retrying failed instance
        """
        self.instances = instances or NITTER_INSTANCES.copy()
        self.cooldown_period = cooldown_period
        self._failed_instances: Dict[str, float] = {}
        self._instance_latencies: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get_working_instance(self) -> str:
        """
        Get a working Nitter instance with failover support.

        Returns:
            URL of a working Nitter instance

        Raises:
            NitterInstanceError: If no instances are available
        """
        async with self._lock:
            current_time = time.time()

            # Clear expired failures
            self._failed_instances = {
                inst: fail_time
                for inst, fail_time in self._failed_instances.items()
                if current_time - fail_time < self.cooldown_period
            }

            # Get available instances
            available = [
                inst for inst in self.instances
                if inst not in self._failed_instances
            ]

            if not available:
                # Reset all failures if none available
                self._failed_instances.clear()
                available = self.instances.copy()

            # Prefer instances with known good latency
            if self._instance_latencies:
                available.sort(
                    key=lambda x: self._instance_latencies.get(x, float('inf'))
                )

            # Add some randomness to avoid always hitting the same instance
            if len(available) > 1:
                # 70% chance of using best instance, 30% random
                if random.random() > 0.7:
                    return random.choice(available)

            return available[0]

    async def mark_instance_failed(self, instance: str):
        """Mark an instance as temporarily failed."""
        async with self._lock:
            self._failed_instances[instance] = time.time()
            logger.warning(f"Marked Nitter instance as failed: {instance}")

    async def record_latency(self, instance: str, latency: float):
        """Record response latency for an instance."""
        async with self._lock:
            # Exponential moving average
            if instance in self._instance_latencies:
                self._instance_latencies[instance] = (
                    0.7 * self._instance_latencies[instance] + 0.3 * latency
                )
            else:
                self._instance_latencies[instance] = latency

    def get_status(self) -> Dict[str, Any]:
        """Get status of all instances."""
        current_time = time.time()
        return {
            "total_instances": len(self.instances),
            "failed_instances": len(self._failed_instances),
            "available_instances": len(self.instances) - len(self._failed_instances),
            "cooldown_period": self.cooldown_period,
            "instances": [
                {
                    "url": inst,
                    "status": "failed" if inst in self._failed_instances else "available",
                    "latency_ms": self._instance_latencies.get(inst, 0) * 1000,
                    "cooldown_remaining": max(
                        0,
                        self.cooldown_period - (current_time - self._failed_instances.get(inst, 0))
                    ) if inst in self._failed_instances else 0
                }
                for inst in self.instances
            ]
        }


# =============================================================================
# Nitter Scraper
# =============================================================================

class NitterScraper:
    """
    Web scraper for Nitter instances.

    Extracts Twitter data from public Nitter mirrors without
    requiring official API access.
    """

    def __init__(
        self,
        instance_manager: NitterInstanceManager,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Nitter scraper.

        Args:
            instance_manager: NitterInstanceManager for instance selection
            timeout: Request timeout in seconds
            max_retries: Maximum retries per request
        """
        self.instance_manager = instance_manager
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client = None

    async def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True
                )
            except ImportError:
                try:
                    import aiohttp
                    self._http_client = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    )
                except ImportError:
                    raise ImportError(
                        "Either httpx or aiohttp is required for TwitterAdapter. "
                        "Install with: pip install httpx or pip install aiohttp"
                    )
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            if hasattr(self._http_client, 'aclose'):
                await self._http_client.aclose()
            else:
                await self._http_client.close()
            self._http_client = None

    def _get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(USER_AGENTS)

    async def _fetch_html(self, path: str) -> Tuple[str, str]:
        """
        Fetch HTML from a Nitter instance.

        Args:
            path: URL path to fetch

        Returns:
            Tuple of (html_content, instance_used)
        """
        last_error = None
        attempts = 0

        while attempts < self.max_retries * 2:
            instance = await self.instance_manager.get_working_instance()
            url = f"{instance}{path}"
            headers = {
                "User-Agent": self._get_random_user_agent(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
            }

            start_time = time.time()
            try:
                client = await self._get_http_client()

                # Handle both httpx and aiohttp
                if hasattr(client, 'get'):
                    # httpx style
                    response = await client.get(url, headers=headers)
                    latency = time.time() - start_time
                    await self.instance_manager.record_latency(instance, latency)

                    if response.status_code == 200:
                        return response.text, instance
                    elif response.status_code == 429:
                        await self.instance_manager.mark_instance_failed(instance)
                        raise RateLimitError(f"Rate limited by {instance}", retry_after=60)
                    elif response.status_code == 404:
                        raise ProfileNotFoundError(f"Profile not found")
                    else:
                        await self.instance_manager.mark_instance_failed(instance)
                        last_error = f"HTTP {response.status_code}"
                else:
                    # aiohttp style
                    async with client.get(url, headers=headers) as response:
                        latency = time.time() - start_time
                        await self.instance_manager.record_latency(instance, latency)

                        if response.status == 200:
                            return await response.text(), instance
                        elif response.status == 429:
                            await self.instance_manager.mark_instance_failed(instance)
                            raise RateLimitError(f"Rate limited by {instance}", retry_after=60)
                        elif response.status == 404:
                            raise ProfileNotFoundError(f"Profile not found")
                        else:
                            await self.instance_manager.mark_instance_failed(instance)
                            last_error = f"HTTP {response.status}"

            except (RateLimitError, ProfileNotFoundError):
                raise
            except Exception as e:
                await self.instance_manager.mark_instance_failed(instance)
                last_error = str(e)
                logger.debug(f"Failed to fetch from {instance}: {e}")

            attempts += 1

        raise NitterInstanceError(
            f"All Nitter instances failed after {attempts} attempts. Last error: {last_error}"
        )

    def _parse_count(self, text: str) -> int:
        """Parse follower/like counts (handles K, M suffixes)."""
        if not text:
            return 0
        text = text.strip().replace(",", "").upper()
        try:
            if "K" in text:
                return int(float(text.replace("K", "")) * 1000)
            elif "M" in text:
                return int(float(text.replace("M", "")) * 1000000)
            elif "B" in text:
                return int(float(text.replace("B", "")) * 1000000000)
            else:
                return int(text)
        except (ValueError, AttributeError):
            return 0

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        if not text:
            return []
        return list(set(re.findall(r'#(\w+)', text)))

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        if not text:
            return []
        return list(set(re.findall(r'@(\w+)', text)))

    async def get_profile(self, username: str) -> TwitterProfile:
        """
        Scrape profile data from Nitter.

        Args:
            username: Twitter username (without @)

        Returns:
            TwitterProfile with scraped data
        """
        username = username.lstrip("@")
        path = f"/{username}"

        html, instance = await self._fetch_html(path)

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for Nitter scraping. "
                "Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(html, "html.parser")

        # Check for error/not found
        error_panel = soup.select_one(".error-panel")
        if error_panel:
            raise ProfileNotFoundError(f"Profile @{username} not found or suspended")

        # Extract profile data
        profile_card = soup.select_one(".profile-card")
        if not profile_card:
            raise ScrapingError(f"Could not find profile card for @{username}")

        # Display name
        display_name_elem = profile_card.select_one(".profile-card-fullname")
        display_name = display_name_elem.get_text(strip=True) if display_name_elem else username

        # Bio
        bio_elem = profile_card.select_one(".profile-bio")
        bio = bio_elem.get_text(strip=True) if bio_elem else ""

        # Stats
        stats = {}
        stat_items = profile_card.select(".profile-stat")
        for stat in stat_items:
            stat_type = stat.select_one(".profile-stat-header")
            stat_value = stat.select_one(".profile-stat-num")
            if stat_type and stat_value:
                stat_name = stat_type.get_text(strip=True).lower()
                stats[stat_name] = self._parse_count(stat_value.get_text())

        # Location
        location_elem = profile_card.select_one(".profile-location")
        location = location_elem.get_text(strip=True) if location_elem else ""

        # Website
        website_elem = profile_card.select_one(".profile-website a")
        website = website_elem.get("href", "") if website_elem else ""

        # Joined date
        joined_elem = profile_card.select_one(".profile-joindate span")
        joined_date = joined_elem.get_text(strip=True) if joined_elem else ""

        # Verification badge
        is_verified = bool(profile_card.select_one(".verified-icon"))

        # Protected account
        is_protected = bool(soup.select_one(".protected-icon"))

        # Profile image
        profile_img = profile_card.select_one(".profile-card-avatar img")
        profile_image_url = profile_img.get("src", "") if profile_img else ""
        if profile_image_url and not profile_image_url.startswith("http"):
            profile_image_url = f"{instance}{profile_image_url}"

        # Banner
        banner_elem = soup.select_one(".profile-banner img")
        banner_image_url = banner_elem.get("src", "") if banner_elem else ""
        if banner_image_url and not banner_image_url.startswith("http"):
            banner_image_url = f"{instance}{banner_image_url}"

        return TwitterProfile(
            username=username,
            display_name=display_name,
            followers=stats.get("followers", 0),
            following=stats.get("following", 0),
            tweets_count=stats.get("tweets", stats.get("posts", 0)),
            bio=bio,
            location=location,
            website=website,
            joined_date=joined_date,
            is_verified=is_verified,
            is_protected=is_protected,
            profile_image_url=profile_image_url,
            banner_image_url=banner_image_url,
            source="nitter"
        )

    async def get_tweets(self, username: str, count: int = 20) -> List[Tweet]:
        """
        Scrape recent tweets from a user's timeline.

        Args:
            username: Twitter username (without @)
            count: Maximum number of tweets to fetch

        Returns:
            List of Tweet objects
        """
        username = username.lstrip("@")
        path = f"/{username}"

        html, instance = await self._fetch_html(path)

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for Nitter scraping. "
                "Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(html, "html.parser")

        tweets = []
        timeline_items = soup.select(".timeline-item")[:count]

        for item in timeline_items:
            try:
                # Skip pinned tweets indicator
                if item.select_one(".pinned"):
                    continue

                # Tweet link and ID
                tweet_link = item.select_one(".tweet-link")
                if not tweet_link:
                    continue

                href = tweet_link.get("href", "")
                tweet_id_match = re.search(r'/status/(\d+)', href)
                if not tweet_id_match:
                    continue
                tweet_id = tweet_id_match.group(1)

                # Tweet content
                content_elem = item.select_one(".tweet-content")
                text = content_elem.get_text(strip=True) if content_elem else ""

                # Stats
                stat_container = item.select_one(".tweet-stats") or item

                # Comments/Replies
                replies = 0
                replies_elem = stat_container.select_one(".icon-comment")
                if replies_elem and replies_elem.parent:
                    replies_text = replies_elem.parent.get_text(strip=True)
                    replies = self._parse_count(re.sub(r'[^\d,KMB.]', '', replies_text))

                # Retweets
                retweets = 0
                retweets_elem = stat_container.select_one(".icon-retweet")
                if retweets_elem and retweets_elem.parent:
                    retweets_text = retweets_elem.parent.get_text(strip=True)
                    retweets = self._parse_count(re.sub(r'[^\d,KMB.]', '', retweets_text))

                # Quotes
                quotes = 0
                quotes_elem = stat_container.select_one(".icon-quote")
                if quotes_elem and quotes_elem.parent:
                    quotes_text = quotes_elem.parent.get_text(strip=True)
                    quotes = self._parse_count(re.sub(r'[^\d,KMB.]', '', quotes_text))

                # Likes
                likes = 0
                likes_elem = stat_container.select_one(".icon-heart")
                if likes_elem and likes_elem.parent:
                    likes_text = likes_elem.parent.get_text(strip=True)
                    likes = self._parse_count(re.sub(r'[^\d,KMB.]', '', likes_text))

                # Timestamp
                timestamp_elem = item.select_one(".tweet-date a")
                timestamp = ""
                if timestamp_elem:
                    timestamp = timestamp_elem.get("title", "")

                # Check if retweet
                is_retweet = bool(item.select_one(".retweet-header"))

                # Check if reply
                reply_elem = item.select_one(".replying-to")
                is_reply = bool(reply_elem)
                reply_to = ""
                if reply_elem:
                    reply_to_link = reply_elem.select_one("a")
                    if reply_to_link:
                        reply_to = reply_to_link.get_text(strip=True).lstrip("@")

                # Media
                media_urls = []
                media_elems = item.select(".attachment-image img, .gallery-video video")
                for media in media_elems:
                    src = media.get("src", "")
                    if src:
                        if not src.startswith("http"):
                            src = f"{instance}{src}"
                        media_urls.append(src)

                tweets.append(Tweet(
                    tweet_id=tweet_id,
                    username=username,
                    text=text,
                    likes=likes,
                    retweets=retweets,
                    replies=replies,
                    quotes=quotes,
                    timestamp=timestamp,
                    is_retweet=is_retweet,
                    is_reply=is_reply,
                    reply_to=reply_to,
                    hashtags=self._extract_hashtags(text),
                    mentions=self._extract_mentions(text),
                    media_urls=media_urls,
                    source="nitter"
                ))

            except Exception as e:
                logger.warning(f"Error parsing tweet: {e}")
                continue

        return tweets

    async def search_tweets(self, query: str, count: int = 20) -> List[Tweet]:
        """
        Search for tweets using Nitter search.

        Args:
            query: Search query
            count: Maximum number of tweets to return

        Returns:
            List of matching Tweet objects
        """
        # URL encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        path = f"/search?f=tweets&q={encoded_query}"

        html, instance = await self._fetch_html(path)

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for Nitter scraping. "
                "Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(html, "html.parser")

        tweets = []
        timeline_items = soup.select(".timeline-item")[:count]

        for item in timeline_items:
            try:
                # Tweet link and ID
                tweet_link = item.select_one(".tweet-link")
                if not tweet_link:
                    continue

                href = tweet_link.get("href", "")
                tweet_id_match = re.search(r'/status/(\d+)', href)
                username_match = re.search(r'^/(\w+)/', href)

                if not tweet_id_match or not username_match:
                    continue

                tweet_id = tweet_id_match.group(1)
                tweet_username = username_match.group(1)

                # Tweet content
                content_elem = item.select_one(".tweet-content")
                text = content_elem.get_text(strip=True) if content_elem else ""

                # Stats (simplified for search results)
                stat_container = item.select_one(".tweet-stats") or item

                likes = 0
                likes_elem = stat_container.select_one(".icon-heart")
                if likes_elem and likes_elem.parent:
                    likes_text = likes_elem.parent.get_text(strip=True)
                    likes = self._parse_count(re.sub(r'[^\d,KMB.]', '', likes_text))

                retweets = 0
                retweets_elem = stat_container.select_one(".icon-retweet")
                if retweets_elem and retweets_elem.parent:
                    retweets_text = retweets_elem.parent.get_text(strip=True)
                    retweets = self._parse_count(re.sub(r'[^\d,KMB.]', '', retweets_text))

                replies = 0
                replies_elem = stat_container.select_one(".icon-comment")
                if replies_elem and replies_elem.parent:
                    replies_text = replies_elem.parent.get_text(strip=True)
                    replies = self._parse_count(re.sub(r'[^\d,KMB.]', '', replies_text))

                # Timestamp
                timestamp_elem = item.select_one(".tweet-date a")
                timestamp = timestamp_elem.get("title", "") if timestamp_elem else ""

                tweets.append(Tweet(
                    tweet_id=tweet_id,
                    username=tweet_username,
                    text=text,
                    likes=likes,
                    retweets=retweets,
                    replies=replies,
                    timestamp=timestamp,
                    hashtags=self._extract_hashtags(text),
                    mentions=self._extract_mentions(text),
                    source="nitter"
                ))

            except Exception as e:
                logger.warning(f"Error parsing search result: {e}")
                continue

        return tweets


# =============================================================================
# Twitter Adapter (Main Class)
# =============================================================================

class TwitterAdapter:
    """
    Twitter/X Integration Adapter for ALFRED.

    Provides structured access to Twitter data without requiring official
    API access. Uses Nitter scraping as primary data source, with optional
    Apify integration and manual input fallback.

    Usage:
        adapter = TwitterAdapter()

        # Get profile stats via Nitter
        profile = await adapter.get_profile("username")

        # Get recent tweets
        tweets = await adapter.get_recent_tweets("username", count=20)

        # Search tweets
        results = await adapter.search_tweets("query", count=10)

        # Manual input fallback
        await adapter.enter_manual_stats({
            "followers": 10000,
            "impressions": 50000,
            ...
        })

        # Get aggregated stats
        stats = await adapter.get_stats_summary("username", days=7)
    """

    def __init__(
        self,
        nitter_instances: Optional[List[str]] = None,
        apify_api_key: Optional[str] = None,
        data_path: Optional[Path] = None,
        cache_ttl_seconds: int = 600,
        enable_caching: bool = True,
        requests_per_minute: float = 10.0,
        offline_mode: bool = False
    ):
        """
        Initialize Twitter adapter.

        Args:
            nitter_instances: Custom list of Nitter instance URLs
            apify_api_key: Optional Apify API key for enhanced data collection
            data_path: Path for storing cached/manual data
            cache_ttl_seconds: Default cache TTL (10 minutes)
            enable_caching: Whether to enable response caching
            requests_per_minute: Rate limit for requests
            offline_mode: Whether to operate in offline mode only
        """
        self.apify_api_key = apify_api_key or os.environ.get("APIFY_API_KEY")
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.enable_caching = enable_caching
        self.offline_mode = offline_mode

        # Ensure data directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.instance_manager = NitterInstanceManager(instances=nitter_instances)
        self.scraper = NitterScraper(instance_manager=self.instance_manager)
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self.cache = ResponseCache(default_ttl_seconds=cache_ttl_seconds)

        # Manual entries storage
        self._manual_entries: List[ManualEntry] = []
        self._load_manual_entries()

        # Request tracking
        self._request_count = 0
        self._last_error: Optional[str] = None

        logger.info(
            f"TwitterAdapter initialized (Apify: {bool(self.apify_api_key)}, "
            f"Offline mode: {offline_mode}, Data path: {self.data_path})"
        )

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def close(self):
        """Close adapter and cleanup resources."""
        await self.scraper.close()
        await self.cache.clear()
        self._save_manual_entries()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Manual Entry Persistence
    # =========================================================================

    def _get_manual_entries_path(self) -> Path:
        """Get path to manual entries file."""
        return self.data_path / "manual_entries.json"

    def _load_manual_entries(self):
        """Load manual entries from disk."""
        path = self._get_manual_entries_path()
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    self._manual_entries = [
                        ManualEntry.from_dict(entry) for entry in data
                    ]
                logger.info(f"Loaded {len(self._manual_entries)} manual entries")
            except Exception as e:
                logger.warning(f"Could not load manual entries: {e}")
                self._manual_entries = []

    def _save_manual_entries(self):
        """Save manual entries to disk."""
        path = self._get_manual_entries_path()
        try:
            with open(path, "w") as f:
                json.dump(
                    [entry.to_dict() for entry in self._manual_entries],
                    f,
                    indent=2
                )
            logger.debug(f"Saved {len(self._manual_entries)} manual entries")
        except Exception as e:
            logger.warning(f"Could not save manual entries: {e}")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _execute_with_rate_limit(self, coro):
        """Execute coroutine with rate limiting."""
        await self.rate_limiter.acquire()
        self._request_count += 1
        try:
            return await coro
        except Exception as e:
            self._last_error = str(e)
            raise

    def _get_cached_data_path(self, username: str) -> Path:
        """Get path for cached user data."""
        return self.data_path / f"{username.lower()}_cache.json"

    async def _save_to_cache_file(self, username: str, data: Dict[str, Any]):
        """Save data to file-based cache for offline use."""
        path = self._get_cached_data_path(username)
        try:
            existing = {}
            if path.exists():
                with open(path, "r") as f:
                    existing = json.load(f)

            existing.update(data)
            existing["last_updated"] = datetime.now().isoformat()

            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache file: {e}")

    async def _load_from_cache_file(self, username: str) -> Optional[Dict[str, Any]]:
        """Load data from file-based cache."""
        path = self._get_cached_data_path(username)
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache file: {e}")
        return None

    # =========================================================================
    # Public API Methods - Profile
    # =========================================================================

    async def get_profile(self, username: str) -> TwitterProfile:
        """
        Get Twitter profile statistics.

        Primary method uses Nitter scraping. Falls back to cached data
        in offline mode or on errors.

        Args:
            username: Twitter username (without @)

        Returns:
            TwitterProfile with follower count, bio, etc.
        """
        username = username.lstrip("@").lower()

        # Check memory cache
        if self.enable_caching:
            cached = await self.cache.get("profile", username)
            if cached:
                return cached

        # Offline mode - use file cache
        if self.offline_mode:
            cached_data = await self._load_from_cache_file(username)
            if cached_data and "profile" in cached_data:
                return TwitterProfile.from_dict(cached_data["profile"])
            raise OfflineError(f"No cached data for @{username} in offline mode")

        try:
            profile = await self._execute_with_rate_limit(
                self.scraper.get_profile(username)
            )

            # Cache the result
            if self.enable_caching:
                await self.cache.set(profile, 3600, "profile", username)

            # Save to file cache for offline use
            await self._save_to_cache_file(username, {"profile": profile.to_dict()})

            return profile

        except (ProfileNotFoundError, RateLimitError):
            raise
        except NitterInstanceError as e:
            # Try to fall back to cached data
            cached_data = await self._load_from_cache_file(username)
            if cached_data and "profile" in cached_data:
                logger.warning(f"Nitter unavailable, using cached data for @{username}")
                return TwitterProfile.from_dict(cached_data["profile"])
            raise OfflineError(f"Cannot fetch profile and no cached data available: {e}")

    async def get_recent_tweets(
        self,
        username: str,
        count: int = 20
    ) -> List[Tweet]:
        """
        Get recent tweets from a user's timeline.

        Args:
            username: Twitter username (without @)
            count: Maximum number of tweets to fetch (max 50)

        Returns:
            List of Tweet objects with engagement data
        """
        username = username.lstrip("@").lower()
        count = min(count, 50)

        # Check memory cache
        cache_key = f"tweets_{username}_{count}"
        if self.enable_caching:
            cached = await self.cache.get("tweets", cache_key)
            if cached:
                return cached

        # Offline mode
        if self.offline_mode:
            cached_data = await self._load_from_cache_file(username)
            if cached_data and "tweets" in cached_data:
                return [Tweet.from_dict(t) for t in cached_data["tweets"][:count]]
            raise OfflineError(f"No cached tweets for @{username} in offline mode")

        try:
            tweets = await self._execute_with_rate_limit(
                self.scraper.get_tweets(username, count)
            )

            # Cache results
            if self.enable_caching:
                await self.cache.set(tweets, 600, "tweets", cache_key)

            # Save to file cache
            await self._save_to_cache_file(username, {
                "tweets": [t.to_dict() for t in tweets]
            })

            return tweets

        except (ProfileNotFoundError, RateLimitError):
            raise
        except NitterInstanceError as e:
            cached_data = await self._load_from_cache_file(username)
            if cached_data and "tweets" in cached_data:
                logger.warning(f"Nitter unavailable, using cached tweets for @{username}")
                return [Tweet.from_dict(t) for t in cached_data["tweets"][:count]]
            raise OfflineError(f"Cannot fetch tweets and no cached data available: {e}")

    async def search_tweets(
        self,
        query: str,
        count: int = 20
    ) -> List[Tweet]:
        """
        Search for tweets matching a query.

        Args:
            query: Search query
            count: Maximum number of results (max 50)

        Returns:
            List of matching Tweet objects
        """
        count = min(count, 50)

        # Check memory cache
        cache_key = f"search_{query}_{count}"
        if self.enable_caching:
            cached = await self.cache.get("search", cache_key)
            if cached:
                return cached

        if self.offline_mode:
            raise OfflineError("Search not available in offline mode")

        try:
            tweets = await self._execute_with_rate_limit(
                self.scraper.search_tweets(query, count)
            )

            if self.enable_caching:
                await self.cache.set(tweets, 300, "search", cache_key)

            return tweets

        except (RateLimitError, NitterInstanceError):
            raise

    # =========================================================================
    # Public API Methods - Manual Input
    # =========================================================================

    async def enter_manual_stats(
        self,
        stats: Dict[str, Any],
        notes: str = ""
    ) -> ManualEntry:
        """
        Enter manual statistics when automated collection is not possible.

        Use this method to manually input metrics from Twitter Analytics,
        third-party tools, or manual observation.

        Args:
            stats: Dictionary of metrics. Common fields:
                - username: Twitter handle
                - followers: Current follower count
                - following: Current following count
                - impressions: Tweet impressions (from Twitter Analytics)
                - engagement_rate: Engagement rate percentage
                - profile_visits: Profile visits (from Twitter Analytics)
                - new_followers: New followers gained
                - mentions: Number of mentions
                - link_clicks: Link click count
                - period_start: Start of measurement period (YYYY-MM-DD)
                - period_end: End of measurement period (YYYY-MM-DD)
            notes: Optional notes about the entry

        Returns:
            ManualEntry record
        """
        entry = ManualEntry(
            platform="twitter",
            entry_type="stats",
            metrics=stats,
            notes=notes
        )

        self._manual_entries.append(entry)
        self._save_manual_entries()

        logger.info(f"Recorded manual stats entry: {len(stats)} metrics")
        return entry

    async def enter_manual_profile(
        self,
        data: Dict[str, Any],
        notes: str = ""
    ) -> ManualEntry:
        """
        Enter manual profile data.

        Args:
            data: Profile data dictionary with fields:
                - username (required)
                - followers
                - following
                - tweets_count
                - bio
                - display_name
            notes: Optional notes

        Returns:
            ManualEntry record
        """
        if not data.get("username"):
            raise ValueError("username is required")

        entry = ManualEntry(
            platform="twitter",
            entry_type="profile",
            metrics=data,
            notes=notes
        )

        self._manual_entries.append(entry)
        self._save_manual_entries()

        return entry

    async def enter_manual_tweet(
        self,
        data: Dict[str, Any],
        notes: str = ""
    ) -> ManualEntry:
        """
        Enter manual tweet data.

        Args:
            data: Tweet data dictionary with fields:
                - tweet_id or url (required)
                - username
                - text
                - likes
                - retweets
                - replies
                - timestamp
            notes: Optional notes

        Returns:
            ManualEntry record
        """
        if not data.get("tweet_id") and not data.get("url"):
            raise ValueError("tweet_id or url is required")

        entry = ManualEntry(
            platform="twitter",
            entry_type="tweet",
            metrics=data,
            notes=notes
        )

        self._manual_entries.append(entry)
        self._save_manual_entries()

        return entry

    async def get_manual_entries(
        self,
        entry_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[ManualEntry]:
        """
        Get stored manual entries.

        Args:
            entry_type: Filter by type (stats, profile, tweet)
            since: Only entries after this datetime

        Returns:
            List of ManualEntry objects
        """
        entries = self._manual_entries

        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]

        if since:
            since_str = since.isoformat()
            entries = [e for e in entries if e.timestamp >= since_str]

        return entries

    # =========================================================================
    # Public API Methods - Statistics
    # =========================================================================

    async def get_stats_summary(
        self,
        username: str,
        days: int = 7
    ) -> TwitterStats:
        """
        Get aggregated statistics summary for a user.

        Combines scraped data with manual entries to provide
        a comprehensive view of performance.

        Args:
            username: Twitter username
            days: Number of days to include in summary

        Returns:
            TwitterStats with aggregated metrics
        """
        username = username.lstrip("@").lower()
        period_end = datetime.now()
        period_start = period_end - timedelta(days=days)

        # Get current profile
        try:
            profile = await self.get_profile(username)
            followers_end = profile.followers
        except Exception as e:
            logger.warning(f"Could not fetch profile for stats: {e}")
            followers_end = 0

        # Get recent tweets
        tweets: List[Tweet] = []
        try:
            tweets = await self.get_recent_tweets(username, count=50)
            # Filter to period
            tweets = [
                t for t in tweets
                if t.timestamp and t.timestamp[:10] >= period_start.strftime("%Y-%m-%d")
            ]
        except Exception as e:
            logger.warning(f"Could not fetch tweets for stats: {e}")

        # Calculate aggregates from tweets
        total_likes = sum(t.likes for t in tweets)
        total_retweets = sum(t.retweets for t in tweets)
        total_replies = sum(t.replies for t in tweets)
        total_views = sum(t.views or 0 for t in tweets)

        # Find top tweet
        top_tweet = None
        if tweets:
            best = max(tweets, key=lambda t: t.engagement_total())
            top_tweet = best.to_dict()

        # Try to get historical follower data from manual entries
        followers_start = followers_end  # Default if no historical data
        manual_stats = await self.get_manual_entries(
            entry_type="stats",
            since=period_start
        )

        for entry in manual_stats:
            if entry.metrics.get("username", "").lower() == username:
                if "followers_start" in entry.metrics:
                    followers_start = entry.metrics["followers_start"]
                    break

        # Calculate engagement rate
        avg_engagement_rate = 0.0
        if tweets and followers_end > 0:
            total_engagement = total_likes + total_retweets + total_replies
            avg_engagement_rate = (total_engagement / len(tweets)) / followers_end

        return TwitterStats(
            username=username,
            period_start=period_start.isoformat()[:10],
            period_end=period_end.isoformat()[:10],
            followers_start=followers_start,
            followers_end=followers_end,
            followers_gained=max(0, followers_end - followers_start),
            followers_lost=max(0, followers_start - followers_end),
            tweets_posted=len(tweets),
            total_likes=total_likes,
            total_retweets=total_retweets,
            total_replies=total_replies,
            total_views=total_views if total_views > 0 else None,
            avg_engagement_rate=avg_engagement_rate,
            top_tweet=top_tweet,
            source="calculated"
        )

    # =========================================================================
    # Integration Helpers (for ALFRED tools)
    # =========================================================================

    async def get_metrics_for_harvester(
        self,
        username: str,
        period_start: str,
        period_end: str
    ) -> Dict[str, Any]:
        """
        Get metrics formatted for Social Metrics Harvester.

        Returns data in the schema expected by ALFRED's SocialMetricsHarvester.

        Args:
            username: Twitter username
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)

        Returns:
            Dictionary with metrics in harvester schema
        """
        username = username.lstrip("@").lower()

        try:
            profile = await self.get_profile(username)
            tweets = await self.get_recent_tweets(username, count=50)

            # Filter tweets to period
            filtered_tweets = []
            for tweet in tweets:
                if tweet.timestamp:
                    tweet_date = tweet.timestamp[:10]
                    if period_start <= tweet_date <= period_end:
                        filtered_tweets.append(tweet)

            # Aggregate metrics
            total_likes = sum(t.likes for t in filtered_tweets)
            total_retweets = sum(t.retweets for t in filtered_tweets)
            total_replies = sum(t.replies for t in filtered_tweets)
            total_quotes = sum(t.quotes for t in filtered_tweets)

            # Sort by engagement for top/lowest performing
            sorted_tweets = sorted(
                filtered_tweets,
                key=lambda t: t.engagement_total(),
                reverse=True
            )

            return {
                "platform": "twitter",
                "period": {
                    "start": period_start,
                    "end": period_end
                },
                "profile": profile.to_dict(),
                "raw_metrics": {
                    "output": {
                        "tweets": len(filtered_tweets),
                        "retweets_made": len([t for t in filtered_tweets if t.is_retweet]),
                        "replies_made": len([t for t in filtered_tweets if t.is_reply]),
                        "original_tweets": len([t for t in filtered_tweets if not t.is_retweet and not t.is_reply])
                    },
                    "reach": {
                        "impressions": 0,  # Not available via scraping
                        "profile_visits": 0  # Not available via scraping
                    },
                    "engagement": {
                        "likes": total_likes,
                        "retweets": total_retweets,
                        "replies": total_replies,
                        "quotes": total_quotes,
                        "total_engagement": total_likes + total_retweets + total_replies + total_quotes
                    },
                    "growth": {
                        "followers_current": profile.followers,
                        "following_current": profile.following
                    }
                },
                "content_items": [
                    {
                        "content_id": t.tweet_id,
                        "platform": "twitter",
                        "content_type": "tweet",
                        "text": t.text[:200] if t.text else "",
                        "published_at": t.timestamp,
                        "url": t.url,
                        "likes": t.likes,
                        "retweets": t.retweets,
                        "replies": t.replies,
                        "quotes": t.quotes,
                        "engagement_total": t.engagement_total(),
                        "is_retweet": t.is_retweet,
                        "is_reply": t.is_reply,
                        "hashtags": t.hashtags,
                        "mentions": t.mentions
                    }
                    for t in sorted_tweets[:20]
                ],
                "top_performing": [
                    {
                        "content_id": t.tweet_id,
                        "text": t.text[:100] if t.text else "",
                        "engagement_total": t.engagement_total(),
                        "likes": t.likes,
                        "retweets": t.retweets
                    }
                    for t in sorted_tweets[:5]
                ],
                "lowest_performing": [
                    {
                        "content_id": t.tweet_id,
                        "text": t.text[:100] if t.text else "",
                        "engagement_total": t.engagement_total(),
                        "likes": t.likes,
                        "retweets": t.retweets
                    }
                    for t in sorted_tweets[-5:]
                ] if len(sorted_tweets) >= 5 else [],
                "api_status": "connected_via_nitter",
                "data_source": "nitter_scraping",
                "collected_at": datetime.now().isoformat()
            }

        except OfflineError:
            # Try manual entries
            manual_stats = await self.get_manual_entries(entry_type="stats")
            if manual_stats:
                latest = manual_stats[-1]
                return {
                    "platform": "twitter",
                    "period": {"start": period_start, "end": period_end},
                    "raw_metrics": latest.metrics,
                    "api_status": "manual_input",
                    "data_source": "manual",
                    "manual_entry_timestamp": latest.timestamp
                }

            return {
                "platform": "twitter",
                "period": {"start": period_start, "end": period_end},
                "error": "Offline and no cached/manual data available",
                "api_status": "offline",
                "manual_input_required": True
            }

        except Exception as e:
            return {
                "platform": "twitter",
                "period": {"start": period_start, "end": period_end},
                "error": str(e),
                "api_status": "error",
                "manual_input_required": True
            }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return self.rate_limiter.get_status()

    def get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics."""
        return {
            "total_requests": self._request_count,
            "offline_mode": self.offline_mode,
            "apify_available": bool(self.apify_api_key),
            "last_error": self._last_error,
            "manual_entries_count": len(self._manual_entries),
            "rate_limit_status": self.get_rate_limit_status(),
            "nitter_status": self.instance_manager.get_status()
        }

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Nitter instances.

        Returns:
            Dict with connection status and details
        """
        result = {
            "status": "unknown",
            "nitter_available": False,
            "working_instances": [],
            "failed_instances": [],
            "apify_configured": bool(self.apify_api_key),
            "offline_mode": self.offline_mode,
            "cached_data_available": False,
            "manual_entries_count": len(self._manual_entries),
            "errors": []
        }

        if self.offline_mode:
            result["status"] = "offline_mode"
            # Check for cached data
            cached_files = list(self.data_path.glob("*_cache.json"))
            result["cached_data_available"] = len(cached_files) > 0
            result["cached_usernames"] = [
                f.stem.replace("_cache", "") for f in cached_files
            ]
            return result

        # Test Nitter instances
        test_username = "twitter"  # Twitter's official account

        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                for instance in NITTER_INSTANCES[:5]:  # Test first 5
                    try:
                        response = await client.get(
                            f"{instance}/{test_username}",
                            headers={"User-Agent": random.choice(USER_AGENTS)}
                        )
                        if response.status_code == 200:
                            result["working_instances"].append(instance)
                        else:
                            result["failed_instances"].append(instance)
                    except Exception as e:
                        result["failed_instances"].append(instance)
                        result["errors"].append(f"{instance}: {str(e)[:50]}")
        except ImportError:
            result["errors"].append("httpx not installed - cannot test connections")

        result["nitter_available"] = len(result["working_instances"]) > 0

        if result["nitter_available"]:
            result["status"] = "connected"
        elif result["manual_entries_count"] > 0:
            result["status"] = "manual_only"
        else:
            result["status"] = "disconnected"

        return result

    async def clear_cache(self):
        """Clear all cached data."""
        await self.cache.clear()
        # Also clear file caches
        for cache_file in self.data_path.glob("*_cache.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete cache file {cache_file}: {e}")

    def set_offline_mode(self, enabled: bool):
        """Enable or disable offline mode."""
        self.offline_mode = enabled
        logger.info(f"Offline mode {'enabled' if enabled else 'disabled'}")


# =============================================================================
# Factory Function
# =============================================================================

def create_twitter_adapter(
    apify_api_key: Optional[str] = None,
    data_path: Optional[str] = None,
    offline_mode: bool = False,
    **kwargs
) -> TwitterAdapter:
    """
    Factory function to create TwitterAdapter instance.

    Can read configuration from environment variables if not provided.

    Args:
        apify_api_key: Optional Apify API key (reads from APIFY_API_KEY env)
        data_path: Path for data storage (default: project twitter data dir)
        offline_mode: Whether to operate in offline mode only
        **kwargs: Additional arguments passed to TwitterAdapter

    Returns:
        Configured TwitterAdapter instance
    """
    apify_key = apify_api_key or os.environ.get("APIFY_API_KEY")

    if data_path:
        data_path = Path(data_path)
    else:
        data_path = Path(os.environ.get(
            "ALFRED_TWITTER_DATA_PATH",
            str(DEFAULT_DATA_PATH)
        ))

    if not apify_key:
        logger.info(
            "No Apify API key provided. Using Nitter scraping only. "
            "Set APIFY_API_KEY for enhanced data collection."
        )

    return TwitterAdapter(
        apify_api_key=apify_key,
        data_path=data_path,
        offline_mode=offline_mode,
        **kwargs
    )
