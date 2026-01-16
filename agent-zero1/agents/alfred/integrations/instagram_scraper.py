"""
Instagram Analytics Integration Adapter for ALFRED

Provides structured access to Instagram data using Instaloader library.
Connects ALFRED's Social Metrics Harvester and Content Strategy Analyst to Instagram.

Features:
- Profile statistics (followers, following, posts count)
- Recent posts with engagement metrics (likes, comments)
- Story indicators (when accessible)
- Hashtag analysis for posts
- Manual data input fallback for restricted access
- Rate limiting with exponential backoff
- Session management (login optional for public profiles)
- Aggressive caching to minimize requests
- Offline mode with cached data fallback

Note: This adapter uses Instaloader, an open-source library that works without
Meta's official API. Some features require login for private profiles.

Instagram Rate Limiting:
- Requests are throttled with random delays (2-5 seconds)
- Session persistence to minimize login frequency
- Automatic backoff on rate limit detection
- Graceful degradation when blocked

Storage: /Users/shaileshsingh/Alfred/agent-zero1/data/alfred/instagram/
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time

# Pydantic imports
try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    raise ImportError(
        "pydantic is required for InstagramAdapter. "
        "Install it with: pip install pydantic"
    )


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

STORAGE_PATH = Path("/Users/shaileshsingh/Alfred/agent-zero1/data/alfred/instagram")
DEFAULT_SESSION_FILE = STORAGE_PATH / "session"
CACHE_FILE = STORAGE_PATH / "cache.json"
MANUAL_ENTRIES_FILE = STORAGE_PATH / "manual_entries.json"
STATS_HISTORY_FILE = STORAGE_PATH / "stats_history.json"

# Rate limiting constants
MIN_REQUEST_DELAY = 2.0  # seconds
MAX_REQUEST_DELAY = 5.0  # seconds
REQUESTS_PER_MINUTE = 12
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_SECONDS = 300


# =============================================================================
# Error Classes
# =============================================================================

class InstagramError(Exception):
    """Base exception for Instagram adapter errors."""
    pass


class RateLimitError(InstagramError):
    """Rate limit exceeded, retry after cooldown."""
    def __init__(self, message: str, retry_after: int = 300):
        super().__init__(message)
        self.retry_after = retry_after


class ProfileNotFoundError(InstagramError):
    """Profile does not exist."""
    pass


class PrivateProfileError(InstagramError):
    """Profile is private and cannot be accessed without login."""
    pass


class LoginRequiredError(InstagramError):
    """Login is required for this operation."""
    pass


class BlockedError(InstagramError):
    """Request was blocked by Instagram."""
    def __init__(self, message: str, blocked_until: Optional[datetime] = None):
        super().__init__(message)
        self.blocked_until = blocked_until


class PostNotFoundError(InstagramError):
    """Post does not exist or is not accessible."""
    pass


class OfflineError(InstagramError):
    """Cannot connect to Instagram, using cached data if available."""
    pass


class SessionError(InstagramError):
    """Session-related error (expired, invalid, etc.)."""
    pass


# =============================================================================
# Pydantic Data Models
# =============================================================================

class InstagramProfile(BaseModel):
    """Instagram profile information and statistics."""
    username: str
    full_name: str = ""
    bio: str = ""
    followers: int = 0
    following: int = 0
    posts_count: int = 0
    is_verified: bool = False
    is_private: bool = False
    is_business: bool = False
    external_url: str = ""
    profile_pic_url: str = ""
    collected_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class InstagramPost(BaseModel):
    """Instagram post with engagement data."""
    shortcode: str
    caption: str = ""
    likes: int = 0
    comments: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    media_type: str = "image"  # image, video, carousel
    video_views: Optional[int] = None
    video_duration: Optional[float] = None
    carousel_count: int = 0
    location: str = ""
    engagement_rate: float = 0.0
    url: str = ""
    collected_at: datetime = Field(default_factory=datetime.now)

    @field_validator('url', mode='before')
    @classmethod
    def generate_url(cls, v: str, info) -> str:
        if v:
            return v
        shortcode = info.data.get('shortcode', '')
        if shortcode:
            return f"https://www.instagram.com/p/{shortcode}/"
        return ""

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class InstagramStats(BaseModel):
    """Aggregate Instagram engagement metrics."""
    username: str
    period_start: datetime
    period_end: datetime
    total_posts: int = 0
    total_likes: int = 0
    total_comments: int = 0
    total_video_views: int = 0
    avg_likes_per_post: float = 0.0
    avg_comments_per_post: float = 0.0
    avg_engagement_rate: float = 0.0
    top_hashtags: List[Dict[str, Any]] = Field(default_factory=list)
    best_performing_post: Optional[Dict[str, Any]] = None
    worst_performing_post: Optional[Dict[str, Any]] = None
    followers_at_start: int = 0
    followers_at_end: int = 0
    follower_growth: int = 0
    follower_growth_rate: float = 0.0
    collected_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class InstagramStory(BaseModel):
    """Instagram story indicator (limited data without official API)."""
    username: str
    has_active_stories: bool = False
    story_count: int = 0
    last_story_timestamp: Optional[datetime] = None
    collected_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class InstagramComment(BaseModel):
    """Instagram comment data."""
    comment_id: str
    post_shortcode: str
    author: str = ""
    text: str = ""
    likes: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    is_reply: bool = False
    parent_id: str = ""
    sentiment: str = ""  # Set by external analysis
    collected_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ManualEntry(BaseModel):
    """Manual data entry for metrics not accessible via scraping."""
    platform: str = "instagram"
    entry_type: str = "stats"  # stats, post, engagement, custom
    metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = "manual"
    notes: str = ""

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CacheEntry(BaseModel):
    """Cache entry with expiration."""
    data: Any
    expires_at: float
    created_at: datetime = Field(default_factory=datetime.now)

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


# =============================================================================
# Rate Limiter with Exponential Backoff
# =============================================================================

class RateLimiter:
    """
    Rate limiter for Instagram requests with exponential backoff.

    Instagram is aggressive with rate limiting. This limiter:
    - Enforces random delays between requests (2-5 seconds)
    - Tracks requests per minute
    - Implements exponential backoff on rate limit hits
    """

    def __init__(
        self,
        requests_per_minute: float = REQUESTS_PER_MINUTE,
        min_delay: float = MIN_REQUEST_DELAY,
        max_delay: float = MAX_REQUEST_DELAY
    ):
        self.requests_per_minute = requests_per_minute
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.request_count = 0
        self.minute_start = time.time()
        self.last_request_time = 0.0
        self.backoff_until = 0.0
        self.consecutive_errors = 0
        self._lock = asyncio.Lock()

    def _random_delay(self) -> float:
        """Generate random delay between min and max."""
        return random.uniform(self.min_delay, self.max_delay)

    async def acquire(self) -> bool:
        """
        Acquire permission for a request with rate limiting.

        Returns:
            True if request can proceed
        Raises:
            RateLimitError if currently in backoff
        """
        async with self._lock:
            current_time = time.time()

            # Check if in backoff period
            if current_time < self.backoff_until:
                wait_time = self.backoff_until - current_time
                raise RateLimitError(
                    f"In backoff period. Wait {wait_time:.1f} seconds.",
                    retry_after=int(wait_time)
                )

            # Reset counter each minute
            if current_time - self.minute_start > 60:
                self.request_count = 0
                self.minute_start = current_time

            # Check minute limit
            if self.request_count >= self.requests_per_minute:
                wait_time = 60 - (current_time - self.minute_start)
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    self.request_count = 0
                    self.minute_start = time.time()

            # Add random delay between requests
            time_since_last = current_time - self.last_request_time
            delay = self._random_delay()
            if time_since_last < delay:
                await asyncio.sleep(delay - time_since_last)

            self.request_count += 1
            self.last_request_time = time.time()
            return True

    def record_success(self):
        """Record successful request, reset error counter."""
        self.consecutive_errors = 0

    def record_error(self, is_rate_limit: bool = False):
        """Record failed request, trigger backoff if rate limited."""
        self.consecutive_errors += 1

        if is_rate_limit:
            # Exponential backoff
            backoff_seconds = min(
                BACKOFF_MULTIPLIER ** self.consecutive_errors * 10,
                MAX_BACKOFF_SECONDS
            )
            self.backoff_until = time.time() + backoff_seconds
            logger.warning(f"Rate limit hit, backing off for {backoff_seconds:.1f}s")

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        current_time = time.time()
        return {
            "requests_this_minute": self.request_count,
            "requests_per_minute_limit": self.requests_per_minute,
            "min_request_delay": self.min_delay,
            "max_request_delay": self.max_delay,
            "seconds_until_reset": max(0, 60 - (current_time - self.minute_start)),
            "in_backoff": current_time < self.backoff_until,
            "backoff_remaining": max(0, self.backoff_until - current_time),
            "consecutive_errors": self.consecutive_errors
        }


# =============================================================================
# Persistent Cache
# =============================================================================

class PersistentCache:
    """
    Persistent cache for Instagram responses.

    Caches data both in memory and on disk for offline mode support.
    """

    def __init__(
        self,
        default_ttl_seconds: int = 300,
        cache_file: Optional[Path] = None
    ):
        self.default_ttl = default_ttl_seconds
        self.cache_file = cache_file or CACHE_FILE
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._load_from_disk()

    def _load_from_disk(self):
        """Load cache from disk file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self._memory_cache[key] = CacheEntry(**entry_data)
                logger.debug(f"Loaded {len(self._memory_cache)} cache entries from disk")
        except Exception as e:
            logger.warning(f"Could not load cache from disk: {e}")

    def _save_to_disk(self):
        """Save cache to disk file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for key, entry in self._memory_cache.items():
                data[key] = {
                    "data": entry.data,
                    "expires_at": entry.expires_at,
                    "created_at": entry.created_at.isoformat()
                }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.warning(f"Could not save cache to disk: {e}")

    def _make_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    async def get(self, *args, allow_expired: bool = False) -> Optional[Any]:
        """
        Get cached value if exists.

        Args:
            *args: Cache key components
            allow_expired: If True, return expired data (for offline mode)

        Returns:
            Cached data or None
        """
        key = self._make_key(*args)
        async with self._lock:
            entry = self._memory_cache.get(key)
            if entry:
                if not entry.is_expired():
                    logger.debug(f"Cache hit for key: {key}")
                    return entry.data
                elif allow_expired:
                    logger.debug(f"Returning expired cache for key: {key} (offline mode)")
                    return entry.data
                else:
                    # Clean up expired entry
                    del self._memory_cache[key]
        return None

    async def set(self, value: Any, *args, ttl_seconds: Optional[int] = None):
        """Cache a value with optional custom TTL."""
        key = self._make_key(*args)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        async with self._lock:
            self._memory_cache[key] = CacheEntry(
                data=value,
                expires_at=time.time() + ttl
            )
            logger.debug(f"Cached value for key: {key}, TTL: {ttl}s")
            self._save_to_disk()

    async def invalidate(self, *args):
        """Invalidate a specific cache entry."""
        key = self._make_key(*args)
        async with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
                self._save_to_disk()

    async def clear(self):
        """Clear all cached entries."""
        async with self._lock:
            self._memory_cache.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()

    async def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        async with self._lock:
            expired_keys = [
                k for k, v in self._memory_cache.items()
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._memory_cache[key]
            if expired_keys:
                self._save_to_disk()
            return len(expired_keys)


# =============================================================================
# Manual Entry Storage
# =============================================================================

class ManualEntryStorage:
    """
    Storage for manually entered Instagram metrics.

    Persists entries to disk with timestamps for historical tracking.
    """

    def __init__(self, storage_file: Optional[Path] = None):
        self.storage_file = storage_file or MANUAL_ENTRIES_FILE
        self._entries: List[ManualEntry] = []
        self._lock = asyncio.Lock()
        self._load_entries()

    def _load_entries(self):
        """Load entries from disk."""
        try:
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self._entries = [ManualEntry(**entry) for entry in data]
                logger.debug(f"Loaded {len(self._entries)} manual entries")
        except Exception as e:
            logger.warning(f"Could not load manual entries: {e}")

    def _save_entries(self):
        """Save entries to disk."""
        try:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            data = [entry.model_dump() for entry in self._entries]
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            logger.warning(f"Could not save manual entries: {e}")

    async def add_entry(self, entry: ManualEntry) -> ManualEntry:
        """Add a new manual entry."""
        async with self._lock:
            self._entries.append(entry)
            self._save_entries()
            return entry

    async def get_entries(
        self,
        entry_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ManualEntry]:
        """Get manual entries with optional filtering."""
        async with self._lock:
            entries = self._entries.copy()

            if entry_type:
                entries = [e for e in entries if e.entry_type == entry_type]

            if since:
                entries = [e for e in entries if e.timestamp >= since]

            # Sort by timestamp descending
            entries.sort(key=lambda e: e.timestamp, reverse=True)

            return entries[:limit]

    async def get_latest_stats(self, username: Optional[str] = None) -> Optional[ManualEntry]:
        """Get the most recent stats entry."""
        async with self._lock:
            stats_entries = [
                e for e in self._entries
                if e.entry_type == "stats"
            ]

            if username:
                stats_entries = [
                    e for e in stats_entries
                    if e.metrics.get("username") == username
                ]

            if stats_entries:
                stats_entries.sort(key=lambda e: e.timestamp, reverse=True)
                return stats_entries[0]
            return None


# =============================================================================
# Instagram Adapter
# =============================================================================

class InstagramAdapter:
    """
    Instagram Analytics Integration Adapter for ALFRED.

    Provides structured access to Instagram data using Instaloader library.
    Designed to work with ALFRED's Social Metrics Harvester and Content Strategy Analyst.

    Features:
    - Profile stats (followers, following, posts count)
    - Recent posts with engagement (likes, comments)
    - Story indicators (if accessible)
    - Hashtag analysis for posts
    - Manual input fallback
    - Aggressive caching with offline mode
    - Rate limit handling with exponential backoff

    Usage:
        adapter = InstagramAdapter()

        # Get profile info
        profile = await adapter.get_profile("username")

        # Get recent posts
        posts = await adapter.get_recent_posts("username", count=12)

        # Calculate engagement rate
        rate = await adapter.get_engagement_rate("username")

        # Manual entry (always available)
        entry = await adapter.enter_manual_stats({
            "followers": 10000,
            "following": 500,
            "posts_count": 150
        })

        # Get stats summary
        summary = await adapter.get_stats_summary("username", days=30)
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        session_file: Optional[Path] = None,
        cache_ttl_seconds: int = 600,
        enable_caching: bool = True,
        storage_path: Optional[Path] = None,
        offline_mode: bool = False
    ):
        """
        Initialize Instagram adapter.

        Args:
            username: Instagram username for login (optional, from env if not provided)
            password: Instagram password for login (optional, from env if not provided)
            session_file: Path to save/load session file
            cache_ttl_seconds: Default cache TTL (aggressive caching by default)
            enable_caching: Whether to enable response caching
            storage_path: Base path for data storage
            offline_mode: If True, return cached data when requests fail
        """
        self.username = username or os.environ.get("INSTAGRAM_USERNAME")
        self.password = password or os.environ.get("INSTAGRAM_PASSWORD")
        self.storage_path = storage_path or STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.session_file = session_file or (self.storage_path / "session")
        self.enable_caching = enable_caching
        self.offline_mode = offline_mode

        # Initialize rate limiter
        self.rate_limiter = RateLimiter()

        # Initialize cache
        self.cache = PersistentCache(
            default_ttl_seconds=cache_ttl_seconds,
            cache_file=self.storage_path / "cache.json"
        )

        # Initialize manual entry storage
        self.manual_storage = ManualEntryStorage(
            storage_file=self.storage_path / "manual_entries.json"
        )

        # Instaloader instance (initialized lazily)
        self._loader = None
        self._logged_in = False

        # Request tracking
        self._request_count = 0
        self._last_error: Optional[str] = None
        self._blocked = False
        self._blocked_until: Optional[datetime] = None

        logger.info(
            f"InstagramAdapter initialized "
            f"(credentials: {bool(self.username)}, "
            f"offline_mode: {offline_mode})"
        )

    # =========================================================================
    # Loader Management
    # =========================================================================

    def _get_loader(self):
        """Get or create Instaloader instance."""
        if self._loader is None:
            try:
                import instaloader
                self._loader = instaloader.Instaloader(
                    download_pictures=False,
                    download_videos=False,
                    download_video_thumbnails=False,
                    download_geotags=False,
                    download_comments=False,
                    save_metadata=False,
                    compress_json=False,
                    quiet=True,
                    request_timeout=30.0
                )
                self._try_load_session()
            except ImportError:
                raise ImportError(
                    "instaloader is required for InstagramAdapter. "
                    "Install it with: pip install instaloader"
                )
        return self._loader

    def _try_load_session(self):
        """Try to load an existing session file."""
        if not self.username:
            return

        try:
            session_path = Path(self.session_file)
            if session_path.exists():
                self._loader.load_session_from_file(self.username, str(session_path))
                self._logged_in = True
                logger.info(f"Loaded existing session for {self.username}")
        except Exception as e:
            logger.debug(f"Could not load session: {e}")

    def _save_session(self):
        """Save current session to file."""
        if not self.username or not self._logged_in or not self._loader:
            return

        try:
            session_path = Path(self.session_file)
            session_path.parent.mkdir(parents=True, exist_ok=True)
            self._loader.save_session_to_file(str(session_path))
            logger.info(f"Session saved to {session_path}")
        except Exception as e:
            logger.warning(f"Could not save session: {e}")

    async def login(self, force: bool = False) -> bool:
        """
        Login to Instagram.

        Args:
            force: Force re-login even if already logged in

        Returns:
            True if login successful

        Raises:
            LoginRequiredError: If credentials not available
            BlockedError: If login is blocked
            SessionError: If login fails
        """
        if self._logged_in and not force:
            return True

        if not self.username or not self.password:
            raise LoginRequiredError(
                "Instagram credentials required. Set INSTAGRAM_USERNAME and "
                "INSTAGRAM_PASSWORD environment variables."
            )

        loader = self._get_loader()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: loader.login(self.username, self.password)
            )
            self._logged_in = True
            self._save_session()
            logger.info(f"Successfully logged in as {self.username}")
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "checkpoint" in error_str or "challenge" in error_str:
                raise BlockedError(
                    "Instagram requires additional verification (checkpoint challenge). "
                    "Please log in via the Instagram app/website first to verify your account."
                )
            elif "bad password" in error_str or "incorrect" in error_str:
                raise SessionError("Invalid Instagram username or password")
            elif "rate" in error_str or "limit" in error_str:
                self.rate_limiter.record_error(is_rate_limit=True)
                raise RateLimitError("Login rate limited. Try again later.", retry_after=600)
            else:
                raise SessionError(f"Login failed: {str(e)}")

    async def close(self):
        """Close adapter and cleanup resources."""
        if self._logged_in:
            self._save_session()
        self._loader = None
        self._logged_in = False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _execute_with_rate_limit(self, func, *args, **kwargs):
        """Execute a function with rate limiting and error handling."""
        # Check if blocked
        if self._blocked:
            if self._blocked_until and datetime.now() < self._blocked_until:
                if self.offline_mode:
                    raise OfflineError("Currently blocked, using cached data")
                raise BlockedError(
                    f"Blocked until {self._blocked_until.isoformat()}",
                    blocked_until=self._blocked_until
                )
            else:
                self._blocked = False
                self._blocked_until = None

        await self.rate_limiter.acquire()
        self._request_count += 1

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            self.rate_limiter.record_success()
            return result
        except Exception as e:
            self._last_error = str(e)
            error_str = str(e).lower()

            if "rate" in error_str or "429" in error_str or "please wait" in error_str:
                self.rate_limiter.record_error(is_rate_limit=True)
                raise RateLimitError(
                    "Instagram rate limit hit. Please wait before trying again.",
                    retry_after=300
                )
            elif "blocked" in error_str or "ban" in error_str:
                self._blocked = True
                self._blocked_until = datetime.now() + timedelta(hours=1)
                raise BlockedError(
                    f"Request blocked by Instagram. Blocked until {self._blocked_until.isoformat()}",
                    blocked_until=self._blocked_until
                )
            elif "not found" in error_str or "does not exist" in error_str:
                raise ProfileNotFoundError(f"Resource not found: {str(e)}")
            elif "private" in error_str:
                raise PrivateProfileError("Profile is private and cannot be accessed")
            elif "login" in error_str or "authentication" in error_str:
                raise LoginRequiredError(f"Authentication required: {str(e)}")
            elif "connection" in error_str or "network" in error_str:
                raise OfflineError(f"Connection error: {str(e)}")
            else:
                raise InstagramError(f"Instagram error: {str(e)}")

    def _extract_shortcode_from_url(self, url: str) -> str:
        """Extract shortcode from Instagram URL."""
        patterns = [
            r'instagram\.com/p/([A-Za-z0-9_-]+)',
            r'instagram\.com/reel/([A-Za-z0-9_-]+)',
            r'instagram\.com/tv/([A-Za-z0-9_-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return url

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        if not text:
            return []
        return re.findall(r'#(\w+)', text)

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text."""
        if not text:
            return []
        return re.findall(r'@(\w+)', text)

    def _analyze_hashtags(self, posts: List[InstagramPost]) -> List[Dict[str, Any]]:
        """Analyze hashtag usage across posts."""
        hashtag_stats: Dict[str, Dict[str, Any]] = {}

        for post in posts:
            for tag in post.hashtags:
                tag_lower = tag.lower()
                if tag_lower not in hashtag_stats:
                    hashtag_stats[tag_lower] = {
                        "hashtag": tag,
                        "count": 0,
                        "total_likes": 0,
                        "total_comments": 0,
                        "posts": []
                    }
                hashtag_stats[tag_lower]["count"] += 1
                hashtag_stats[tag_lower]["total_likes"] += post.likes
                hashtag_stats[tag_lower]["total_comments"] += post.comments
                hashtag_stats[tag_lower]["posts"].append(post.shortcode)

        # Calculate averages and sort by usage
        result = []
        for tag, stats in hashtag_stats.items():
            if stats["count"] > 0:
                stats["avg_likes"] = stats["total_likes"] / stats["count"]
                stats["avg_comments"] = stats["total_comments"] / stats["count"]
                stats["avg_engagement"] = (stats["avg_likes"] + stats["avg_comments"])
                del stats["posts"]  # Remove post list for cleaner output
                result.append(stats)

        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:20]  # Top 20 hashtags

    # =========================================================================
    # Public API Methods
    # =========================================================================

    async def get_profile(self, username: str) -> InstagramProfile:
        """
        Get profile information for a user.

        Args:
            username: Instagram username (without @)

        Returns:
            InstagramProfile with stats and metadata

        Raises:
            ProfileNotFoundError: If profile doesn't exist
            PrivateProfileError: If profile is private
            RateLimitError: If rate limited
            OfflineError: If offline and no cache available
        """
        username = username.lstrip("@").lower()

        # Check cache first
        if self.enable_caching:
            cached = await self.cache.get("profile", username)
            if cached:
                return InstagramProfile(**cached) if isinstance(cached, dict) else cached

        try:
            loader = self._get_loader()
            import instaloader

            def fetch_profile():
                return instaloader.Profile.from_username(loader.context, username)

            profile = await self._execute_with_rate_limit(fetch_profile)

            result = InstagramProfile(
                username=profile.username,
                full_name=profile.full_name or "",
                bio=profile.biography or "",
                followers=profile.followers,
                following=profile.followees,
                posts_count=profile.mediacount,
                is_verified=profile.is_verified,
                is_private=profile.is_private,
                is_business=profile.is_business_account,
                external_url=profile.external_url or "",
                profile_pic_url=profile.profile_pic_url or ""
            )

            # Cache the result
            if self.enable_caching:
                await self.cache.set(result.model_dump(), "profile", username, ttl_seconds=3600)

            return result

        except (RateLimitError, OfflineError, BlockedError):
            # Try offline mode with cached data
            if self.offline_mode:
                cached = await self.cache.get("profile", username, allow_expired=True)
                if cached:
                    logger.info(f"Returning cached profile for {username} (offline mode)")
                    return InstagramProfile(**cached) if isinstance(cached, dict) else cached
            raise

    async def get_recent_posts(
        self,
        username: str,
        count: int = 12
    ) -> List[InstagramPost]:
        """
        Get recent posts for a user with engagement data.

        Args:
            username: Instagram username
            count: Number of posts to retrieve (default 12)

        Returns:
            List of InstagramPost objects with engagement metrics

        Raises:
            PrivateProfileError: If profile is private
            RateLimitError: If rate limited
        """
        username = username.lstrip("@").lower()
        cache_key = f"posts_{username}_{count}"

        # Check cache
        if self.enable_caching:
            cached = await self.cache.get("recent_posts", cache_key)
            if cached:
                return [InstagramPost(**p) if isinstance(p, dict) else p for p in cached]

        try:
            loader = self._get_loader()
            import instaloader

            def fetch_posts():
                profile = instaloader.Profile.from_username(loader.context, username)
                if profile.is_private and not self._logged_in:
                    raise PrivateProfileError(f"Profile @{username} is private")

                posts = []
                for i, post in enumerate(profile.get_posts()):
                    if i >= count:
                        break

                    # Determine media type
                    if post.typename == "GraphSidecar":
                        media_type = "carousel"
                        carousel_count = len(list(post.get_sidecar_nodes()))
                    elif post.is_video:
                        media_type = "video"
                        carousel_count = 0
                    else:
                        media_type = "image"
                        carousel_count = 0

                    # Calculate engagement rate
                    total_engagement = post.likes + post.comments
                    engagement_rate = total_engagement / profile.followers if profile.followers > 0 else 0

                    posts.append(InstagramPost(
                        shortcode=post.shortcode,
                        caption=post.caption or "",
                        likes=post.likes,
                        comments=post.comments,
                        timestamp=post.date_utc if post.date_utc else datetime.now(),
                        hashtags=self._extract_hashtags(post.caption),
                        mentions=self._extract_mentions(post.caption),
                        media_type=media_type,
                        video_views=post.video_view_count if post.is_video else None,
                        video_duration=post.video_duration if post.is_video else None,
                        carousel_count=carousel_count,
                        location=post.location.name if post.location else "",
                        engagement_rate=engagement_rate
                    ))

                    # Add small delay between post fetches
                    time.sleep(random.uniform(0.5, 1.5))

                return posts

            posts = await self._execute_with_rate_limit(fetch_posts)

            # Cache the result
            if self.enable_caching:
                await self.cache.set(
                    [p.model_dump() for p in posts],
                    "recent_posts",
                    cache_key,
                    ttl_seconds=600
                )

            return posts

        except (RateLimitError, OfflineError, BlockedError):
            if self.offline_mode:
                cached = await self.cache.get("recent_posts", cache_key, allow_expired=True)
                if cached:
                    logger.info(f"Returning cached posts for {username} (offline mode)")
                    return [InstagramPost(**p) if isinstance(p, dict) else p for p in cached]
            raise

    async def get_post_stats(self, shortcode: str) -> InstagramPost:
        """
        Get statistics for a single post.

        Args:
            shortcode: Post shortcode or full URL

        Returns:
            InstagramPost with engagement data

        Raises:
            PostNotFoundError: If post doesn't exist
        """
        shortcode = self._extract_shortcode_from_url(shortcode)

        # Check cache
        if self.enable_caching:
            cached = await self.cache.get("post", shortcode)
            if cached:
                return InstagramPost(**cached) if isinstance(cached, dict) else cached

        try:
            loader = self._get_loader()
            import instaloader

            def fetch_post():
                post = instaloader.Post.from_shortcode(loader.context, shortcode)

                # Determine media type
                if post.typename == "GraphSidecar":
                    media_type = "carousel"
                    carousel_count = len(list(post.get_sidecar_nodes()))
                elif post.is_video:
                    media_type = "video"
                    carousel_count = 0
                else:
                    media_type = "image"
                    carousel_count = 0

                return InstagramPost(
                    shortcode=post.shortcode,
                    caption=post.caption or "",
                    likes=post.likes,
                    comments=post.comments,
                    timestamp=post.date_utc if post.date_utc else datetime.now(),
                    hashtags=self._extract_hashtags(post.caption),
                    mentions=self._extract_mentions(post.caption),
                    media_type=media_type,
                    video_views=post.video_view_count if post.is_video else None,
                    video_duration=post.video_duration if post.is_video else None,
                    carousel_count=carousel_count,
                    location=post.location.name if post.location else ""
                )

            post = await self._execute_with_rate_limit(fetch_post)

            # Cache the result
            if self.enable_caching:
                await self.cache.set(post.model_dump(), "post", shortcode, ttl_seconds=600)

            return post

        except (RateLimitError, OfflineError, BlockedError):
            if self.offline_mode:
                cached = await self.cache.get("post", shortcode, allow_expired=True)
                if cached:
                    return InstagramPost(**cached) if isinstance(cached, dict) else cached
            raise
        except Exception as e:
            raise PostNotFoundError(f"Could not fetch post {shortcode}: {str(e)}")

    async def get_engagement_rate(self, username: str, post_count: int = 12) -> float:
        """
        Calculate engagement rate for a user.

        Engagement rate = (Total likes + comments) / (Posts * Followers) * 100

        Args:
            username: Instagram username
            post_count: Number of recent posts to analyze

        Returns:
            Engagement rate as percentage
        """
        username = username.lstrip("@").lower()

        try:
            profile = await self.get_profile(username)
            posts = await self.get_recent_posts(username, count=post_count)

            if not posts or profile.followers == 0:
                return 0.0

            total_engagement = sum(p.likes + p.comments for p in posts)
            engagement_rate = (total_engagement / len(posts)) / profile.followers * 100

            return round(engagement_rate, 4)

        except Exception as e:
            logger.error(f"Error calculating engagement rate: {e}")
            return 0.0

    async def get_story_indicators(self, username: str) -> InstagramStory:
        """
        Get story indicators for a user.

        Note: Full story content requires login and may be limited.

        Args:
            username: Instagram username

        Returns:
            InstagramStory with availability indicators
        """
        username = username.lstrip("@").lower()

        # Check cache
        if self.enable_caching:
            cached = await self.cache.get("story", username)
            if cached:
                return InstagramStory(**cached) if isinstance(cached, dict) else cached

        try:
            loader = self._get_loader()
            import instaloader

            def check_stories():
                profile = instaloader.Profile.from_username(loader.context, username)

                # Check if profile has stories (requires login for actual content)
                has_stories = False
                story_count = 0
                last_timestamp = None

                # Try to get story count if logged in
                if self._logged_in:
                    try:
                        stories = list(loader.get_stories(userids=[profile.userid]))
                        if stories:
                            for story in stories:
                                story_items = list(story.get_items())
                                story_count += len(story_items)
                                if story_items:
                                    has_stories = True
                                    last_timestamp = max(
                                        s.date_utc for s in story_items
                                    )
                    except Exception as e:
                        logger.debug(f"Could not fetch stories: {e}")

                return InstagramStory(
                    username=username,
                    has_active_stories=has_stories,
                    story_count=story_count,
                    last_story_timestamp=last_timestamp
                )

            story_info = await self._execute_with_rate_limit(check_stories)

            # Cache the result (short TTL since stories expire)
            if self.enable_caching:
                await self.cache.set(
                    story_info.model_dump(),
                    "story",
                    username,
                    ttl_seconds=300
                )

            return story_info

        except Exception as e:
            logger.warning(f"Could not fetch story indicators: {e}")
            return InstagramStory(username=username)

    # =========================================================================
    # Manual Input Methods
    # =========================================================================

    async def enter_manual_stats(
        self,
        stats: Dict[str, Any],
        notes: str = ""
    ) -> ManualEntry:
        """
        Enter manual statistics for Instagram.

        Use this method when automated collection is not possible
        (e.g., rate limited, blocked, or for metrics only available in-app).

        Args:
            stats: Dictionary with metrics:
                - username (recommended)
                - followers (optional)
                - following (optional)
                - posts_count (optional)
                - impressions (optional, from Instagram Insights)
                - reach (optional, from Instagram Insights)
                - profile_visits (optional)
                - website_clicks (optional)
                - engagement_rate (optional)
                - story_views (optional)
                - reels_views (optional)
                - Any other custom metrics
            notes: Optional notes about the entry

        Returns:
            ManualEntry object with timestamp

        Example:
            entry = await adapter.enter_manual_stats({
                "username": "myaccount",
                "followers": 10500,
                "following": 450,
                "posts_count": 180,
                "impressions": 25000,
                "reach": 18000,
                "profile_visits": 500,
                "engagement_rate": 4.5
            }, notes="Weekly stats from Instagram Insights")
        """
        entry = ManualEntry(
            platform="instagram",
            entry_type="stats",
            metrics=stats,
            timestamp=datetime.now(),
            source="manual",
            notes=notes
        )

        await self.manual_storage.add_entry(entry)
        logger.info(f"Manual stats entry saved with timestamp {entry.timestamp}")
        return entry

    async def get_manual_entries(
        self,
        entry_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50
    ) -> List[ManualEntry]:
        """
        Get stored manual entries.

        Args:
            entry_type: Filter by entry type (stats, post, engagement, custom)
            since: Only return entries after this datetime
            limit: Maximum number of entries to return

        Returns:
            List of ManualEntry objects, most recent first
        """
        return await self.manual_storage.get_entries(
            entry_type=entry_type,
            since=since,
            limit=limit
        )

    # =========================================================================
    # Stats Summary Methods
    # =========================================================================

    async def get_stats_summary(
        self,
        username: str,
        days: int = 30
    ) -> InstagramStats:
        """
        Get aggregate statistics summary for a period.

        Combines automated scraping with manual entries if available.

        Args:
            username: Instagram username
            days: Number of days to analyze (default 30)

        Returns:
            InstagramStats with aggregate metrics
        """
        username = username.lstrip("@").lower()
        period_start = datetime.now() - timedelta(days=days)
        period_end = datetime.now()

        try:
            # Get profile
            profile = await self.get_profile(username)

            # Get posts
            posts = await self.get_recent_posts(username, count=50)

            # Filter posts by period
            period_posts = [
                p for p in posts
                if p.timestamp >= period_start
            ]

            if not period_posts:
                period_posts = posts[:12]  # Use recent posts if none in period

            # Calculate aggregate stats
            total_likes = sum(p.likes for p in period_posts)
            total_comments = sum(p.comments for p in period_posts)
            total_video_views = sum(p.video_views or 0 for p in period_posts)

            avg_likes = total_likes / len(period_posts) if period_posts else 0
            avg_comments = total_comments / len(period_posts) if period_posts else 0
            avg_engagement = sum(p.engagement_rate for p in period_posts) / len(period_posts) if period_posts else 0

            # Analyze hashtags
            hashtag_analysis = self._analyze_hashtags(period_posts)

            # Find best/worst performing posts
            sorted_posts = sorted(period_posts, key=lambda p: p.engagement_rate, reverse=True)
            best_post = sorted_posts[0] if sorted_posts else None
            worst_post = sorted_posts[-1] if len(sorted_posts) > 1 else None

            # Check for manual entries with follower history
            manual_entries = await self.manual_storage.get_entries(
                entry_type="stats",
                since=period_start
            )
            manual_entries = [e for e in manual_entries if e.metrics.get("username") == username]

            followers_at_start = profile.followers
            if manual_entries:
                # Get earliest entry in period for comparison
                oldest_entry = min(manual_entries, key=lambda e: e.timestamp)
                followers_at_start = oldest_entry.metrics.get("followers", profile.followers)

            follower_growth = profile.followers - followers_at_start
            follower_growth_rate = (follower_growth / followers_at_start * 100) if followers_at_start > 0 else 0

            return InstagramStats(
                username=username,
                period_start=period_start,
                period_end=period_end,
                total_posts=len(period_posts),
                total_likes=total_likes,
                total_comments=total_comments,
                total_video_views=total_video_views,
                avg_likes_per_post=round(avg_likes, 2),
                avg_comments_per_post=round(avg_comments, 2),
                avg_engagement_rate=round(avg_engagement, 4),
                top_hashtags=hashtag_analysis,
                best_performing_post={
                    "shortcode": best_post.shortcode,
                    "likes": best_post.likes,
                    "comments": best_post.comments,
                    "engagement_rate": best_post.engagement_rate,
                    "url": best_post.url
                } if best_post else None,
                worst_performing_post={
                    "shortcode": worst_post.shortcode,
                    "likes": worst_post.likes,
                    "comments": worst_post.comments,
                    "engagement_rate": worst_post.engagement_rate,
                    "url": worst_post.url
                } if worst_post else None,
                followers_at_start=followers_at_start,
                followers_at_end=profile.followers,
                follower_growth=follower_growth,
                follower_growth_rate=round(follower_growth_rate, 2)
            )

        except Exception as e:
            logger.error(f"Error generating stats summary: {e}")
            # Return minimal stats from manual entries if available
            latest_manual = await self.manual_storage.get_latest_stats(username)
            if latest_manual:
                return InstagramStats(
                    username=username,
                    period_start=period_start,
                    period_end=period_end,
                    followers_at_end=latest_manual.metrics.get("followers", 0),
                    avg_engagement_rate=latest_manual.metrics.get("engagement_rate", 0)
                )
            raise

    # =========================================================================
    # Comments Methods
    # =========================================================================

    async def get_post_comments(
        self,
        shortcode: str,
        limit: int = 50
    ) -> List[InstagramComment]:
        """
        Get comments for a specific post.

        Note: May require login for some posts.

        Args:
            shortcode: Post shortcode or URL
            limit: Maximum number of comments

        Returns:
            List of InstagramComment objects
        """
        shortcode = self._extract_shortcode_from_url(shortcode)
        cache_key = f"comments_{shortcode}_{limit}"

        # Check cache
        if self.enable_caching:
            cached = await self.cache.get("comments", cache_key)
            if cached:
                return [InstagramComment(**c) if isinstance(c, dict) else c for c in cached]

        # Login may be required
        if not self._logged_in and self.username and self.password:
            await self.login()

        try:
            loader = self._get_loader()
            import instaloader

            def fetch_comments():
                post = instaloader.Post.from_shortcode(loader.context, shortcode)
                comments = []

                for i, comment in enumerate(post.get_comments()):
                    if i >= limit:
                        break

                    comments.append(InstagramComment(
                        comment_id=str(comment.id),
                        post_shortcode=shortcode,
                        author=comment.owner.username,
                        text=comment.text,
                        likes=comment.likes_count,
                        timestamp=comment.created_at_utc if comment.created_at_utc else datetime.now()
                    ))

                    # Rate limiting
                    if i > 0 and i % 10 == 0:
                        time.sleep(random.uniform(1, 2))

                return comments

            comments = await self._execute_with_rate_limit(fetch_comments)

            # Cache the result
            if self.enable_caching:
                await self.cache.set(
                    [c.model_dump() for c in comments],
                    "comments",
                    cache_key,
                    ttl_seconds=600
                )

            return comments

        except Exception as e:
            logger.warning(f"Could not fetch comments for {shortcode}: {e}")
            return []

    # =========================================================================
    # Integration Helpers (for ALFRED tools)
    # =========================================================================

    async def get_metrics_for_harvester(
        self,
        username: str,
        period_start: str,
        period_end: str,
        post_limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get metrics formatted for Social Metrics Harvester.

        Returns data in the schema expected by SocialMetricsHarvester.
        """
        try:
            profile = await self.get_profile(username)
            posts = await self.get_recent_posts(username, count=post_limit)

            # Filter posts by date range
            start_dt = datetime.fromisoformat(period_start)
            end_dt = datetime.fromisoformat(period_end)

            filtered_posts = [
                p for p in posts
                if start_dt <= p.timestamp <= end_dt
            ]

            # Use all fetched posts if none in range
            if not filtered_posts:
                filtered_posts = posts

            # Aggregate metrics
            total_likes = sum(p.likes for p in filtered_posts)
            total_comments = sum(p.comments for p in filtered_posts)
            video_views = sum(p.video_views or 0 for p in filtered_posts)

            # Sort by engagement
            sorted_posts = sorted(
                filtered_posts,
                key=lambda p: p.engagement_rate,
                reverse=True
            )

            # Hashtag analysis
            hashtag_analysis = self._analyze_hashtags(filtered_posts)

            return {
                "platform": "instagram",
                "period": {
                    "start": period_start,
                    "end": period_end
                },
                "profile": profile.model_dump(),
                "raw_metrics": {
                    "output": {
                        "posts": len(filtered_posts),
                        "videos": len([p for p in filtered_posts if p.media_type == "video"]),
                        "images": len([p for p in filtered_posts if p.media_type == "image"]),
                        "carousels": len([p for p in filtered_posts if p.media_type == "carousel"])
                    },
                    "reach": {
                        "video_views": video_views,
                        "estimated_impressions": 0
                    },
                    "engagement": {
                        "likes": total_likes,
                        "comments": total_comments,
                        "total_engagement": total_likes + total_comments,
                        "avg_engagement_rate": sum(p.engagement_rate for p in filtered_posts) / len(
                            filtered_posts) if filtered_posts else 0
                    },
                    "growth": {
                        "followers_current": profile.followers,
                        "following_current": profile.following
                    }
                },
                "content_items": [p.model_dump() for p in sorted_posts],
                "top_performing": [
                    {
                        "shortcode": p.shortcode,
                        "engagement_rate": p.engagement_rate,
                        "likes": p.likes,
                        "comments": p.comments,
                        "url": p.url
                    }
                    for p in sorted_posts[:5]
                ],
                "lowest_performing": [
                    {
                        "shortcode": p.shortcode,
                        "engagement_rate": p.engagement_rate,
                        "likes": p.likes,
                        "comments": p.comments,
                        "url": p.url
                    }
                    for p in sorted_posts[-5:]
                ] if len(sorted_posts) >= 5 else [],
                "hashtag_analysis": hashtag_analysis,
                "api_status": "connected",
                "collected_at": datetime.now().isoformat()
            }

        except PrivateProfileError:
            return {
                "platform": "instagram",
                "period": {"start": period_start, "end": period_end},
                "error": "Profile is private",
                "api_status": "private_profile",
                "manual_input_required": True
            }
        except (RateLimitError, BlockedError) as e:
            return {
                "platform": "instagram",
                "period": {"start": period_start, "end": period_end},
                "error": str(e),
                "api_status": "rate_limited",
                "manual_input_required": True
            }
        except Exception as e:
            return {
                "platform": "instagram",
                "period": {"start": period_start, "end": period_end},
                "error": str(e),
                "api_status": "error",
                "manual_input_required": True
            }

    async def get_comments_for_audience_extractor(
        self,
        username: str,
        post_limit: int = 5,
        comments_per_post: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get comments formatted for Audience Signals Extractor.

        Returns comments in the schema expected by AudienceSignalsExtractor.
        """
        all_comments = []

        try:
            posts = await self.get_recent_posts(username, count=post_limit)

            for post in posts:
                try:
                    comments = await self.get_post_comments(
                        post.shortcode,
                        limit=comments_per_post
                    )

                    for comment in comments:
                        all_comments.append({
                            "platform": "instagram",
                            "content_id": post.shortcode,
                            "text": comment.text,
                            "author": comment.author,
                            "timestamp": comment.timestamp.isoformat(),
                            "like_count": comment.likes,
                            "comment_id": comment.comment_id
                        })

                except Exception as e:
                    logger.warning(f"Could not fetch comments for post {post.shortcode}: {e}")

        except Exception as e:
            logger.error(f"Error fetching comments for audience extraction: {e}")

        return all_comments

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
            "logged_in": self._logged_in,
            "username": self.username if self._logged_in else None,
            "blocked": self._blocked,
            "blocked_until": self._blocked_until.isoformat() if self._blocked_until else None,
            "last_error": self._last_error,
            "offline_mode": self.offline_mode,
            "rate_limit_status": self.get_rate_limit_status()
        }

    async def test_connection(self, test_username: str = "instagram") -> Dict[str, Any]:
        """
        Test connection to Instagram.

        Args:
            test_username: Username to test with (default: official instagram account)

        Returns:
            Dict with connection status and any error details
        """
        result = {
            "status": "unknown",
            "can_fetch_public": False,
            "logged_in": self._logged_in,
            "session_file_exists": self.session_file.exists() if self.session_file else False,
            "credentials_available": bool(self.username and self.password),
            "offline_mode": self.offline_mode,
            "blocked": self._blocked,
            "errors": []
        }

        try:
            profile = await self.get_profile(test_username)
            result["can_fetch_public"] = True
            result["test_profile"] = {
                "username": profile.username,
                "followers": profile.followers,
                "is_verified": profile.is_verified
            }
        except RateLimitError as e:
            result["errors"].append(f"Rate limited: {str(e)}")
        except BlockedError as e:
            result["errors"].append(f"Blocked: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Connection error: {str(e)}")

        # Determine overall status
        if result["can_fetch_public"]:
            result["status"] = "connected"
        elif result["credentials_available"]:
            result["status"] = "credentials_available"
        elif result["offline_mode"]:
            result["status"] = "offline_mode"
        else:
            result["status"] = "disconnected"

        return result

    async def clear_cache(self):
        """Clear all cached data."""
        await self.cache.clear()
        logger.info("Cache cleared")

    async def enable_offline_mode(self):
        """Enable offline mode (return cached data when requests fail)."""
        self.offline_mode = True
        logger.info("Offline mode enabled")

    async def disable_offline_mode(self):
        """Disable offline mode."""
        self.offline_mode = False
        logger.info("Offline mode disabled")


# =============================================================================
# Factory Function
# =============================================================================

def create_instagram_adapter(
    username: Optional[str] = None,
    password: Optional[str] = None,
    offline_mode: bool = False,
    **kwargs
) -> InstagramAdapter:
    """
    Factory function to create InstagramAdapter instance.

    Can read credentials from environment variables if not provided.

    Args:
        username: Instagram username (optional, reads from INSTAGRAM_USERNAME env)
        password: Instagram password (optional, reads from INSTAGRAM_PASSWORD env)
        offline_mode: Enable offline mode by default
        **kwargs: Additional arguments passed to InstagramAdapter

    Returns:
        Configured InstagramAdapter instance
    """
    username = username or os.environ.get("INSTAGRAM_USERNAME")
    password = password or os.environ.get("INSTAGRAM_PASSWORD")

    # Ensure storage directory exists
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    if not username:
        logger.info(
            "No Instagram credentials provided. "
            "Public profiles can still be accessed. "
            "Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD for full access."
        )

    return InstagramAdapter(
        username=username,
        password=password,
        offline_mode=offline_mode,
        **kwargs
    )
