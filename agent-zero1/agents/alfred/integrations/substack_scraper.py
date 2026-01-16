"""
Substack Integration Adapter for ALFRED

Provides access to Substack publication data through RSS feeds, web scraping,
and manual input. Since Substack has no official API, this adapter combines
multiple data sources for comprehensive analytics.

Data Sources:
- RSS Feed: Every Substack has RSS at {publication}.substack.com/feed
- Web Scraping: Parse public publication pages for metadata and stats
- Manual Input: Dashboard stats (subscribers, email metrics) from user

Features:
- RSS feed parsing with feedparser for post retrieval
- BeautifulSoup for web scraping public stats
- Pydantic models for data validation
- Persistent storage for manual entries and growth tracking
- Caching and offline mode support
- Publication change tracking
- Output formats compatible with Social Metrics Harvester
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse, urljoin
import asyncio
import hashlib
import json
import logging
import os
import re
import time

from pydantic import BaseModel, Field, field_validator, model_validator

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================

class SubstackError(Exception):
    """Base exception for Substack adapter errors."""
    pass


class InvalidPublicationError(SubstackError):
    """Invalid or non-existent publication URL."""
    pass


class FeedNotFoundError(SubstackError):
    """RSS feed not found or inaccessible."""
    pass


class ParseError(SubstackError):
    """Failed to parse content (RSS or HTML)."""
    pass


class RateLimitError(SubstackError):
    """Rate limit exceeded, retry after cooldown."""
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class ScrapingError(SubstackError):
    """Failed to scrape page content."""
    pass


class OfflineError(SubstackError):
    """Cannot connect to Substack - offline mode."""
    pass


# =============================================================================
# Pydantic Data Models
# =============================================================================

class SubstackPublication(BaseModel):
    """Publication metadata and information."""
    name: str = Field(..., description="Publication name/title")
    url: str = Field(..., description="Publication base URL")
    subdomain: str = Field(default="", description="Substack subdomain")
    description: str = Field(default="", description="Publication description/about")
    author_name: str = Field(default="", description="Author/creator name")
    subscriber_count: Optional[int] = Field(default=None, description="Total subscribers (if publicly visible)")
    logo_url: str = Field(default="", description="Publication logo URL")
    twitter_handle: str = Field(default="", description="Twitter handle if available")
    collected_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator('url', mode='before')
    @classmethod
    def normalize_url(cls, v):
        if v and not v.startswith('http'):
            return f"https://{v}"
        return v.rstrip('/') if v else v

    class Config:
        extra = "allow"


class SubstackPost(BaseModel):
    """Individual Substack post data from RSS feed."""
    post_id: str = Field(..., description="Unique post identifier")
    title: str = Field(..., description="Post title")
    subtitle: str = Field(default="", description="Post subtitle/preview")
    url: str = Field(default="", description="Full post URL")
    published_date: str = Field(default="", description="Publication date (ISO format)")
    is_paid: bool = Field(default=False, description="Whether this is a paid-only post")
    preview_text: str = Field(default="", description="Text preview/excerpt")
    likes: int = Field(default=0, description="Like/heart count")
    comments_count: int = Field(default=0, description="Number of comments")
    restacks: int = Field(default=0, description="Restack/share count")
    collected_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator('post_id', mode='before')
    @classmethod
    def generate_post_id(cls, v):
        if not v:
            return hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        return v

    class Config:
        extra = "allow"


class SubstackStats(BaseModel):
    """Publication-level statistics calculated from posts."""
    publication_url: str = Field(..., description="Publication URL")
    posts_count: int = Field(default=0, description="Total posts analyzed")
    paid_posts: int = Field(default=0, description="Number of paid posts")
    free_posts: int = Field(default=0, description="Number of free posts")
    avg_frequency_days: float = Field(default=0.0, description="Average days between posts")
    total_likes: int = Field(default=0, description="Total likes across posts")
    total_comments: int = Field(default=0, description="Total comments across posts")
    total_restacks: int = Field(default=0, description="Total restacks across posts")
    first_post_date: str = Field(default="", description="Earliest post date analyzed")
    last_post_date: str = Field(default="", description="Most recent post date")
    calculated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        extra = "allow"


class DashboardMetrics(BaseModel):
    """Manual entry metrics from Substack dashboard."""
    subscribers: int = Field(default=0, description="Total subscriber count")
    paid_subscribers: int = Field(default=0, description="Paid subscriber count")
    free_subscribers: int = Field(default=0, description="Free subscriber count")
    open_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Email open rate (0-1)")
    click_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Email click rate (0-1)")
    monthly_revenue: float = Field(default=0.0, description="Monthly revenue if applicable")
    annual_revenue: float = Field(default=0.0, description="Annual revenue if applicable")
    gross_annualized_revenue: float = Field(default=0.0, description="GAR metric")
    churn_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Monthly churn rate (0-1)")
    entered_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @model_validator(mode='before')
    @classmethod
    def calculate_free_subscribers(cls, values):
        if isinstance(values, dict):
            if 'subscribers' in values and 'paid_subscribers' in values:
                if 'free_subscribers' not in values or values.get('free_subscribers') == 0:
                    values['free_subscribers'] = values['subscribers'] - values.get('paid_subscribers', 0)
        return values

    class Config:
        extra = "allow"


class ManualEntry(BaseModel):
    """Generic manual entry container for storing dashboard data."""
    platform: str = Field(default="substack", description="Platform identifier")
    entry_type: str = Field(default="metrics", description="Type of entry")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Metrics data")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str = Field(default="", description="Optional notes")

    class Config:
        extra = "allow"


class PostEngagement(BaseModel):
    """Individual post engagement metrics from manual input."""
    post_id: str
    post_title: str = ""
    post_url: str = ""
    views: int = 0
    reads: int = 0
    read_ratio: float = 0.0
    likes: int = 0
    comments: int = 0
    restacks: int = 0
    email_opens: int = 0
    email_clicks: int = 0
    entered_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        extra = "allow"


class CacheEntry(BaseModel):
    """Cache entry with expiration."""
    data: Any
    expires_at: float

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Rate limiter for web scraping requests.
    Implements a simple token bucket to be respectful of Substack servers.
    """

    def __init__(self, requests_per_minute: float = 10.0):
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        async with self._lock:
            current_time = time.time()
            min_interval = 60.0 / self.requests_per_minute
            time_since_last = current_time - self.last_request_time

            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)

            self.last_request_time = time.time()
            return True


# =============================================================================
# Response Cache
# =============================================================================

class ResponseCache:
    """Simple in-memory cache for responses with file persistence."""

    def __init__(self, default_ttl_seconds: int = 300, cache_file: Optional[str] = None):
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.cache_file = cache_file

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


# =============================================================================
# Storage Manager
# =============================================================================

class StorageManager:
    """Manages persistent storage for Substack data."""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.manual_entries_path = self.storage_path / "manual_entries"
        self.publications_path = self.storage_path / "publications"
        self.history_path = self.storage_path / "history"

        for path in [self.manual_entries_path, self.publications_path, self.history_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _get_publication_filename(self, publication_url: str) -> str:
        """Generate filename from publication URL."""
        parsed = urlparse(publication_url)
        subdomain = parsed.netloc.replace('.substack.com', '').replace('www.', '')
        return f"{subdomain}.json"

    def save_manual_entry(self, entry: ManualEntry, publication_url: str) -> str:
        """Save a manual entry to storage."""
        filename = self._get_publication_filename(publication_url)
        entries_file = self.manual_entries_path / filename

        # Load existing entries
        entries = []
        if entries_file.exists():
            with open(entries_file, 'r') as f:
                entries = json.load(f)

        # Add new entry
        entries.append(entry.dict())

        # Save
        with open(entries_file, 'w') as f:
            json.dump(entries, f, indent=2, default=str)

        logger.info(f"Saved manual entry to {entries_file}")
        return str(entries_file)

    def load_manual_entries(self, publication_url: str) -> List[ManualEntry]:
        """Load all manual entries for a publication."""
        filename = self._get_publication_filename(publication_url)
        entries_file = self.manual_entries_path / filename

        if not entries_file.exists():
            return []

        with open(entries_file, 'r') as f:
            data = json.load(f)

        return [ManualEntry(**entry) for entry in data]

    def save_dashboard_metrics(self, metrics: DashboardMetrics, publication_url: str) -> str:
        """Save dashboard metrics with timestamp."""
        entry = ManualEntry(
            platform="substack",
            entry_type="dashboard_metrics",
            metrics=metrics.dict(),
            timestamp=datetime.now().isoformat()
        )
        return self.save_manual_entry(entry, publication_url)

    def load_dashboard_metrics_history(self, publication_url: str) -> List[DashboardMetrics]:
        """Load historical dashboard metrics."""
        entries = self.load_manual_entries(publication_url)
        metrics_list = []
        for entry in entries:
            if entry.entry_type == "dashboard_metrics":
                metrics_list.append(DashboardMetrics(**entry.metrics))
        return metrics_list

    def save_publication_snapshot(self, publication: SubstackPublication) -> str:
        """Save publication snapshot for change tracking."""
        filename = self._get_publication_filename(publication.url)
        pub_file = self.publications_path / filename

        # Load existing snapshots
        snapshots = []
        if pub_file.exists():
            with open(pub_file, 'r') as f:
                snapshots = json.load(f)

        snapshots.append(publication.dict())

        # Keep last 100 snapshots
        if len(snapshots) > 100:
            snapshots = snapshots[-100:]

        with open(pub_file, 'w') as f:
            json.dump(snapshots, f, indent=2, default=str)

        return str(pub_file)

    def load_publication_history(self, publication_url: str) -> List[SubstackPublication]:
        """Load publication snapshot history."""
        filename = self._get_publication_filename(publication_url)
        pub_file = self.publications_path / filename

        if not pub_file.exists():
            return []

        with open(pub_file, 'r') as f:
            data = json.load(f)

        return [SubstackPublication(**snapshot) for snapshot in data]


# =============================================================================
# Substack Adapter
# =============================================================================

class SubstackAdapter:
    """
    Substack Integration Adapter for ALFRED.

    Provides structured access to Substack publication data through
    RSS feeds, web scraping, and manual input. Designed to work with
    ALFRED's Social Metrics Harvester and Content Strategy Analyst.

    Since Substack has no official API, this adapter combines:
    - RSS feed parsing for post lists (using feedparser)
    - Web scraping for engagement metrics and publication info (using BeautifulSoup)
    - Manual input for dashboard-only stats (subscribers, email metrics)

    Usage:
        adapter = SubstackAdapter()

        # Get publication info
        pub_info = await adapter.get_publication_info("https://yourpub.substack.com")

        # Get recent posts from RSS
        posts = await adapter.get_recent_posts("https://yourpub.substack.com", count=10)

        # Get post details with engagement
        details = await adapter.get_post_details("https://yourpub.substack.com/p/my-post")

        # Enter dashboard stats manually
        adapter.enter_manual_stats({
            "subscribers": 5000,
            "paid_subscribers": 500,
            "open_rate": 0.45
        })

        # Get publishing frequency analysis
        frequency = await adapter.get_publishing_frequency("https://yourpub.substack.com")
    """

    # RSS namespace definitions
    RSS_NAMESPACES = {
        'atom': 'http://www.w3.org/2005/Atom',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'content': 'http://purl.org/rss/1.0/modules/content/',
    }

    def __init__(
        self,
        publication_url: Optional[str] = None,
        storage_path: str = "/Users/shaileshsingh/Alfred/agent-zero1/data/alfred/substack",
        cache_ttl_seconds: int = 300,
        enable_caching: bool = True,
        requests_per_minute: float = 10.0,
        offline_mode: bool = False
    ):
        """
        Initialize Substack adapter.

        Args:
            publication_url: Default Substack publication URL
            storage_path: Path for persistent storage
            cache_ttl_seconds: Default cache TTL
            enable_caching: Whether to enable response caching
            requests_per_minute: Rate limit for web scraping
            offline_mode: Enable offline mode (use cached/stored data only)
        """
        self.publication_url: Optional[str] = None
        self.publication_name: Optional[str] = None
        self.enable_caching = enable_caching
        self.offline_mode = offline_mode

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)

        # Initialize cache
        self.cache = ResponseCache(default_ttl_seconds=cache_ttl_seconds)

        # Initialize storage manager
        self.storage = StorageManager(storage_path)

        # HTTP client placeholder
        self._http_client = None

        # Store for current session manual stats
        self._current_dashboard_metrics: Optional[DashboardMetrics] = None
        self._manual_post_stats: Dict[str, PostEngagement] = {}

        # Set publication if provided
        if publication_url:
            self.set_publication(publication_url)

        logger.info(f"SubstackAdapter initialized (offline_mode={offline_mode})")

    # =========================================================================
    # HTTP Client Management
    # =========================================================================

    async def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import aiohttp
                self._http_client = aiohttp.ClientSession(
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                    }
                )
            except ImportError:
                raise ImportError(
                    "aiohttp is required for SubstackAdapter. "
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
    # Publication Configuration
    # =========================================================================

    def set_publication(self, publication_url: str) -> None:
        """
        Set the Substack publication to work with.

        Args:
            publication_url: Full URL to the Substack publication
                            (e.g., "https://yourpub.substack.com")

        Raises:
            InvalidPublicationError: If URL is not a valid Substack URL
        """
        # Normalize URL
        if not publication_url.startswith('http'):
            publication_url = f"https://{publication_url}"

        parsed = urlparse(publication_url)

        # Validate it's a Substack URL or custom domain
        if 'substack.com' not in parsed.netloc:
            logger.warning(f"URL may not be a Substack publication: {publication_url}")

        # Extract publication name
        if 'substack.com' in parsed.netloc:
            self.publication_name = parsed.netloc.split('.')[0]
        else:
            self.publication_name = parsed.netloc.replace('www.', '')

        self.publication_url = f"{parsed.scheme}://{parsed.netloc}"

        logger.info(f"Publication set to: {self.publication_url}")

    def _get_rss_url(self, publication_url: Optional[str] = None) -> str:
        """Get the RSS feed URL for the publication."""
        url = publication_url or self.publication_url
        if not url:
            raise InvalidPublicationError("Publication URL not set")
        return f"{url.rstrip('/')}/feed"

    def _get_archive_url(self, publication_url: Optional[str] = None) -> str:
        """Get the archive page URL for the publication."""
        url = publication_url or self.publication_url
        if not url:
            raise InvalidPublicationError("Publication URL not set")
        return f"{url.rstrip('/')}/archive"

    # =========================================================================
    # RSS Feed Methods
    # =========================================================================

    async def get_recent_posts(
        self,
        publication_url: Optional[str] = None,
        count: int = 20
    ) -> List[SubstackPost]:
        """
        Get recent posts from the publication's RSS feed.

        Args:
            publication_url: Publication URL (uses default if not provided)
            count: Maximum number of posts to return

        Returns:
            List of SubstackPost objects

        Raises:
            InvalidPublicationError: If publication not set
            FeedNotFoundError: If RSS feed not found
            ParseError: If RSS feed cannot be parsed
        """
        url = publication_url or self.publication_url
        if not url:
            raise InvalidPublicationError("Publication URL not set. Call set_publication() first.")

        cache_key = f"rss_{url}"

        # Check cache
        if self.enable_caching:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached[:count]

        # Check offline mode
        if self.offline_mode:
            raise OfflineError("Offline mode enabled. Use cached data or manual input.")

        # Fetch and parse RSS
        await self.rate_limiter.acquire()

        try:
            client = await self._get_http_client()
            rss_url = self._get_rss_url(url)

            async with client.get(rss_url) as response:
                if response.status == 404:
                    raise FeedNotFoundError(f"RSS feed not found at {rss_url}")
                if response.status != 200:
                    raise ParseError(f"Failed to fetch RSS feed: HTTP {response.status}")

                content = await response.text()
        except FeedNotFoundError:
            raise
        except Exception as e:
            if isinstance(e, SubstackError):
                raise
            raise OfflineError(f"Failed to connect to Substack: {str(e)}")

        # Parse RSS using feedparser
        posts = await self._parse_rss_with_feedparser(content)

        # Cache results
        if self.enable_caching:
            await self.cache.set(posts, 600, cache_key)  # 10 min cache

        return posts[:count]

    async def _parse_rss_with_feedparser(self, content: str) -> List[SubstackPost]:
        """Parse RSS feed content using feedparser library."""
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed, falling back to XML parsing")
            return self._parse_rss_xml(content)

        feed = feedparser.parse(content)

        if feed.bozo and not feed.entries:
            raise ParseError(f"Invalid RSS feed: {feed.bozo_exception}")

        posts = []
        for entry in feed.entries:
            try:
                # Extract post ID from link
                link = entry.get('link', '')
                post_id = hashlib.md5(link.encode()).hexdigest()[:12] if link else ''

                # Parse published date
                published = entry.get('published', entry.get('updated', ''))
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6]).isoformat()

                # Get content/description for preview
                content_text = ''
                if 'content' in entry and entry.content:
                    content_text = entry.content[0].get('value', '')
                elif 'summary' in entry:
                    content_text = entry.summary
                elif 'description' in entry:
                    content_text = entry.description

                preview = self._clean_html(content_text)[:500]

                # Determine if paid post
                is_paid = False
                title = entry.get('title', '')
                if any(indicator in content_text.lower() for indicator in
                       ['subscriber-only', 'for paid subscribers', 'paid post']):
                    is_paid = True
                if any(indicator in title.lower() for indicator in ['[paid]', '[subscriber only]']):
                    is_paid = True

                # Extract subtitle if present
                subtitle = ''
                if '<h2' in content_text:
                    match = re.search(r'<h2[^>]*>([^<]+)</h2>', content_text)
                    if match:
                        subtitle = match.group(1)

                post = SubstackPost(
                    post_id=post_id,
                    title=title,
                    subtitle=subtitle,
                    url=link,
                    published_date=published,
                    is_paid=is_paid,
                    preview_text=preview,
                )
                posts.append(post)
            except Exception as e:
                logger.warning(f"Failed to parse RSS entry: {e}")
                continue

        return posts

    def _parse_rss_xml(self, xml_content: str) -> List[SubstackPost]:
        """Fallback XML parsing if feedparser is not available."""
        import xml.etree.ElementTree as ET

        posts = []
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ParseError(f"Invalid RSS XML: {str(e)}")

        # Find all items (RSS 2.0)
        items = root.findall('.//item')

        for item in items:
            try:
                title = self._get_xml_text(item.find('title'))
                link = self._get_xml_text(item.find('link'))
                pub_date = self._get_xml_text(item.find('pubDate'))
                description = self._get_xml_text(item.find('description'))

                if not title or not link:
                    continue

                post_id = hashlib.md5(link.encode()).hexdigest()[:12]
                preview = self._clean_html(description)[:500]
                is_paid = 'subscriber-only' in description.lower() or 'for paid subscribers' in description.lower()

                post = SubstackPost(
                    post_id=post_id,
                    title=title,
                    url=link,
                    published_date=pub_date,
                    is_paid=is_paid,
                    preview_text=preview,
                )
                posts.append(post)
            except Exception as e:
                logger.warning(f"Failed to parse RSS item: {e}")
                continue

        return posts

    def _get_xml_text(self, elem) -> str:
        """Helper to get text from XML element."""
        return elem.text.strip() if elem is not None and elem.text else ""

    def _clean_html(self, html_content: str) -> str:
        """Remove HTML tags and clean up text."""
        if not html_content:
            return ""

        # Try using BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
        except ImportError:
            # Fallback to regex
            text = re.sub(r'<[^>]+>', ' ', html_content)
            text = re.sub(r'\s+', ' ', text).strip()

        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")

        return text

    # =========================================================================
    # Publication Info Methods
    # =========================================================================

    async def get_publication_info(self, publication_url: Optional[str] = None) -> SubstackPublication:
        """
        Get publication metadata by scraping the publication homepage.

        Args:
            publication_url: Publication URL (uses default if not provided)

        Returns:
            SubstackPublication with metadata

        Raises:
            InvalidPublicationError: If publication not set or invalid
            ScrapingError: If page cannot be scraped
        """
        url = publication_url or self.publication_url
        if not url:
            raise InvalidPublicationError("Publication URL not set")

        cache_key = f"pub_info_{url}"

        # Check cache
        if self.enable_caching:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Check offline mode
        if self.offline_mode:
            # Try to load from storage
            history = self.storage.load_publication_history(url)
            if history:
                return history[-1]
            raise OfflineError("Offline mode enabled and no cached publication data available.")

        await self.rate_limiter.acquire()

        try:
            client = await self._get_http_client()

            async with client.get(url) as response:
                if response.status != 200:
                    raise ScrapingError(f"Failed to fetch publication page: HTTP {response.status}")
                html = await response.text()
        except Exception as e:
            if isinstance(e, SubstackError):
                raise
            raise OfflineError(f"Failed to connect: {str(e)}")

        publication = await self._parse_publication_page(html, url)

        # Cache and save snapshot
        if self.enable_caching:
            await self.cache.set(publication, 3600, cache_key)  # 1 hour cache

        self.storage.save_publication_snapshot(publication)

        return publication

    async def _parse_publication_page(self, html: str, url: str) -> SubstackPublication:
        """Parse publication metadata from homepage HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
        except ImportError:
            return self._parse_publication_regex(html, url)

        # Extract publication name from various sources
        name = ""

        # Try og:site_name or og:title
        og_site = soup.find('meta', property='og:site_name')
        if og_site:
            name = og_site.get('content', '')

        if not name:
            og_title = soup.find('meta', property='og:title')
            if og_title:
                name = og_title.get('content', '')

        if not name:
            title_tag = soup.find('title')
            if title_tag:
                name = title_tag.text.split('|')[0].strip()

        # Extract description
        description = ""
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            description = og_desc.get('content', '')

        if not description:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '')

        # Extract author name
        author_name = ""
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            author_name = author_meta.get('content', '')

        # Extract logo
        logo_url = ""
        og_image = soup.find('meta', property='og:image')
        if og_image:
            logo_url = og_image.get('content', '')

        # Try to extract subscriber count if publicly visible
        subscriber_count = None
        subscriber_patterns = [
            r'"subscriberCount"\s*:\s*(\d+)',
            r'(\d+(?:,\d+)?)\s*subscriber',
            r'(\d+(?:,\d+)?)\s*reader',
        ]
        for pattern in subscriber_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                count_str = match.group(1).replace(',', '')
                try:
                    subscriber_count = int(count_str)
                    break
                except ValueError:
                    continue

        # Extract Twitter handle
        twitter_handle = ""
        twitter_meta = soup.find('meta', attrs={'name': 'twitter:creator'})
        if twitter_meta:
            twitter_handle = twitter_meta.get('content', '')

        # Extract subdomain
        parsed = urlparse(url)
        subdomain = parsed.netloc.replace('.substack.com', '').replace('www.', '')

        return SubstackPublication(
            name=name or subdomain,
            url=url,
            subdomain=subdomain,
            description=description,
            author_name=author_name,
            subscriber_count=subscriber_count,
            logo_url=logo_url,
            twitter_handle=twitter_handle,
        )

    def _parse_publication_regex(self, html: str, url: str) -> SubstackPublication:
        """Fallback regex parsing for publication info."""
        # Extract name from title
        name_match = re.search(r'<title>([^<]+)</title>', html)
        name = name_match.group(1).split('|')[0].strip() if name_match else ""

        # Extract description
        desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', html, re.IGNORECASE)
        description = desc_match.group(1) if desc_match else ""

        # Extract subdomain
        parsed = urlparse(url)
        subdomain = parsed.netloc.replace('.substack.com', '').replace('www.', '')

        return SubstackPublication(
            name=name or subdomain,
            url=url,
            subdomain=subdomain,
            description=description,
        )

    # =========================================================================
    # Post Details Methods
    # =========================================================================

    async def get_post_details(self, post_url: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific post by scraping.

        Args:
            post_url: Full URL to the post

        Returns:
            Dictionary with post details and engagement stats

        Raises:
            ScrapingError: If page cannot be scraped
        """
        cache_key = f"post_{post_url}"

        # Check cache
        if self.enable_caching:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Check for manual override
        post_id = hashlib.md5(post_url.encode()).hexdigest()[:12]
        if post_id in self._manual_post_stats:
            return self._manual_post_stats[post_id].dict()

        # Check offline mode
        if self.offline_mode:
            raise OfflineError("Offline mode enabled. Use manual input for post details.")

        await self.rate_limiter.acquire()

        try:
            client = await self._get_http_client()

            async with client.get(post_url) as response:
                if response.status != 200:
                    raise ScrapingError(f"Failed to fetch post page: HTTP {response.status}")
                html = await response.text()
        except Exception as e:
            if isinstance(e, SubstackError):
                raise
            raise OfflineError(f"Failed to connect: {str(e)}")

        details = await self._parse_post_page(html, post_url)

        # Cache results
        if self.enable_caching:
            await self.cache.set(details, 300, cache_key)  # 5 min cache

        return details

    async def _parse_post_page(self, html: str, post_url: str) -> Dict[str, Any]:
        """Parse post details from HTML."""
        details = {
            "post_url": post_url,
            "post_id": hashlib.md5(post_url.encode()).hexdigest()[:12],
            "title": "",
            "subtitle": "",
            "author": "",
            "published_date": "",
            "likes": 0,
            "comments_count": 0,
            "restacks": 0,
            "is_paid": False,
            "read_time_minutes": 0,
            "collected_at": datetime.now().isoformat(),
        }

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title
            og_title = soup.find('meta', property='og:title')
            if og_title:
                details["title"] = og_title.get('content', '')
            else:
                h1 = soup.find('h1')
                if h1:
                    details["title"] = h1.get_text(strip=True)

            # Extract author
            author_meta = soup.find('meta', attrs={'name': 'author'})
            if author_meta:
                details["author"] = author_meta.get('content', '')

            # Extract published date
            time_tag = soup.find('time')
            if time_tag:
                details["published_date"] = time_tag.get('datetime', time_tag.get_text(strip=True))

        except ImportError:
            # Fallback to regex
            title_match = re.search(r'"headline"\s*:\s*"([^"]+)"', html)
            if title_match:
                details["title"] = title_match.group(1)

            date_match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
            if date_match:
                details["published_date"] = date_match.group(1)

        # Extract engagement metrics using regex (works regardless of BeautifulSoup)
        like_patterns = [
            r'"like_count"\s*:\s*(\d+)',
            r'"likeCount"\s*:\s*(\d+)',
            r'(\d+)\s*like',
        ]
        for pattern in like_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                details["likes"] = int(match.group(1))
                break

        comment_patterns = [
            r'"comment_count"\s*:\s*(\d+)',
            r'"commentCount"\s*:\s*(\d+)',
            r'(\d+)\s*comment',
        ]
        for pattern in comment_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                details["comments_count"] = int(match.group(1))
                break

        restack_patterns = [
            r'"restack_count"\s*:\s*(\d+)',
            r'"restackCount"\s*:\s*(\d+)',
            r'(\d+)\s*restack',
        ]
        for pattern in restack_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                details["restacks"] = int(match.group(1))
                break

        # Check if paid post
        paid_indicators = [
            'subscriber-only',
            'for paid subscribers',
            '"is_paid":true',
            '"audience":"only_paid"',
        ]
        for indicator in paid_indicators:
            if indicator.lower() in html.lower():
                details["is_paid"] = True
                break

        return details

    # =========================================================================
    # Manual Input Methods
    # =========================================================================

    def enter_manual_stats(
        self,
        stats: Dict[str, Any],
        publication_url: Optional[str] = None
    ) -> DashboardMetrics:
        """
        Enter dashboard metrics manually from Substack dashboard.

        These stats are only available in the Substack dashboard and
        cannot be scraped from public pages.

        Args:
            stats: Dictionary with dashboard stats:
                {
                    "subscribers": int,         # Total subscriber count
                    "paid_subscribers": int,    # Paid subscriber count
                    "open_rate": float,         # Email open rate (0-1)
                    "click_rate": float,        # Email click rate (0-1)
                    "monthly_revenue": float,   # Monthly revenue
                    "churn_rate": float,        # Monthly churn rate (0-1)
                }
            publication_url: Publication URL (uses default if not provided)

        Returns:
            DashboardMetrics object

        Example:
            adapter.enter_manual_stats({
                "subscribers": 5000,
                "paid_subscribers": 500,
                "open_rate": 0.45,
                "click_rate": 0.08
            })
        """
        url = publication_url or self.publication_url
        if not url:
            raise InvalidPublicationError("Publication URL not set")

        # Create DashboardMetrics
        metrics = DashboardMetrics(
            subscribers=stats.get("subscribers", 0),
            paid_subscribers=stats.get("paid_subscribers", 0),
            free_subscribers=stats.get("free_subscribers", 0),
            open_rate=stats.get("open_rate", 0.0),
            click_rate=stats.get("click_rate", 0.0),
            monthly_revenue=stats.get("monthly_revenue", 0.0),
            annual_revenue=stats.get("annual_revenue", 0.0),
            gross_annualized_revenue=stats.get("gross_annualized_revenue", 0.0),
            churn_rate=stats.get("churn_rate", 0.0),
        )

        # Store in memory and persistent storage
        self._current_dashboard_metrics = metrics
        self.storage.save_dashboard_metrics(metrics, url)

        logger.info(f"Dashboard metrics saved: {metrics.subscribers} subscribers")

        return metrics

    def enter_post_engagement(
        self,
        post_data: Dict[str, Any]
    ) -> PostEngagement:
        """
        Enter engagement metrics for a specific post manually.

        Args:
            post_data: Dictionary with post engagement:
                {
                    "post_id": str,       # Required
                    "post_title": str,
                    "post_url": str,
                    "views": int,
                    "reads": int,
                    "likes": int,
                    "comments": int,
                    "restacks": int,
                    "email_opens": int,
                    "email_clicks": int,
                }

        Returns:
            PostEngagement object
        """
        if not post_data.get("post_id"):
            raise ValueError("post_id is required")

        engagement = PostEngagement(
            post_id=post_data["post_id"],
            post_title=post_data.get("post_title", ""),
            post_url=post_data.get("post_url", ""),
            views=post_data.get("views", 0),
            reads=post_data.get("reads", 0),
            read_ratio=post_data.get("read_ratio", 0.0),
            likes=post_data.get("likes", 0),
            comments=post_data.get("comments", 0),
            restacks=post_data.get("restacks", 0),
            email_opens=post_data.get("email_opens", 0),
            email_clicks=post_data.get("email_clicks", 0),
        )

        self._manual_post_stats[engagement.post_id] = engagement

        logger.info(f"Post engagement saved for: {engagement.post_id}")

        return engagement

    # =========================================================================
    # Analytics Methods
    # =========================================================================

    async def get_publishing_frequency(
        self,
        publication_url: Optional[str] = None,
        post_count: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze publishing frequency and patterns.

        Args:
            publication_url: Publication URL (uses default if not provided)
            post_count: Number of posts to analyze

        Returns:
            Dictionary with frequency analysis:
            {
                "avg_days_between_posts": float,
                "posts_per_week": float,
                "posts_per_month": float,
                "most_active_day": str,
                "posting_consistency": str,
                "date_range": {"start": str, "end": str}
            }
        """
        url = publication_url or self.publication_url
        if not url:
            raise InvalidPublicationError("Publication URL not set")

        posts = await self.get_recent_posts(url, count=post_count)

        if len(posts) < 2:
            return {
                "avg_days_between_posts": 0,
                "posts_per_week": 0,
                "posts_per_month": 0,
                "most_active_day": "unknown",
                "posting_consistency": "insufficient_data",
                "date_range": {"start": "", "end": ""},
                "total_posts_analyzed": len(posts),
            }

        # Parse dates and calculate intervals
        dates = []
        day_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # Mon-Sun
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        for post in posts:
            if post.published_date:
                try:
                    dt = self._parse_date_to_datetime(post.published_date)
                    if dt:
                        dates.append(dt)
                        day_counts[dt.weekday()] += 1
                except Exception:
                    continue

        if len(dates) < 2:
            return {
                "avg_days_between_posts": 0,
                "posts_per_week": 0,
                "posts_per_month": 0,
                "most_active_day": "unknown",
                "posting_consistency": "insufficient_data",
                "total_posts_analyzed": len(posts),
            }

        # Sort dates
        dates.sort(reverse=True)

        # Calculate intervals
        intervals = []
        for i in range(len(dates) - 1):
            delta = dates[i] - dates[i + 1]
            intervals.append(delta.days)

        avg_interval = sum(intervals) / len(intervals) if intervals else 0

        # Calculate date range
        date_range = {
            "start": dates[-1].isoformat()[:10],
            "end": dates[0].isoformat()[:10],
        }
        total_days = (dates[0] - dates[-1]).days or 1

        # Calculate rates
        posts_per_day = len(dates) / total_days
        posts_per_week = posts_per_day * 7
        posts_per_month = posts_per_day * 30

        # Find most active day
        most_active_day_idx = max(day_counts, key=day_counts.get)
        most_active_day = day_names[most_active_day_idx]

        # Assess consistency
        if intervals:
            std_dev = (sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5
            cv = std_dev / avg_interval if avg_interval > 0 else 0

            if cv < 0.3:
                consistency = "very_consistent"
            elif cv < 0.6:
                consistency = "moderately_consistent"
            elif cv < 1.0:
                consistency = "somewhat_irregular"
            else:
                consistency = "irregular"
        else:
            consistency = "insufficient_data"

        return {
            "avg_days_between_posts": round(avg_interval, 1),
            "posts_per_week": round(posts_per_week, 2),
            "posts_per_month": round(posts_per_month, 1),
            "most_active_day": most_active_day,
            "posting_consistency": consistency,
            "date_range": date_range,
            "total_posts_analyzed": len(posts),
            "day_distribution": {day_names[k]: v for k, v in day_counts.items()},
        }

    async def get_growth_summary(
        self,
        days: int = 30,
        publication_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize growth metrics over a period (from manual entries).

        Args:
            days: Number of days to look back
            publication_url: Publication URL (uses default if not provided)

        Returns:
            Dictionary with growth summary from manual entries
        """
        url = publication_url or self.publication_url
        if not url:
            raise InvalidPublicationError("Publication URL not set")

        # Load historical metrics
        metrics_history = self.storage.load_dashboard_metrics_history(url)

        if not metrics_history:
            return {
                "period_days": days,
                "has_data": False,
                "message": "No manual metrics entries found. Use enter_manual_stats() to add data.",
            }

        # Filter to requested period
        cutoff = datetime.now() - timedelta(days=days)
        relevant_metrics = []

        for m in metrics_history:
            try:
                entry_date = datetime.fromisoformat(m.entered_at.replace('Z', '+00:00'))
                if entry_date >= cutoff:
                    relevant_metrics.append(m)
            except Exception:
                continue

        if len(relevant_metrics) < 2:
            # Return latest data if available
            if metrics_history:
                latest = metrics_history[-1]
                return {
                    "period_days": days,
                    "has_data": True,
                    "entries_count": 1,
                    "latest_metrics": latest.dict(),
                    "message": "Insufficient data points for growth analysis. Add more entries over time.",
                }
            return {
                "period_days": days,
                "has_data": False,
                "message": "No metrics in the specified period.",
            }

        # Sort by date
        relevant_metrics.sort(key=lambda x: x.entered_at)

        first = relevant_metrics[0]
        last = relevant_metrics[-1]

        # Calculate growth
        subscriber_growth = last.subscribers - first.subscribers
        subscriber_growth_pct = (subscriber_growth / first.subscribers * 100) if first.subscribers > 0 else 0

        paid_growth = last.paid_subscribers - first.paid_subscribers
        paid_growth_pct = (paid_growth / first.paid_subscribers * 100) if first.paid_subscribers > 0 else 0

        return {
            "period_days": days,
            "has_data": True,
            "entries_count": len(relevant_metrics),
            "subscriber_growth": {
                "start": first.subscribers,
                "end": last.subscribers,
                "change": subscriber_growth,
                "change_percent": round(subscriber_growth_pct, 2),
            },
            "paid_subscriber_growth": {
                "start": first.paid_subscribers,
                "end": last.paid_subscribers,
                "change": paid_growth,
                "change_percent": round(paid_growth_pct, 2),
            },
            "latest_open_rate": last.open_rate,
            "latest_click_rate": last.click_rate,
            "period_start": first.entered_at,
            "period_end": last.entered_at,
        }

    def _parse_date_to_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats to datetime object."""
        if not date_str:
            return None

        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        # Try ISO format parsing
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            pass

        return None

    # =========================================================================
    # Integration Helpers (for ALFRED tools)
    # =========================================================================

    async def get_metrics_for_harvester(
        self,
        period_start: str,
        period_end: str,
        publication_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metrics formatted for Social Metrics Harvester.

        Returns data in the schema expected by SocialMetricsHarvester.

        Args:
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)
            publication_url: Publication URL (uses default if not provided)

        Returns:
            Dictionary with metrics in harvester format
        """
        url = publication_url or self.publication_url
        if not url:
            raise InvalidPublicationError("Publication URL not set")

        # Get recent posts
        posts = await self.get_recent_posts(url, count=50)

        # Filter posts by date range
        filtered_posts = []
        for post in posts:
            if post.published_date:
                pub_date = self._parse_date_to_datetime(post.published_date)
                if pub_date:
                    pub_date_str = pub_date.strftime("%Y-%m-%d")
                    if period_start <= pub_date_str <= period_end:
                        filtered_posts.append(post)

        # Try to get stats for each post
        content_items = []
        total_likes = 0
        total_comments = 0
        total_restacks = 0

        for post in filtered_posts:
            try:
                if not self.offline_mode:
                    stats = await self.get_post_details(post.url)
                    post.likes = stats.get("likes", 0)
                    post.comments_count = stats.get("comments_count", 0)
                    post.restacks = stats.get("restacks", 0)
            except Exception as e:
                logger.debug(f"Could not get stats for {post.url}: {e}")

            total_likes += post.likes
            total_comments += post.comments_count
            total_restacks += post.restacks

            content_items.append({
                "content_id": post.post_id,
                "platform": "substack",
                "content_type": "newsletter",
                "title": post.title,
                "published_at": post.published_date,
                "url": post.url,
                "is_paid": post.is_paid,
                "likes": post.likes,
                "comments": post.comments_count,
                "restacks": post.restacks,
                "engagement_rate": 0.0,
            })

        # Get subscriber count from manual metrics
        subscriber_count = 0
        open_rate = 0.0
        click_rate = 0.0

        if self._current_dashboard_metrics:
            subscriber_count = self._current_dashboard_metrics.subscribers
            open_rate = self._current_dashboard_metrics.open_rate
            click_rate = self._current_dashboard_metrics.click_rate

            # Calculate engagement rates
            for item in content_items:
                total_engagement = item["likes"] + item["comments"] + item["restacks"]
                if subscriber_count > 0:
                    item["engagement_rate"] = total_engagement / subscriber_count

        # Sort by engagement
        sorted_items = sorted(
            content_items,
            key=lambda x: x["likes"] + x["comments"] + x["restacks"],
            reverse=True
        )

        return {
            "platform": "substack",
            "period": {
                "start": period_start,
                "end": period_end
            },
            "raw_metrics": {
                "output": {
                    "posts": len(filtered_posts),
                    "newsletters": len(filtered_posts),
                },
                "reach": {
                    "subscribers": subscriber_count,
                    "free_subscribers": self._current_dashboard_metrics.free_subscribers if self._current_dashboard_metrics else 0,
                    "paid_subscribers": self._current_dashboard_metrics.paid_subscribers if self._current_dashboard_metrics else 0,
                },
                "engagement": {
                    "likes": total_likes,
                    "comments": total_comments,
                    "restacks": total_restacks,
                    "total": total_likes + total_comments + total_restacks,
                },
                "email": {
                    "open_rate": open_rate,
                    "click_rate": click_rate,
                },
                "growth": {
                    "subscriber_count": subscriber_count,
                },
            },
            "content_items": content_items,
            "top_performing": sorted_items[:5] if sorted_items else [],
            "lowest_performing": sorted_items[-5:] if len(sorted_items) >= 5 else [],
            "data_source": "mixed",
            "notes": [
                "Subscriber stats require manual input from Substack dashboard",
                "Email stats (open/click rates) require manual input",
                "Engagement stats scraped from public post pages",
            ]
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_dashboard_metrics(self) -> Optional[DashboardMetrics]:
        """Get currently stored dashboard metrics."""
        return self._current_dashboard_metrics

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Substack publication.

        Returns:
            Dictionary with connection status and details
        """
        result = {
            "status": "unknown",
            "publication_url": self.publication_url,
            "publication_name": self.publication_name,
            "rss_accessible": False,
            "post_count": 0,
            "offline_mode": self.offline_mode,
            "errors": [],
        }

        if not self.publication_url:
            result["status"] = "not_configured"
            result["errors"].append("Publication URL not set")
            return result

        if self.offline_mode:
            result["status"] = "offline_mode"
            result["errors"].append("Offline mode enabled")
            return result

        try:
            posts = await self.get_recent_posts(count=5)
            result["rss_accessible"] = True
            result["post_count"] = len(posts)
            result["status"] = "connected"

            if posts:
                result["latest_post"] = {
                    "title": posts[0].title,
                    "published_at": posts[0].published_date,
                    "url": posts[0].url,
                }
        except FeedNotFoundError as e:
            result["status"] = "error"
            result["errors"].append(f"RSS feed not found: {str(e)}")
        except ParseError as e:
            result["status"] = "error"
            result["errors"].append(f"Parse error: {str(e)}")
        except OfflineError as e:
            result["status"] = "offline"
            result["errors"].append(f"Connection error: {str(e)}")
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Unexpected error: {str(e)}")

        return result

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        asyncio.create_task(self.cache.clear())

    def get_data_source_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data sources and their status.

        Returns:
            Dictionary describing data sources and availability
        """
        return {
            "rss_feed": {
                "available": True,
                "url": self._get_rss_url() if self.publication_url else None,
                "provides": ["post_list", "titles", "dates", "urls", "content_preview"],
            },
            "web_scraping": {
                "available": not self.offline_mode,
                "provides": ["publication_info", "likes", "comments_count", "restacks", "is_paid"],
                "limitations": ["Rate limited", "May not work for all posts"],
            },
            "manual_input": {
                "available": True,
                "provides": ["subscriber_count", "open_rate", "click_rate", "detailed_stats", "revenue"],
                "has_data": {
                    "dashboard_metrics": self._current_dashboard_metrics is not None,
                    "post_stats_count": len(self._manual_post_stats),
                },
            },
            "storage": {
                "path": str(self.storage.storage_path),
                "available": True,
            },
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_substack_adapter(
    publication_url: Optional[str] = None,
    **kwargs
) -> SubstackAdapter:
    """
    Factory function to create SubstackAdapter instance.

    Can read publication URL from environment variable if not provided.

    Args:
        publication_url: Substack publication URL
        **kwargs: Additional arguments passed to SubstackAdapter

    Returns:
        Configured SubstackAdapter instance
    """
    publication_url = publication_url or os.environ.get("SUBSTACK_PUBLICATION_URL")

    if not publication_url:
        logger.warning(
            "No Substack publication URL provided. "
            "Set SUBSTACK_PUBLICATION_URL environment variable or call set_publication()."
        )

    return SubstackAdapter(
        publication_url=publication_url,
        **kwargs
    )
