"""
Unified Manual Input Interface for ALFRED

Provides a consistent way to manually enter metrics for ANY platform when
automated scraping isn't available. This is a critical fallback component
that ensures ALFRED can always receive data even when automated methods fail.

Features:
- Universal manual entry system for any platform
- Standardized entry format with validation
- Timestamp all entries
- Support for notes/context with entries
- Batch entry (multiple metrics at once)
- Historical tracking with time series
- Growth calculation between entries
- Export to JSON/CSV
- Import from JSON/CSV
- Merge with automated data

Storage: JSON files per platform with combined index file.
Path: {data_dir}/manual_input/
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import csv
import hashlib
import io
import json
import logging
import os
import statistics

try:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class Platform(str, Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    SUBSTACK = "substack"
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    THREADS = "threads"
    BLUESKY = "bluesky"
    MASTODON = "mastodon"
    MEDIUM = "medium"
    PODCAST = "podcast"
    OTHER = "other"


# Platform-specific metric schemas
PLATFORM_METRICS = {
    Platform.TWITTER: [
        "followers", "following", "tweets", "impressions",
        "profile_visits", "mentions", "retweets", "likes",
        "replies", "link_clicks", "engagement_rate"
    ],
    Platform.INSTAGRAM: [
        "followers", "following", "posts", "reach", "impressions",
        "engagement_rate", "likes", "comments", "shares", "saves",
        "stories_views", "reels_views", "profile_visits"
    ],
    Platform.SUBSTACK: [
        "subscribers", "paid_subscribers", "free_subscribers",
        "open_rate", "click_rate", "posts", "total_views",
        "email_sends", "email_opens", "restacks"
    ],
    Platform.YOUTUBE: [
        "subscribers", "views", "watch_hours", "watch_time_minutes",
        "videos", "likes", "comments", "shares", "impressions",
        "click_through_rate", "average_view_duration"
    ],
    Platform.LINKEDIN: [
        "connections", "followers", "posts", "impressions",
        "engagement_rate", "profile_views", "search_appearances",
        "article_views", "reactions", "comments", "shares"
    ],
    Platform.TIKTOK: [
        "followers", "following", "likes", "videos", "views",
        "profile_views", "shares", "comments", "engagement_rate"
    ],
    Platform.THREADS: [
        "followers", "following", "posts", "likes", "replies",
        "reposts", "quotes"
    ],
    Platform.BLUESKY: [
        "followers", "following", "posts", "likes", "reposts",
        "replies", "lists"
    ],
    Platform.MASTODON: [
        "followers", "following", "posts", "favourites",
        "boosts", "replies"
    ],
    Platform.MEDIUM: [
        "followers", "stories", "total_views", "total_reads",
        "fans", "claps", "read_ratio"
    ],
    Platform.PODCAST: [
        "episodes", "total_downloads", "unique_listeners",
        "average_completion_rate", "subscribers", "reviews",
        "rating"
    ],
    Platform.OTHER: []  # Accept any metric for other platforms
}

# Default storage path
DEFAULT_DATA_DIR = "/Users/shaileshsingh/Alfred/agent-zero1/data/alfred/manual_input"


# =============================================================================
# Error Classes
# =============================================================================

class ManualInputError(Exception):
    """Base exception for manual input errors."""
    pass


class ValidationError(ManualInputError):
    """Validation error for metric entries."""
    pass


class PlatformNotFoundError(ManualInputError):
    """Platform data not found."""
    pass


class MetricNotFoundError(ManualInputError):
    """Metric data not found."""
    pass


class DataImportError(ManualInputError):
    """Error importing data from file."""
    pass


# =============================================================================
# Data Classes (Pydantic if available, otherwise dataclass)
# =============================================================================

if PYDANTIC_AVAILABLE:
    class ManualMetricEntry(BaseModel):
        """Single manual metric entry."""
        id: str = Field(default_factory=lambda: hashlib.md5(
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12])
        platform: Platform
        metric_name: str
        value: float
        timestamp: datetime = Field(default_factory=datetime.now)
        notes: Optional[str] = None
        source: str = "manual"

        class Config:
            use_enum_values = True

        @validator('value')
        def validate_value(cls, v, values):
            if v < 0:
                logger.warning(f"Negative value {v} for metric. This may be intentional for loss metrics.")
            return v

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary for serialization."""
            return {
                "id": self.id,
                "platform": self.platform if isinstance(self.platform, str) else self.platform.value,
                "metric_name": self.metric_name,
                "value": self.value,
                "timestamp": self.timestamp.isoformat(),
                "notes": self.notes,
                "source": self.source
            }

    class ManualSnapshot(BaseModel):
        """Snapshot of multiple metrics at a point in time."""
        id: str = Field(default_factory=lambda: hashlib.md5(
            f"snapshot_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12])
        platform: Platform
        timestamp: datetime = Field(default_factory=datetime.now)
        metrics: Dict[str, float]
        notes: Optional[str] = None
        source: str = "manual"

        class Config:
            use_enum_values = True

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary for serialization."""
            return {
                "id": self.id,
                "platform": self.platform if isinstance(self.platform, str) else self.platform.value,
                "timestamp": self.timestamp.isoformat(),
                "metrics": self.metrics,
                "notes": self.notes,
                "source": self.source
            }
else:
    @dataclass
    class ManualMetricEntry:
        """Single manual metric entry."""
        platform: str
        metric_name: str
        value: float
        timestamp: datetime = field(default_factory=datetime.now)
        notes: Optional[str] = None
        source: str = "manual"
        id: str = field(default_factory=lambda: "")

        def __post_init__(self):
            if not self.id:
                self.id = hashlib.md5(
                    f"{self.platform}_{self.metric_name}_{self.timestamp.isoformat()}".encode()
                ).hexdigest()[:12]
            if self.value < 0:
                logger.warning(f"Negative value {self.value} for metric. This may be intentional.")

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary for serialization."""
            return {
                "id": self.id,
                "platform": self.platform.value if hasattr(self.platform, 'value') else self.platform,
                "metric_name": self.metric_name,
                "value": self.value,
                "timestamp": self.timestamp.isoformat(),
                "notes": self.notes,
                "source": self.source
            }

    @dataclass
    class ManualSnapshot:
        """Snapshot of multiple metrics at a point in time."""
        platform: str
        metrics: Dict[str, float]
        timestamp: datetime = field(default_factory=datetime.now)
        notes: Optional[str] = None
        source: str = "manual"
        id: str = field(default_factory=lambda: "")

        def __post_init__(self):
            if not self.id:
                self.id = hashlib.md5(
                    f"snapshot_{self.platform}_{self.timestamp.isoformat()}".encode()
                ).hexdigest()[:12]

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary for serialization."""
            return {
                "id": self.id,
                "platform": self.platform.value if hasattr(self.platform, 'value') else self.platform,
                "timestamp": self.timestamp.isoformat(),
                "metrics": self.metrics,
                "notes": self.notes,
                "source": self.source
            }


@dataclass
class GrowthReport:
    """Growth calculation report between two time periods."""
    platform: str
    metric_name: str
    start_value: float
    end_value: float
    start_date: datetime
    end_date: datetime
    absolute_change: float
    percent_change: float
    daily_average_change: float
    data_points: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform,
            "metric_name": self.metric_name,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "absolute_change": self.absolute_change,
            "percent_change": self.percent_change,
            "daily_average_change": self.daily_average_change,
            "data_points": self.data_points
        }


@dataclass
class PlatformSummary:
    """Summary of all metrics for a platform."""
    platform: str
    total_entries: int
    unique_metrics: int
    metrics: Dict[str, Dict[str, Any]]  # metric_name -> {latest, min, max, avg, count}
    first_entry: Optional[datetime]
    last_entry: Optional[datetime]
    snapshots_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform,
            "total_entries": self.total_entries,
            "unique_metrics": self.unique_metrics,
            "metrics": self.metrics,
            "first_entry": self.first_entry.isoformat() if self.first_entry else None,
            "last_entry": self.last_entry.isoformat() if self.last_entry else None,
            "snapshots_count": self.snapshots_count
        }


# =============================================================================
# Manual Input Manager
# =============================================================================

class ManualInputManager:
    """
    Unified Manual Input Interface for ALFRED.

    Provides a consistent way to manually enter metrics for any platform when
    automated scraping isn't available. Supports single entries, batch entries,
    historical tracking, growth calculations, and data import/export.

    Usage:
        manager = ManualInputManager()

        # Single metric entry
        manager.enter_metric(Platform.TWITTER, "followers", 15000, "As of Jan 2026")

        # Snapshot (multiple metrics at once)
        manager.enter_snapshot(Platform.INSTAGRAM, {
            "followers": 25000,
            "posts": 150,
            "engagement_rate": 4.5
        })

        # Get historical data
        history = manager.get_history(Platform.TWITTER, "followers", days=30)

        # Calculate growth
        growth = manager.calculate_growth(Platform.TWITTER, "followers", days=30)

        # Export data
        manager.export_data(Platform.TWITTER, format="json")
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        auto_save: bool = True,
        validate_metrics: bool = True
    ):
        """
        Initialize Manual Input Manager.

        Args:
            data_dir: Directory for storing data files
            auto_save: Automatically save after each operation
            validate_metrics: Validate metric names against platform schemas
        """
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.auto_save = auto_save
        self.validate_metrics = validate_metrics

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self._entries: Dict[str, List[ManualMetricEntry]] = {}  # platform -> entries
        self._snapshots: Dict[str, List[ManualSnapshot]] = {}   # platform -> snapshots

        # Load existing data
        self._load_all_data()

        logger.info(f"ManualInputManager initialized. Data dir: {self.data_dir}")

    # =========================================================================
    # File Operations
    # =========================================================================

    def _get_platform_file(self, platform: Union[Platform, str]) -> Path:
        """Get the JSON file path for a platform."""
        platform_str = platform.value if isinstance(platform, Platform) else platform
        return self.data_dir / f"{platform_str}_manual.json"

    def _get_index_file(self) -> Path:
        """Get the index file path."""
        return self.data_dir / "manual_entries_index.json"

    def _load_all_data(self) -> None:
        """Load all existing data from files."""
        # Load index
        index_file = self._get_index_file()
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index = json.load(f)
                    platforms = index.get("platforms", [])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load index file: {e}")
                platforms = []
        else:
            # Scan for existing platform files
            platforms = [
                f.stem.replace("_manual", "")
                for f in self.data_dir.glob("*_manual.json")
            ]

        # Load each platform's data
        for platform in platforms:
            self._load_platform_data(platform)

    def _load_platform_data(self, platform: str) -> None:
        """Load data for a specific platform."""
        file_path = self._get_platform_file(platform)

        if not file_path.exists():
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Parse entries
            entries = []
            for entry_data in data.get("entries", []):
                try:
                    entry = ManualMetricEntry(
                        id=entry_data.get("id", ""),
                        platform=entry_data["platform"],
                        metric_name=entry_data["metric_name"],
                        value=float(entry_data["value"]),
                        timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                        notes=entry_data.get("notes"),
                        source=entry_data.get("source", "manual")
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Could not parse entry: {e}")

            self._entries[platform] = entries

            # Parse snapshots
            snapshots = []
            for snapshot_data in data.get("snapshots", []):
                try:
                    snapshot = ManualSnapshot(
                        id=snapshot_data.get("id", ""),
                        platform=snapshot_data["platform"],
                        timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
                        metrics=snapshot_data["metrics"],
                        notes=snapshot_data.get("notes"),
                        source=snapshot_data.get("source", "manual")
                    )
                    snapshots.append(snapshot)
                except Exception as e:
                    logger.warning(f"Could not parse snapshot: {e}")

            self._snapshots[platform] = snapshots

            logger.debug(f"Loaded {len(entries)} entries and {len(snapshots)} snapshots for {platform}")

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load data for platform {platform}: {e}")

    def _save_platform_data(self, platform: str) -> None:
        """Save data for a specific platform."""
        file_path = self._get_platform_file(platform)

        entries = self._entries.get(platform, [])
        snapshots = self._snapshots.get(platform, [])

        data = {
            "platform": platform,
            "last_updated": datetime.now().isoformat(),
            "entries_count": len(entries),
            "snapshots_count": len(snapshots),
            "entries": [e.to_dict() for e in entries],
            "snapshots": [s.to_dict() for s in snapshots]
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved data for platform {platform}")
        except IOError as e:
            logger.error(f"Could not save data for platform {platform}: {e}")
            raise ManualInputError(f"Failed to save data: {e}")

    def _update_index(self) -> None:
        """Update the index file."""
        index = {
            "last_updated": datetime.now().isoformat(),
            "platforms": list(set(list(self._entries.keys()) + list(self._snapshots.keys()))),
            "total_entries": sum(len(e) for e in self._entries.values()),
            "total_snapshots": sum(len(s) for s in self._snapshots.values())
        }

        try:
            with open(self._get_index_file(), 'w') as f:
                json.dump(index, f, indent=2)
        except IOError as e:
            logger.warning(f"Could not update index file: {e}")

    def _save_if_auto(self, platform: str) -> None:
        """Save if auto_save is enabled."""
        if self.auto_save:
            self._save_platform_data(platform)
            self._update_index()

    # =========================================================================
    # Validation
    # =========================================================================

    def _normalize_platform(self, platform: Union[Platform, str]) -> str:
        """Normalize platform to string."""
        if isinstance(platform, Platform):
            return platform.value
        return platform.lower()

    def _validate_metric_name(self, platform: str, metric_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate metric name against platform schema.

        Returns:
            Tuple of (is_valid, warning_message)
        """
        if not self.validate_metrics:
            return True, None

        try:
            platform_enum = Platform(platform)
            expected_metrics = PLATFORM_METRICS.get(platform_enum, [])

            if not expected_metrics:  # OTHER or unknown platform
                return True, None

            if metric_name.lower() not in [m.lower() for m in expected_metrics]:
                return True, f"Metric '{metric_name}' is not in standard schema for {platform}. Expected: {expected_metrics}"

            return True, None
        except ValueError:
            return True, f"Unknown platform '{platform}'. Accepting any metric."

    def _validate_value(self, metric_name: str, value: float) -> Tuple[bool, Optional[str]]:
        """
        Validate metric value for outliers.

        Returns:
            Tuple of (is_valid, warning_message)
        """
        warnings = []

        # Check for negative values on typically positive metrics
        negative_suspicious = [
            "followers", "following", "subscribers", "views", "likes",
            "comments", "posts", "impressions", "reach"
        ]

        if value < 0 and any(m in metric_name.lower() for m in negative_suspicious):
            warnings.append(f"Negative value {value} for '{metric_name}' - this is unusual")

        # Check for suspiciously large values
        if value > 1_000_000_000:
            warnings.append(f"Very large value {value} for '{metric_name}' - please verify")

        # Check rate metrics are in expected range
        rate_metrics = ["rate", "percentage", "percent"]
        if any(m in metric_name.lower() for m in rate_metrics):
            if value > 100:
                warnings.append(f"Rate metric '{metric_name}' has value {value} > 100 - should this be a decimal?")

        return True, "; ".join(warnings) if warnings else None

    def _detect_outliers(
        self,
        platform: str,
        metric_name: str,
        value: float,
        std_threshold: float = 3.0
    ) -> Optional[str]:
        """
        Detect if a value is an outlier based on historical data.

        Returns warning message if outlier detected, None otherwise.
        """
        entries = self._entries.get(platform, [])
        relevant = [
            e.value for e in entries
            if e.metric_name.lower() == metric_name.lower()
        ]

        if len(relevant) < 5:
            return None  # Not enough data for outlier detection

        mean = statistics.mean(relevant)
        std = statistics.stdev(relevant)

        if std == 0:
            return None

        z_score = abs(value - mean) / std

        if z_score > std_threshold:
            return f"Value {value} appears to be an outlier (z-score: {z_score:.2f}). Historical mean: {mean:.2f}, std: {std:.2f}"

        return None

    # =========================================================================
    # Entry Methods
    # =========================================================================

    def enter_metric(
        self,
        platform: Union[Platform, str],
        metric_name: str,
        value: float,
        notes: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> ManualMetricEntry:
        """
        Enter a single metric value.

        Args:
            platform: Platform name or Platform enum
            metric_name: Name of the metric (e.g., "followers", "impressions")
            value: Numeric value of the metric
            notes: Optional notes/context for this entry
            timestamp: Optional timestamp (defaults to now)

        Returns:
            ManualMetricEntry object

        Raises:
            ValidationError: If validation fails

        Example:
            manager.enter_metric(Platform.TWITTER, "followers", 15000, "Monthly snapshot")
        """
        platform_str = self._normalize_platform(platform)
        timestamp = timestamp or datetime.now()

        # Validate metric name
        is_valid, warning = self._validate_metric_name(platform_str, metric_name)
        if warning:
            logger.warning(warning)

        # Validate value
        is_valid, warning = self._validate_value(metric_name, value)
        if warning:
            logger.warning(warning)

        # Check for outliers
        outlier_warning = self._detect_outliers(platform_str, metric_name, value)
        if outlier_warning:
            logger.warning(outlier_warning)

        # Create entry
        entry = ManualMetricEntry(
            platform=platform_str,
            metric_name=metric_name,
            value=value,
            timestamp=timestamp,
            notes=notes,
            source="manual"
        )

        # Store entry
        if platform_str not in self._entries:
            self._entries[platform_str] = []
        self._entries[platform_str].append(entry)

        # Sort by timestamp
        self._entries[platform_str].sort(key=lambda e: e.timestamp)

        logger.info(f"Entered metric: {platform_str}/{metric_name} = {value}")

        self._save_if_auto(platform_str)

        return entry

    def enter_snapshot(
        self,
        platform: Union[Platform, str],
        metrics: Dict[str, float],
        notes: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> ManualSnapshot:
        """
        Enter multiple metrics at once as a snapshot.

        Args:
            platform: Platform name or Platform enum
            metrics: Dictionary of metric_name -> value
            notes: Optional notes/context for this snapshot
            timestamp: Optional timestamp (defaults to now)

        Returns:
            ManualSnapshot object

        Example:
            manager.enter_snapshot(Platform.INSTAGRAM, {
                "followers": 25000,
                "posts": 150,
                "engagement_rate": 4.5
            }, notes="Weekly snapshot")
        """
        platform_str = self._normalize_platform(platform)
        timestamp = timestamp or datetime.now()

        # Validate each metric
        for metric_name, value in metrics.items():
            is_valid, warning = self._validate_metric_name(platform_str, metric_name)
            if warning:
                logger.warning(warning)

            is_valid, warning = self._validate_value(metric_name, value)
            if warning:
                logger.warning(warning)

        # Create snapshot
        snapshot = ManualSnapshot(
            platform=platform_str,
            timestamp=timestamp,
            metrics=metrics,
            notes=notes,
            source="manual"
        )

        # Store snapshot
        if platform_str not in self._snapshots:
            self._snapshots[platform_str] = []
        self._snapshots[platform_str].append(snapshot)

        # Sort by timestamp
        self._snapshots[platform_str].sort(key=lambda s: s.timestamp)

        # Also create individual entries for each metric
        for metric_name, value in metrics.items():
            entry = ManualMetricEntry(
                platform=platform_str,
                metric_name=metric_name,
                value=value,
                timestamp=timestamp,
                notes=f"From snapshot: {notes}" if notes else "From snapshot",
                source="snapshot"
            )
            if platform_str not in self._entries:
                self._entries[platform_str] = []
            self._entries[platform_str].append(entry)

        # Sort entries
        self._entries[platform_str].sort(key=lambda e: e.timestamp)

        logger.info(f"Entered snapshot for {platform_str} with {len(metrics)} metrics")

        self._save_if_auto(platform_str)

        return snapshot

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_history(
        self,
        platform: Union[Platform, str],
        metric_name: str,
        days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ManualMetricEntry]:
        """
        Get historical values for a specific metric.

        Args:
            platform: Platform name or Platform enum
            metric_name: Name of the metric
            days: Number of days to look back (alternative to date range)
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of ManualMetricEntry objects sorted by timestamp
        """
        platform_str = self._normalize_platform(platform)

        entries = self._entries.get(platform_str, [])
        if not entries:
            return []

        # Filter by metric name
        filtered = [
            e for e in entries
            if e.metric_name.lower() == metric_name.lower()
        ]

        # Apply date filter
        if days is not None:
            cutoff = datetime.now() - timedelta(days=days)
            filtered = [e for e in filtered if e.timestamp >= cutoff]
        elif start_date or end_date:
            if start_date:
                filtered = [e for e in filtered if e.timestamp >= start_date]
            if end_date:
                filtered = [e for e in filtered if e.timestamp <= end_date]

        return sorted(filtered, key=lambda e: e.timestamp)

    def get_latest(
        self,
        platform: Union[Platform, str],
        metric_name: str
    ) -> Optional[ManualMetricEntry]:
        """
        Get the most recent value for a specific metric.

        Args:
            platform: Platform name or Platform enum
            metric_name: Name of the metric

        Returns:
            Most recent ManualMetricEntry or None if not found
        """
        history = self.get_history(platform, metric_name)
        return history[-1] if history else None

    def get_all_platforms(self) -> List[str]:
        """
        Get list of all platforms with manual data.

        Returns:
            List of platform names
        """
        platforms = set(list(self._entries.keys()) + list(self._snapshots.keys()))
        return sorted(list(platforms))

    def get_all_metrics(self, platform: Union[Platform, str]) -> List[str]:
        """
        Get list of all metrics recorded for a platform.

        Args:
            platform: Platform name or Platform enum

        Returns:
            List of unique metric names
        """
        platform_str = self._normalize_platform(platform)
        entries = self._entries.get(platform_str, [])
        return sorted(list(set(e.metric_name for e in entries)))

    def get_expected_metrics(self, platform: Union[Platform, str]) -> List[str]:
        """
        Get list of expected metrics for a platform based on schema.

        Args:
            platform: Platform name or Platform enum

        Returns:
            List of expected metric names
        """
        platform_str = self._normalize_platform(platform)
        try:
            platform_enum = Platform(platform_str)
            return PLATFORM_METRICS.get(platform_enum, [])
        except ValueError:
            return []

    # =========================================================================
    # Growth Calculation
    # =========================================================================

    def calculate_growth(
        self,
        platform: Union[Platform, str],
        metric_name: str,
        days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[GrowthReport]:
        """
        Calculate growth rate for a metric over a time period.

        Args:
            platform: Platform name or Platform enum
            metric_name: Name of the metric
            days: Number of days to analyze (alternative to date range)
            start_date: Start of date range
            end_date: End of date range

        Returns:
            GrowthReport object or None if insufficient data

        Example:
            growth = manager.calculate_growth(Platform.TWITTER, "followers", days=30)
            print(f"Growth: {growth.percent_change:.1f}%")
        """
        history = self.get_history(
            platform, metric_name,
            days=days, start_date=start_date, end_date=end_date
        )

        if len(history) < 2:
            logger.warning(f"Insufficient data for growth calculation (need at least 2 entries)")
            return None

        platform_str = self._normalize_platform(platform)

        start_entry = history[0]
        end_entry = history[-1]

        start_value = start_entry.value
        end_value = end_entry.value

        # Calculate changes
        absolute_change = end_value - start_value

        if start_value != 0:
            percent_change = (absolute_change / start_value) * 100
        else:
            percent_change = 100.0 if end_value > 0 else 0.0

        # Calculate daily average
        days_elapsed = (end_entry.timestamp - start_entry.timestamp).days
        daily_average = absolute_change / days_elapsed if days_elapsed > 0 else absolute_change

        return GrowthReport(
            platform=platform_str,
            metric_name=metric_name,
            start_value=start_value,
            end_value=end_value,
            start_date=start_entry.timestamp,
            end_date=end_entry.timestamp,
            absolute_change=absolute_change,
            percent_change=percent_change,
            daily_average_change=daily_average,
            data_points=len(history)
        )

    # =========================================================================
    # Platform Summary
    # =========================================================================

    def get_platform_summary(
        self,
        platform: Union[Platform, str]
    ) -> PlatformSummary:
        """
        Get summary of all metrics for a platform.

        Args:
            platform: Platform name or Platform enum

        Returns:
            PlatformSummary object with aggregated statistics

        Example:
            summary = manager.get_platform_summary(Platform.TWITTER)
            print(f"Total entries: {summary.total_entries}")
            print(f"Metrics tracked: {summary.unique_metrics}")
        """
        platform_str = self._normalize_platform(platform)

        entries = self._entries.get(platform_str, [])
        snapshots = self._snapshots.get(platform_str, [])

        if not entries:
            return PlatformSummary(
                platform=platform_str,
                total_entries=0,
                unique_metrics=0,
                metrics={},
                first_entry=None,
                last_entry=None,
                snapshots_count=len(snapshots)
            )

        # Group by metric
        metrics_data: Dict[str, Dict[str, Any]] = {}

        for metric_name in self.get_all_metrics(platform):
            history = self.get_history(platform, metric_name)
            if not history:
                continue

            values = [e.value for e in history]

            metrics_data[metric_name] = {
                "latest": values[-1],
                "latest_timestamp": history[-1].timestamp.isoformat(),
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "count": len(values)
            }

            # Add growth if possible
            growth = self.calculate_growth(platform, metric_name)
            if growth:
                metrics_data[metric_name]["growth_percent"] = growth.percent_change
                metrics_data[metric_name]["growth_absolute"] = growth.absolute_change

        return PlatformSummary(
            platform=platform_str,
            total_entries=len(entries),
            unique_metrics=len(metrics_data),
            metrics=metrics_data,
            first_entry=entries[0].timestamp if entries else None,
            last_entry=entries[-1].timestamp if entries else None,
            snapshots_count=len(snapshots)
        )

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_data(
        self,
        platform: Union[Platform, str],
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export platform data to JSON or CSV.

        Args:
            platform: Platform name or Platform enum
            format: Output format ("json" or "csv")
            output_path: Optional output file path (returns string if None)

        Returns:
            String containing exported data (or file path if output_path provided)

        Example:
            # Export to string
            json_data = manager.export_data(Platform.TWITTER, format="json")

            # Export to file
            manager.export_data(Platform.TWITTER, format="csv", output_path="twitter_data.csv")
        """
        platform_str = self._normalize_platform(platform)

        entries = self._entries.get(platform_str, [])
        snapshots = self._snapshots.get(platform_str, [])

        if format.lower() == "json":
            data = {
                "platform": platform_str,
                "exported_at": datetime.now().isoformat(),
                "entries_count": len(entries),
                "snapshots_count": len(snapshots),
                "entries": [e.to_dict() for e in entries],
                "snapshots": [s.to_dict() for s in snapshots]
            }

            output = json.dumps(data, indent=2)

        elif format.lower() == "csv":
            output_buffer = io.StringIO()
            writer = csv.writer(output_buffer)

            # Header
            writer.writerow([
                "id", "platform", "metric_name", "value",
                "timestamp", "notes", "source"
            ])

            # Entries
            for entry in entries:
                writer.writerow([
                    entry.id,
                    entry.platform if isinstance(entry.platform, str) else entry.platform.value,
                    entry.metric_name,
                    entry.value,
                    entry.timestamp.isoformat(),
                    entry.notes or "",
                    entry.source
                ])

            output = output_buffer.getvalue()

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

        if output_path:
            with open(output_path, 'w') as f:
                f.write(output)
            logger.info(f"Exported {platform_str} data to {output_path}")
            return output_path

        return output

    def export_all_data(
        self,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export all platform data to a single file.

        Args:
            format: Output format ("json" or "csv")
            output_path: Optional output file path

        Returns:
            String containing exported data
        """
        all_entries = []
        all_snapshots = []

        for platform in self.get_all_platforms():
            entries = self._entries.get(platform, [])
            snapshots = self._snapshots.get(platform, [])
            all_entries.extend(entries)
            all_snapshots.extend(snapshots)

        if format.lower() == "json":
            data = {
                "exported_at": datetime.now().isoformat(),
                "platforms": self.get_all_platforms(),
                "total_entries": len(all_entries),
                "total_snapshots": len(all_snapshots),
                "entries": [e.to_dict() for e in all_entries],
                "snapshots": [s.to_dict() for s in all_snapshots]
            }
            output = json.dumps(data, indent=2)

        elif format.lower() == "csv":
            output_buffer = io.StringIO()
            writer = csv.writer(output_buffer)

            writer.writerow([
                "id", "platform", "metric_name", "value",
                "timestamp", "notes", "source"
            ])

            for entry in all_entries:
                writer.writerow([
                    entry.id,
                    entry.platform if isinstance(entry.platform, str) else entry.platform.value,
                    entry.metric_name,
                    entry.value,
                    entry.timestamp.isoformat(),
                    entry.notes or "",
                    entry.source
                ])

            output = output_buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write(output)
            logger.info(f"Exported all data to {output_path}")
            return output_path

        return output

    # =========================================================================
    # Import Methods
    # =========================================================================

    def import_data(
        self,
        file_path: str,
        merge: bool = True
    ) -> Dict[str, int]:
        """
        Import data from JSON or CSV file.

        Args:
            file_path: Path to the import file
            merge: If True, merge with existing data; if False, replace

        Returns:
            Dictionary with import statistics

        Example:
            stats = manager.import_data("twitter_backup.json")
            print(f"Imported {stats['entries']} entries")
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataImportError(f"File not found: {file_path}")

        if file_path.suffix.lower() == ".json":
            return self._import_json(file_path, merge)
        elif file_path.suffix.lower() == ".csv":
            return self._import_csv(file_path, merge)
        else:
            raise DataImportError(f"Unsupported file format: {file_path.suffix}")

    def _import_json(self, file_path: Path, merge: bool) -> Dict[str, int]:
        """Import from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise DataImportError(f"Failed to read JSON file: {e}")

        stats = {"entries": 0, "snapshots": 0, "platforms": 0}
        platforms_updated = set()

        # Import entries
        for entry_data in data.get("entries", []):
            try:
                platform = entry_data["platform"]

                if not merge and platform not in platforms_updated:
                    self._entries[platform] = []
                    platforms_updated.add(platform)

                entry = ManualMetricEntry(
                    id=entry_data.get("id", ""),
                    platform=platform,
                    metric_name=entry_data["metric_name"],
                    value=float(entry_data["value"]),
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    notes=entry_data.get("notes"),
                    source=entry_data.get("source", "imported")
                )

                if platform not in self._entries:
                    self._entries[platform] = []
                self._entries[platform].append(entry)
                stats["entries"] += 1

            except Exception as e:
                logger.warning(f"Could not import entry: {e}")

        # Import snapshots
        for snapshot_data in data.get("snapshots", []):
            try:
                platform = snapshot_data["platform"]

                if not merge and platform not in platforms_updated:
                    self._snapshots[platform] = []
                    platforms_updated.add(platform)

                snapshot = ManualSnapshot(
                    id=snapshot_data.get("id", ""),
                    platform=platform,
                    timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
                    metrics=snapshot_data["metrics"],
                    notes=snapshot_data.get("notes"),
                    source=snapshot_data.get("source", "imported")
                )

                if platform not in self._snapshots:
                    self._snapshots[platform] = []
                self._snapshots[platform].append(snapshot)
                stats["snapshots"] += 1

            except Exception as e:
                logger.warning(f"Could not import snapshot: {e}")

        # Sort and save
        for platform in platforms_updated.union(
            set(e["platform"] for e in data.get("entries", []) if "platform" in e)
        ):
            if platform in self._entries:
                self._entries[platform].sort(key=lambda e: e.timestamp)
            if platform in self._snapshots:
                self._snapshots[platform].sort(key=lambda s: s.timestamp)
            self._save_platform_data(platform)

        stats["platforms"] = len(platforms_updated)
        self._update_index()

        logger.info(f"Imported {stats['entries']} entries and {stats['snapshots']} snapshots")
        return stats

    def _import_csv(self, file_path: Path, merge: bool) -> Dict[str, int]:
        """Import from CSV file."""
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except (csv.Error, IOError) as e:
            raise DataImportError(f"Failed to read CSV file: {e}")

        stats = {"entries": 0, "snapshots": 0, "platforms": 0}
        platforms_updated = set()

        for row in rows:
            try:
                platform = row["platform"]

                if not merge and platform not in platforms_updated:
                    self._entries[platform] = []
                    platforms_updated.add(platform)

                entry = ManualMetricEntry(
                    id=row.get("id", ""),
                    platform=platform,
                    metric_name=row["metric_name"],
                    value=float(row["value"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    notes=row.get("notes") or None,
                    source=row.get("source", "imported")
                )

                if platform not in self._entries:
                    self._entries[platform] = []
                self._entries[platform].append(entry)
                stats["entries"] += 1
                platforms_updated.add(platform)

            except Exception as e:
                logger.warning(f"Could not import row: {e}")

        # Sort and save
        for platform in platforms_updated:
            self._entries[platform].sort(key=lambda e: e.timestamp)
            self._save_platform_data(platform)

        stats["platforms"] = len(platforms_updated)
        self._update_index()

        logger.info(f"Imported {stats['entries']} entries from CSV")
        return stats

    # =========================================================================
    # Integration with Automated Data
    # =========================================================================

    def merge_with_automated(
        self,
        platform: Union[Platform, str],
        automated_data: Dict[str, Any],
        prefer_manual: bool = True
    ) -> Dict[str, Any]:
        """
        Merge manual data with automated scraped data.

        Args:
            platform: Platform name or Platform enum
            automated_data: Data from automated scraping
            prefer_manual: If True, manual data takes precedence on conflicts

        Returns:
            Merged data dictionary

        Example:
            automated = await twitter_adapter.get_metrics()
            merged = manager.merge_with_automated(Platform.TWITTER, automated)
        """
        platform_str = self._normalize_platform(platform)

        # Start with automated data as base
        merged = automated_data.copy()
        merged["data_sources"] = ["automated"]

        # Get latest manual values
        manual_metrics = {}
        for metric_name in self.get_all_metrics(platform):
            latest = self.get_latest(platform, metric_name)
            if latest:
                manual_metrics[metric_name] = {
                    "value": latest.value,
                    "timestamp": latest.timestamp.isoformat(),
                    "notes": latest.notes
                }

        if manual_metrics:
            merged["data_sources"].append("manual")

            # Merge raw_metrics if present
            if "raw_metrics" in merged:
                for category in merged["raw_metrics"]:
                    if isinstance(merged["raw_metrics"][category], dict):
                        for metric_name, value in merged["raw_metrics"][category].items():
                            if metric_name in manual_metrics:
                                manual_value = manual_metrics[metric_name]["value"]
                                if prefer_manual:
                                    merged["raw_metrics"][category][metric_name] = manual_value
                                elif value == 0 or value is None:
                                    # Use manual if automated is empty
                                    merged["raw_metrics"][category][metric_name] = manual_value

            # Add manual_supplements section
            merged["manual_supplements"] = manual_metrics

        merged["merge_timestamp"] = datetime.now().isoformat()

        return merged

    def get_metrics_for_harvester(
        self,
        platform: Union[Platform, str],
        period_start: Optional[str] = None,
        period_end: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metrics formatted for Social Metrics Harvester.

        Args:
            platform: Platform name or Platform enum
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)

        Returns:
            Dictionary in harvester format
        """
        platform_str = self._normalize_platform(platform)

        period_end = period_end or datetime.now().strftime("%Y-%m-%d")
        period_start = period_start or (
            datetime.now() - timedelta(days=30)
        ).strftime("%Y-%m-%d")

        start_dt = datetime.fromisoformat(period_start)
        end_dt = datetime.fromisoformat(period_end)

        summary = self.get_platform_summary(platform)

        # Get entries in period
        entries_in_period = []
        for entry in self._entries.get(platform_str, []):
            if start_dt <= entry.timestamp <= end_dt:
                entries_in_period.append(entry)

        # Build raw_metrics from latest values
        raw_metrics = {
            "output": {},
            "reach": {},
            "engagement": {},
            "growth": {}
        }

        for metric_name, data in summary.metrics.items():
            # Categorize metrics
            if metric_name in ["posts", "videos", "stories", "tweets", "newsletters"]:
                raw_metrics["output"][metric_name] = data["latest"]
            elif metric_name in ["followers", "subscribers", "reach", "impressions", "views"]:
                raw_metrics["reach"][metric_name] = data["latest"]
            elif metric_name in ["likes", "comments", "shares", "engagement_rate", "saves"]:
                raw_metrics["engagement"][metric_name] = data["latest"]
            else:
                raw_metrics["growth"][metric_name] = data["latest"]

        return {
            "platform": platform_str,
            "source": "manual_input",
            "period": {
                "start": period_start,
                "end": period_end
            },
            "raw_metrics": raw_metrics,
            "summary": summary.to_dict(),
            "entries_count": len(entries_in_period),
            "collected_at": datetime.now().isoformat()
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def save_all(self) -> None:
        """Force save all data to disk."""
        for platform in self.get_all_platforms():
            self._save_platform_data(platform)
        self._update_index()
        logger.info("Saved all manual input data")

    def clear_platform(self, platform: Union[Platform, str]) -> None:
        """
        Clear all data for a platform.

        Args:
            platform: Platform name or Platform enum
        """
        platform_str = self._normalize_platform(platform)

        self._entries[platform_str] = []
        self._snapshots[platform_str] = []

        # Remove file
        file_path = self._get_platform_file(platform_str)
        if file_path.exists():
            file_path.unlink()

        self._update_index()
        logger.info(f"Cleared all data for {platform_str}")

    def delete_entry(
        self,
        platform: Union[Platform, str],
        entry_id: str
    ) -> bool:
        """
        Delete a specific entry.

        Args:
            platform: Platform name or Platform enum
            entry_id: ID of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        platform_str = self._normalize_platform(platform)

        entries = self._entries.get(platform_str, [])
        original_len = len(entries)

        self._entries[platform_str] = [
            e for e in entries if e.id != entry_id
        ]

        if len(self._entries[platform_str]) < original_len:
            self._save_if_auto(platform_str)
            logger.info(f"Deleted entry {entry_id}")
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get overall status of the manual input system.

        Returns:
            Dictionary with status information
        """
        platforms = self.get_all_platforms()

        platform_stats = {}
        for platform in platforms:
            entries = self._entries.get(platform, [])
            snapshots = self._snapshots.get(platform, [])
            platform_stats[platform] = {
                "entries": len(entries),
                "snapshots": len(snapshots),
                "metrics": len(self.get_all_metrics(platform)),
                "last_entry": entries[-1].timestamp.isoformat() if entries else None
            }

        return {
            "data_dir": str(self.data_dir),
            "platforms_count": len(platforms),
            "platforms": platforms,
            "total_entries": sum(len(e) for e in self._entries.values()),
            "total_snapshots": sum(len(s) for s in self._snapshots.values()),
            "platform_stats": platform_stats,
            "auto_save": self.auto_save,
            "validate_metrics": self.validate_metrics
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_manual_input_manager(
    data_dir: Optional[str] = None,
    **kwargs
) -> ManualInputManager:
    """
    Factory function to create ManualInputManager instance.

    Args:
        data_dir: Directory for storing data files
        **kwargs: Additional arguments passed to ManualInputManager

    Returns:
        Configured ManualInputManager instance

    Example:
        manager = create_manual_input_manager()
        manager.enter_metric(Platform.TWITTER, "followers", 15000)
    """
    data_dir = data_dir or os.environ.get(
        "ALFRED_MANUAL_INPUT_DIR",
        DEFAULT_DATA_DIR
    )

    return ManualInputManager(
        data_dir=data_dir,
        **kwargs
    )


# =============================================================================
# CLI Interface (for standalone usage)
# =============================================================================

def main():
    """CLI interface for manual input."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ALFRED Manual Input Interface"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Enter metric command
    enter_parser = subparsers.add_parser("enter", help="Enter a metric")
    enter_parser.add_argument("platform", help="Platform name")
    enter_parser.add_argument("metric", help="Metric name")
    enter_parser.add_argument("value", type=float, help="Metric value")
    enter_parser.add_argument("--notes", help="Optional notes")

    # List command
    list_parser = subparsers.add_parser("list", help="List platforms or metrics")
    list_parser.add_argument("--platform", help="Show metrics for platform")

    # History command
    history_parser = subparsers.add_parser("history", help="Show metric history")
    history_parser.add_argument("platform", help="Platform name")
    history_parser.add_argument("metric", help="Metric name")
    history_parser.add_argument("--days", type=int, default=30, help="Days to show")

    # Growth command
    growth_parser = subparsers.add_parser("growth", help="Calculate growth")
    growth_parser.add_argument("platform", help="Platform name")
    growth_parser.add_argument("metric", help="Metric name")
    growth_parser.add_argument("--days", type=int, default=30, help="Days to analyze")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show platform summary")
    summary_parser.add_argument("platform", help="Platform name")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export data")
    export_parser.add_argument("platform", help="Platform name or 'all'")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json")
    export_parser.add_argument("--output", help="Output file path")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import data")
    import_parser.add_argument("file", help="File path to import")
    import_parser.add_argument("--no-merge", action="store_true", help="Replace instead of merge")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")

    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Enter snapshot (multiple metrics)")
    snapshot_parser.add_argument("platform", help="Platform name")
    snapshot_parser.add_argument("--metrics", help="JSON string of metrics dict")
    snapshot_parser.add_argument("--notes", help="Optional notes")

    args = parser.parse_args()

    manager = create_manual_input_manager()

    if args.command == "enter":
        entry = manager.enter_metric(
            args.platform, args.metric, args.value, args.notes
        )
        print(f"Entered: {entry.platform}/{entry.metric_name} = {entry.value}")

    elif args.command == "list":
        if args.platform:
            metrics = manager.get_all_metrics(args.platform)
            expected = manager.get_expected_metrics(args.platform)
            print(f"Metrics for {args.platform}:")
            print(f"  Recorded: {len(metrics)}")
            for m in metrics:
                latest = manager.get_latest(args.platform, m)
                print(f"    {m}: {latest.value if latest else 'N/A'}")
            if expected:
                print(f"  Expected (schema): {expected}")
        else:
            platforms = manager.get_all_platforms()
            print("Platforms with data:")
            for p in platforms:
                entries = len(manager._entries.get(p, []))
                print(f"  {p}: {entries} entries")

    elif args.command == "history":
        history = manager.get_history(args.platform, args.metric, days=args.days)
        print(f"History for {args.platform}/{args.metric} (last {args.days} days):")
        for entry in history:
            print(f"  {entry.timestamp.strftime('%Y-%m-%d %H:%M')}: {entry.value}")
            if entry.notes:
                print(f"    Notes: {entry.notes}")

    elif args.command == "growth":
        growth = manager.calculate_growth(args.platform, args.metric, days=args.days)
        if growth:
            print(f"Growth for {args.platform}/{args.metric}:")
            print(f"  Period: {growth.start_date.strftime('%Y-%m-%d')} to {growth.end_date.strftime('%Y-%m-%d')}")
            print(f"  Start value: {growth.start_value:,.0f}")
            print(f"  End value: {growth.end_value:,.0f}")
            print(f"  Absolute change: {growth.absolute_change:+,.0f}")
            print(f"  Percent change: {growth.percent_change:+.1f}%")
            print(f"  Daily average: {growth.daily_average_change:+,.1f}")
            print(f"  Data points: {growth.data_points}")
        else:
            print("Insufficient data for growth calculation")

    elif args.command == "summary":
        summary = manager.get_platform_summary(args.platform)
        print(f"Summary for {summary.platform}:")
        print(f"  Total entries: {summary.total_entries}")
        print(f"  Unique metrics: {summary.unique_metrics}")
        print(f"  Snapshots: {summary.snapshots_count}")
        if summary.first_entry:
            print(f"  First entry: {summary.first_entry.strftime('%Y-%m-%d')}")
        if summary.last_entry:
            print(f"  Last entry: {summary.last_entry.strftime('%Y-%m-%d')}")
        print("  Metrics:")
        for name, data in summary.metrics.items():
            growth_str = ""
            if "growth_percent" in data:
                growth_str = f" ({data['growth_percent']:+.1f}%)"
            print(f"    {name}: {data['latest']:,.0f}{growth_str}")

    elif args.command == "export":
        if args.platform.lower() == "all":
            output = manager.export_all_data(
                format=args.format, output_path=args.output
            )
        else:
            output = manager.export_data(
                args.platform, format=args.format, output_path=args.output
            )
        if args.output:
            print(f"Exported to {args.output}")
        else:
            print(output)

    elif args.command == "import":
        stats = manager.import_data(args.file, merge=not args.no_merge)
        print(f"Imported:")
        print(f"  Entries: {stats['entries']}")
        print(f"  Snapshots: {stats['snapshots']}")
        print(f"  Platforms: {stats['platforms']}")

    elif args.command == "status":
        status = manager.get_status()
        print("Manual Input System Status:")
        print(f"  Data directory: {status['data_dir']}")
        print(f"  Platforms: {status['platforms_count']}")
        print(f"  Total entries: {status['total_entries']}")
        print(f"  Total snapshots: {status['total_snapshots']}")
        print(f"  Auto-save: {status['auto_save']}")
        print(f"  Validate metrics: {status['validate_metrics']}")
        if status['platform_stats']:
            print("  Platform details:")
            for p, s in status['platform_stats'].items():
                print(f"    {p}: {s['entries']} entries, {s['metrics']} metrics")

    elif args.command == "snapshot":
        if args.metrics:
            try:
                metrics = json.loads(args.metrics)
            except json.JSONDecodeError:
                print("Error: --metrics must be valid JSON")
                return
            snapshot = manager.enter_snapshot(args.platform, metrics, args.notes)
            print(f"Entered snapshot for {snapshot.platform} with {len(snapshot.metrics)} metrics")
        else:
            print("Error: --metrics is required for snapshot command")
            print("Example: --metrics '{\"followers\": 15000, \"posts\": 500}'")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
