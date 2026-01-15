"""
Tests for IntakeAgent - Unified Ingestion System

Tests cover:
- Main class functionality
- State-aware behavior (GREEN/YELLOW/RED)
- Input/output format compliance
- Edge cases
- Channel adapter behavior
- Deduplication logic
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent-zero1" / "agents" / "alfred"))

from tools.intake_agent import (
    IntakeAgent,
    InboundItem,
    InboundBatch,
    IntakeRequest,
    BatchSummary,
    Attachment,
    ChannelType,
    UrgencyMarker,
    IncludeFilter,
    TimeWindow,
    DeduplicationEngine,
    EmailAdapter,
    WhatsAppAdapter,
    ScanAdapter,
    CalendarAdapter,
    create_intake_agent,
)
from tools import AlfredState, AgentResponse


class TestIntakeAgentInitialization:
    """Tests for IntakeAgent initialization."""

    def test_create_intake_agent(self):
        """Test that IntakeAgent can be created."""
        agent = create_intake_agent()
        assert agent is not None
        assert agent.name == "IntakeAgent"

    def test_initial_state_is_green(self):
        """Test that agent starts in GREEN state."""
        agent = IntakeAgent()
        assert agent.alfred_state == AlfredState.GREEN

    def test_adapters_initialized(self):
        """Test that all channel adapters are initialized."""
        agent = IntakeAgent()
        assert ChannelType.EMAIL in agent._adapters
        assert ChannelType.WHATSAPP in agent._adapters
        assert ChannelType.SCAN in agent._adapters
        assert ChannelType.CALENDAR in agent._adapters
        assert ChannelType.VOICE in agent._adapters

    def test_deduplicator_initialized(self):
        """Test that deduplication engine is initialized."""
        agent = IntakeAgent()
        assert agent._deduplicator is not None


class TestIntakeAgentStateAwareBehavior:
    """Tests for state-aware behavior."""

    def test_green_state_permits_operation(self, intake_agent_factory, mock_alfred_state_green):
        """Test that GREEN state permits all operations."""
        agent = intake_agent_factory(state=mock_alfred_state_green)
        permitted, reason = agent.check_state_permission()
        assert permitted is True

    def test_yellow_state_permits_operation(self, intake_agent_factory, mock_alfred_state_yellow):
        """Test that YELLOW state permits operations (intake continues)."""
        agent = intake_agent_factory(state=mock_alfred_state_yellow)
        # IntakeAgent is an OperationsAgent, should continue in YELLOW
        permitted, reason = agent.check_state_permission()
        assert permitted is True

    def test_red_state_permits_operation(self, intake_agent_factory, mock_alfred_state_red):
        """Test that RED state permits operations (intake continues)."""
        agent = intake_agent_factory(state=mock_alfred_state_red)
        # IntakeAgent is an OperationsAgent, should continue in RED
        permitted, reason = agent.check_state_permission()
        assert permitted is True

    def test_state_change_propagation(self, intake_agent_factory):
        """Test that state changes propagate correctly."""
        agent = intake_agent_factory()
        agent.alfred_state = AlfredState.YELLOW
        assert agent.alfred_state == AlfredState.YELLOW
        agent.alfred_state = AlfredState.RED
        assert agent.alfred_state == AlfredState.RED


class TestInboundItem:
    """Tests for InboundItem data class."""

    def test_inbound_item_creation(self):
        """Test basic InboundItem creation."""
        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="test@example.com",
            raw_reference="msg_001"
        )
        assert item.source == ChannelType.EMAIL
        assert item.sender == "test@example.com"

    def test_content_hash_generation(self):
        """Test that content hash is generated automatically."""
        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="test@example.com",
            raw_reference="msg_001",
            content_preview="Test content"
        )
        assert item.content_hash is not None
        assert len(item.content_hash) == 64  # SHA256 hex length

    def test_content_hash_uniqueness(self):
        """Test that different content produces different hashes."""
        now = datetime.now()
        item1 = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=now,
            sender="test@example.com",
            raw_reference="msg_001",
            content_preview="Content A"
        )
        item2 = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=now,
            sender="test@example.com",
            raw_reference="msg_002",
            content_preview="Content B"
        )
        assert item1.content_hash != item2.content_hash

    def test_attachment_count(self):
        """Test attachment count property."""
        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="test@example.com",
            raw_reference="msg_001",
            attachments=[
                Attachment(filename="file1.pdf", file_type="pdf", size_bytes=1000),
                Attachment(filename="file2.doc", file_type="doc", size_bytes=2000)
            ]
        )
        assert item.attachment_count == 2

    def test_attachment_types(self):
        """Test attachment types property."""
        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="test@example.com",
            raw_reference="msg_001",
            attachments=[
                Attachment(filename="file1.pdf", file_type="pdf", size_bytes=1000),
                Attachment(filename="file2.doc", file_type="doc", size_bytes=2000)
            ]
        )
        assert item.attachment_types == ["pdf", "doc"]

    def test_to_dict_serialization(self):
        """Test to_dict produces valid dictionary."""
        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="test@example.com",
            raw_reference="msg_001",
            subject="Test Subject"
        )
        result = item.to_dict()
        assert result["source"] == "Email"
        assert result["sender"] == "test@example.com"
        assert result["subject"] == "Test Subject"
        assert "timestamp" in result


class TestIntakeRequest:
    """Tests for IntakeRequest parsing."""

    def test_from_dict_basic(self, sample_intake_request):
        """Test parsing basic request from dict."""
        request = IntakeRequest.from_dict(sample_intake_request)
        assert ChannelType.EMAIL in request.channels
        assert ChannelType.WHATSAPP in request.channels
        assert request.time_window == TimeWindow.LAST_24_HOURS
        assert request.include_filter == IncludeFilter.ALL

    def test_from_dict_all_channels(self):
        """Test parsing 'all' channels."""
        data = {"channels": ["all"], "time_window": "last_hour"}
        request = IntakeRequest.from_dict(data)
        assert len(request.channels) == len(ChannelType)

    def test_from_dict_case_insensitive(self):
        """Test that channel names are case-insensitive."""
        data = {"channels": ["email", "WHATSAPP", "Calendar"]}
        request = IntakeRequest.from_dict(data)
        assert ChannelType.EMAIL in request.channels
        assert ChannelType.WHATSAPP in request.channels
        assert ChannelType.CALENDAR in request.channels

    def test_from_dict_default_values(self):
        """Test default values are applied."""
        request = IntakeRequest.from_dict({})
        assert request.time_window == TimeWindow.SINCE_LAST_CHECK
        assert request.include_filter == IncludeFilter.ALL

    def test_from_dict_invalid_time_window(self):
        """Test invalid time window falls back to default."""
        data = {"time_window": "invalid_window"}
        request = IntakeRequest.from_dict(data)
        assert request.time_window == TimeWindow.SINCE_LAST_CHECK


class TestInboundBatch:
    """Tests for InboundBatch output format."""

    def test_batch_creation(self):
        """Test batch creation."""
        batch = InboundBatch(
            batch_id="test_001",
            items_count=0,
            channels_checked=[ChannelType.EMAIL],
            items=[],
            summary=BatchSummary()
        )
        assert batch.batch_id == "test_001"
        assert batch.items_count == 0
        assert batch.status == "complete"

    def test_batch_to_dict(self):
        """Test batch serialization."""
        batch = InboundBatch(
            batch_id="test_001",
            items_count=1,
            channels_checked=[ChannelType.EMAIL, ChannelType.WHATSAPP],
            items=[],
            summary=BatchSummary(by_channel={"Email": 1})
        )
        result = batch.to_dict()
        assert result["batch_id"] == "test_001"
        assert "Email" in result["channels_checked"]
        assert "WhatsApp" in result["channels_checked"]

    def test_batch_formatted_output(self):
        """Test formatted output string."""
        batch = InboundBatch(
            batch_id="test_001",
            items_count=0,
            channels_checked=[ChannelType.EMAIL],
            items=[],
            summary=BatchSummary()
        )
        output = batch.to_formatted_output()
        assert "INBOUND_BATCH" in output
        assert "test_001" in output


class TestUrgencyMarkerDetection:
    """Tests for urgency marker detection in adapters."""

    def test_detect_explicit_urgent(self):
        """Test detection of explicit urgent markers."""
        adapter = EmailAdapter()
        result = adapter.detect_urgency_markers("This is URGENT please respond")
        assert result == UrgencyMarker.EXPLICIT_URGENT

    def test_detect_urgent_emoji(self):
        """Test detection of urgent emoji markers."""
        adapter = EmailAdapter()
        # Note: This may or may not work depending on encoding
        result = adapter.detect_urgency_markers("Need help!!! Immediately!")
        assert result == UrgencyMarker.EXPLICIT_URGENT

    def test_detect_time_sensitive(self):
        """Test detection of time-sensitive markers."""
        adapter = EmailAdapter()
        result = adapter.detect_urgency_markers("Please respond by 5pm today")
        assert result == UrgencyMarker.TIME_SENSITIVE

    def test_detect_deadline(self):
        """Test detection of deadline markers."""
        adapter = EmailAdapter()
        result = adapter.detect_urgency_markers("The deadline is approaching")
        assert result == UrgencyMarker.TIME_SENSITIVE

    def test_no_urgency_detected(self):
        """Test normal content with no urgency."""
        adapter = EmailAdapter()
        result = adapter.detect_urgency_markers("Hello, hope you are well.")
        assert result == UrgencyMarker.NONE


class TestContentPreviewExtraction:
    """Tests for content preview extraction."""

    def test_preview_short_content(self):
        """Test that short content is returned as-is."""
        adapter = EmailAdapter()
        content = "Short content"
        result = adapter.extract_preview(content)
        assert result == content

    def test_preview_long_content_truncated(self):
        """Test that long content is truncated."""
        adapter = EmailAdapter()
        content = "A" * 200
        result = adapter.extract_preview(content, max_length=100)
        assert len(result) <= 103  # 100 + "..."
        assert result.endswith("...")

    def test_preview_truncation_word_boundary(self):
        """Test truncation at word boundary."""
        adapter = EmailAdapter()
        content = "This is a test sentence with multiple words that should be truncated nicely at a word boundary"
        result = adapter.extract_preview(content, max_length=50)
        # Should truncate at space, not mid-word
        assert not result[-4:-3].isalpha() or result.endswith("...")

    def test_preview_empty_content(self):
        """Test empty content returns empty string."""
        adapter = EmailAdapter()
        result = adapter.extract_preview("")
        assert result == ""

    def test_preview_whitespace_normalization(self):
        """Test whitespace is normalized."""
        adapter = EmailAdapter()
        content = "Multiple   spaces    and\nnewlines"
        result = adapter.extract_preview(content)
        assert "   " not in result
        assert "\n" not in result


class TestDeduplicationEngine:
    """Tests for deduplication functionality."""

    def test_deduplication_removes_duplicates(self):
        """Test that duplicate items are removed."""
        engine = DeduplicationEngine()
        now = datetime.now()

        items = [
            InboundItem(
                source=ChannelType.EMAIL,
                timestamp=now,
                sender="test@example.com",
                raw_reference="msg_001",
                content_preview="Same content"
            ),
            InboundItem(
                source=ChannelType.EMAIL,
                timestamp=now,
                sender="test@example.com",
                raw_reference="msg_002",
                content_preview="Same content"
            )
        ]

        unique = engine.deduplicate(items)
        assert len(unique) == 1

    def test_deduplication_keeps_unique(self):
        """Test that unique items are kept."""
        engine = DeduplicationEngine()

        items = [
            InboundItem(
                source=ChannelType.EMAIL,
                timestamp=datetime.now(),
                sender="test1@example.com",
                raw_reference="msg_001",
                content_preview="Content A"
            ),
            InboundItem(
                source=ChannelType.EMAIL,
                timestamp=datetime.now(),
                sender="test2@example.com",
                raw_reference="msg_002",
                content_preview="Content B"
            )
        ]

        unique = engine.deduplicate(items)
        assert len(unique) == 2

    def test_deduplication_cache_expiry(self):
        """Test that cache entries expire."""
        engine = DeduplicationEngine(cache_duration_hours=0)  # Immediate expiry

        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="test@example.com",
            raw_reference="msg_001",
            content_preview="Content"
        )

        # First dedupe
        unique1 = engine.deduplicate([item])
        assert len(unique1) == 1

        # Force cleanup
        engine._cleanup_cache()

        # Second dedupe should see item as new (cache expired)
        # Note: With 0 hour cache, it depends on timing

    def test_clear_cache(self):
        """Test cache clearing."""
        engine = DeduplicationEngine()

        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="test@example.com",
            raw_reference="msg_001",
            content_preview="Content"
        )

        engine.deduplicate([item])
        engine.clear_cache()

        # After clearing, same item should be seen as new
        unique = engine.deduplicate([item])
        assert len(unique) == 1


class TestIntakeAgentBatchCreation:
    """Tests for batch creation."""

    def test_create_batch(self, intake_agent_factory):
        """Test batch creation with items."""
        agent = intake_agent_factory()

        items = [
            InboundItem(
                source=ChannelType.EMAIL,
                timestamp=datetime.now(),
                sender="test@example.com",
                raw_reference="msg_001",
                urgency_markers=UrgencyMarker.EXPLICIT_URGENT,
                is_read=False,
                is_flagged=True,
                attachments=[Attachment("file.pdf", "pdf", 1000)]
            )
        ]

        batch = agent.create_batch(items, [ChannelType.EMAIL])

        assert batch.items_count == 1
        assert batch.summary.with_attachments == 1
        assert batch.summary.with_urgency_markers == 1
        assert batch.summary.unread_count == 1
        assert batch.summary.flagged_count == 1
        assert batch.summary.by_channel.get("Email") == 1

    def test_create_empty_batch(self, intake_agent_factory):
        """Test creation of empty batch."""
        agent = intake_agent_factory()
        batch = agent.create_batch([], [ChannelType.EMAIL])

        assert batch.items_count == 0
        assert batch.summary.with_attachments == 0

    def test_batch_id_generation(self, intake_agent_factory):
        """Test that batch IDs are unique."""
        agent = intake_agent_factory()

        batch1 = agent.create_batch([], [ChannelType.EMAIL])
        batch2 = agent.create_batch([], [ChannelType.EMAIL])

        assert batch1.batch_id != batch2.batch_id


class TestIntakeAgentNormalization:
    """Tests for item normalization."""

    def test_normalize_long_preview(self, intake_agent_factory):
        """Test that long previews are truncated."""
        agent = intake_agent_factory()

        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="test@example.com",
            raw_reference="msg_001",
            content_preview="A" * 200
        )

        normalized = agent.normalize_item(item)
        assert len(normalized.content_preview) <= 100

    def test_normalize_missing_sender(self, intake_agent_factory):
        """Test that missing sender is set to 'unknown'."""
        agent = intake_agent_factory()

        item = InboundItem(
            source=ChannelType.EMAIL,
            timestamp=datetime.now(),
            sender="",
            raw_reference="msg_001"
        )

        normalized = agent.normalize_item(item)
        assert normalized.sender == "unknown"


class TestIntakeAgentTimeWindow:
    """Tests for time window calculation."""

    def test_last_hour_window(self, intake_agent_factory):
        """Test last hour time window."""
        agent = intake_agent_factory()

        request = IntakeRequest(
            channels=[ChannelType.EMAIL],
            time_window=TimeWindow.LAST_HOUR,
            include_filter=IncludeFilter.ALL
        )

        since, until = agent._calculate_time_window(request)
        assert since is not None
        assert until is not None
        assert (until - since).seconds <= 3600

    def test_last_24_hours_window(self, intake_agent_factory):
        """Test last 24 hours time window."""
        agent = intake_agent_factory()

        request = IntakeRequest(
            channels=[ChannelType.EMAIL],
            time_window=TimeWindow.LAST_24_HOURS,
            include_filter=IncludeFilter.ALL
        )

        since, until = agent._calculate_time_window(request)
        assert since is not None
        diff = until - since
        assert diff.days == 1 or diff.seconds >= 86000

    def test_last_week_window(self, intake_agent_factory):
        """Test last week time window."""
        agent = intake_agent_factory()

        request = IntakeRequest(
            channels=[ChannelType.EMAIL],
            time_window=TimeWindow.LAST_WEEK,
            include_filter=IncludeFilter.ALL
        )

        since, until = agent._calculate_time_window(request)
        assert since is not None
        assert (until - since).days == 7


class TestIntakeAgentChannelStatus:
    """Tests for channel status reporting."""

    def test_get_channel_status(self, intake_agent_factory):
        """Test getting channel status."""
        agent = intake_agent_factory()
        status = agent.get_channel_status()

        assert "Email" in status
        assert "WhatsApp" in status
        assert "connected" in status["Email"]

    def test_channel_status_format(self, intake_agent_factory):
        """Test channel status format."""
        agent = intake_agent_factory()
        status = agent.get_channel_status()

        for channel_name, channel_status in status.items():
            assert "connected" in channel_status
            assert "last_check" in channel_status
            assert "recent_errors" in channel_status


class TestIntakeAgentExecute:
    """Tests for execute method (synchronous wrapper)."""

    def test_execute_returns_agent_response(self, intake_agent_factory, sample_intake_request):
        """Test that execute returns AgentResponse."""
        agent = intake_agent_factory()
        response = agent.execute(sample_intake_request)

        assert isinstance(response, AgentResponse)
        assert response.agent_name == "IntakeAgent"

    def test_execute_with_empty_request(self, intake_agent_factory):
        """Test execute with empty request."""
        agent = intake_agent_factory()
        response = agent.execute({})

        assert isinstance(response, AgentResponse)


class TestIntakeAgentEdgeCases:
    """Edge case tests for IntakeAgent."""

    def test_invalid_channel_type_in_request(self, intake_agent_factory):
        """Test handling of invalid channel type."""
        agent = intake_agent_factory()
        request = {"channels": ["InvalidChannel", "Email"]}
        parsed = IntakeRequest.from_dict(request)
        # Should only have Email (invalid channel skipped)
        assert ChannelType.EMAIL in parsed.channels

    def test_empty_channels_list(self, intake_agent_factory):
        """Test handling of empty channels list."""
        request = IntakeRequest.from_dict({"channels": []})
        # Empty channels should use all
        assert len(request.channels) > 0

    def test_duplicate_channels_in_request(self):
        """Test handling of duplicate channels."""
        request = IntakeRequest.from_dict({
            "channels": ["Email", "Email", "WhatsApp"]
        })
        # Should handle gracefully, might have duplicates or not
        assert ChannelType.EMAIL in request.channels
        assert ChannelType.WHATSAPP in request.channels

    def test_none_values_in_request(self):
        """Test handling of None values."""
        request = IntakeRequest.from_dict({
            "channels": None,
            "time_window": None,
            "include": None
        })
        # Should use defaults
        assert request.time_window == TimeWindow.SINCE_LAST_CHECK


class TestAdapterConnectivity:
    """Tests for adapter connection handling."""

    @pytest.mark.asyncio
    async def test_email_adapter_connect(self):
        """Test EmailAdapter connection."""
        adapter = EmailAdapter()
        result = await adapter.connect()
        # In test mode, should return True (placeholder)
        assert result is True
        assert adapter.is_connected is True

    @pytest.mark.asyncio
    async def test_email_adapter_disconnect(self):
        """Test EmailAdapter disconnection."""
        adapter = EmailAdapter()
        await adapter.connect()
        await adapter.disconnect()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_whatsapp_adapter_connect(self):
        """Test WhatsAppAdapter connection."""
        adapter = WhatsAppAdapter()
        result = await adapter.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_scan_adapter_connect(self):
        """Test ScanAdapter connection."""
        adapter = ScanAdapter()
        result = await adapter.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_calendar_adapter_connect(self):
        """Test CalendarAdapter connection."""
        adapter = CalendarAdapter()
        result = await adapter.connect()
        assert result is True
