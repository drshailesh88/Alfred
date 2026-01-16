"""
Comprehensive tests for Discord Gateway Adapter.

Tests:
1. DiscordAdapter instantiation
2. Config loading and validation
3. Owner ID validation (single and multiple)
4. Message handling flow
5. send_message method signature
6. Error handling
"""
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from datetime import datetime


# ============================================================
# Test Results Tracking
# ============================================================
class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def record_pass(self, test_name):
        self.passed.append(test_name)
        print(f"  [PASS] {test_name}")

    def record_fail(self, test_name, reason):
        self.failed.append((test_name, reason))
        print(f"  [FAIL] {test_name}: {reason}")

    def record_error(self, test_name, error):
        self.errors.append((test_name, error))
        print(f"  [ERROR] {test_name}: {error}")

    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {len(self.passed)}/{total} passed")
        print("=" * 60)
        if self.failed:
            print("\nFailed tests:")
            for name, reason in self.failed:
                print(f"  - {name}: {reason}")
        if self.errors:
            print("\nErrored tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return len(self.failed) == 0 and len(self.errors) == 0


results = TestResults()


# ============================================================
# Mock Discord.py Library
# ============================================================
@dataclass
class MockUser:
    id: int
    name: str = "TestUser"
    discriminator: str = "0001"


@dataclass
class MockChannel:
    id: int

    async def send(self, content):
        return MagicMock(id=12345)

    async def trigger_typing(self):
        pass


class MockDMChannel(MockChannel):
    """Mock DM Channel."""
    pass


@dataclass
class MockMessage:
    author: MockUser
    channel: MockChannel
    content: str
    id: int = 1
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class MockIntents:
    @classmethod
    def default(cls):
        intents = MagicMock()
        intents.message_content = False
        intents.dm_messages = False
        intents.guilds = True
        return intents


class MockClient:
    def __init__(self, intents=None):
        self.intents = intents
        self.user = MockUser(id=999, name="TestBot", discriminator="0000")
        self._event_handlers = {}

    def event(self, func):
        """Decorator to register event handlers."""
        self._event_handlers[func.__name__] = func
        return func

    def get_channel(self, channel_id):
        return MockDMChannel(id=channel_id)

    async def fetch_channel(self, channel_id):
        return MockDMChannel(id=channel_id)

    async def start(self, token):
        pass

    async def close(self):
        pass


# Create mock discord module
mock_discord = MagicMock()
mock_discord.Intents = MockIntents
mock_discord.Client = MockClient
mock_discord.DMChannel = MockDMChannel
mock_discord.Message = MockMessage
mock_discord.LoginFailure = Exception
mock_discord.Forbidden = Exception
mock_discord.NotFound = Exception

# Patch discord module before importing adapter
sys.modules['discord'] = mock_discord


# ============================================================
# Now import the adapter (after mocking)
# ============================================================
sys.path.insert(0, '/home/user/Alfred/gateway')

# Import base directly to avoid loading all adapters via __init__.py
import importlib.util
import types

# Create the adapters package namespace
adapters_pkg = types.ModuleType('adapters')
adapters_pkg.__path__ = ['/home/user/Alfred/gateway/adapters']
sys.modules['adapters'] = adapters_pkg

# Load base adapter directly
base_spec = importlib.util.spec_from_file_location(
    "adapters.base", "/home/user/Alfred/gateway/adapters/base.py",
    submodule_search_locations=[]
)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules['adapters.base'] = base_module
base_spec.loader.exec_module(base_module)
BaseAdapter = base_module.BaseAdapter
IncomingMessage = base_module.IncomingMessage

# Load discord adapter directly
discord_spec = importlib.util.spec_from_file_location(
    "adapters.discord_adapter", "/home/user/Alfred/gateway/adapters/discord_adapter.py",
    submodule_search_locations=[]
)
discord_module = importlib.util.module_from_spec(discord_spec)
sys.modules['adapters.discord_adapter'] = discord_module
discord_spec.loader.exec_module(discord_module)
DiscordAdapter = discord_module.DiscordAdapter


# ============================================================
# Test Functions
# ============================================================

def test_1_import_and_class_structure():
    """Test 1: Import dependencies and class structure."""
    print("\n[Test 1] Import and Class Structure")

    try:
        # Check class exists
        assert DiscordAdapter is not None, "DiscordAdapter class not found"

        # Check inheritance - use module's own reference to avoid module loading artifacts
        base_classes = [c.__name__ for c in DiscordAdapter.__mro__]
        assert 'BaseAdapter' in base_classes, f"DiscordAdapter should inherit from BaseAdapter, got: {base_classes}"

        # Check required methods exist
        required_methods = ['start', 'stop', 'send_message', 'send_typing', 'platform_name']
        for method in required_methods:
            assert hasattr(DiscordAdapter, method), f"Missing method: {method}"

        results.record_pass("Import and class structure")
    except AssertionError as e:
        results.record_fail("Import and class structure", str(e))
    except Exception as e:
        results.record_error("Import and class structure", str(e))


def test_2_instantiation():
    """Test 2: DiscordAdapter can be instantiated."""
    print("\n[Test 2] Instantiation")

    try:
        config = {"bot_token": "test_token_123"}
        owner_id = "12345"
        callback = AsyncMock()

        adapter = DiscordAdapter(config, owner_id, callback)

        assert adapter is not None, "Adapter instantiation returned None"
        assert adapter.config == config, "Config not set correctly"
        assert adapter.owner_id == "12345", "Owner ID not set correctly"
        assert adapter.on_message == callback, "Callback not set correctly"

        results.record_pass("Instantiation")
    except AssertionError as e:
        results.record_fail("Instantiation", str(e))
    except Exception as e:
        results.record_error("Instantiation", str(e))


def test_3_platform_name():
    """Test 3: Platform name property returns 'discord'."""
    print("\n[Test 3] Platform Name")

    try:
        config = {"bot_token": "test_token_123"}
        adapter = DiscordAdapter(config, "12345", AsyncMock())

        assert adapter.platform_name == "discord", f"Expected 'discord', got '{adapter.platform_name}'"

        results.record_pass("Platform name")
    except AssertionError as e:
        results.record_fail("Platform name", str(e))
    except Exception as e:
        results.record_error("Platform name", str(e))


def test_4_config_loading_with_token():
    """Test 4: Config loading with valid token."""
    print("\n[Test 4] Config Loading (with token)")

    async def run_test():
        config = {"bot_token": "valid_test_token"}
        adapter = DiscordAdapter(config, "12345", AsyncMock())

        # Mock asyncio.create_task to prevent actual task creation
        with patch('asyncio.create_task') as mock_task:
            with patch('asyncio.sleep'):
                result = await adapter.start()

        # Should return True (no immediate failure with valid token)
        return result

    try:
        result = asyncio.get_event_loop().run_until_complete(run_test())
        # Note: start() returns True even if connection is pending
        assert result is True, f"Expected True, got {result}"
        results.record_pass("Config loading with token")
    except AssertionError as e:
        results.record_fail("Config loading with token", str(e))
    except Exception as e:
        results.record_error("Config loading with token", str(e))


def test_5_config_missing_token():
    """Test 5: Config validation - missing token should fail."""
    print("\n[Test 5] Config Missing Token")

    async def run_test():
        config = {}  # No token
        adapter = DiscordAdapter(config, "12345", AsyncMock())
        result = await adapter.start()
        return result

    try:
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result is False, f"Expected False for missing token, got {result}"
        results.record_pass("Config missing token")
    except AssertionError as e:
        results.record_fail("Config missing token", str(e))
    except Exception as e:
        results.record_error("Config missing token", str(e))


def test_6_config_empty_token():
    """Test 6: Config validation - empty token should fail."""
    print("\n[Test 6] Config Empty Token")

    async def run_test():
        config = {"bot_token": ""}  # Empty token
        adapter = DiscordAdapter(config, "12345", AsyncMock())
        result = await adapter.start()
        return result

    try:
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result is False, f"Expected False for empty token, got {result}"
        results.record_pass("Config empty token")
    except AssertionError as e:
        results.record_fail("Config empty token", str(e))
    except Exception as e:
        results.record_error("Config empty token", str(e))


def test_7_single_owner_id():
    """Test 7: Single owner ID validation."""
    print("\n[Test 7] Single Owner ID")

    try:
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "428068098", AsyncMock())

        # Check owner_ids list
        assert adapter.owner_ids == ["428068098"], f"Expected ['428068098'], got {adapter.owner_ids}"

        # Check is_owner method
        assert adapter.is_owner("428068098") is True, "is_owner should return True for owner"
        assert adapter.is_owner("999999999") is False, "is_owner should return False for non-owner"

        results.record_pass("Single owner ID")
    except AssertionError as e:
        results.record_fail("Single owner ID", str(e))
    except Exception as e:
        results.record_error("Single owner ID", str(e))


def test_8_multiple_owner_ids():
    """Test 8: Multiple owner IDs (comma-separated)."""
    print("\n[Test 8] Multiple Owner IDs")

    try:
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "428068098,1803061323,999888777", AsyncMock())

        # Check owner_ids list
        expected = ["428068098", "1803061323", "999888777"]
        assert adapter.owner_ids == expected, f"Expected {expected}, got {adapter.owner_ids}"

        # Check primary owner (backwards compatibility)
        assert adapter.owner_id == "428068098", f"Primary owner should be first ID"

        # Check is_owner for all
        assert adapter.is_owner("428068098") is True, "First owner should be valid"
        assert adapter.is_owner("1803061323") is True, "Second owner should be valid"
        assert adapter.is_owner("999888777") is True, "Third owner should be valid"
        assert adapter.is_owner("111111111") is False, "Non-owner should be invalid"

        results.record_pass("Multiple owner IDs")
    except AssertionError as e:
        results.record_fail("Multiple owner IDs", str(e))
    except Exception as e:
        results.record_error("Multiple owner IDs", str(e))


def test_9_owner_id_with_spaces():
    """Test 9: Owner IDs with spaces should be trimmed."""
    print("\n[Test 9] Owner IDs with Spaces")

    try:
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "  428068098 , 1803061323  ", AsyncMock())

        # Check spaces are trimmed
        assert adapter.owner_ids == ["428068098", "1803061323"], f"Spaces should be trimmed, got {adapter.owner_ids}"
        assert adapter.is_owner("428068098") is True, "Trimmed owner should be valid"

        results.record_pass("Owner IDs with spaces")
    except AssertionError as e:
        results.record_fail("Owner IDs with spaces", str(e))
    except Exception as e:
        results.record_error("Owner IDs with spaces", str(e))


def test_10_message_handling_owner():
    """Test 10: Message from owner should be processed."""
    print("\n[Test 10] Message Handling (Owner)")

    async def run_test():
        callback = AsyncMock()
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "428068098", callback)

        # Create mock incoming message
        incoming = IncomingMessage(
            platform="discord",
            user_id="428068098",
            chat_id="123456",
            text="Hello Alfred",
            timestamp=datetime.now().timestamp(),
            raw=MagicMock()
        )

        # Process the message
        await adapter.handle_message(incoming)

        # Callback should have been called
        return callback.called

    try:
        called = asyncio.get_event_loop().run_until_complete(run_test())
        assert called is True, "Callback should be called for owner message"
        results.record_pass("Message handling (owner)")
    except AssertionError as e:
        results.record_fail("Message handling (owner)", str(e))
    except Exception as e:
        results.record_error("Message handling (owner)", str(e))


def test_11_message_handling_non_owner():
    """Test 11: Message from non-owner should be ignored."""
    print("\n[Test 11] Message Handling (Non-Owner)")

    async def run_test():
        callback = AsyncMock()
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "428068098", callback)

        # Create mock incoming message from non-owner
        incoming = IncomingMessage(
            platform="discord",
            user_id="999999999",  # Not the owner
            chat_id="123456",
            text="Hello Alfred",
            timestamp=datetime.now().timestamp(),
            raw=MagicMock()
        )

        # Process the message
        await adapter.handle_message(incoming)

        # Callback should NOT have been called
        return callback.called

    try:
        called = asyncio.get_event_loop().run_until_complete(run_test())
        assert called is False, "Callback should NOT be called for non-owner message"
        results.record_pass("Message handling (non-owner)")
    except AssertionError as e:
        results.record_fail("Message handling (non-owner)", str(e))
    except Exception as e:
        results.record_error("Message handling (non-owner)", str(e))


def test_12_send_message_signature():
    """Test 12: send_message method has correct signature."""
    print("\n[Test 12] send_message Signature")

    try:
        import inspect
        sig = inspect.signature(DiscordAdapter.send_message)
        params = list(sig.parameters.keys())

        # Expected: self, chat_id, text, reply_to (optional)
        assert 'self' in params, "Missing 'self' parameter"
        assert 'chat_id' in params, "Missing 'chat_id' parameter"
        assert 'text' in params, "Missing 'text' parameter"
        assert 'reply_to' in params, "Missing 'reply_to' parameter"

        # Check reply_to has default value
        reply_to_param = sig.parameters['reply_to']
        assert reply_to_param.default is None, "reply_to should default to None"

        results.record_pass("send_message signature")
    except AssertionError as e:
        results.record_fail("send_message signature", str(e))
    except Exception as e:
        results.record_error("send_message signature", str(e))


def test_13_send_message_chunking():
    """Test 13: send_message should chunk long messages."""
    print("\n[Test 13] send_message Chunking")

    async def run_test():
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "12345", AsyncMock())
        adapter._running = True

        # Create mock client with channel tracking
        send_calls = []

        class TrackingChannel:
            id = 123456
            async def send(self, content):
                send_calls.append(content)
                return MagicMock()

        mock_client = MagicMock()
        mock_client.get_channel = MagicMock(return_value=TrackingChannel())
        adapter.client = mock_client

        # Create message longer than 2000 chars
        long_message = "A" * 4500  # Should be split into 3 chunks

        with patch('asyncio.sleep'):
            await adapter.send_message("123456", long_message)

        return send_calls

    try:
        chunks = asyncio.get_event_loop().run_until_complete(run_test())
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        assert len(chunks[0]) == 2000, f"First chunk should be 2000 chars"
        assert len(chunks[1]) == 2000, f"Second chunk should be 2000 chars"
        assert len(chunks[2]) == 500, f"Third chunk should be 500 chars"
        results.record_pass("send_message chunking")
    except AssertionError as e:
        results.record_fail("send_message chunking", str(e))
    except Exception as e:
        results.record_error("send_message chunking", str(e))


def test_14_send_message_not_running():
    """Test 14: send_message should not send when adapter not running."""
    print("\n[Test 14] send_message (Not Running)")

    async def run_test():
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "12345", AsyncMock())
        adapter._running = False  # Not running

        # This should not raise, just log a warning
        await adapter.send_message("123456", "Test message")

        return True

    try:
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result is True, "send_message should handle not-running state gracefully"
        results.record_pass("send_message (not running)")
    except AssertionError as e:
        results.record_fail("send_message (not running)", str(e))
    except Exception as e:
        results.record_error("send_message (not running)", str(e))


def test_15_handle_discord_message_dm():
    """Test 15: _handle_message correctly processes DM."""
    print("\n[Test 15] _handle_message (DM)")

    async def run_test():
        callback = AsyncMock()
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "428068098", callback)

        # Set up mock client
        adapter.client = MagicMock()
        adapter.client.user = MockUser(id=999, name="Bot", discriminator="0000")

        # Create mock Discord message (DM from owner)
        dm_channel = MockDMChannel(id=123456)
        message = MockMessage(
            author=MockUser(id=428068098, name="Owner", discriminator="1234"),
            channel=dm_channel,
            content="Hello Alfred"
        )

        # Process the message
        await adapter._handle_message(message)

        return callback.called

    try:
        called = asyncio.get_event_loop().run_until_complete(run_test())
        assert called is True, "Callback should be called for DM from owner"
        results.record_pass("_handle_message (DM)")
    except AssertionError as e:
        results.record_fail("_handle_message (DM)", str(e))
    except Exception as e:
        results.record_error("_handle_message (DM)", str(e))


def test_16_handle_discord_message_ignore_self():
    """Test 16: _handle_message ignores bot's own messages."""
    print("\n[Test 16] _handle_message (Ignore Self)")

    async def run_test():
        callback = AsyncMock()
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "428068098", callback)

        # Set up mock client
        bot_user = MockUser(id=999, name="Bot", discriminator="0000")
        adapter.client = MagicMock()
        adapter.client.user = bot_user

        # Create mock Discord message from bot itself
        dm_channel = MockDMChannel(id=123456)
        message = MockMessage(
            author=bot_user,  # Bot's own message
            channel=dm_channel,
            content="Bot response"
        )

        # Process the message
        await adapter._handle_message(message)

        return callback.called

    try:
        called = asyncio.get_event_loop().run_until_complete(run_test())
        assert called is False, "Callback should NOT be called for bot's own message"
        results.record_pass("_handle_message (ignore self)")
    except AssertionError as e:
        results.record_fail("_handle_message (ignore self)", str(e))
    except Exception as e:
        results.record_error("_handle_message (ignore self)", str(e))


def test_17_handle_discord_message_ignore_non_dm():
    """Test 17: _handle_message ignores non-DM messages."""
    print("\n[Test 17] _handle_message (Ignore Non-DM)")

    async def run_test():
        callback = AsyncMock()
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "428068098", callback)

        # Set up mock client
        adapter.client = MagicMock()
        adapter.client.user = MockUser(id=999, name="Bot", discriminator="0000")

        # Create mock Discord message in a guild channel (not DM)
        guild_channel = MockChannel(id=123456)  # Not MockDMChannel
        message = MockMessage(
            author=MockUser(id=428068098, name="Owner", discriminator="1234"),
            channel=guild_channel,  # Guild channel, not DM
            content="Hello Alfred"
        )

        # Process the message
        await adapter._handle_message(message)

        return callback.called

    try:
        called = asyncio.get_event_loop().run_until_complete(run_test())
        assert called is False, "Callback should NOT be called for non-DM message"
        results.record_pass("_handle_message (ignore non-DM)")
    except AssertionError as e:
        results.record_fail("_handle_message (ignore non-DM)", str(e))
    except Exception as e:
        results.record_error("_handle_message (ignore non-DM)", str(e))


def test_18_stop_gracefully():
    """Test 18: stop() handles graceful shutdown."""
    print("\n[Test 18] stop() Graceful Shutdown")

    async def run_test():
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "12345", AsyncMock())
        adapter._running = True
        adapter.client = MagicMock()
        adapter.client.close = AsyncMock()
        adapter._task = asyncio.create_task(asyncio.sleep(10))

        await adapter.stop()

        return adapter._running is False

    try:
        stopped = asyncio.get_event_loop().run_until_complete(run_test())
        assert stopped is True, "_running should be False after stop()"
        results.record_pass("stop() graceful shutdown")
    except AssertionError as e:
        results.record_fail("stop() graceful shutdown", str(e))
    except Exception as e:
        results.record_error("stop() graceful shutdown", str(e))


def test_19_empty_owner_id_blocks_all():
    """Test 19: Empty owner ID should block all messages."""
    print("\n[Test 19] Empty Owner ID (Security)")

    async def run_test():
        callback = AsyncMock()
        config = {"bot_token": "test"}
        # Empty owner ID simulates misconfigured system
        adapter = DiscordAdapter(config, "", callback)

        # Verify owner_ids is empty list
        assert adapter.owner_ids == [], f"Expected empty list, got {adapter.owner_ids}"
        assert adapter.owner_id == "", f"Expected empty string, got '{adapter.owner_id}'"

        # Verify no one is owner
        assert adapter.is_owner("428068098") is False, "Should not be owner"
        assert adapter.is_owner("") is False, "Empty string should not be owner"

        # Create mock incoming message
        incoming = IncomingMessage(
            platform="discord",
            user_id="428068098",
            chat_id="123456",
            text="Hello Alfred",
            timestamp=datetime.now().timestamp(),
            raw=MagicMock()
        )

        await adapter.handle_message(incoming)
        return callback.called

    try:
        called = asyncio.get_event_loop().run_until_complete(run_test())
        assert called is False, "Callback should NOT be called with empty owner_id"
        results.record_pass("Empty owner ID blocks all")
    except AssertionError as e:
        results.record_fail("Empty owner ID blocks all", str(e))
    except Exception as e:
        results.record_error("Empty owner ID blocks all", str(e))


def test_20_handle_empty_message():
    """Test 20: Empty message content should be ignored."""
    print("\n[Test 20] Empty Message Content")

    async def run_test():
        callback = AsyncMock()
        config = {"bot_token": "test"}
        adapter = DiscordAdapter(config, "428068098", callback)

        # Set up mock client
        adapter.client = MagicMock()
        adapter.client.user = MockUser(id=999, name="Bot", discriminator="0000")

        # Create mock Discord message with empty content
        dm_channel = MockDMChannel(id=123456)
        message = MockMessage(
            author=MockUser(id=428068098, name="Owner", discriminator="1234"),
            channel=dm_channel,
            content=""  # Empty content
        )

        await adapter._handle_message(message)
        return callback.called

    try:
        called = asyncio.get_event_loop().run_until_complete(run_test())
        assert called is False, "Callback should NOT be called for empty message"
        results.record_pass("Empty message content ignored")
    except AssertionError as e:
        results.record_fail("Empty message content ignored", str(e))
    except Exception as e:
        results.record_error("Empty message content ignored", str(e))


# ============================================================
# Run All Tests
# ============================================================
def run_all_tests():
    print("=" * 60)
    print("DISCORD ADAPTER TEST SUITE")
    print("=" * 60)

    test_1_import_and_class_structure()
    test_2_instantiation()
    test_3_platform_name()
    test_4_config_loading_with_token()
    test_5_config_missing_token()
    test_6_config_empty_token()
    test_7_single_owner_id()
    test_8_multiple_owner_ids()
    test_9_owner_id_with_spaces()
    test_10_message_handling_owner()
    test_11_message_handling_non_owner()
    test_12_send_message_signature()
    test_13_send_message_chunking()
    test_14_send_message_not_running()
    test_15_handle_discord_message_dm()
    test_16_handle_discord_message_ignore_self()
    test_17_handle_discord_message_ignore_non_dm()
    test_18_stop_gracefully()
    test_19_empty_owner_id_blocks_all()
    test_20_handle_empty_message()

    all_passed = results.summary()

    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL STATUS: PASS")
    else:
        print("OVERALL STATUS: FAIL")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
