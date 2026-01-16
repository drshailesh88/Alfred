"""
Test suite for TelegramAdapter.

This test validates:
1. TelegramAdapter class can be instantiated
2. Config loading works
3. Owner ID validation works (including multiple IDs)
4. Message handling flow is correct
5. send_message method signature is correct
6. Error handling is proper
"""
import asyncio
import sys
import os
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

# Add parent directory to path for imports
gateway_path = '/home/user/Alfred/gateway'
adapters_path = os.path.join(gateway_path, 'adapters')
sys.path.insert(0, gateway_path)

# Import directly from specific modules to avoid __init__.py importing all adapters
# This avoids requiring discord and signal dependencies for telegram tests
import importlib.util


def load_module_directly(module_name, file_path):
    """Load a module directly from file path without going through __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load base adapter directly
base_module = load_module_directly(
    'adapters.base',
    os.path.join(adapters_path, 'base.py')
)
BaseAdapter = base_module.BaseAdapter
IncomingMessage = base_module.IncomingMessage
OutgoingMessage = base_module.OutgoingMessage

# Load telegram adapter directly
telegram_module = load_module_directly(
    'adapters.telegram_adapter',
    os.path.join(adapters_path, 'telegram_adapter.py')
)
TelegramAdapter = telegram_module.TelegramAdapter


class TestTelegramAdapter(unittest.TestCase):
    """Test cases for TelegramAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "bot_token": "test_token_12345",
            "use_polling": True
        }
        self.single_owner_id = "428068098"
        self.multiple_owner_ids = "428068098,1803061323"
        self.on_message_callback = AsyncMock()

    def test_01_import_dependencies(self):
        """Test that all import dependencies work."""
        print("\n[TEST 01] Testing import dependencies...")
        try:
            # BaseAdapter already loaded above
            self.assertIsNotNone(BaseAdapter)
            self.assertIsNotNone(IncomingMessage)
            self.assertIsNotNone(OutgoingMessage)
            print("  [OK] BaseAdapter, IncomingMessage, OutgoingMessage imported")
        except Exception as e:
            self.fail(f"Failed to import base classes: {e}")

        try:
            # TelegramAdapter already loaded above
            self.assertIsNotNone(TelegramAdapter)
            print("  [OK] TelegramAdapter imported")
        except Exception as e:
            self.fail(f"Failed to import TelegramAdapter: {e}")

        try:
            from telegram import Update
            from telegram.ext import Application
            print("  [OK] python-telegram-bot dependencies imported")
        except ImportError as e:
            self.fail(f"Failed to import telegram dependencies: {e}")

        print("  [PASS] All imports successful")

    def test_02_class_instantiation(self):
        """Test TelegramAdapter can be instantiated."""
        print("\n[TEST 02] Testing class instantiation...")

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.config, self.test_config)
        print("  [OK] TelegramAdapter instantiated successfully")
        print("  [PASS] Class instantiation works")

    def test_03_platform_name(self):
        """Test platform_name property returns correct value."""
        print("\n[TEST 03] Testing platform_name property...")

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        self.assertEqual(adapter.platform_name, "telegram")
        print("  [OK] platform_name == 'telegram'")
        print("  [PASS] Platform name is correct")

    def test_04_single_owner_id(self):
        """Test single owner ID validation."""
        print("\n[TEST 04] Testing single owner ID validation...")

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        # Test owner recognition
        self.assertTrue(adapter.is_owner("428068098"))
        print("  [OK] Owner ID '428068098' recognized")

        # Test non-owner rejection
        self.assertFalse(adapter.is_owner("999999999"))
        print("  [OK] Non-owner ID '999999999' rejected")

        # Check owner_ids list
        self.assertEqual(adapter.owner_ids, ["428068098"])
        print("  [OK] owner_ids list is correct")

        print("  [PASS] Single owner ID works correctly")

    def test_05_multiple_owner_ids(self):
        """Test multiple owner IDs (comma-separated) validation."""
        print("\n[TEST 05] Testing multiple owner IDs (comma-separated)...")

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.multiple_owner_ids,
            on_message=self.on_message_callback
        )

        # Test first owner
        self.assertTrue(adapter.is_owner("428068098"))
        print("  [OK] First owner ID '428068098' recognized")

        # Test second owner
        self.assertTrue(adapter.is_owner("1803061323"))
        print("  [OK] Second owner ID '1803061323' recognized")

        # Test non-owner
        self.assertFalse(adapter.is_owner("999999999"))
        print("  [OK] Non-owner ID rejected")

        # Check owner_ids list
        self.assertEqual(adapter.owner_ids, ["428068098", "1803061323"])
        print("  [OK] owner_ids list contains both IDs")

        # Check backwards compatibility (owner_id should be first)
        self.assertEqual(adapter.owner_id, "428068098")
        print("  [OK] owner_id (primary) is first ID for backwards compatibility")

        print("  [PASS] Multiple owner IDs work correctly")

    def test_06_owner_id_with_spaces(self):
        """Test owner IDs with extra spaces are handled."""
        print("\n[TEST 06] Testing owner IDs with extra spaces...")

        # Test with spaces around IDs
        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=" 428068098 , 1803061323 ",
            on_message=self.on_message_callback
        )

        self.assertTrue(adapter.is_owner("428068098"))
        self.assertTrue(adapter.is_owner("1803061323"))
        self.assertEqual(adapter.owner_ids, ["428068098", "1803061323"])
        print("  [OK] Spaces trimmed correctly")
        print("  [PASS] Owner ID space handling works")

    def test_07_config_loading(self):
        """Test configuration is properly loaded."""
        print("\n[TEST 07] Testing config loading...")

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        self.assertEqual(adapter.config.get("bot_token"), "test_token_12345")
        print("  [OK] bot_token loaded from config")
        self.assertTrue(adapter.config.get("use_polling"))
        print("  [OK] use_polling loaded from config")
        print("  [PASS] Config loading works correctly")

    def test_08_send_message_signature(self):
        """Test send_message method has correct signature."""
        print("\n[TEST 08] Testing send_message method signature...")
        import inspect

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        # Check method exists
        self.assertTrue(hasattr(adapter, 'send_message'))
        print("  [OK] send_message method exists")

        # Check signature
        sig = inspect.signature(adapter.send_message)
        params = list(sig.parameters.keys())

        self.assertIn('chat_id', params)
        print("  [OK] 'chat_id' parameter present")
        self.assertIn('text', params)
        print("  [OK] 'text' parameter present")
        self.assertIn('reply_to', params)
        print("  [OK] 'reply_to' parameter present")

        # Check reply_to is optional
        reply_to_param = sig.parameters['reply_to']
        self.assertEqual(reply_to_param.default, None)
        print("  [OK] 'reply_to' has default value None")

        print("  [PASS] send_message signature is correct")

    def test_09_incoming_message_dataclass(self):
        """Test IncomingMessage dataclass structure."""
        print("\n[TEST 09] Testing IncomingMessage dataclass...")

        msg = IncomingMessage(
            platform="telegram",
            user_id="428068098",
            chat_id="428068098",
            text="Hello, Alfred!",
            timestamp=datetime.now().timestamp(),
            raw=None,
            metadata={"message_id": 12345, "username": "testuser"}
        )

        self.assertEqual(msg.platform, "telegram")
        print("  [OK] platform field works")
        self.assertEqual(msg.user_id, "428068098")
        print("  [OK] user_id field works")
        self.assertEqual(msg.chat_id, "428068098")
        print("  [OK] chat_id field works")
        self.assertEqual(msg.text, "Hello, Alfred!")
        print("  [OK] text field works")
        self.assertEqual(msg.metadata["message_id"], 12345)
        print("  [OK] metadata field works")

        print("  [PASS] IncomingMessage dataclass is correct")

    def test_10_message_handler_owner_check(self):
        """Test message handler properly checks owner."""
        print("\n[TEST 10] Testing message handler owner validation...")

        callback = AsyncMock()
        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=callback
        )

        # Create message from owner
        owner_msg = IncomingMessage(
            platform="telegram",
            user_id="428068098",
            chat_id="428068098",
            text="Test message",
            timestamp=datetime.now().timestamp(),
            raw=None
        )

        # Create message from non-owner
        non_owner_msg = IncomingMessage(
            platform="telegram",
            user_id="999999999",
            chat_id="999999999",
            text="Test message",
            timestamp=datetime.now().timestamp(),
            raw=None
        )

        # Test owner message - callback should be called
        asyncio.run(adapter.handle_message(owner_msg))
        callback.assert_called_once()
        print("  [OK] Owner message triggers callback")

        # Reset mock
        callback.reset_mock()

        # Test non-owner message - callback should NOT be called
        asyncio.run(adapter.handle_message(non_owner_msg))
        callback.assert_not_called()
        print("  [OK] Non-owner message does NOT trigger callback")

        print("  [PASS] Message handler owner check works")

    def test_11_start_method_no_token(self):
        """Test start method handles missing bot_token."""
        print("\n[TEST 11] Testing start method with missing token...")

        # Config without token
        config_no_token = {"use_polling": True}
        adapter = TelegramAdapter(
            config=config_no_token,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        # Start should return False when no token
        result = asyncio.run(adapter.start())
        self.assertFalse(result)
        print("  [OK] start() returns False when bot_token missing")
        print("  [PASS] Missing token handling works")

    def test_12_running_state(self):
        """Test adapter running state tracking."""
        print("\n[TEST 12] Testing running state tracking...")

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        # Should not be running initially
        self.assertFalse(adapter.is_running)
        print("  [OK] is_running is False initially")

        # _running is directly accessible
        self.assertFalse(adapter._running)
        print("  [OK] _running is False initially")

        print("  [PASS] Running state tracking works")

    def test_13_uptime_tracking(self):
        """Test uptime tracking."""
        print("\n[TEST 13] Testing uptime tracking...")

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        # Initially uptime should be 0
        self.assertEqual(adapter.uptime_seconds, 0)
        print("  [OK] uptime_seconds is 0 initially")

        # Simulate started state
        adapter._started_at = datetime.now()
        self.assertGreaterEqual(adapter.uptime_seconds, 0)
        print("  [OK] uptime_seconds works after start")

        print("  [PASS] Uptime tracking works")

    def test_14_abstract_methods_implemented(self):
        """Test all abstract methods from BaseAdapter are implemented."""
        print("\n[TEST 14] Testing abstract method implementations...")
        import inspect

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        # Check all abstract methods are implemented
        required_methods = ['start', 'stop', 'send_message', 'send_typing', 'platform_name']

        for method in required_methods:
            self.assertTrue(hasattr(adapter, method))
            attr = getattr(adapter, method)
            # platform_name is a property
            if method == 'platform_name':
                self.assertIsInstance(attr, str)
                print(f"  [OK] {method} property implemented")
            else:
                self.assertTrue(callable(attr) or inspect.iscoroutinefunction(attr))
                print(f"  [OK] {method} method implemented")

        print("  [PASS] All abstract methods implemented")

    def test_15_send_typing_signature(self):
        """Test send_typing method has correct signature."""
        print("\n[TEST 15] Testing send_typing method signature...")
        import inspect

        adapter = TelegramAdapter(
            config=self.test_config,
            owner_id=self.single_owner_id,
            on_message=self.on_message_callback
        )

        # Check method exists
        self.assertTrue(hasattr(adapter, 'send_typing'))
        print("  [OK] send_typing method exists")

        # Check signature
        sig = inspect.signature(adapter.send_typing)
        params = list(sig.parameters.keys())

        self.assertIn('chat_id', params)
        print("  [OK] 'chat_id' parameter present")

        print("  [PASS] send_typing signature is correct")


class TestTelegramAdapterIntegration(unittest.TestCase):
    """Integration-level tests with mocked Telegram library."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "bot_token": "test_token_12345",
            "use_polling": True
        }
        self.owner_id = "428068098"
        self.on_message_callback = AsyncMock()

    def test_start_creates_application(self):
        """Test start() creates Telegram Application."""
        print("\n[INTEGRATION TEST] Testing Application creation...")

        # Create mock application
        mock_app = MagicMock()
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.updater.start_polling = AsyncMock()
        mock_bot = MagicMock()
        mock_bot.get_me = AsyncMock(return_value=MagicMock(username="test_bot"))
        mock_app.bot = mock_bot

        # Mock the builder pattern
        mock_builder = MagicMock()
        mock_builder.token.return_value = mock_builder
        mock_builder.build.return_value = mock_app

        # Use patch as context manager with the telegram_module reference
        with patch.object(telegram_module, 'Application') as mock_app_class:
            mock_app_class.builder.return_value = mock_builder

            adapter = TelegramAdapter(
                config=self.test_config,
                owner_id=self.owner_id,
                on_message=self.on_message_callback
            )

            result = asyncio.run(adapter.start())

            self.assertTrue(result)
            mock_builder.token.assert_called_once_with("test_token_12345")
            print("  [OK] Application.builder().token() called with correct token")
            mock_app.initialize.assert_called_once()
            print("  [OK] app.initialize() called")
            mock_app.start.assert_called_once()
            print("  [OK] app.start() called")
            mock_app.updater.start_polling.assert_called_once()
            print("  [OK] app.updater.start_polling() called")

        print("  [PASS] Application creation works")


def run_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("TELEGRAM ADAPTER TEST SUITE")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests in order
    suite.addTests(loader.loadTestsFromTestCase(TestTelegramAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestTelegramAdapterIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback}")

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
