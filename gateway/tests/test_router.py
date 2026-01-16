#!/usr/bin/env python3
"""
Router Test Suite for Alfred Gateway.

Tests the AlfredRouter class to ensure:
- Proper initialization in direct vs api mode
- API URL is correctly configured
- Message processing flow works
- Response extraction is correct
- Error handling is proper

Run with: python -m pytest tests/test_router.py -v
Or directly: python tests/test_router.py
"""

import asyncio
import json
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from router import AlfredRouter


@dataclass
class MockIncomingMessage:
    """Mock message for testing."""
    text: str
    platform: str = "test"
    chat_id: str = "test_chat_123"
    user_id: str = "test_user"
    timestamp: float = 0.0


class MockResponse:
    """Mock aiohttp response."""
    def __init__(self, status, json_data):
        self.status = status
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockClientSession:
    """Mock aiohttp ClientSession."""
    def __init__(self, response_status=200, response_data=None, capture_dict=None):
        self.response_status = response_status
        self.response_data = response_data or {}
        self.capture_dict = capture_dict

    def post(self, url, json, timeout):
        if self.capture_dict is not None:
            self.capture_dict['url'] = url
            self.capture_dict['body'] = json
            self.capture_dict['timeout'] = timeout.total if hasattr(timeout, 'total') else timeout
        return MockResponse(self.response_status, self.response_data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestRouterInitialization(unittest.TestCase):
    """Test router initialization modes."""

    def test_init_echo_mode_default(self):
        """Test router defaults to echo mode when no config."""
        config = {}
        router = AlfredRouter(config)
        self.assertEqual(router.mode, "echo")
        self.assertFalse(router.is_connected)

    def test_init_api_mode_sets_url(self):
        """Test API mode correctly sets api_url."""
        config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://alfred-ai:50001/api_message"
            }
        }
        router = AlfredRouter(config)

        self.assertEqual(router.mode, "api")
        self.assertTrue(router.is_connected)
        self.assertEqual(router.api_url, "http://alfred-ai:50001/api_message")

    def test_init_api_mode_default_url(self):
        """Test API mode uses correct default URL when not specified."""
        config = {
            "agent_zero": {
                "mode": "api"
            }
        }
        router = AlfredRouter(config)

        self.assertEqual(router.mode, "api")
        # CRITICAL CHECK: Default URL should be correct
        # The default should point to /api_message endpoint
        expected_url = "http://localhost:50001/api_message"
        self.assertEqual(
            router.api_url,
            expected_url,
            f"Default API URL is wrong! Got '{router.api_url}', expected '{expected_url}'"
        )

    def test_init_unknown_mode_falls_back_to_echo(self):
        """Test unknown mode falls back to echo."""
        config = {
            "agent_zero": {
                "mode": "unknown_mode"
            }
        }
        router = AlfredRouter(config)
        self.assertEqual(router.mode, "echo")
        self.assertFalse(router.is_connected)

    def test_init_direct_mode_missing_path_falls_back(self):
        """Test direct mode falls back to echo when path doesn't exist."""
        config = {
            "agent_zero": {
                "mode": "direct",
                "agent_zero_path": "/nonexistent/path"
            }
        }
        router = AlfredRouter(config)
        self.assertEqual(router.mode, "echo")


class TestApiMessageFormat(unittest.TestCase):
    """Test API message format and processing."""

    def setUp(self):
        """Set up test router in API mode."""
        self.config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://alfred-ai:50001/api_message"
            }
        }
        self.router = AlfredRouter(self.config)

    def test_api_message_format_correct(self):
        """Test that message sent to API has correct format."""
        captured = {}

        mock_session = MockClientSession(
            response_status=200,
            response_data={"context_id": "ctx_123", "response": "Test response"},
            capture_dict=captured
        )

        async def run_test():
            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await self.router._process_api("Hello Alfred")
                return result

        result = asyncio.run(run_test())

        # Verify message format
        self.assertEqual(captured['url'], "http://alfred-ai:50001/api_message")
        self.assertEqual(captured['body'], {"message": "Hello Alfred"})
        self.assertEqual(result, "Test response")

    def test_api_response_extraction(self):
        """Test response is correctly extracted from API response."""
        mock_session = MockClientSession(
            response_status=200,
            response_data={
                "context_id": "ctx_456",
                "response": "Greetings, sir. Alfred at your service."
            }
        )

        async def run_test():
            with patch('aiohttp.ClientSession', return_value=mock_session):
                return await self.router._process_api("Test message")

        result = asyncio.run(run_test())
        self.assertEqual(result, "Greetings, sir. Alfred at your service.")

    def test_api_missing_response_field(self):
        """Test handling when response field is missing."""
        mock_session = MockClientSession(
            response_status=200,
            response_data={"context_id": "ctx_789"}  # No "response" field!
        )

        async def run_test():
            with patch('aiohttp.ClientSession', return_value=mock_session):
                return await self.router._process_api("Test")

        result = asyncio.run(run_test())
        self.assertEqual(result, "No response received")


class TestApiErrorHandling(unittest.TestCase):
    """Test API error handling."""

    def setUp(self):
        """Set up test router in API mode."""
        self.config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://alfred-ai:50001/api_message"
            }
        }
        self.router = AlfredRouter(self.config)

    def test_api_error_status(self):
        """Test handling of non-200 status codes."""
        mock_session = MockClientSession(response_status=500, response_data={})

        async def run_test():
            with patch('aiohttp.ClientSession', return_value=mock_session):
                return await self.router._process_api("Test")

        result = asyncio.run(run_test())
        self.assertEqual(result, "API error: 500")

    def test_api_timeout_setting(self):
        """Test that timeout is set to 300 seconds."""
        captured = {}
        mock_session = MockClientSession(
            response_status=200,
            response_data={"response": "ok"},
            capture_dict=captured
        )

        async def run_test():
            with patch('aiohttp.ClientSession', return_value=mock_session):
                await self.router._process_api("Test")

        asyncio.run(run_test())
        self.assertEqual(captured['timeout'], 300, "Timeout should be 300 seconds")


class TestProcessMessage(unittest.TestCase):
    """Test message processing flow."""

    def setUp(self):
        """Set up test router."""
        self.config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://alfred-ai:50001/api_message"
            }
        }
        self.router = AlfredRouter(self.config)

    def test_process_message_calls_api(self):
        """Test process_message routes to API correctly."""
        mock_session = MockClientSession(
            response_status=200,
            response_data={"context_id": "ctx", "response": "Alfred response"}
        )

        async def run_test():
            message = MockIncomingMessage(text="Hello")
            send_reply = AsyncMock()
            send_typing = AsyncMock()

            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await self.router.process_message(message, send_reply, send_typing)
                return result, send_typing

        result, send_typing = asyncio.run(run_test())

        self.assertEqual(result, "Alfred response")
        send_typing.assert_called_once_with("test_chat_123")

    def test_process_message_echo_mode(self):
        """Test echo mode response."""
        config = {}
        router = AlfredRouter(config)
        self.assertEqual(router.mode, "echo")

        async def run_test():
            message = MockIncomingMessage(text="Hello from echo test")
            send_reply = AsyncMock()
            send_typing = AsyncMock()

            result = await router.process_message(message, send_reply, send_typing)
            return result

        result = asyncio.run(run_test())

        self.assertIn("[Echo Mode", result)
        self.assertIn("Hello from echo test", result)


class TestIntegration(unittest.TestCase):
    """Integration tests for router with real config patterns."""

    def test_docker_config_pattern(self):
        """Test typical Docker configuration."""
        # This is the config pattern used in docker-compose
        config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://alfred-ai:50001/api_message"
            }
        }

        router = AlfredRouter(config)

        self.assertEqual(router.mode, "api")
        self.assertEqual(router.api_url, "http://alfred-ai:50001/api_message")
        self.assertTrue(router.is_connected)

    def test_local_dev_config_pattern(self):
        """Test typical local development configuration."""
        config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://localhost:50001/api_message"
            }
        }

        router = AlfredRouter(config)

        self.assertEqual(router.mode, "api")
        self.assertEqual(router.api_url, "http://localhost:50001/api_message")


class TestApiEndpointContract(unittest.TestCase):
    """Test the contract between Gateway and API endpoint."""

    def test_api_message_endpoint_format(self):
        """Verify the API endpoint URL format is /api_message."""
        # The api_message.py handler expects requests at /api_message
        # Not /api/chat or any other path
        config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://alfred-ai:50001/api_message"
            }
        }
        router = AlfredRouter(config)

        # URL should end with /api_message
        self.assertTrue(
            router.api_url.endswith("/api_message"),
            f"API URL should end with /api_message, got: {router.api_url}"
        )

    def test_request_body_format(self):
        """Verify request body matches what api_message.py expects."""
        # api_message.py expects: {"message": "...", "context_id": "...", ...}
        # Router sends: {"message": "..."}
        captured = {}
        mock_session = MockClientSession(
            response_status=200,
            response_data={"context_id": "ctx", "response": "ok"},
            capture_dict=captured
        )

        config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://test:50001/api_message"
            }
        }
        router = AlfredRouter(config)

        async def run_test():
            with patch('aiohttp.ClientSession', return_value=mock_session):
                await router._process_api("Test message")

        asyncio.run(run_test())

        # Verify request has 'message' key
        self.assertIn("message", captured['body'])
        self.assertEqual(captured['body']['message'], "Test message")

    def test_response_format(self):
        """Verify router correctly handles api_message.py response format."""
        # api_message.py returns: {"context_id": "...", "response": "..."}
        mock_session = MockClientSession(
            response_status=200,
            response_data={
                "context_id": "ctx_test",
                "response": "Alfred's response here"
            }
        )

        config = {
            "agent_zero": {
                "mode": "api",
                "api_url": "http://test:50001/api_message"
            }
        }
        router = AlfredRouter(config)

        async def run_test():
            with patch('aiohttp.ClientSession', return_value=mock_session):
                return await router._process_api("Test")

        result = asyncio.run(run_test())
        self.assertEqual(result, "Alfred's response here")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("ALFRED GATEWAY - ROUTER TEST SUITE")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRouterInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestApiMessageFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestApiErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessMessage))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestApiEndpointContract))

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("RESULT: ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("RESULT: SOME TESTS FAILED")
        print()
        if result.failures:
            print("FAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("ERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
