#!/usr/bin/env python3
"""
Comprehensive tests for Signal adapter.
Tests config loading, class instantiation, phone number validation,
message handling, JSON-RPC communication, and error handling.

Ralph Loop: READ -> TEST -> FIX -> VERIFY
"""
import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime
from io import StringIO
import importlib.util

# Add parent directory to path for imports
GATEWAY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GATEWAY_DIR)

# Direct module loading to avoid importing all adapters via __init__.py
# This prevents issues with telegram/discord dependencies
def load_module_direct(name, filepath):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load base first (no dependencies)
base_module = load_module_direct(
    "adapters.base",
    os.path.join(GATEWAY_DIR, "adapters", "base.py")
)
BaseAdapter = base_module.BaseAdapter
IncomingMessage = base_module.IncomingMessage
OutgoingMessage = base_module.OutgoingMessage

# Load signal_adapter (depends on base only)
signal_module = load_module_direct(
    "adapters.signal_adapter",
    os.path.join(GATEWAY_DIR, "adapters", "signal_adapter.py")
)
SignalAdapter = signal_module.SignalAdapter

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "errors": []
}


def run_test(test_name, test_func):
    """Run a single test and track results."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    try:
        result = test_func()
        if result is True or result is None:
            print(f"[PASS] {test_name}")
            test_results["passed"].append(test_name)
            return True
        else:
            print(f"[FAIL] {test_name}: {result}")
            test_results["failed"].append((test_name, result))
            return False
    except Exception as e:
        import traceback
        print(f"[ERROR] {test_name}: {e}")
        traceback.print_exc()
        test_results["errors"].append((test_name, str(e)))
        return False


def run_async_test(test_name, coro_func):
    """Run an async test and track results."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    try:
        result = asyncio.get_event_loop().run_until_complete(coro_func())
        if result is True or result is None:
            print(f"[PASS] {test_name}")
            test_results["passed"].append(test_name)
            return True
        else:
            print(f"[FAIL] {test_name}: {result}")
            test_results["failed"].append((test_name, result))
            return False
    except Exception as e:
        import traceback
        print(f"[ERROR] {test_name}: {e}")
        traceback.print_exc()
        test_results["errors"].append((test_name, str(e)))
        return False


# ============================================================
# TEST 1: Import dependencies
# ============================================================
def test_imports():
    """Test that all required imports work."""
    try:
        import asyncio
        import json
        import os
        from datetime import datetime
        from typing import Optional
        print("  - asyncio: OK")
        print("  - json: OK")
        print("  - os: OK")
        print("  - datetime: OK")
        print("  - typing: OK")
        return True
    except ImportError as e:
        return f"Import failed: {e}"


# ============================================================
# TEST 2: Import adapter modules
# ============================================================
def test_adapter_imports():
    """Test that adapter modules can be imported."""
    try:
        # Using pre-loaded modules (avoids telegram/discord dependency issues)
        if BaseAdapter is None:
            return "BaseAdapter not loaded"
        print("  - BaseAdapter: OK")

        if IncomingMessage is None:
            return "IncomingMessage not loaded"
        print("  - IncomingMessage: OK")

        if OutgoingMessage is None:
            return "OutgoingMessage not loaded"
        print("  - OutgoingMessage: OK")

        if SignalAdapter is None:
            return "SignalAdapter not loaded"
        print("  - SignalAdapter: OK")
        return True
    except Exception as e:
        return f"Adapter import failed: {e}"


# ============================================================
# TEST 3: SignalAdapter class structure
# ============================================================
def test_class_structure():
    """Test SignalAdapter has correct class structure."""
    # Using pre-loaded modules
    # Check inheritance
    if not issubclass(SignalAdapter, BaseAdapter):
        return "SignalAdapter does not inherit from BaseAdapter"
    print("  - Inherits from BaseAdapter: OK")

    # Check required methods exist
    required_methods = ['start', 'stop', 'send_message', 'send_typing', 'platform_name']
    for method in required_methods:
        if not hasattr(SignalAdapter, method):
            return f"Missing method: {method}"
        print(f"  - Has {method}: OK")

    # Check internal methods
    internal_methods = ['_receive_loop', '_stderr_loop', '_handle_jsonrpc', '_send_jsonrpc']
    for method in internal_methods:
        if not hasattr(SignalAdapter, method):
            return f"Missing internal method: {method}"
        print(f"  - Has {method}: OK")

    return True


# ============================================================
# TEST 4: Config loading
# ============================================================
def test_config_loading():
    """Test that config values are properly loaded."""
    # Using pre-loaded SignalAdapter

    # Create mock callback
    mock_callback = AsyncMock()

    # Test config
    config = {
        "signal_cli_path": "/custom/path/signal-cli",
        "data_dir": "/custom/data",
        "phone_number": "+1234567890"
    }

    adapter = SignalAdapter(config, "+1234567890", mock_callback)

    # Verify config loaded correctly
    if adapter.signal_cli != "/custom/path/signal-cli":
        return f"signal_cli wrong: {adapter.signal_cli}"
    print(f"  - signal_cli_path loaded: {adapter.signal_cli}")

    if adapter.data_dir != "/custom/data":
        return f"data_dir wrong: {adapter.data_dir}"
    print(f"  - data_dir loaded: {adapter.data_dir}")

    if adapter.phone != "+1234567890":
        return f"phone wrong: {adapter.phone}"
    print(f"  - phone_number loaded: {adapter.phone}")

    return True


# ============================================================
# TEST 5: Config defaults
# ============================================================
def test_config_defaults():
    """Test that missing config values use correct defaults."""
    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()
    config = {}  # Empty config

    adapter = SignalAdapter(config, "", mock_callback)

    # Check defaults
    if adapter.signal_cli != "signal-cli":
        return f"Default signal_cli wrong: {adapter.signal_cli}"
    print(f"  - Default signal_cli: {adapter.signal_cli}")

    if adapter.data_dir != "/root/.signal-cli-alfred":
        return f"Default data_dir wrong: {adapter.data_dir}"
    print(f"  - Default data_dir: {adapter.data_dir}")

    if adapter.phone != "":
        return f"Default phone wrong: {adapter.phone}"
    print(f"  - Default phone (empty): OK")

    return True


# ============================================================
# TEST 6: Phone number format validation
# ============================================================
def test_phone_number_formats():
    """Test phone number format handling."""
    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()

    # Test various phone formats
    test_cases = [
        ("+918527086157", "+918527086157"),  # Indian number
        ("+1234567890", "+1234567890"),       # US format
        ("+442012345678", "+442012345678"),   # UK format
    ]

    for input_phone, expected in test_cases:
        config = {"phone_number": input_phone}
        adapter = SignalAdapter(config, input_phone, mock_callback)
        if adapter.phone != expected:
            return f"Phone format wrong for {input_phone}: got {adapter.phone}"
        print(f"  - Phone {input_phone}: OK")

    return True


# ============================================================
# TEST 7: Platform name property
# ============================================================
def test_platform_name():
    """Test platform_name property returns 'signal'."""
    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()
    config = {"phone_number": "+1234567890"}
    adapter = SignalAdapter(config, "+1234567890", mock_callback)

    if adapter.platform_name != "signal":
        return f"platform_name wrong: {adapter.platform_name}"
    print(f"  - platform_name: {adapter.platform_name}")

    return True


# ============================================================
# TEST 8: Owner ID handling
# ============================================================
def test_owner_id_handling():
    """Test owner ID validation (inherited from BaseAdapter)."""
    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()
    config = {"phone_number": "+1234567890"}

    # Test single owner
    adapter = SignalAdapter(config, "+918527086157", mock_callback)
    if not adapter.is_owner("+918527086157"):
        return "Single owner check failed"
    print("  - Single owner check: OK")

    # Test multiple owners (comma-separated)
    adapter = SignalAdapter(config, "+918527086157,+1234567890", mock_callback)
    if not adapter.is_owner("+918527086157"):
        return "Multiple owner check 1 failed"
    if not adapter.is_owner("+1234567890"):
        return "Multiple owner check 2 failed"
    print("  - Multiple owner check: OK")

    # Test non-owner rejection
    if adapter.is_owner("+9999999999"):
        return "Non-owner should be rejected"
    print("  - Non-owner rejection: OK")

    return True


# ============================================================
# TEST 9: send_message method signature
# ============================================================
def test_send_message_signature():
    """Test send_message has correct signature."""
    # Using pre-loaded SignalAdapter
    import inspect

    sig = inspect.signature(SignalAdapter.send_message)
    params = list(sig.parameters.keys())

    expected_params = ['self', 'chat_id', 'text', 'reply_to']
    if params != expected_params:
        return f"Wrong params: {params}, expected {expected_params}"
    print(f"  - Parameters: {params}")

    # Check reply_to has default None
    reply_to_param = sig.parameters['reply_to']
    if reply_to_param.default is not None:
        return f"reply_to default wrong: {reply_to_param.default}"
    print("  - reply_to default is None: OK")

    return True


# ============================================================
# TEST 10: IncomingMessage creation
# ============================================================
def test_incoming_message_creation():
    """Test IncomingMessage dataclass creation."""
    # Using pre-loaded IncomingMessage

    msg = IncomingMessage(
        platform="signal",
        user_id="+918527086157",
        chat_id="+918527086157",
        text="Hello Alfred",
        timestamp=1234567890.0,
        raw={"test": "data"},
        metadata={"timestamp": 1234567890000}
    )

    if msg.platform != "signal":
        return f"platform wrong: {msg.platform}"
    print(f"  - platform: {msg.platform}")

    if msg.user_id != "+918527086157":
        return f"user_id wrong: {msg.user_id}"
    print(f"  - user_id: {msg.user_id}")

    if msg.text != "Hello Alfred":
        return f"text wrong: {msg.text}"
    print(f"  - text: {msg.text}")

    return True


# ============================================================
# TEST 11: JSON-RPC request format
# ============================================================
async def test_jsonrpc_request_format():
    """Test JSON-RPC request is formatted correctly."""
    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()
    config = {"phone_number": "+1234567890"}
    adapter = SignalAdapter(config, "+1234567890", mock_callback)

    # Create a mock process with stdin/stdout
    mock_process = MagicMock()
    mock_stdin = MagicMock()
    mock_stdin.write = MagicMock()
    mock_stdin.drain = AsyncMock()
    mock_process.stdin = mock_stdin

    # Mock stdout to return empty (timeout behavior)
    mock_stdout = MagicMock()

    async def mock_readline():
        await asyncio.sleep(0.1)
        return b''

    mock_stdout.readline = mock_readline
    mock_process.stdout = mock_stdout

    adapter._process = mock_process
    adapter._running = True

    # Call _send_jsonrpc (it will timeout, but we can check the write call)
    try:
        await asyncio.wait_for(
            adapter._send_jsonrpc("send", {"recipient": ["+1111"], "message": "test"}),
            timeout=1.0
        )
    except asyncio.TimeoutError:
        pass

    # Check that write was called
    if mock_stdin.write.called:
        call_args = mock_stdin.write.call_args[0][0]
        request = json.loads(call_args.decode())

        if request.get("jsonrpc") != "2.0":
            return f"jsonrpc version wrong: {request.get('jsonrpc')}"
        print(f"  - jsonrpc version: {request.get('jsonrpc')}")

        if request.get("method") != "send":
            return f"method wrong: {request.get('method')}"
        print(f"  - method: {request.get('method')}")

        if "id" not in request:
            return "Missing id field"
        print(f"  - id present: OK")

        if request.get("params") != {"recipient": ["+1111"], "message": "test"}:
            return f"params wrong: {request.get('params')}"
        print(f"  - params: OK")

    return True


# ============================================================
# TEST 12: Handle incoming JSON-RPC message
# ============================================================
async def test_handle_jsonrpc_receive():
    """Test handling of incoming JSON-RPC receive notification."""
    # Using pre-loaded SignalAdapter

    received_messages = []

    async def mock_callback(msg):
        received_messages.append(msg)

    config = {"phone_number": "+1234567890"}
    adapter = SignalAdapter(config, "+918527086157", mock_callback)
    adapter._running = True

    # Simulate incoming message from signal-cli
    jsonrpc_data = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {
            "envelope": {
                "source": "+918527086157",
                "sourceNumber": "+918527086157",
                "timestamp": 1234567890000,
                "dataMessage": {
                    "message": "Hello from Signal",
                    "timestamp": 1234567890000
                }
            }
        }
    }

    await adapter._handle_jsonrpc(jsonrpc_data)

    # Check message was processed
    if len(received_messages) != 1:
        return f"Expected 1 message, got {len(received_messages)}"
    print(f"  - Message received: OK")

    msg = received_messages[0]
    if msg.platform != "signal":
        return f"platform wrong: {msg.platform}"
    print(f"  - platform: {msg.platform}")

    if msg.user_id != "+918527086157":
        return f"user_id wrong: {msg.user_id}"
    print(f"  - user_id: {msg.user_id}")

    if msg.text != "Hello from Signal":
        return f"text wrong: {msg.text}"
    print(f"  - text: {msg.text}")

    return True


# ============================================================
# TEST 13: Ignore non-owner messages
# ============================================================
async def test_ignore_non_owner():
    """Test that messages from non-owners are ignored."""
    # Using pre-loaded SignalAdapter

    received_messages = []

    async def mock_callback(msg):
        received_messages.append(msg)

    config = {"phone_number": "+1234567890"}
    # Owner is +918527086157, message from +9999999999 should be ignored
    adapter = SignalAdapter(config, "+918527086157", mock_callback)
    adapter._running = True

    # Simulate message from non-owner
    jsonrpc_data = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {
            "envelope": {
                "source": "+9999999999",
                "timestamp": 1234567890000,
                "dataMessage": {
                    "message": "Spam message",
                    "timestamp": 1234567890000
                }
            }
        }
    }

    await adapter._handle_jsonrpc(jsonrpc_data)

    # Message should NOT be in received_messages (ignored by handle_message)
    if len(received_messages) != 0:
        return f"Non-owner message should be ignored, got {len(received_messages)}"
    print("  - Non-owner message ignored: OK")

    return True


# ============================================================
# TEST 14: Message chunking for long messages
# ============================================================
async def test_message_chunking():
    """Test that long messages are chunked correctly."""
    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()
    config = {"phone_number": "+1234567890"}
    adapter = SignalAdapter(config, "+1234567890", mock_callback)

    # Track _send_jsonrpc calls
    send_calls = []

    async def mock_send_jsonrpc(method, params):
        send_calls.append((method, params))

    adapter._send_jsonrpc = mock_send_jsonrpc
    adapter._running = True

    # Create message longer than 2000 chars
    long_message = "A" * 4500

    await adapter.send_message("+1111111111", long_message)

    # Should be split into 3 chunks (2000 + 2000 + 500)
    if len(send_calls) != 3:
        return f"Expected 3 chunks, got {len(send_calls)}"
    print(f"  - Chunked into {len(send_calls)} parts: OK")

    # Check chunk sizes
    total_length = 0
    for i, (method, params) in enumerate(send_calls):
        chunk_len = len(params.get("message", ""))
        total_length += chunk_len
        print(f"  - Chunk {i+1} size: {chunk_len}")

    if total_length != 4500:
        return f"Total length wrong: {total_length}"
    print(f"  - Total reconstructed length: {total_length}")

    return True


# ============================================================
# TEST 15: Start without phone number fails
# ============================================================
async def test_start_without_phone():
    """Test that start fails without phone number configured."""
    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()
    config = {}  # No phone number
    adapter = SignalAdapter(config, "", mock_callback)

    result = await adapter.start()

    if result is not False:
        return f"start() should return False without phone, got {result}"
    print("  - Start without phone returns False: OK")

    return True


# ============================================================
# TEST 16: Timestamp normalization
# ============================================================
async def test_timestamp_normalization():
    """Test timestamp conversion from milliseconds to seconds."""
    # Using pre-loaded SignalAdapter

    received_messages = []

    async def mock_callback(msg):
        received_messages.append(msg)

    config = {"phone_number": "+1234567890"}
    adapter = SignalAdapter(config, "+918527086157", mock_callback)
    adapter._running = True

    # Test millisecond timestamp (common from Signal)
    jsonrpc_data = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {
            "envelope": {
                "source": "+918527086157",
                "timestamp": 1704067200000,  # Milliseconds
                "dataMessage": {
                    "message": "Test",
                    "timestamp": 1704067200000
                }
            }
        }
    }

    await adapter._handle_jsonrpc(jsonrpc_data)

    if len(received_messages) != 1:
        return f"Expected 1 message, got {len(received_messages)}"

    msg = received_messages[0]
    # Should be converted to seconds (around 1704067200)
    if msg.timestamp > 1e12:
        return f"Timestamp not normalized: {msg.timestamp}"
    print(f"  - Timestamp normalized: {msg.timestamp}")

    return True


# ============================================================
# TEST 17: Skip messages without text
# ============================================================
async def test_skip_empty_messages():
    """Test that messages without text are skipped."""
    # Using pre-loaded SignalAdapter

    received_messages = []

    async def mock_callback(msg):
        received_messages.append(msg)

    config = {"phone_number": "+1234567890"}
    adapter = SignalAdapter(config, "+918527086157", mock_callback)
    adapter._running = True

    # Message with empty text
    jsonrpc_data = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {
            "envelope": {
                "source": "+918527086157",
                "timestamp": 1234567890000,
                "dataMessage": {
                    "message": "",  # Empty
                    "timestamp": 1234567890000
                }
            }
        }
    }

    await adapter._handle_jsonrpc(jsonrpc_data)

    if len(received_messages) != 0:
        return f"Empty message should be skipped, got {len(received_messages)}"
    print("  - Empty message skipped: OK")

    return True


# ============================================================
# TEST 18: Skip non-data messages
# ============================================================
async def test_skip_non_data_messages():
    """Test that non-dataMessage envelopes are skipped."""
    # Using pre-loaded SignalAdapter

    received_messages = []

    async def mock_callback(msg):
        received_messages.append(msg)

    config = {"phone_number": "+1234567890"}
    adapter = SignalAdapter(config, "+918527086157", mock_callback)
    adapter._running = True

    # Receipt/typing indicator (no dataMessage)
    jsonrpc_data = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {
            "envelope": {
                "source": "+918527086157",
                "timestamp": 1234567890000,
                "receiptMessage": {
                    "type": "DELIVERY"
                }
            }
        }
    }

    await adapter._handle_jsonrpc(jsonrpc_data)

    if len(received_messages) != 0:
        return f"Receipt message should be skipped, got {len(received_messages)}"
    print("  - Non-data message skipped: OK")

    return True


# ============================================================
# TEST 19: Stop gracefully
# ============================================================
async def test_stop_gracefully():
    """Test that stop() terminates cleanly."""
    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()
    config = {"phone_number": "+1234567890"}
    adapter = SignalAdapter(config, "+1234567890", mock_callback)

    # Mock process
    mock_process = MagicMock()
    mock_process.terminate = MagicMock()
    mock_process.kill = MagicMock()

    async def mock_wait():
        return 0

    mock_process.wait = mock_wait

    adapter._process = mock_process
    adapter._running = True
    adapter._receive_task = None

    await adapter.stop()

    if adapter._running:
        return "_running should be False after stop"
    print("  - _running set to False: OK")

    if not mock_process.terminate.called:
        return "terminate() should be called"
    print("  - Process terminated: OK")

    return True


# ============================================================
# TEST 20: YAML config integration
# ============================================================
def test_yaml_config_integration():
    """Test loading config from YAML format matches adapter expectations."""
    import yaml

    # Sample config that would come from config.yaml
    yaml_config = """
signal:
  enabled: true
  data_dir: "/root/.signal-cli-alfred"
  phone_number: "+918527086157"
  signal_cli_path: "/usr/local/bin/signal-cli"
"""

    config = yaml.safe_load(yaml_config)
    signal_config = config.get("signal", {})

    # Using pre-loaded SignalAdapter

    mock_callback = AsyncMock()
    adapter = SignalAdapter(signal_config, signal_config.get("phone_number", ""), mock_callback)

    if adapter.phone != "+918527086157":
        return f"Phone from YAML wrong: {adapter.phone}"
    print(f"  - Phone from YAML: {adapter.phone}")

    if adapter.data_dir != "/root/.signal-cli-alfred":
        return f"data_dir from YAML wrong: {adapter.data_dir}"
    print(f"  - data_dir from YAML: {adapter.data_dir}")

    if adapter.signal_cli != "/usr/local/bin/signal-cli":
        return f"signal_cli from YAML wrong: {adapter.signal_cli}"
    print(f"  - signal_cli from YAML: {adapter.signal_cli}")

    return True


# ============================================================
# MAIN TEST RUNNER
# ============================================================
def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SIGNAL ADAPTER TEST SUITE")
    print("="*70)

    # Sync tests
    run_test("1. Import dependencies", test_imports)
    run_test("2. Import adapter modules", test_adapter_imports)
    run_test("3. Class structure", test_class_structure)
    run_test("4. Config loading", test_config_loading)
    run_test("5. Config defaults", test_config_defaults)
    run_test("6. Phone number formats", test_phone_number_formats)
    run_test("7. Platform name property", test_platform_name)
    run_test("8. Owner ID handling", test_owner_id_handling)
    run_test("9. send_message signature", test_send_message_signature)
    run_test("10. IncomingMessage creation", test_incoming_message_creation)
    run_test("20. YAML config integration", test_yaml_config_integration)

    # Async tests
    run_async_test("11. JSON-RPC request format", test_jsonrpc_request_format)
    run_async_test("12. Handle JSON-RPC receive", test_handle_jsonrpc_receive)
    run_async_test("13. Ignore non-owner messages", test_ignore_non_owner)
    run_async_test("14. Message chunking", test_message_chunking)
    run_async_test("15. Start without phone fails", test_start_without_phone)
    run_async_test("16. Timestamp normalization", test_timestamp_normalization)
    run_async_test("17. Skip empty messages", test_skip_empty_messages)
    run_async_test("18. Skip non-data messages", test_skip_non_data_messages)
    run_async_test("19. Stop gracefully", test_stop_gracefully)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {len(test_results['passed'])}")
    print(f"Failed: {len(test_results['failed'])}")
    print(f"Errors: {len(test_results['errors'])}")

    if test_results['failed']:
        print("\nFailed tests:")
        for name, reason in test_results['failed']:
            print(f"  - {name}: {reason}")

    if test_results['errors']:
        print("\nError tests:")
        for name, error in test_results['errors']:
            print(f"  - {name}: {error}")

    total = len(test_results['passed']) + len(test_results['failed']) + len(test_results['errors'])

    print("\n" + "="*70)
    if test_results['failed'] or test_results['errors']:
        print(f"OVERALL: FAIL ({len(test_results['passed'])}/{total} passed)")
        print("="*70)
        return 1
    else:
        print(f"OVERALL: PASS ({len(test_results['passed'])}/{total} passed)")
        print("="*70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
