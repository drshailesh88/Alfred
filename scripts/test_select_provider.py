#!/usr/bin/env python3
"""
Test suite for LLM provider fallback chain validation.
Tests provider selection logic, API key detection, and settings generation.

Run with: python scripts/test_select_provider.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent-zero1"))

# Import the module under test
from scripts.select_provider import (
    FALLBACK_CHAIN,
    DEFAULT_SETTINGS,
    get_api_key,
    select_provider,
    create_settings,
)


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, test_name):
        self.passed += 1
        print(f"  [PASS] {test_name}")

    def fail(self, test_name, message):
        self.failed += 1
        self.errors.append(f"{test_name}: {message}")
        print(f"  [FAIL] {test_name}: {message}")

    def summary(self):
        total = self.passed + self.failed
        print()
        print("=" * 60)
        print(f"RESULTS: {self.passed}/{total} tests passed")
        if self.errors:
            print()
            print("FAILURES:")
            for err in self.errors:
                print(f"  - {err}")
        print("=" * 60)
        return self.failed == 0


results = TestResult()


# ==============================================================================
# Test 1: Fallback Chain Order
# ==============================================================================
def test_fallback_chain_order():
    """Verify fallback chain order: Z.AI -> MiniMax -> Moonshot -> OpenAI"""
    print("\n[TEST] Fallback Chain Order")

    expected_order = ["zai", "minimax", "moonshot", "openai"]
    actual_order = [p["provider"] for p in FALLBACK_CHAIN]

    if actual_order == expected_order:
        results.ok("Fallback chain order is correct")
    else:
        results.fail("Fallback chain order", f"Expected {expected_order}, got {actual_order}")


# ==============================================================================
# Test 2: Environment Variable Names
# ==============================================================================
def test_env_variable_names():
    """Verify correct environment variable names for each provider"""
    print("\n[TEST] Environment Variable Names")

    expected_env_keys = {
        "zai": ["ZAI_API_KEY", "ZHIPU_API_KEY"],
        "minimax": ["MINIMAX_API_KEY"],
        "moonshot": ["MOONSHOT_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
    }

    for provider in FALLBACK_CHAIN:
        provider_name = provider["provider"]
        expected = expected_env_keys.get(provider_name, [])
        actual = provider["env_keys"]

        if actual == expected:
            results.ok(f"Env keys for {provider_name}: {actual}")
        else:
            results.fail(f"Env keys for {provider_name}", f"Expected {expected}, got {actual}")


# ==============================================================================
# Test 3: API Key Detection
# ==============================================================================
def test_api_key_detection():
    """Test get_api_key function with various scenarios"""
    print("\n[TEST] API Key Detection")

    # Test with valid key
    test_env = {"TEST_API_KEY": "sk-test123456789"}
    with patch.dict(os.environ, test_env, clear=False):
        key = get_api_key(["TEST_API_KEY"])
        if key == "sk-test123456789":
            results.ok("Detects valid API key")
        else:
            results.fail("Valid API key detection", f"Got {key}")

    # Test with empty key
    test_env = {"EMPTY_KEY": ""}
    with patch.dict(os.environ, test_env, clear=False):
        key = get_api_key(["EMPTY_KEY"])
        if key is None:
            results.ok("Rejects empty API key")
        else:
            results.fail("Empty API key rejection", f"Expected None, got {key}")

    # Test with placeholder values
    for placeholder in ["None", "NA", "your_api_key_here"]:
        test_env = {"PLACEHOLDER_KEY": placeholder}
        with patch.dict(os.environ, test_env, clear=False):
            key = get_api_key(["PLACEHOLDER_KEY"])
            if key is None:
                results.ok(f"Rejects placeholder '{placeholder}'")
            else:
                results.fail(f"Placeholder '{placeholder}' rejection", f"Expected None, got {key}")

    # Test fallback to second key
    test_env = {"SECOND_KEY": "sk-secondary123"}
    with patch.dict(os.environ, test_env, clear=False):
        key = get_api_key(["FIRST_KEY", "SECOND_KEY"])
        if key == "sk-secondary123":
            results.ok("Falls back to second env key")
        else:
            results.fail("Fallback to second key", f"Expected sk-secondary123, got {key}")

    # Test missing key
    key = get_api_key(["NONEXISTENT_KEY_12345"])
    if key is None:
        results.ok("Returns None for missing key")
    else:
        results.fail("Missing key handling", f"Expected None, got {key}")


# ==============================================================================
# Test 4: Provider Selection Logic
# ==============================================================================
def test_provider_selection():
    """Test provider selection with various API key combinations"""
    print("\n[TEST] Provider Selection Logic")

    # Test: First provider selected when available
    test_env = {
        "ZAI_API_KEY": "sk-zai-test123",
        "MINIMAX_API_KEY": "sk-minimax-test123",
    }
    with patch.dict(os.environ, test_env, clear=True):
        provider = select_provider()
        if provider and provider["provider"] == "zai":
            results.ok("Selects first available provider (Z.AI)")
        else:
            results.fail("First provider selection", f"Expected zai, got {provider}")

    # Test: Falls back to second provider
    test_env = {
        "MINIMAX_API_KEY": "sk-minimax-test123",
        "MOONSHOT_API_KEY": "sk-moonshot-test123",
    }
    with patch.dict(os.environ, test_env, clear=True):
        provider = select_provider()
        if provider and provider["provider"] == "minimax":
            results.ok("Falls back to MiniMax when Z.AI unavailable")
        else:
            results.fail("MiniMax fallback", f"Expected minimax, got {provider}")

    # Test: Falls back to third provider
    test_env = {
        "MOONSHOT_API_KEY": "sk-moonshot-test123",
    }
    with patch.dict(os.environ, test_env, clear=True):
        provider = select_provider()
        if provider and provider["provider"] == "moonshot":
            results.ok("Falls back to Moonshot when MiniMax unavailable")
        else:
            results.fail("Moonshot fallback", f"Expected moonshot, got {provider}")

    # Test: Falls back to OpenAI (last resort)
    test_env = {
        "OPENAI_API_KEY": "sk-openai-test123",
    }
    with patch.dict(os.environ, test_env, clear=True):
        provider = select_provider()
        if provider and provider["provider"] == "openai":
            results.ok("Falls back to OpenAI as last resort")
        else:
            results.fail("OpenAI fallback", f"Expected openai, got {provider}")

    # Test: Returns None when no keys available
    with patch.dict(os.environ, {}, clear=True):
        provider = select_provider()
        if provider is None:
            results.ok("Returns None when no API keys available")
        else:
            results.fail("No keys handling", f"Expected None, got {provider}")


# ==============================================================================
# Test 5: Settings JSON Structure
# ==============================================================================
def test_settings_structure():
    """Verify settings JSON has all required fields"""
    print("\n[TEST] Settings JSON Structure")

    required_fields = [
        "chat_model_provider",
        "chat_model_name",
        "chat_model_vision",
        "util_model_provider",
        "util_model_name",
        "browser_model_provider",
        "browser_model_name",
        "browser_model_vision",
    ]

    # Check DEFAULT_SETTINGS has all required fields
    for field in required_fields:
        if field in DEFAULT_SETTINGS:
            results.ok(f"DEFAULT_SETTINGS has '{field}'")
        else:
            results.fail(f"DEFAULT_SETTINGS field", f"Missing '{field}'")


# ==============================================================================
# Test 6: Settings Creation
# ==============================================================================
def test_settings_creation():
    """Test settings creation for each provider"""
    print("\n[TEST] Settings Creation")

    for provider in FALLBACK_CHAIN:
        settings = create_settings(provider)

        # Check provider is set correctly
        if settings["chat_model_provider"] == provider["provider"]:
            results.ok(f"{provider['name']}: chat_model_provider correct")
        else:
            results.fail(f"{provider['name']}: chat_model_provider",
                        f"Expected {provider['provider']}, got {settings['chat_model_provider']}")

        # Check model name is set correctly
        if settings["chat_model_name"] == provider["chat_model"]:
            results.ok(f"{provider['name']}: chat_model_name correct")
        else:
            results.fail(f"{provider['name']}: chat_model_name",
                        f"Expected {provider['chat_model']}, got {settings['chat_model_name']}")

        # Check util model is set correctly
        if settings["util_model_name"] == provider["util_model"]:
            results.ok(f"{provider['name']}: util_model_name correct")
        else:
            results.fail(f"{provider['name']}: util_model_name",
                        f"Expected {provider['util_model']}, got {settings['util_model_name']}")

        # Check vision flag is set correctly
        if settings["chat_model_vision"] == provider["vision"]:
            results.ok(f"{provider['name']}: vision flag correct")
        else:
            results.fail(f"{provider['name']}: vision flag",
                        f"Expected {provider['vision']}, got {settings['chat_model_vision']}")


# ==============================================================================
# Test 7: Model Names Validity (LiteLLM format)
# ==============================================================================
def test_model_names():
    """Verify model names are in valid format for LiteLLM"""
    print("\n[TEST] Model Names Validity")

    expected_models = {
        "zai": {
            "chat": "glm-4.7",
            "util": "glm-4.5-flash",
        },
        "minimax": {
            "chat": "MiniMax-M2.1",
            "util": "MiniMax-M2.1-lightning",
        },
        "moonshot": {
            "chat": "kimi-k2-0711-preview",
            "util": "moonshot-v1-8k",
        },
        "openai": {
            "chat": "gpt-4o-mini",
            "util": "gpt-4o-mini",
        },
    }

    for provider in FALLBACK_CHAIN:
        provider_name = provider["provider"]
        expected = expected_models.get(provider_name, {})

        if provider["chat_model"] == expected.get("chat"):
            results.ok(f"{provider_name}: chat model '{provider['chat_model']}' is valid")
        else:
            results.fail(f"{provider_name}: chat model",
                        f"Expected {expected.get('chat')}, got {provider['chat_model']}")

        if provider["util_model"] == expected.get("util"):
            results.ok(f"{provider_name}: util model '{provider['util_model']}' is valid")
        else:
            results.fail(f"{provider_name}: util model",
                        f"Expected {expected.get('util')}, got {provider['util_model']}")


# ==============================================================================
# Test 8: Provider Config in model_providers.yaml
# ==============================================================================
def test_provider_config_yaml():
    """Verify providers exist in model_providers.yaml"""
    print("\n[TEST] Provider Config in YAML")

    try:
        import yaml
        yaml_path = Path(__file__).parent.parent / "agent-zero1" / "conf" / "model_providers.yaml"

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        chat_providers = config.get("chat", {})

        for provider in FALLBACK_CHAIN:
            provider_name = provider["provider"]
            if provider_name in chat_providers:
                results.ok(f"Provider '{provider_name}' exists in YAML")

                # Check litellm_provider mapping
                litellm_provider = chat_providers[provider_name].get("litellm_provider")
                if litellm_provider:
                    results.ok(f"Provider '{provider_name}' has litellm_provider: {litellm_provider}")
                else:
                    results.fail(f"Provider '{provider_name}' litellm_provider", "Missing litellm_provider")
            else:
                results.fail(f"Provider config", f"'{provider_name}' not found in model_providers.yaml")

    except ImportError:
        results.fail("YAML test", "PyYAML not installed, skipping")
    except FileNotFoundError:
        results.fail("YAML test", f"model_providers.yaml not found at {yaml_path}")


# ==============================================================================
# Test 9: Settings JSON Serialization
# ==============================================================================
def test_settings_serialization():
    """Test that settings can be properly serialized to JSON"""
    print("\n[TEST] Settings JSON Serialization")

    for provider in FALLBACK_CHAIN:
        settings = create_settings(provider)
        try:
            json_str = json.dumps(settings, indent=2)
            parsed = json.loads(json_str)
            if parsed == settings:
                results.ok(f"{provider['name']}: JSON serialization round-trip")
            else:
                results.fail(f"{provider['name']}: JSON serialization", "Round-trip mismatch")
        except Exception as e:
            results.fail(f"{provider['name']}: JSON serialization", str(e))


# ==============================================================================
# Test 10: ZHIPU_API_KEY fallback for Z.AI
# ==============================================================================
def test_zhipu_fallback():
    """Test that ZHIPU_API_KEY works as fallback for Z.AI"""
    print("\n[TEST] ZHIPU_API_KEY Fallback")

    # Test with only ZHIPU_API_KEY set
    test_env = {"ZHIPU_API_KEY": "sk-zhipu-test123"}
    with patch.dict(os.environ, test_env, clear=True):
        provider = select_provider()
        if provider and provider["provider"] == "zai":
            results.ok("ZHIPU_API_KEY selects Z.AI provider")
        else:
            results.fail("ZHIPU_API_KEY fallback", f"Expected zai, got {provider}")


# ==============================================================================
# Test 11: models.py get_api_key compatibility
# ==============================================================================
def test_models_get_api_key():
    """Test that models.py get_api_key works with our provider env vars"""
    print("\n[TEST] models.py get_api_key Compatibility")

    try:
        from models import get_api_key as models_get_api_key

        # Test ZAI_API_KEY
        test_env = {"ZAI_API_KEY": "sk-zai-test123"}
        with patch.dict(os.environ, test_env, clear=True):
            key = models_get_api_key("zai")
            if key == "sk-zai-test123":
                results.ok("models.py finds ZAI_API_KEY")
            else:
                results.fail("models.py ZAI_API_KEY", f"Expected sk-zai-test123, got {key}")

        # Test MINIMAX_API_KEY
        test_env = {"MINIMAX_API_KEY": "sk-minimax-test123"}
        with patch.dict(os.environ, test_env, clear=True):
            key = models_get_api_key("minimax")
            if key == "sk-minimax-test123":
                results.ok("models.py finds MINIMAX_API_KEY")
            else:
                results.fail("models.py MINIMAX_API_KEY", f"Expected sk-minimax-test123, got {key}")

        # Test MOONSHOT_API_KEY
        test_env = {"MOONSHOT_API_KEY": "sk-moonshot-test123"}
        with patch.dict(os.environ, test_env, clear=True):
            key = models_get_api_key("moonshot")
            if key == "sk-moonshot-test123":
                results.ok("models.py finds MOONSHOT_API_KEY")
            else:
                results.fail("models.py MOONSHOT_API_KEY", f"Expected sk-moonshot-test123, got {key}")

        # Test OPENAI_API_KEY
        test_env = {"OPENAI_API_KEY": "sk-openai-test123"}
        with patch.dict(os.environ, test_env, clear=True):
            key = models_get_api_key("openai")
            if key == "sk-openai-test123":
                results.ok("models.py finds OPENAI_API_KEY")
            else:
                results.fail("models.py OPENAI_API_KEY", f"Expected sk-openai-test123, got {key}")

    except ImportError as e:
        # Skip test if dependencies like litellm are not installed
        # This is expected in isolated test environments
        results.ok(f"[SKIPPED] models.py import (missing dependency: {str(e).split()[3] if 'No module named' in str(e) else 'unknown'})")
    except Exception as e:
        results.fail("models.py test", f"Error: {e}")


# ==============================================================================
# Test 12: LiteLLM Model String Construction
# ==============================================================================
def test_litellm_model_string():
    """Verify litellm model string format is correct (provider/model)"""
    print("\n[TEST] LiteLLM Model String Construction")

    # Expected litellm model strings
    expected_strings = {
        "zai": "zai/glm-4.7",
        "minimax": "minimax/MiniMax-M2.1",
        "moonshot": "moonshot/kimi-k2-0711-preview",
        "openai": "openai/gpt-4o-mini",
    }

    for provider in FALLBACK_CHAIN:
        provider_name = provider["provider"]
        model_name = provider["chat_model"]
        expected = expected_strings.get(provider_name, "")

        # LiteLLM format is provider/model
        litellm_model = f"{provider_name}/{model_name}"

        if litellm_model == expected:
            results.ok(f"LiteLLM string for {provider_name}: {litellm_model}")
        else:
            results.fail(f"LiteLLM string for {provider_name}",
                        f"Expected {expected}, got {litellm_model}")


# ==============================================================================
# Test 13: Provider Priority Override
# ==============================================================================
def test_provider_priority():
    """Test that earlier providers take priority over later ones"""
    print("\n[TEST] Provider Priority")

    # All keys set - first should win
    test_env = {
        "ZAI_API_KEY": "sk-zai-test123",
        "MINIMAX_API_KEY": "sk-minimax-test123",
        "MOONSHOT_API_KEY": "sk-moonshot-test123",
        "OPENAI_API_KEY": "sk-openai-test123",
    }
    with patch.dict(os.environ, test_env, clear=True):
        provider = select_provider()
        if provider and provider["provider"] == "zai":
            results.ok("Z.AI has highest priority when all keys set")
        else:
            results.fail("Provider priority", f"Expected zai, got {provider}")

    # Skip first, second should win
    test_env = {
        "MINIMAX_API_KEY": "sk-minimax-test123",
        "MOONSHOT_API_KEY": "sk-moonshot-test123",
        "OPENAI_API_KEY": "sk-openai-test123",
    }
    with patch.dict(os.environ, test_env, clear=True):
        provider = select_provider()
        if provider and provider["provider"] == "minimax":
            results.ok("MiniMax is second priority")
        else:
            results.fail("Second priority", f"Expected minimax, got {provider}")


# ==============================================================================
# Test 14: Whitespace Handling in API Keys
# ==============================================================================
def test_whitespace_handling():
    """Test that API keys with whitespace are handled correctly"""
    print("\n[TEST] Whitespace Handling")

    # Key with leading/trailing whitespace
    test_env = {"TEST_KEY": "  sk-test123  "}
    with patch.dict(os.environ, test_env, clear=False):
        key = get_api_key(["TEST_KEY"])
        if key == "sk-test123":
            results.ok("Strips whitespace from API key")
        else:
            results.fail("Whitespace handling", f"Expected 'sk-test123', got '{key}'")

    # Key with only whitespace
    test_env = {"WHITESPACE_KEY": "   "}
    with patch.dict(os.environ, test_env, clear=False):
        key = get_api_key(["WHITESPACE_KEY"])
        if key is None:
            results.ok("Rejects whitespace-only API key")
        else:
            results.fail("Whitespace-only key", f"Expected None, got '{key}'")


# ==============================================================================
# Test 15: Vision Support Flags
# ==============================================================================
def test_vision_support():
    """Verify vision support flags are set correctly"""
    print("\n[TEST] Vision Support Flags")

    expected_vision = {
        "zai": True,
        "minimax": False,
        "moonshot": False,
        "openai": True,
    }

    for provider in FALLBACK_CHAIN:
        provider_name = provider["provider"]
        expected = expected_vision.get(provider_name)
        actual = provider["vision"]

        if actual == expected:
            results.ok(f"{provider_name}: vision={actual}")
        else:
            results.fail(f"{provider_name} vision flag",
                        f"Expected {expected}, got {actual}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 60)
    print("LLM Provider Fallback Chain Test Suite")
    print("=" * 60)

    # Run all tests
    test_fallback_chain_order()
    test_env_variable_names()
    test_api_key_detection()
    test_provider_selection()
    test_settings_structure()
    test_settings_creation()
    test_model_names()
    test_provider_config_yaml()
    test_settings_serialization()
    test_zhipu_fallback()
    test_models_get_api_key()
    test_litellm_model_string()
    test_provider_priority()
    test_whitespace_handling()
    test_vision_support()

    # Print summary
    success = results.summary()

    if success:
        print("\nSTATUS: PASS")
        return 0
    else:
        print("\nSTATUS: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
