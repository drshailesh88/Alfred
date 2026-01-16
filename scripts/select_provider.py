#!/usr/bin/env python3
"""
Auto-select LLM provider based on available API keys.
Fallback order: Z.AI (GLM 4.7) → MiniMax (M2.1) → Moonshot (Kimi K2) → OpenAI (GPT-4o-mini)

This script runs at container startup to configure the LLM provider.
"""

import os
import json
from pathlib import Path

# Fallback chain configuration
FALLBACK_CHAIN = [
    {
        "name": "Z.AI (GLM 4.7)",
        "env_keys": ["ZAI_API_KEY", "ZHIPU_API_KEY"],
        "provider": "zai",
        "chat_model": "glm-4.7",
        "util_model": "glm-4.5-flash",
        "vision": True,
    },
    {
        "name": "MiniMax (M2.1)",
        "env_keys": ["MINIMAX_API_KEY"],
        "provider": "minimax",
        "chat_model": "MiniMax-M2.1",
        "util_model": "MiniMax-M2.1-lightning",
        "vision": False,
    },
    {
        "name": "Moonshot (Kimi K2 Thinking)",
        "env_keys": ["MOONSHOT_API_KEY"],
        "provider": "moonshot",
        "chat_model": "kimi-k2-thinking-preview",
        "util_model": "moonshot-v1-8k",
        "vision": False,
    },
    {
        "name": "OpenAI (GPT-4o-mini)",
        "env_keys": ["OPENAI_API_KEY"],
        "provider": "openai",
        "chat_model": "gpt-4o-mini",
        "util_model": "gpt-4o-mini",
        "vision": True,
    },
]

# Settings file path
SETTINGS_FILE = Path(__file__).parent.parent / "agent-zero1" / "tmp" / "settings.json"

# Default settings template
DEFAULT_SETTINGS = {
    "version": "0.8",
    "chat_model_provider": "",
    "chat_model_name": "",
    "chat_model_api_base": "",
    "chat_model_kwargs": {"temperature": "0"},
    "chat_model_ctx_length": 128000,
    "chat_model_ctx_history": 0.7,
    "chat_model_vision": True,
    "chat_model_rl_requests": 0,
    "chat_model_rl_input": 0,
    "chat_model_rl_output": 0,
    "util_model_provider": "",
    "util_model_name": "",
    "util_model_api_base": "",
    "util_model_ctx_length": 32000,
    "util_model_ctx_input": 0.7,
    "util_model_kwargs": {"temperature": "0"},
    "util_model_rl_requests": 0,
    "util_model_rl_input": 0,
    "util_model_rl_output": 0,
    "embed_model_provider": "huggingface",
    "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embed_model_api_base": "",
    "embed_model_kwargs": {},
    "embed_model_rl_requests": 0,
    "embed_model_rl_input": 0,
    "browser_model_provider": "",
    "browser_model_name": "",
    "browser_model_api_base": "",
    "browser_model_vision": True,
    "browser_model_rl_requests": 0,
    "browser_model_rl_input": 0,
    "browser_model_rl_output": 0,
    "browser_model_kwargs": {"temperature": "0"},
    "browser_http_headers": {},
    "memory_recall_enabled": True,
    "memory_recall_delayed": False,
    "memory_recall_interval": 3,
    "memory_recall_history_len": 10000,
    "memory_recall_memories_max_search": 12,
    "memory_recall_solutions_max_search": 8,
    "memory_recall_memories_max_result": 5,
    "memory_recall_solutions_max_result": 4,
    "memory_save_enabled": True,
    "agent_profile": "alfred",
    "agent_memory_subdir": "default",
    "agent_knowledge_subdir": "custom",
    "stt_model_size": "base",
    "code_exec_docker_enabled": False,
    "code_exec_ssh_enabled": False,
    "mcp_servers": "",
    "secrets": ""
}


def get_api_key(env_keys: list[str]) -> str | None:
    """Check if any of the environment variables contain a valid API key."""
    for key in env_keys:
        value = os.environ.get(key, "").strip()
        if value and value not in ("", "None", "NA", "your_api_key_here"):
            return value
    return None


def select_provider() -> dict | None:
    """Select the first available provider from the fallback chain."""
    for provider in FALLBACK_CHAIN:
        api_key = get_api_key(provider["env_keys"])
        if api_key:
            print(f"✓ Selected: {provider['name']}")
            print(f"  API Key: {provider['env_keys'][0]} = {api_key[:8]}...")
            return provider
        else:
            print(f"✗ Skipped: {provider['name']} (no API key)")
    return None


def create_settings(provider: dict) -> dict:
    """Create settings dictionary for the selected provider."""
    settings = DEFAULT_SETTINGS.copy()

    # Chat model
    settings["chat_model_provider"] = provider["provider"]
    settings["chat_model_name"] = provider["chat_model"]
    settings["chat_model_vision"] = provider["vision"]

    # Utility model
    settings["util_model_provider"] = provider["provider"]
    settings["util_model_name"] = provider["util_model"]

    # Browser model (same as chat)
    settings["browser_model_provider"] = provider["provider"]
    settings["browser_model_name"] = provider["chat_model"]
    settings["browser_model_vision"] = provider["vision"]

    return settings


def save_settings(settings: dict):
    """Save settings to the settings file."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"✓ Settings saved to: {SETTINGS_FILE}")


def main():
    print("=" * 60)
    print("Alfred LLM Provider Auto-Selection")
    print("=" * 60)
    print()
    print("Fallback chain:")
    for i, p in enumerate(FALLBACK_CHAIN, 1):
        print(f"  {i}. {p['name']}")
    print()

    # Check if settings already exist
    if SETTINGS_FILE.exists():
        print(f"⚠ Settings file already exists: {SETTINGS_FILE}")
        print("  Delete it to re-run auto-selection.")

        # Still show which provider would be selected
        print()
        print("Current API key status:")
        provider = select_provider()
        return

    # Select provider
    print("Checking available API keys...")
    provider = select_provider()

    if not provider:
        print()
        print("✗ ERROR: No LLM API keys found!")
        print()
        print("Please set one of these in your .env file:")
        for p in FALLBACK_CHAIN:
            print(f"  - {p['env_keys'][0]}")
        return

    print()

    # Create and save settings
    settings = create_settings(provider)
    save_settings(settings)

    print()
    print("=" * 60)
    print(f"Alfred will use: {provider['name']}")
    print(f"  Chat model: {provider['chat_model']}")
    print(f"  Util model: {provider['util_model']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
