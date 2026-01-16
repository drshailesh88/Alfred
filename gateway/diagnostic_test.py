#!/usr/bin/env python3
"""
Ralph Loop Diagnostic Test for Alfred Gateway
Tests each component of the message flow:
1. Config loading
2. Router initialization
3. API connectivity (mock)
4. Message processing
5. Response handling
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent-zero1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent-zero1" / "python"))

print("=" * 70)
print("ALFRED GATEWAY DIAGNOSTIC TEST")
print("=" * 70)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Track issues
issues = []
warnings = []

# ============================================================
# TEST 1: Config Loading
# ============================================================
print("[TEST 1] Loading configuration...")
try:
    import yaml
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"  ✓ Config loaded from: {config_path}")

    # Check critical fields
    checks = [
        ("telegram.enabled", config.get("telegram", {}).get("enabled")),
        ("telegram.bot_token", bool(config.get("telegram", {}).get("bot_token"))),
        ("owner.telegram_id", config.get("owner", {}).get("telegram_id")),
        ("agent_zero.mode", config.get("agent_zero", {}).get("mode")),
        ("agent_zero.api_url", config.get("agent_zero", {}).get("api_url")),
        ("signal.enabled", config.get("signal", {}).get("enabled")),
        ("signal.phone_number", config.get("signal", {}).get("phone_number")),
    ]

    for name, value in checks:
        status = "✓" if value else "✗"
        print(f"    {status} {name}: {value}")
        if not value and "enabled" not in name:
            if "signal" in name.lower():
                warnings.append(f"Config: {name} not set")
            else:
                issues.append(f"Config: {name} not set")

except Exception as e:
    issues.append(f"Config loading failed: {e}")
    print(f"  ✗ Config loading failed: {e}")

print()

# ============================================================
# TEST 2: Router Initialization
# ============================================================
print("[TEST 2] Router initialization...")
try:
    from router import AlfredRouter
    router = AlfredRouter(config)
    print(f"  ✓ Router initialized")
    print(f"    Mode: {router.mode}")
    print(f"    Connected: {router.is_connected}")

    if router.mode == "echo":
        issues.append("Router is in ECHO mode - Agent Zero not connected!")
    elif router.mode == "api":
        print(f"    API URL: {router.api_url}")
    elif router.mode == "direct":
        print(f"    Direct mode - Agent instance: {router.agent}")

except Exception as e:
    issues.append(f"Router initialization failed: {e}")
    print(f"  ✗ Router init failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================
# TEST 3: API Connectivity Test
# ============================================================
print("[TEST 3] API connectivity test...")
api_url = config.get("agent_zero", {}).get("api_url", "")
if api_url:
    print(f"  Testing: {api_url}")

    # Parse URL
    from urllib.parse import urlparse
    parsed = urlparse(api_url)
    print(f"    Host: {parsed.hostname}")
    print(f"    Port: {parsed.port}")
    print(f"    Path: {parsed.path}")

    # Test with aiohttp
    async def test_api():
        import aiohttp
        test_message = "Hello Alfred, this is a diagnostic test."

        # First try a simple GET to check if server is up
        health_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}/api/health"
        print(f"  Testing health endpoint: {health_url}")

        try:
            async with aiohttp.ClientSession() as session:
                # Try health check first
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        print(f"    Health check: {resp.status}")
                        if resp.status == 200:
                            print(f"    ✓ Server is running")
                        else:
                            text = await resp.text()
                            print(f"    Response: {text[:200]}")
                except Exception as e:
                    print(f"    ✗ Health check failed: {e}")
                    issues.append(f"Health check failed: {e}")

                # Test actual API endpoint
                print(f"  Testing message API: {api_url}")
                try:
                    async with session.post(
                        api_url,
                        json={"message": test_message},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        print(f"    Status: {resp.status}")
                        if resp.status == 200:
                            data = await resp.json()
                            response = data.get("response", "")
                            context_id = data.get("context_id", "")
                            print(f"    ✓ API responded!")
                            print(f"    Context ID: {context_id}")
                            print(f"    Response preview: {response[:150]}..." if len(response) > 150 else f"    Response: {response}")
                            return True
                        else:
                            text = await resp.text()
                            print(f"    ✗ API error: {text[:300]}")
                            issues.append(f"API returned status {resp.status}: {text[:100]}")
                            return False
                except asyncio.TimeoutError:
                    print(f"    ✗ API timeout (30s)")
                    issues.append("API request timed out after 30 seconds")
                    return False
                except Exception as e:
                    print(f"    ✗ API request failed: {e}")
                    issues.append(f"API request failed: {e}")
                    return False

        except Exception as e:
            print(f"    ✗ Connection failed: {e}")
            issues.append(f"Cannot connect to API: {e}")
            return False

    # Run async test
    try:
        asyncio.run(test_api())
    except Exception as e:
        print(f"  ✗ Async test failed: {e}")
        issues.append(f"Async test error: {e}")
else:
    print("  ✗ No API URL configured")
    issues.append("No API URL configured in config.yaml")

print()

# ============================================================
# TEST 4: Telegram Adapter Test
# ============================================================
print("[TEST 4] Telegram adapter validation...")
telegram_config = config.get("telegram", {})
if telegram_config.get("enabled"):
    token = telegram_config.get("bot_token", "")
    if token:
        # Mask token for display
        masked = token[:10] + "..." + token[-5:] if len(token) > 20 else "***"
        print(f"  ✓ Bot token configured: {masked}")

        # Test token validity
        async def test_telegram():
            try:
                from telegram import Bot
                bot = Bot(token=token)
                me = await bot.get_me()
                print(f"  ✓ Bot valid: @{me.username} (ID: {me.id})")
                return True
            except Exception as e:
                print(f"  ✗ Bot token invalid: {e}")
                issues.append(f"Telegram bot token invalid: {e}")
                return False

        try:
            asyncio.run(test_telegram())
        except Exception as e:
            print(f"  ✗ Telegram test failed: {e}")
    else:
        print("  ✗ No bot token")
        issues.append("Telegram enabled but no bot_token configured")
else:
    print("  - Telegram disabled in config")

print()

# ============================================================
# TEST 5: Signal Adapter Validation
# ============================================================
print("[TEST 5] Signal adapter validation...")
signal_config = config.get("signal", {})
if signal_config.get("enabled"):
    phone = signal_config.get("phone_number", "")
    data_dir = signal_config.get("data_dir", "/root/.signal-cli-alfred")

    print(f"  Phone: {phone}")
    print(f"  Data dir: {data_dir}")

    if phone:
        print(f"  ✓ Phone number configured: {phone}")
    else:
        warnings.append("Signal enabled but no phone_number configured")
        print("  ✗ No phone number configured")

    # Check if data dir exists and has account
    if os.path.exists(data_dir):
        print(f"  ✓ Data directory exists")
        account_path = os.path.join(data_dir, "data", phone.replace("+", ""))
        if os.path.exists(account_path):
            print(f"  ✓ Account directory exists")
        else:
            alt_account_path = os.path.join(data_dir, "data", phone)
            if os.path.exists(alt_account_path):
                print(f"  ✓ Account directory exists (with +)")
            else:
                warnings.append(f"Signal account not registered in {data_dir}")
                print(f"  ⚠ Account not found - needs registration")
    else:
        warnings.append(f"Signal data directory doesn't exist: {data_dir}")
        print(f"  ⚠ Data directory doesn't exist - needs setup")
else:
    print("  - Signal disabled in config")

print()

# ============================================================
# TEST 6: LLM Provider Check
# ============================================================
print("[TEST 6] LLM provider check...")
try:
    # Check for API keys in environment
    llm_keys = [
        ("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY")),
        ("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")),
        ("ZHIPUAI_API_KEY", os.environ.get("ZHIPUAI_API_KEY")),
        ("MINIMAX_API_KEY", os.environ.get("MINIMAX_API_KEY")),
        ("MOONSHOT_API_KEY", os.environ.get("MOONSHOT_API_KEY")),
    ]

    has_llm = False
    for name, value in llm_keys:
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"  ✓ {name}: {masked}")
            has_llm = True
        else:
            print(f"  - {name}: not set")

    if not has_llm:
        issues.append("No LLM API keys found in environment")
        print("  ✗ No LLM API keys configured!")

except Exception as e:
    print(f"  ✗ Error checking LLM keys: {e}")

print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if issues:
    print(f"\n🔴 ISSUES FOUND ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print("\n✅ No critical issues found!")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

print()
print("=" * 70)

# Exit with error code if issues found
sys.exit(1 if issues else 0)
