#!/bin/bash
#
# Alfred Gateway - Conflict Checker
#
# Run this script BEFORE starting Alfred Gateway to verify
# there are no conflicts with Clawdbot or other services.
#
# Usage: ./check_conflicts.sh
#

echo "=============================================="
echo "    Alfred Gateway - Conflict Checker"
echo "=============================================="
echo ""

ISSUES=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ISSUES=$((ISSUES + 1))
}

check_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ISSUES=$((ISSUES + 1))
}

# ============================================================
# CHECK 1: Port Conflicts
# ============================================================
echo "Checking ports..."

# Alfred uses 8765, Clawdbot uses 18789
if netstat -tlpn 2>/dev/null | grep -q ":8765 "; then
    check_fail "Port 8765 already in use (Alfred's health check port)"
else
    check_pass "Port 8765 available"
fi

# ============================================================
# CHECK 2: Signal-CLI Conflicts
# ============================================================
echo ""
echo "Checking signal-cli..."

# Check if signal-cli is locked by another process
if pgrep -f "signal-cli.*jsonRpc" > /dev/null; then
    SIGNAL_PID=$(pgrep -f "signal-cli.*jsonRpc")
    SIGNAL_CONFIG=$(ps -p $SIGNAL_PID -o args= | grep -oP '(?<=--config )[^ ]+')

    if [ "$SIGNAL_CONFIG" = "/root/.signal-cli-alfred" ]; then
        check_fail "signal-cli already running for Alfred"
    else
        check_pass "signal-cli running for different config: $SIGNAL_CONFIG"
    fi
else
    check_pass "No signal-cli jsonRpc processes running"
fi

# ============================================================
# CHECK 3: Config File Validation
# ============================================================
echo ""
echo "Checking configuration..."

CONFIG_FILE="/opt/alfred/gateway/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    # Check for empty tokens
    if grep -q 'bot_token: ""' "$CONFIG_FILE" 2>/dev/null; then
        check_warn "Some bot tokens are not configured in config.yaml"
    else
        check_pass "Bot tokens appear to be configured"
    fi

    # Check for owner IDs
    if grep -q 'telegram_id: ""' "$CONFIG_FILE" 2>/dev/null; then
        check_warn "owner.telegram_id not set"
    fi
else
    check_warn "Config file not found at $CONFIG_FILE"
fi

# ============================================================
# CHECK 4: Clawdbot Status
# ============================================================
echo ""
echo "Checking Clawdbot..."

if pgrep -f "clawdbot" > /dev/null; then
    check_pass "Clawdbot is running (good - they can coexist)"
else
    check_pass "Clawdbot not running (or using different process name)"
fi

if [ -f "/root/.clawdbot/clawdbot.json" ]; then
    check_pass "Clawdbot config exists at /root/.clawdbot/"
fi

# ============================================================
# CHECK 5: Alfred Gateway Status
# ============================================================
echo ""
echo "Checking Alfred Gateway..."

if systemctl is-active --quiet alfred-gateway; then
    check_pass "alfred-gateway service is running"
else
    check_warn "alfred-gateway service is not running"
fi

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "=============================================="
if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Safe to start Alfred Gateway.${NC}"
else
    echo -e "${YELLOW}$ISSUES issue(s) found. Review above warnings.${NC}"
fi
echo "=============================================="
echo ""
