#!/bin/bash
#
# Alfred Gateway Deployment Script
#
# This script deploys the Alfred Gateway to your VPS with ZERO CONFLICT
# with existing Clawdbot installation.
#
# Conflict Prevention:
#   - Installs to /opt/alfred (not near Clawdbot)
#   - Uses separate Python virtualenv
#   - Uses separate systemd service
#   - Uses separate signal-cli data directory
#   - Uses separate bot tokens (you must create these)
#
# Usage:
#   1. Upload gateway folder to VPS
#   2. cd /path/to/gateway
#   3. chmod +x deploy.sh
#   4. ./deploy.sh
#

set -e

# ============================================================
# CONFIGURATION
# ============================================================
INSTALL_DIR="/opt/alfred"
GATEWAY_DIR="$INSTALL_DIR/gateway"
VENV_DIR="$INSTALL_DIR/venv"
SERVICE_NAME="alfred-gateway"
SIGNAL_DATA_DIR="/root/.signal-cli-alfred"  # SEPARATE from Clawdbot
LOG_DIR="/var/log/alfred"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# HELPER FUNCTIONS
# ============================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo ""
echo "=============================================="
echo "    Alfred Gateway Deployment"
echo "=============================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "Please run as root (sudo ./deploy.sh)"
    exit 1
fi

# Check for existing Clawdbot
log_info "Checking for existing Clawdbot installation..."
if pgrep -f "clawdbot" > /dev/null; then
    log_success "Clawdbot is running - will ensure no conflicts"
fi

if [ -d "/root/.clawdbot" ]; then
    log_success "Clawdbot config found at /root/.clawdbot"
fi

if [ -d "/root/.signal-cli" ]; then
    log_warning "Existing signal-cli data at /root/.signal-cli (Clawdbot's)"
    log_info "Alfred will use separate directory: $SIGNAL_DATA_DIR"
fi

echo ""

# ============================================================
# STEP 1: CREATE DIRECTORIES
# ============================================================
log_info "[1/7] Creating directories..."

mkdir -p "$INSTALL_DIR"
mkdir -p "$GATEWAY_DIR"
mkdir -p "$SIGNAL_DATA_DIR"
mkdir -p "$LOG_DIR"

log_success "Directories created"

# ============================================================
# STEP 2: INSTALL SYSTEM DEPENDENCIES
# ============================================================
log_info "[2/7] Installing system dependencies..."

apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv curl wget

log_success "System dependencies installed"

# ============================================================
# STEP 3: INSTALL SIGNAL-CLI (if not present)
# ============================================================
log_info "[3/7] Checking signal-cli..."

if command -v signal-cli &> /dev/null; then
    log_success "signal-cli already installed"
else
    log_info "Installing signal-cli..."

    SIGNAL_CLI_VERSION="0.13.4"
    SIGNAL_CLI_URL="https://github.com/AsamK/signal-cli/releases/download/v${SIGNAL_CLI_VERSION}/signal-cli-${SIGNAL_CLI_VERSION}.tar.gz"

    # Check if Java is installed (required for signal-cli)
    if ! command -v java &> /dev/null; then
        log_info "Installing Java (required for signal-cli)..."
        apt-get install -y -qq default-jre-headless
    fi

    cd /tmp
    wget -q "$SIGNAL_CLI_URL" -O signal-cli.tar.gz
    tar -xzf signal-cli.tar.gz -C /usr/local/
    ln -sf "/usr/local/signal-cli-${SIGNAL_CLI_VERSION}/bin/signal-cli" /usr/local/bin/signal-cli
    rm signal-cli.tar.gz

    log_success "signal-cli installed"
fi

# ============================================================
# STEP 4: SETUP PYTHON VIRTUAL ENVIRONMENT
# ============================================================
log_info "[4/7] Setting up Python virtual environment..."

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q

log_success "Python environment ready"

# ============================================================
# STEP 5: COPY GATEWAY FILES
# ============================================================
log_info "[5/7] Copying gateway files..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Copy all files
cp -r "$SCRIPT_DIR"/* "$GATEWAY_DIR/"

# Make gateway.py executable
chmod +x "$GATEWAY_DIR/gateway.py"

log_success "Files copied to $GATEWAY_DIR"

# ============================================================
# STEP 6: CREATE SYSTEMD SERVICE
# ============================================================
log_info "[6/7] Creating systemd service..."

cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=Alfred Gateway - Multi-Platform Messaging
Documentation=https://github.com/your-repo/alfred
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=$GATEWAY_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"

ExecStart=$VENV_DIR/bin/python gateway.py config.yaml

# Reliability - auto restart
Restart=always
RestartSec=5

# Don't restart more than 5 times in 60 seconds
StartLimitIntervalSec=60
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=alfred-gateway

# Resource limits (prevent runaway)
MemoryMax=512M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

log_success "Systemd service created and enabled"

# ============================================================
# STEP 7: FINAL SUMMARY
# ============================================================
echo ""
echo "=============================================="
echo -e "${GREEN}    Deployment Complete!${NC}"
echo "=============================================="
echo ""
echo "Installation directory: $GATEWAY_DIR"
echo "Signal data directory:  $SIGNAL_DATA_DIR"
echo "Service name:           $SERVICE_NAME"
echo ""
echo "=============================================="
echo "    NEXT STEPS"
echo "=============================================="
echo ""
echo "1. Edit configuration file:"
echo "   nano $GATEWAY_DIR/config.yaml"
echo ""
echo "2. Add your bot tokens:"
echo "   - Telegram: Create NEW bot via @BotFather"
echo "   - Discord:  Create NEW app at discord.com/developers"
echo "   - Signal:   Register number (see step 3)"
echo ""
echo "3. Setup Signal (SEPARATE from Clawdbot):"
echo "   signal-cli --config $SIGNAL_DATA_DIR -u +YOURPHONE register"
echo "   signal-cli --config $SIGNAL_DATA_DIR -u +YOURPHONE verify CODE"
echo ""
echo "   OR link as secondary device:"
echo "   signal-cli --config $SIGNAL_DATA_DIR link -n 'Alfred Bot'"
echo ""
echo "4. Start the gateway:"
echo "   systemctl start $SERVICE_NAME"
echo ""
echo "5. View logs:"
echo "   journalctl -u $SERVICE_NAME -f"
echo ""
echo "=============================================="
echo -e "${YELLOW}    CONFLICT PREVENTION SUMMARY${NC}"
echo "=============================================="
echo ""
echo "This installation is ISOLATED from Clawdbot:"
echo "  - Different install dir:    /opt/alfred (not Clawdbot's)"
echo "  - Different signal-cli dir: $SIGNAL_DATA_DIR"
echo "  - Different service name:   $SERVICE_NAME"
echo "  - Different health port:    8765 (vs Clawdbot's 18789)"
echo ""
echo "You MUST use:"
echo "  - DIFFERENT Telegram bot token"
echo "  - DIFFERENT Discord application"
echo "  - DIFFERENT Signal phone (or link as separate device)"
echo ""
