# Alfred AI - Deployment Guide

> Complete guide for deploying Alfred to a VPS with Telegram, Signal, and Discord integration.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start (Docker)](#quick-start-docker)
3. [VPS Deployment](#vps-deployment)
4. [Messaging Integration](#messaging-integration)
   - [Telegram Setup](#telegram-setup)
   - [Signal Setup](#signal-setup)
   - [Discord Setup](#discord-setup)
5. [Configuration Reference](#configuration-reference)
6. [Health Monitoring](#health-monitoring)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum VPS Specifications

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Storage | 20 GB SSD | 50 GB SSD |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |

### Required Software

- Docker 24.0+ and Docker Compose 2.0+
- Git
- curl (for health checks)

### API Keys Required

| Service | Required | Purpose |
|---------|----------|---------|
| Anthropic API | **Yes** | Primary LLM (Claude) |
| OpenAI API | Optional | Embeddings, Whisper |
| Google OAuth | Optional | Calendar, Gmail |
| Telegram Bot | Optional | Telegram messaging |
| Signal CLI | Optional | Signal messaging |

---

## Quick Start (Docker)

### 1. Clone and Configure

```bash
# Clone repository
git clone https://github.com/yourusername/Alfred.git
cd Alfred

# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env  # or vim .env
```

### 2. Set Required API Key

At minimum, set your Anthropic API key in `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
```

### 3. Start Services

```bash
# Build and start (Alfred + Qdrant)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f alfred
```

### 4. Access Alfred

Open your browser to: `http://localhost:50001`

---

## VPS Deployment

### Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Log out and back in for group changes
exit
```

### Step 2: Clone Repository

```bash
cd /opt
sudo git clone https://github.com/yourusername/Alfred.git
sudo chown -R $USER:$USER Alfred
cd Alfred
```

### Step 3: Configure Environment

```bash
cp .env.example .env
nano .env
```

**Required settings for VPS:**

```bash
# Core API
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx

# Security (generate these!)
MEMORY_ENCRYPTION_KEY=$(openssl rand -hex 32)
FLASK_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Timezone
TZ=Asia/Kolkata  # or your timezone
```

### Step 4: Deploy

```bash
# Build and start
docker-compose up -d --build

# Verify running
docker-compose ps

# Check logs
docker-compose logs -f alfred
```

### Step 5: Configure Firewall

```bash
# Allow SSH
sudo ufw allow 22

# Allow Alfred web UI (optional - for direct access)
sudo ufw allow 50001

# Enable firewall
sudo ufw enable
```

### Step 6: Setup Reverse Proxy (Optional but Recommended)

For HTTPS access, use nginx:

```bash
sudo apt install nginx certbot python3-certbot-nginx -y

# Create nginx config
sudo nano /etc/nginx/sites-available/alfred
```

```nginx
server {
    listen 80;
    server_name alfred.yourdomain.com;

    location / {
        proxy_pass http://localhost:50001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/alfred /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get SSL certificate
sudo certbot --nginx -d alfred.yourdomain.com
```

---

## Messaging Integration

Alfred can be accessed via Telegram, Signal, or Discord instead of the web UI.

### Telegram Setup

**Step 1: Create a Telegram Bot**

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`
3. Choose a name: `Alfred Personal AI`
4. Choose a username: `your_alfred_bot`
5. Copy the **bot token** (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

**Step 2: Get Your Telegram User ID**

1. Message [@userinfobot](https://t.me/userinfobot)
2. It will reply with your user ID (a number like `123456789`)

**Step 3: Configure**

Add to your `.env`:

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_OWNER_ID=123456789
```

**Step 4: Create Gateway Config**

Create `gateway/config.yaml`:

```yaml
# Alfred Gateway Configuration
# See gateway/config.example.yaml for full reference

owner:
  telegram_id: "123456789"  # Your Telegram user ID

alfred:
  api_url: "http://alfred:50001"  # Docker internal URL
  timeout: 120

telegram:
  enabled: true
  bot_token: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"

discord:
  enabled: false

signal:
  enabled: false
```

**Step 5: Start Gateway**

```bash
# Start with gateway profile
docker-compose --profile gateway up -d

# Check gateway logs
docker-compose logs -f gateway
```

**Step 6: Test**

1. Open Telegram
2. Start a chat with your bot
3. Send `/start`
4. Alfred should respond

---

### Signal Setup

Signal requires `signal-cli` which needs a phone number to register.

**Step 1: Install signal-cli (on VPS)**

```bash
# Install Java (required)
sudo apt install default-jre-headless -y

# Download signal-cli
SIGNAL_CLI_VERSION="0.13.4"
wget https://github.com/AsamK/signal-cli/releases/download/v${SIGNAL_CLI_VERSION}/signal-cli-${SIGNAL_CLI_VERSION}.tar.gz
sudo tar xf signal-cli-${SIGNAL_CLI_VERSION}.tar.gz -C /opt
sudo ln -sf /opt/signal-cli-${SIGNAL_CLI_VERSION}/bin/signal-cli /usr/local/bin/signal-cli
```

**Step 2: Register Phone Number**

Option A - New number (requires SMS):
```bash
# Register (you'll receive SMS verification)
signal-cli --config /root/.signal-cli-alfred -u +1234567890 register

# Verify with code from SMS
signal-cli --config /root/.signal-cli-alfred -u +1234567890 verify 123456
```

Option B - Link as secondary device (recommended):
```bash
# Generate QR code link
signal-cli --config /root/.signal-cli-alfred link -n "Alfred Bot"

# Scan the QR code with your Signal app:
# Signal > Settings > Linked Devices > Link New Device
```

**Step 3: Configure**

Add to `.env`:

```bash
SIGNAL_PHONE_NUMBER=+1234567890
```

Update `gateway/config.yaml`:

```yaml
owner:
  signal_number: "+1987654321"  # Your personal Signal number

signal:
  enabled: true
  phone_number: "+1234567890"   # Alfred's Signal number
  signal_cli_path: "signal-cli"
  data_dir: "/root/.signal-cli-alfred"
```

**Step 4: Start Gateway**

```bash
docker-compose --profile gateway up -d
```

**Step 5: Test**

Send a message to Alfred's Signal number from your phone.

---

### Discord Setup

**Step 1: Create Discord Application**

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Name it "Alfred AI"
4. Go to "Bot" section
5. Click "Add Bot"
6. Copy the **bot token**
7. Enable "Message Content Intent" under Privileged Intents

**Step 2: Get Your Discord User ID**

1. Enable Developer Mode in Discord settings
2. Right-click your username
3. Click "Copy User ID"

**Step 3: Invite Bot to Server**

1. Go to OAuth2 > URL Generator
2. Select scopes: `bot`
3. Select permissions: `Send Messages`, `Read Message History`
4. Copy the generated URL
5. Open URL and add bot to your server

**Step 4: Configure**

Add to `.env`:

```bash
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_OWNER_ID=your_user_id_here
```

Update `gateway/config.yaml`:

```yaml
owner:
  discord_id: "123456789012345678"

discord:
  enabled: true
  bot_token: "your_discord_bot_token"
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `OPENAI_API_KEY` | No | For embeddings |
| `GOOGLE_CLIENT_ID` | No | Google OAuth |
| `GOOGLE_CLIENT_SECRET` | No | Google OAuth |
| `GOOGLE_REFRESH_TOKEN` | No | Google OAuth |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot |
| `TELEGRAM_OWNER_ID` | No | Your Telegram ID |
| `SIGNAL_PHONE_NUMBER` | No | Signal phone |
| `DISCORD_BOT_TOKEN` | No | Discord bot |
| `DISCORD_OWNER_ID` | No | Your Discord ID |
| `QDRANT_URL` | Auto | Vector database URL |
| `MEMORY_ENCRYPTION_KEY` | Recommended | Memory encryption |
| `FLASK_SECRET_KEY` | Recommended | Session security |
| `TZ` | No | Timezone (UTC default) |
| `LOG_LEVEL` | No | INFO/DEBUG/WARNING |

### Docker Compose Profiles

| Profile | Services | Command |
|---------|----------|---------|
| (default) | alfred, qdrant | `docker-compose up -d` |
| gateway | + gateway | `docker-compose --profile gateway up -d` |

### Ports

| Service | Port | Purpose |
|---------|------|---------|
| Alfred | 50001 | Web UI / API |
| Qdrant | 6333 | Vector DB HTTP |
| Qdrant | 6334 | Vector DB gRPC |

---

## Health Monitoring

### Check Service Status

```bash
# All services
docker-compose ps

# Alfred logs
docker-compose logs -f alfred

# Gateway logs
docker-compose logs -f gateway

# Qdrant logs
docker-compose logs -f qdrant
```

### Health Endpoints

```bash
# Alfred health
curl http://localhost:50001/api/health

# Qdrant health
curl http://localhost:6333/readiness
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart alfred

# Rebuild and restart
docker-compose up -d --build
```

### View Resource Usage

```bash
docker stats
```

---

## Troubleshooting

### Alfred Won't Start

```bash
# Check logs
docker-compose logs alfred

# Common issues:
# 1. Missing ANTHROPIC_API_KEY
# 2. Port 50001 already in use
# 3. Out of memory
```

### Telegram Bot Not Responding

```bash
# Check gateway logs
docker-compose logs gateway

# Common issues:
# 1. Wrong bot token
# 2. Another process using same token (Conflict error)
# 3. Gateway not started (use --profile gateway)
```

### Signal Not Working

```bash
# Check if signal-cli is registered
signal-cli --config /root/.signal-cli-alfred -u +1234567890 listGroups

# Common issues:
# 1. Phone number not registered
# 2. signal-cli data directory wrong
# 3. Java not installed
```

### Qdrant Connection Failed

```bash
# Check Qdrant is running
docker-compose logs qdrant

# Test connection
curl http://localhost:6333/readiness

# Common issues:
# 1. Qdrant container not started
# 2. QDRANT_URL wrong in docker internal network
```

### Out of Memory

```bash
# Check memory usage
free -h
docker stats

# Solutions:
# 1. Increase VPS RAM
# 2. Add swap space:
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Backup and Restore

### Backup Data

```bash
# Stop services
docker-compose down

# Backup volumes
docker run --rm -v alfred_memory:/data -v $(pwd):/backup alpine tar czf /backup/alfred_memory.tar.gz -C /data .
docker run --rm -v qdrant_storage:/data -v $(pwd):/backup alpine tar czf /backup/qdrant_storage.tar.gz -C /data .
```

### Restore Data

```bash
# Restore volumes
docker run --rm -v alfred_memory:/data -v $(pwd):/backup alpine tar xzf /backup/alfred_memory.tar.gz -C /data
docker run --rm -v qdrant_storage:/data -v $(pwd):/backup alpine tar xzf /backup/qdrant_storage.tar.gz -C /data

# Start services
docker-compose up -d
```

---

## Security Recommendations

1. **Always use HTTPS** in production (nginx + certbot)
2. **Generate strong keys** for MEMORY_ENCRYPTION_KEY and FLASK_SECRET_KEY
3. **Restrict Telegram/Discord** to owner IDs only (already enforced)
4. **Use firewall** to block unnecessary ports
5. **Keep Docker updated** for security patches
6. **Regular backups** of memory and knowledge volumes

---

## Quick Reference

### Start Everything

```bash
docker-compose --profile gateway up -d
```

### Stop Everything

```bash
docker-compose --profile gateway down
```

### View All Logs

```bash
docker-compose --profile gateway logs -f
```

### Update Alfred

```bash
git pull
docker-compose --profile gateway up -d --build
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-16*
