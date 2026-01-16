# Alfred AI Personal Governance System
# Docker deployment based on Agent Zero framework
#
# Build: docker build -t alfred:latest .
# Run:   docker run -p 50001:50001 --env-file .env alfred:latest

FROM python:3.11-slim

LABEL maintainer="Alfred AI System"
LABEL description="Personal AI Governance System - Built on Agent Zero"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=UTC \
    TOKENIZERS_PARALLELISM=false \
    # Agent Zero settings
    AGENT_PROFILE=alfred \
    ALFRED_PORT=50001 \
    # MCP Server settings
    MCP_ENABLED=true

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    # Git for version control
    git \
    # Network tools
    curl \
    wget \
    # For PDF processing
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    # For image processing
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # For audio processing (whisper)
    ffmpeg \
    # Node.js for MCP servers
    nodejs \
    npm \
    # Signal CLI support
    default-jre-headless \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install playwright dependencies
RUN playwright install-deps chromium || true

# Copy requirements files
COPY requirements.txt /app/requirements.txt
COPY agent-zero1/requirements.txt /app/agent-zero1/requirements.txt
COPY agent-zero1/requirements2.txt /app/agent-zero1/requirements2.txt
COPY gateway/requirements.txt /app/gateway/requirements.txt

# Install Python dependencies (requirements2.txt contains litellm)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir -r /app/agent-zero1/requirements.txt \
    && pip install --no-cache-dir -r /app/agent-zero1/requirements2.txt \
    && pip install --no-cache-dir -r /app/gateway/requirements.txt

# Install Playwright browser
RUN pip install playwright && playwright install chromium || true

# Pre-install MCP servers globally via npm
RUN npm install -g \
    task-master-ai \
    mcp-simple-pubmed \
    @qdrant/mcp-server-qdrant \
    || true

# Copy application code
COPY . /app/

# Copy MCP configuration
COPY .mcp.json /app/.mcp.json

# Create necessary directories
RUN mkdir -p \
    /app/data/memory \
    /app/data/knowledge \
    /app/data/chats \
    /app/data/work_dir \
    /app/data/logs \
    /app/agent-zero1/tmp \
    /var/log/alfred

# Set permissions (tmp directory needs write access for settings)
RUN chmod -R 755 /app \
    && chmod 777 /app/data /app/data/* /var/log/alfred /app/agent-zero1/tmp

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash alfred \
    && chown -R alfred:alfred /app /var/log/alfred

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${ALFRED_PORT}/health || exit 1

# Expose port
EXPOSE 50001

# Switch to non-root user
USER alfred

# Default command - run Agent Zero UI with Alfred profile
CMD ["python", "agent-zero1/run_ui.py", "--port", "50001"]
