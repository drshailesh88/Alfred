# ALFRED - Personal AI Governance System
# Docker deployment configuration
# Phase 9: Operational Deployment

FROM python:3.11-slim

LABEL maintainer="ALFRED System"
LABEL description="Personal AI Governance System for Agent Zero"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ALFRED_HOME=/app/agents/alfred
ENV DATA_PATH=/app/data/alfred

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Create data directories
RUN mkdir -p ${DATA_PATH}/state \
    ${DATA_PATH}/briefs \
    ${DATA_PATH}/briefs/weekly \
    ${DATA_PATH}/orchestration \
    ${DATA_PATH}/memory_service \
    ${DATA_PATH}/relationships \
    ${DATA_PATH}/content_pipeline \
    ${DATA_PATH}/learning_curator \
    ${DATA_PATH}/learning_scout \
    ${DATA_PATH}/learning_distiller \
    ${DATA_PATH}/metrics \
    ${DATA_PATH}/audience_signals \
    ${DATA_PATH}/strategy \
    ${DATA_PATH}/reputation \
    ${DATA_PATH}/world_radar \
    ${DATA_PATH}/social_triage \
    ${DATA_PATH}/research \
    ${DATA_PATH}/substack \
    ${DATA_PATH}/twitter \
    ${DATA_PATH}/youtube \
    ${DATA_PATH}/shipping \
    ${DATA_PATH}/financial \
    ${DATA_PATH}/scheduling \
    ${DATA_PATH}/patient_data \
    ${DATA_PATH}/intake

# Set permissions
RUN chmod -R 755 ${DATA_PATH}

# Expose port for Agent Zero UI
EXPOSE 50001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:50001/health || exit 1

# Default command
CMD ["python", "run_ui.py"]
