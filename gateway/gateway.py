#!/usr/bin/env python3
"""
Alfred Gateway - Multi-Platform Messaging Gateway for Alfred AI.

This is the main entry point for the gateway. It orchestrates all
platform adapters (Telegram, Discord, Signal) and routes messages
to Agent Zero (Alfred).

CONFLICT PREVENTION:
- Uses different ports than Clawdbot (8765 vs 18789)
- Uses separate bot tokens for each platform
- Uses separate signal-cli data directory
- Runs as a separate systemd service

Usage:
    python gateway.py [config.yaml]

    If no config file is specified, looks for config.yaml in the current directory.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import yaml

from adapters import TelegramAdapter, DiscordAdapter, SignalAdapter, IncomingMessage
from router import AlfredRouter


# Banner
BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     █████╗ ██╗     ███████╗██████╗ ███████╗██████╗        ║
║    ██╔══██╗██║     ██╔════╝██╔══██╗██╔════╝██╔══██╗       ║
║    ███████║██║     █████╗  ██████╔╝█████╗  ██║  ██║       ║
║    ██╔══██║██║     ██╔══╝  ██╔══██╗██╔══╝  ██║  ██║       ║
║    ██║  ██║███████╗██║     ██║  ██║███████╗██████╔╝       ║
║    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝        ║
║                                                           ║
║              Multi-Platform Messaging Gateway             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


def setup_logging(level: str = "INFO"):
    """Configure logging for the gateway."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # File handler (optional)
    log_dir = Path("/var/log/alfred")
    if log_dir.exists():
        file_handler = logging.FileHandler(log_dir / "gateway.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
    else:
        file_handler = None

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("discord").setLevel(logging.WARNING)


class AlfredGateway:
    """Main gateway orchestrating all platform adapters."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the gateway.

        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.adapters: Dict[str, any] = {}
        self.router: Optional[AlfredRouter] = None
        self._shutdown_event = asyncio.Event()
        self._started_at: Optional[datetime] = None
        self.logger = logging.getLogger("Alfred.Gateway")

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(path)

        if not config_file.exists():
            # Try in script directory
            script_dir = Path(__file__).parent
            config_file = script_dir / path

        if not config_file.exists():
            print(f"ERROR: Config file not found: {path}")
            print("Create a config.yaml file with your bot tokens.")
            sys.exit(1)

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Validate required fields
        self._validate_config(config)

        return config

    def _validate_config(self, config: dict):
        """Validate configuration and warn about missing items."""
        owner = config.get("owner", {})

        warnings = []

        # Check owner IDs
        if config.get("telegram", {}).get("enabled") and not owner.get("telegram_id"):
            warnings.append("Telegram enabled but owner.telegram_id not set")

        if config.get("discord", {}).get("enabled") and not owner.get("discord_id"):
            warnings.append("Discord enabled but owner.discord_id not set")

        if config.get("signal", {}).get("enabled") and not owner.get("signal_number"):
            warnings.append("Signal enabled but owner.signal_number not set")

        # Check bot tokens
        if config.get("telegram", {}).get("enabled") and not config.get("telegram", {}).get("bot_token"):
            warnings.append("Telegram enabled but bot_token not set")

        if config.get("discord", {}).get("enabled") and not config.get("discord", {}).get("bot_token"):
            warnings.append("Discord enabled but bot_token not set")

        for warning in warnings:
            print(f"WARNING: {warning}")

    async def _on_message(self, message: IncomingMessage):
        """Handle incoming message from any platform."""
        adapter = self.adapters.get(message.platform)
        if not adapter:
            self.logger.error(f"No adapter for platform: {message.platform}")
            return

        # Send typing indicator
        await adapter.send_typing(message.chat_id)

        # Process message through router
        response = await self.router.process_message(
            message,
            send_reply=adapter.send_message,
            send_typing=adapter.send_typing
        )

        # Send response
        await adapter.send_message(message.chat_id, response)

    async def start(self):
        """Start all enabled adapters."""
        self.logger.info("Starting Alfred Gateway...")

        # Initialize router
        self.router = AlfredRouter(self.config)
        self.logger.info(f"Router mode: {self.router.mode}")

        owner = self.config.get("owner", {})
        started_adapters = []

        # Start Telegram
        if self.config.get("telegram", {}).get("enabled"):
            telegram_owner = owner.get("telegram_id", "")
            telegram_token = self.config.get("telegram", {}).get("bot_token", "")

            if telegram_owner and telegram_token:
                adapter = TelegramAdapter(
                    config=self.config["telegram"],
                    owner_id=telegram_owner,
                    on_message=self._on_message
                )
                if await adapter.start():
                    self.adapters["telegram"] = adapter
                    started_adapters.append("Telegram")
            else:
                self.logger.warning("Telegram not configured properly")

        # Start Discord
        if self.config.get("discord", {}).get("enabled"):
            discord_owner = owner.get("discord_id", "")
            discord_token = self.config.get("discord", {}).get("bot_token", "")

            if discord_owner and discord_token:
                adapter = DiscordAdapter(
                    config=self.config["discord"],
                    owner_id=discord_owner,
                    on_message=self._on_message
                )
                if await adapter.start():
                    self.adapters["discord"] = adapter
                    started_adapters.append("Discord")
            else:
                self.logger.warning("Discord not configured properly")

        # Start Signal
        if self.config.get("signal", {}).get("enabled"):
            signal_owner = owner.get("signal_number", "")

            if signal_owner:
                adapter = SignalAdapter(
                    config=self.config["signal"],
                    owner_id=signal_owner,
                    on_message=self._on_message
                )
                if await adapter.start():
                    self.adapters["signal"] = adapter
                    started_adapters.append("Signal")
            else:
                self.logger.warning("Signal not configured properly")

        if not self.adapters:
            self.logger.error("No adapters started! Check your configuration.")
            return

        self._started_at = datetime.now()

        self.logger.info("=" * 50)
        self.logger.info(f"Alfred Gateway running")
        self.logger.info(f"Platforms: {', '.join(started_adapters)}")
        self.logger.info(f"Router mode: {self.router.mode}")
        self.logger.info("=" * 50)

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def stop(self):
        """Stop all adapters gracefully."""
        self.logger.info("Shutting down Alfred Gateway...")

        # Stop all adapters concurrently
        stop_tasks = []
        for name, adapter in self.adapters.items():
            self.logger.info(f"Stopping {name}...")
            stop_tasks.append(adapter.stop())

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        self.logger.info("Alfred Gateway stopped. Goodbye, sir.")

    def request_shutdown(self):
        """Request graceful shutdown."""
        self._shutdown_event.set()


async def main():
    """Main entry point."""
    print(BANNER)

    # Determine config path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    # Setup logging
    setup_logging("INFO")

    # Create gateway
    gateway = AlfredGateway(config_path)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, gateway.request_shutdown)

    try:
        await gateway.start()
    finally:
        await gateway.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
