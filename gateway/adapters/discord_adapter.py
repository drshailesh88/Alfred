"""
Discord adapter using discord.py.

CONFLICT PREVENTION:
- Uses a completely separate Discord application/bot
- Each Discord bot token can only be used by ONE process
- Only responds to DMs from the owner
"""
import asyncio
from datetime import datetime
from typing import Optional

import discord
from discord import DMChannel, Message

from .base import BaseAdapter, IncomingMessage


class DiscordAdapter(BaseAdapter):
    """Discord messaging adapter using discord.py."""

    @property
    def platform_name(self) -> str:
        return "discord"

    async def start(self) -> bool:
        """Start Discord bot."""
        token = self.config.get("bot_token")
        if not token:
            self.logger.error("Discord bot_token not configured!")
            self.logger.error("Create a new app at: https://discord.com/developers/applications")
            return False

        # Set up intents - we need message content and DMs
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        intents.guilds = False  # We only care about DMs

        self.client = discord.Client(intents=intents)

        # Register event handlers
        @self.client.event
        async def on_ready():
            self._running = True
            self._started_at = datetime.now()
            self.logger.info(f"Discord connected as {self.client.user.name}#{self.client.user.discriminator}")

        @self.client.event
        async def on_message(message: Message):
            await self._handle_message(message)

        @self.client.event
        async def on_disconnect():
            self.logger.warning("Discord disconnected")

        @self.client.event
        async def on_resumed():
            self.logger.info("Discord connection resumed")

        try:
            # Start in background task
            self._task = asyncio.create_task(self._run_client(token))

            # Wait a bit for connection
            await asyncio.sleep(3)

            if not self._running:
                self.logger.warning("Discord connection pending...")

            return True

        except discord.LoginFailure as e:
            self.logger.error(f"Discord login failed: {e}")
            self.logger.error("Check that your bot token is correct")
            return False

        except Exception as e:
            self.logger.error(f"Discord error: {e}")
            return False

    async def _run_client(self, token: str):
        """Run the Discord client with auto-reconnect."""
        while True:
            try:
                await self.client.start(token)
            except discord.LoginFailure:
                self.logger.error("Invalid Discord token")
                break
            except Exception as e:
                if not self._running:
                    break
                self.logger.error(f"Discord error, reconnecting: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        """Stop Discord bot gracefully."""
        self._running = False
        if hasattr(self, 'client') and self.client:
            await self.client.close()
        if hasattr(self, '_task'):
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Discord adapter stopped")

    async def _handle_message(self, message: Message):
        """Handle incoming Discord message."""
        # Ignore bot's own messages
        if message.author == self.client.user:
            return

        # Only respond to DMs, not guild messages
        if not isinstance(message.channel, DMChannel):
            return

        # Ignore empty messages
        if not message.content:
            return

        incoming = IncomingMessage(
            platform="discord",
            user_id=str(message.author.id),
            chat_id=str(message.channel.id),
            text=message.content,
            timestamp=message.created_at.timestamp(),
            raw=message,
            metadata={
                "message_id": message.id,
                "username": f"{message.author.name}#{message.author.discriminator}"
            }
        )

        await self.handle_message(incoming)

    async def send_message(self, chat_id: str, text: str, reply_to: Optional[str] = None):
        """Send message to Discord channel with chunking."""
        if not self._running or not hasattr(self, 'client'):
            self.logger.warning("Cannot send - Discord adapter not running")
            return

        try:
            # Get or fetch channel
            channel = self.client.get_channel(int(chat_id))
            if not channel:
                channel = await self.client.fetch_channel(int(chat_id))

            # Discord limit is 2000 characters
            max_len = 2000
            chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]

            for chunk in chunks:
                await channel.send(chunk)
                if len(chunks) > 1:
                    await asyncio.sleep(0.3)

        except discord.Forbidden:
            self.logger.error("Cannot send message - missing permissions")
        except discord.NotFound:
            self.logger.error(f"Channel not found: {chat_id}")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")

    async def send_typing(self, chat_id: str):
        """Send typing indicator."""
        if not self._running or not hasattr(self, 'client'):
            return

        try:
            channel = self.client.get_channel(int(chat_id))
            if channel:
                await channel.trigger_typing()
        except Exception:
            pass  # Typing indicators are non-critical
