"""
Telegram adapter using python-telegram-bot.

CONFLICT PREVENTION:
- Uses polling (not webhooks) to avoid port conflicts
- Requires a DIFFERENT bot token than any other Telegram bot
- Each bot token can only be used by ONE process at a time
"""
import asyncio
from datetime import datetime
from typing import Optional

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from telegram.constants import ChatAction
from telegram.error import TelegramError, Conflict

from .base import BaseAdapter, IncomingMessage


class TelegramAdapter(BaseAdapter):
    """Telegram messaging adapter using python-telegram-bot v20+."""

    @property
    def platform_name(self) -> str:
        return "telegram"

    async def start(self) -> bool:
        """Start Telegram bot with long polling."""
        token = self.config.get("bot_token")
        if not token:
            self.logger.error("Telegram bot_token not configured!")
            self.logger.error("Create a new bot via @BotFather and add the token to config.yaml")
            return False

        try:
            # Build application
            self.app = Application.builder().token(token).build()

            # Add handlers
            self.app.add_handler(CommandHandler("start", self._handle_start))
            self.app.add_handler(CommandHandler("status", self._handle_status))
            self.app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_message
            ))

            # Error handler
            self.app.add_error_handler(self._handle_error)

            # Initialize
            await self.app.initialize()
            await self.app.start()

            # Start polling - drop pending updates to avoid processing old messages
            await self.app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=["message"]
            )

            self._running = True
            self._started_at = datetime.now()

            # Get bot info for logging
            bot_info = await self.app.bot.get_me()
            self.logger.info(f"Telegram connected as @{bot_info.username}")

            return True

        except Conflict as e:
            self.logger.error("=" * 60)
            self.logger.error("TELEGRAM CONFLICT ERROR!")
            self.logger.error("Another process is using this bot token.")
            self.logger.error("This usually means:")
            self.logger.error("  1. Clawdbot is using the same token (use a DIFFERENT bot)")
            self.logger.error("  2. Another instance of Alfred Gateway is running")
            self.logger.error("=" * 60)
            return False

        except TelegramError as e:
            self.logger.error(f"Telegram error: {e}")
            return False

    async def stop(self):
        """Stop Telegram bot gracefully."""
        self._running = False
        if hasattr(self, 'app') and self.app:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
            except Exception as e:
                self.logger.warning(f"Error during Telegram shutdown: {e}")
        self.logger.info("Telegram adapter stopped")

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = str(update.message.from_user.id)

        if self.is_owner(user_id):
            await update.message.reply_text(
                "Good evening, sir. Alfred at your service.\n\n"
                "I am connected and ready to assist you.\n"
                "Simply send me a message and I shall respond."
            )
        else:
            # Politely decline non-owners
            await update.message.reply_text(
                "I apologize, but I serve only one master.\n"
                "This is a private assistant."
            )

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - owner only."""
        user_id = str(update.message.from_user.id)

        if not self.is_owner(user_id):
            return

        uptime = self.uptime_seconds
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)

        await update.message.reply_text(
            f"Alfred Gateway Status\n"
            f"----------------------\n"
            f"Platform: Telegram\n"
            f"Status: Running\n"
            f"Uptime: {hours}h {minutes}m\n"
            f"Owner ID: {self.owner_id}"
        )

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        if not update.message or not update.message.text:
            return

        message = IncomingMessage(
            platform="telegram",
            user_id=str(update.message.from_user.id),
            chat_id=str(update.message.chat_id),
            text=update.message.text,
            timestamp=update.message.date.timestamp(),
            raw=update,
            metadata={
                "message_id": update.message.message_id,
                "username": update.message.from_user.username
            }
        )

        await self.handle_message(message)

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle Telegram errors."""
        self.logger.error(f"Telegram error: {context.error}")

        # Check for conflict errors
        if isinstance(context.error, Conflict):
            self.logger.error("Token conflict detected! Another process may be using this bot.")

    async def send_message(self, chat_id: str, text: str, reply_to: Optional[str] = None):
        """Send message to Telegram chat with chunking for long messages."""
        if not self._running or not hasattr(self, 'app'):
            self.logger.warning("Cannot send - Telegram adapter not running")
            return

        # Telegram limit is 4096 characters
        max_len = 4096
        chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]

        for i, chunk in enumerate(chunks):
            try:
                await self.app.bot.send_message(
                    chat_id=int(chat_id),
                    text=chunk,
                    reply_to_message_id=int(reply_to) if reply_to and i == 0 else None
                )
                # Small delay between chunks
                if len(chunks) > 1 and i < len(chunks) - 1:
                    await asyncio.sleep(0.3)
            except TelegramError as e:
                self.logger.error(f"Failed to send message: {e}")

    async def send_typing(self, chat_id: str):
        """Send typing indicator."""
        if not self._running or not hasattr(self, 'app'):
            return

        try:
            await self.app.bot.send_chat_action(
                chat_id=int(chat_id),
                action=ChatAction.TYPING
            )
        except TelegramError:
            pass  # Typing indicators are non-critical
