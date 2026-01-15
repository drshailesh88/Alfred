"""
Signal adapter using signal-cli subprocess.

CONFLICT PREVENTION:
- Uses a SEPARATE data directory (/root/.signal-cli-alfred)
- Clawdbot typically uses ~/.signal-cli or ~/.local/share/signal-cli
- signal-cli locks its config files - only one process can use each directory
- Uses a separate phone number or links as a secondary device
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Optional

from .base import BaseAdapter, IncomingMessage


class SignalAdapter(BaseAdapter):
    """
    Signal messaging adapter via signal-cli JSON-RPC.

    Requires signal-cli to be installed and a phone number to be registered
    in the SEPARATE data directory specified in config.
    """

    @property
    def platform_name(self) -> str:
        return "signal"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_cli = self.config.get("signal_cli_path", "signal-cli")
        self.data_dir = self.config.get("data_dir", "/root/.signal-cli-alfred")
        self.phone = self.config.get("phone_number", "")
        self._process: Optional[asyncio.subprocess.Process] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """Start signal-cli in JSON-RPC mode."""
        if not self.phone:
            self.logger.error("Signal phone_number not configured!")
            return False

        # Check if signal-cli exists
        if not os.path.exists(self.signal_cli) and self.signal_cli != "signal-cli":
            self.logger.error(f"signal-cli not found at: {self.signal_cli}")
            return False

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Check if the account is registered
        account_dir = os.path.join(self.data_dir, "data", self.phone)
        if not os.path.exists(account_dir):
            self.logger.error(f"Signal account not found in: {self.data_dir}")
            self.logger.error("You need to register or link first:")
            self.logger.error(f"  signal-cli --config {self.data_dir} -u {self.phone} register")
            self.logger.error(f"  signal-cli --config {self.data_dir} -u {self.phone} verify CODE")
            self.logger.error("Or link as secondary device:")
            self.logger.error(f"  signal-cli --config {self.data_dir} link -n 'Alfred Bot'")
            return False

        try:
            # Start signal-cli in JSON-RPC mode
            cmd = [
                self.signal_cli,
                "--config", self.data_dir,
                "-u", self.phone,
                "jsonRpc"
            ]

            self.logger.info(f"Starting signal-cli: {' '.join(cmd)}")

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            self._running = True
            self._started_at = datetime.now()

            # Start background task to receive messages
            self._receive_task = asyncio.create_task(self._receive_loop())
            # Start background task to log stderr
            asyncio.create_task(self._stderr_loop())

            self.logger.info(f"Signal adapter running for {self.phone}")
            return True

        except FileNotFoundError:
            self.logger.error("signal-cli not found. Install it first:")
            self.logger.error("  apt install signal-cli")
            self.logger.error("  or download from: https://github.com/AsamK/signal-cli/releases")
            return False

        except Exception as e:
            self.logger.error(f"Failed to start Signal adapter: {e}")
            return False

    async def stop(self):
        """Stop signal-cli process."""
        self._running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                self.logger.warning(f"Error stopping signal-cli: {e}")

        self.logger.info("Signal adapter stopped")

    async def _receive_loop(self):
        """Continuously receive messages from signal-cli stdout."""
        while self._running and self._process:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    if self._running:
                        self.logger.warning("signal-cli stdout closed")
                    break

                try:
                    data = json.loads(line.decode().strip())
                    await self._handle_jsonrpc(data)
                except json.JSONDecodeError:
                    continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Signal receive error: {e}")
                await asyncio.sleep(1)

    async def _stderr_loop(self):
        """Log stderr from signal-cli."""
        while self._running and self._process:
            try:
                line = await self._process.stderr.readline()
                if not line:
                    break
                msg = line.decode().strip()
                if msg:
                    self.logger.debug(f"signal-cli: {msg}")
            except Exception:
                break

    async def _handle_jsonrpc(self, data: dict):
        """Handle JSON-RPC message from signal-cli."""
        # Check if it's a receive notification
        method = data.get("method", "")

        if method == "receive":
            params = data.get("params", {})
            envelope = params.get("envelope", {})

            # Get source phone number
            source = envelope.get("source", "") or envelope.get("sourceNumber", "")

            # Check for data message
            if "dataMessage" not in envelope:
                return

            data_msg = envelope["dataMessage"]
            text = data_msg.get("message", "")
            timestamp = envelope.get("timestamp", 0)

            if not text or not source:
                return

            message = IncomingMessage(
                platform="signal",
                user_id=source,
                chat_id=source,  # In Signal, chat_id is the phone number
                text=text,
                timestamp=timestamp / 1000 if timestamp > 1e12 else timestamp,
                raw=data,
                metadata={
                    "timestamp": timestamp
                }
            )

            await self.handle_message(message)

    async def _send_jsonrpc(self, method: str, params: dict) -> Optional[dict]:
        """Send JSON-RPC request to signal-cli."""
        if not self._process or not self._process.stdin:
            return None

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": str(datetime.now().timestamp())
        }

        try:
            line = json.dumps(request) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()

            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=30
            )

            if response_line:
                return json.loads(response_line.decode())

        except asyncio.TimeoutError:
            self.logger.warning("signal-cli request timed out")
        except Exception as e:
            self.logger.error(f"signal-cli request failed: {e}")

        return None

    async def send_message(self, chat_id: str, text: str, reply_to: Optional[str] = None):
        """Send message via Signal."""
        if not self._running:
            self.logger.warning("Cannot send - Signal adapter not running")
            return

        # Chunk if too long (Signal limit is ~2000 chars)
        max_len = 2000
        chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]

        for chunk in chunks:
            await self._send_jsonrpc("send", {
                "recipient": [chat_id],
                "message": chunk
            })
            if len(chunks) > 1:
                await asyncio.sleep(0.3)

    async def send_typing(self, chat_id: str):
        """Signal doesn't have typing indicators via CLI."""
        pass
