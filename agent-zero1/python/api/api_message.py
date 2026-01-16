import base64
import os
from datetime import datetime, timedelta
from agent import AgentContext, UserMessage, AgentContextType
from python.helpers.api import ApiHandler, Request, Response
from python.helpers import files
from python.helpers.print_style import PrintStyle
from werkzeug.utils import secure_filename
from initialize import initialize_agent
import threading


class ApiMessage(ApiHandler):
    # Track chat lifetimes for cleanup
    _chat_lifetimes = {}
    _cleanup_lock = threading.Lock()

    @classmethod
    def requires_auth(cls) -> bool:
        return False  # No web auth required

    @classmethod
    def requires_csrf(cls) -> bool:
        return False  # No CSRF required

    @classmethod
    def requires_api_key(cls) -> bool:
        return False  # Disabled for internal Docker network access

    async def process(self, input: dict, request: Request) -> dict | Response:
        # Extract parameters
        context_id = input.get("context_id", "")
        message = input.get("message", "")
        attachments = input.get("attachments", [])
        lifetime_hours = input.get("lifetime_hours", 24)  # Default 24 hours

        if not message:
            return Response('{"error": "Message is required"}', status=400, mimetype="application/json")

        # Handle attachments (base64 encoded)
        attachment_paths = []
        if attachments:
            upload_folder_int = "/a0/tmp/uploads"
            upload_folder_ext = files.get_abs_path("tmp/uploads")
            os.makedirs(upload_folder_ext, exist_ok=True)

            for attachment in attachments:
                if not isinstance(attachment, dict) or "filename" not in attachment or "base64" not in attachment:
                    continue

                try:
                    filename = secure_filename(attachment["filename"])
                    if not filename:
                        continue

                    # Decode base64 content
                    file_content = base64.b64decode(attachment["base64"])

                    # Save to temp file
                    save_path = os.path.join(upload_folder_ext, filename)
                    with open(save_path, "wb") as f:
                        f.write(file_content)

                    attachment_paths.append(os.path.join(upload_folder_int, filename))
                except Exception as e:
                    PrintStyle.error(f"Failed to process attachment {attachment.get('filename', 'unknown')}: {e}")
                    continue

        # Get or create context
        if context_id:
            context = AgentContext.use(context_id)
            if not context:
                return Response('{"error": "Context not found"}', status=404, mimetype="application/json")
        else:
            config = initialize_agent()
            context = AgentContext(config=config, type=AgentContextType.USER)
            AgentContext.use(context.id)
            context_id = context.id

        # Update chat lifetime
        with self._cleanup_lock:
            self._chat_lifetimes[context_id] = datetime.now() + timedelta(hours=lifetime_hours)

        # Process message
        try:
            import asyncio

            # Log the message
            attachment_filenames = [os.path.basename(path) for path in attachment_paths] if attachment_paths else []

            PrintStyle(
                background_color="#6C3483", font_color="white", bold=True, padding=True
            ).print("External API message:")
            PrintStyle(font_color="white", padding=False).print(f"> {message}")
            if attachment_filenames:
                PrintStyle(font_color="white", padding=False).print("Attachments:")
                for filename in attachment_filenames:
                    PrintStyle(font_color="white", padding=False).print(f"- {filename}")

            # Add user message to chat history so it's visible in the UI
            context.log.log(
                type="user",
                heading="User message",
                content=message,
                kvps={"attachments": attachment_filenames},
            )

            # Send message to agent with timeout
            task = context.communicate(UserMessage(message, attachment_paths))
            try:
                result = await asyncio.wait_for(task.result(), timeout=60.0)
            except asyncio.TimeoutError:
                # On timeout, try to extract any response that was generated
                PrintStyle.error("API timeout - extracting partial response")
                result = self._extract_response_from_context(context)
                if result:
                    return {
                        "context_id": context_id,
                        "response": result
                    }
                return Response('{"error": "Request timeout - no response generated"}', status=504, mimetype="application/json")

            # Clean up expired chats
            self._cleanup_expired_chats()

            return {
                "context_id": context_id,
                "response": result
            }

        except Exception as e:
            PrintStyle.error(f"External API error: {e}")
            # Try to extract response from context log even if there was an error
            extracted = self._extract_response_from_context(context)
            if extracted:
                return {
                    "context_id": context_id,
                    "response": extracted
                }
            return Response(f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json")

    def _extract_response_from_context(self, context) -> str | None:
        """Extract the last response from context logs"""
        try:
            import re
            import json

            # Try to get from agent0's history (most reliable)
            agent = context.agent0 if hasattr(context, 'agent0') else None
            if agent and hasattr(agent, 'history') and agent.history:
                history = agent.history
                # Check current topic messages
                if hasattr(history, 'current') and history.current:
                    for msg in reversed(history.current.messages):
                        if msg.ai:  # AI message
                            content = msg.content
                            extracted = self._extract_message_from_content(content)
                            if extracted:
                                PrintStyle().print(f"Extracted from current topic: {extracted[:100]}...")
                                return extracted

                # Check historical topics
                if hasattr(history, 'topics'):
                    for topic in reversed(history.topics):
                        for msg in reversed(topic.messages):
                            if msg.ai:
                                content = msg.content
                                extracted = self._extract_message_from_content(content)
                                if extracted:
                                    PrintStyle().print(f"Extracted from topic: {extracted[:100]}...")
                                    return extracted

            # Fallback: Search context logs
            for entry in reversed(context.log.logs):
                content = entry.get("content", "")
                if not content:
                    continue

                # Skip system/error messages
                entry_type = entry.get("type", "")
                if entry_type in ["error", "warning"]:
                    continue

                if isinstance(content, str):
                    extracted = self._extract_message_from_content(content)
                    if extracted:
                        PrintStyle().print(f"Extracted from log: {extracted[:100]}...")
                        return extracted

            # Debug: print what we have
            PrintStyle().print(f"Extraction failed. Log count: {len(context.log.logs) if hasattr(context.log, 'logs') else 'N/A'}")
            if agent and hasattr(agent, 'history') and agent.history:
                h = agent.history
                msg_count = len(h.current.messages) if hasattr(h, 'current') and h.current else 0
                PrintStyle().print(f"History current topic messages: {msg_count}")
                if msg_count > 0:
                    for i, msg in enumerate(h.current.messages[-3:]):
                        content_preview = str(msg.content)[:150] if msg.content else "None"
                        PrintStyle().print(f"Msg {i}: ai={msg.ai}, content={content_preview}...")

            return None
        except Exception as e:
            PrintStyle.error(f"Failed to extract response: {e}")
            import traceback
            PrintStyle.error(traceback.format_exc())
            return None

    def _extract_message_from_content(self, content) -> str | None:
        """Extract message from various content formats"""
        import re
        import json

        if not content:
            return None

        # If content is a dict, try to get from tool_args
        if isinstance(content, dict):
            # Check for assistant content with tool_args
            inner = content.get("content", content)
            if isinstance(inner, str):
                content = inner
            elif isinstance(inner, dict):
                # Try tool_args.text or tool_args.message
                tool_args = inner.get("tool_args", {})
                if isinstance(tool_args, dict):
                    msg = tool_args.get("text") or tool_args.get("message")
                    if msg and isinstance(msg, str) and len(msg) > 10:
                        return self._unescape_string(msg)
                content = str(inner)

        if not isinstance(content, str):
            content = str(content)

        # Try to parse as JSON and extract tool_args
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                tool_args = data.get("tool_args", {})
                if isinstance(tool_args, dict):
                    # Look for text (response tool) or message (a2a_chat tool)
                    msg = tool_args.get("text") or tool_args.get("message")
                    if msg and isinstance(msg, str) and len(msg) > 10:
                        return self._unescape_string(msg)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try regex extraction for "text" field (response tool)
        match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]', content, re.DOTALL)
        if match:
            msg = match.group(1)
            if len(msg) > 10:
                return self._unescape_string(msg)

        # Try regex extraction for "message" field (a2a_chat or direct message)
        match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]', content, re.DOTALL)
        if match:
            msg = match.group(1)
            if len(msg) > 10:
                return self._unescape_string(msg)

        # Check if content itself is a reasonable response (for plain text responses)
        if len(content) > 20 and len(content) < 3000:
            # Skip if it looks like JSON structure
            if content.strip().startswith('{') or '"tool_name"' in content:
                return None
            # Look for Alfred-like responses
            if any(phrase in content.lower() for phrase in ["greetings", "sir", "alfred", "steward", "assist", "may i", "how can", "good evening", "at your service"]):
                return content

        return None

    def _unescape_string(self, s: str) -> str:
        """Unescape JSON string escape sequences"""
        return s.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')

    @classmethod
    def _cleanup_expired_chats(cls):
        """Clean up expired chats"""
        with cls._cleanup_lock:
            now = datetime.now()
            expired_contexts = [
                context_id for context_id, expiry in cls._chat_lifetimes.items()
                if now > expiry
            ]

            for context_id in expired_contexts:
                try:
                    context = AgentContext.get(context_id)
                    if context:
                        context.reset()
                        AgentContext.remove(context_id)
                    del cls._chat_lifetimes[context_id]
                    PrintStyle().print(f"Cleaned up expired chat: {context_id}")
                except Exception as e:
                    PrintStyle.error(f"Failed to cleanup chat {context_id}: {e}")
