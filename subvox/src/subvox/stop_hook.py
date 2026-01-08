#!/usr/bin/env python3
"""Stop hook - extracts recent exchange and asks OLMo what's memorable.

This is the heart of Subvox. When Alpha stops talking:
1. Read the transcript backwards to find the last user message
2. Extract that exchange (user message + assistant responses)
3. Add to Redis STM messages list
4. Build conversation from all STM messages
5. Ask OLMo what's memorable
6. Store memorables to Redis
"""

import json
import re
import sys
from pathlib import Path

import httpx
import logfire
import redis

from .config import (
    OLLAMA_CONTEXT,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROMPT_FILE,
    REDIS_URL,
    STM_MEMORABLES_KEY,
    STM_MESSAGES_KEY,
    STM_TTL,
)

# Configure logfire and instrument httpx
# Disable scrubbing - we need to see the actual prompts for debugging
logfire.configure(
    service_name="subvox-stop",
    send_to_logfire="if-token-present",
    scrubbing=False,
)
logfire.instrument_httpx()


def load_prompt_template() -> str:
    """Load the OLMo prompt template."""
    with open(PROMPT_FILE) as f:
        return f.read()


def parse_memorables(response: str) -> list[str]:
    """Extract memorables from OLMo's response.

    Returns a list of memorable items, or empty list if none found.
    """
    match = re.search(r'<memorables>(.*?)</memorables>', response, re.DOTALL)
    if not match:
        return []

    content = match.group(1).strip()
    if not content or content.lower() in ('', 'none', 'nothing notable', 'nothing notable.'):
        return []

    # Parse bullet points
    memorables = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('- ') or line.startswith('* '):
            memorables.append(line[2:].strip())
        elif line and not line.startswith('<'):
            memorables.append(line)

    return [m for m in memorables if m]


def parse_transcript_backwards(transcript_path: str) -> dict | None:
    """
    Read transcript JSONL backwards to extract the last exchange.

    Returns a dict with:
        - user_message: str (the user's message)
        - assistant_messages: list[str] (all assistant text responses)
    """
    path = Path(transcript_path)
    if not path.exists():
        logfire.warning("Transcript file not found", path=transcript_path)
        return None

    # Read all lines
    with open(path) as f:
        lines = f.readlines()

    # Parse backwards to find last user message
    user_message = None
    user_index = None
    assistant_messages = []

    for i in range(len(lines) - 1, -1, -1):
        try:
            entry = json.loads(lines[i])
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type")

        if entry_type == "user" and user_message is None:
            # Found the last user message
            msg = entry.get("message", {})
            content = msg.get("content")

            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                # Extract text from content array (skip tool results)
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                user_message = "\n".join(text_parts)

            user_index = i
            break

    if user_message is None:
        logfire.info("No user message found in transcript")
        return None

    # Now collect assistant messages AFTER this user message
    for i in range(user_index + 1, len(lines)):
        try:
            entry = json.loads(lines[i])
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type")

        if entry_type == "assistant":
            msg = entry.get("message", {})
            content = msg.get("content", [])

            # Extract only text blocks, skip tool_use
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        assistant_messages.append(text)

    return {
        "user_message": user_message,
        "assistant_messages": assistant_messages,
    }


def format_exchange(exchange: dict) -> str:
    """Format an exchange for the conversation snippet."""
    parts = [f"Jeffery: {exchange['user_message']}"]

    for msg in exchange["assistant_messages"]:
        parts.append(f"Alpha: {msg}")

    return "\n\n".join(parts)


def build_conversation_from_stm(r: redis.Redis, current_exchange: dict) -> str:
    """Build full conversation from all STM messages plus current exchange."""
    # Get existing messages from Redis
    existing = r.lrange(STM_MESSAGES_KEY, 0, -1)

    parts = []
    for msg_bytes in reversed(existing):  # oldest first
        try:
            exchange = json.loads(msg_bytes)
            parts.append(format_exchange(exchange))
        except json.JSONDecodeError:
            continue

    # Add current exchange
    parts.append(format_exchange(current_exchange))

    return "\n\n---\n\n".join(parts)


def ask_olmo(conversation: str, existing_memorables: str, prompt_template: str) -> str:
    """Ask OLMo what's memorable in this conversation."""
    # Build the full prompt
    prompt = f"""{prompt_template}

<conversation>
{conversation}
</conversation>

<memory-candidates>
{existing_memorables}
</memory-candidates>
"""

    with logfire.span("olmo-generate", prompt_length=len(prompt)):
        # Log the full prompt for debugging
        logfire.info("OLMo prompt", prompt=prompt)

        # Call Ollama API directly via httpx (instrumented by logfire)
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": OLLAMA_CONTEXT},
            },
            timeout=120.0,  # 2 minute timeout for cold starts
        )
        response.raise_for_status()
        data = response.json()

        result = data.get("response", "").strip()

        logfire.info(
            "OLMo response",
            response=result,
            eval_count=data.get("eval_count"),
            prompt_eval_count=data.get("prompt_eval_count"),
            total_duration_ms=data.get("total_duration", 0) / 1_000_000,
        )

    return result


def main():
    """Main entry point for the stop hook."""
    with logfire.span("subvox-stop-hook"):
        # Read input from stdin
        try:
            input_data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            logfire.error("Invalid JSON input", error=str(e))
            print(f"Subvox: Invalid JSON input: {e}", file=sys.stderr)
            return

        transcript_path = input_data.get("transcript_path")
        if not transcript_path:
            logfire.error("No transcript_path in input")
            print("Subvox: No transcript_path provided", file=sys.stderr)
            return

        # Parse transcript to get the last exchange
        exchange = parse_transcript_backwards(transcript_path)
        if not exchange:
            logfire.info("No exchange found, exiting cleanly")
            return

        logfire.info(
            "Extracted exchange",
            user_message_length=len(exchange["user_message"]),
            assistant_message_count=len(exchange["assistant_messages"]),
        )

        # Connect to Redis
        try:
            r = redis.from_url(REDIS_URL)
        except redis.RedisError as e:
            logfire.error("Redis connection failed", error=str(e))
            print(f"Subvox: Redis connection failed: {e}", file=sys.stderr)
            return

        # Build conversation from STM + current exchange
        conversation = build_conversation_from_stm(r, exchange)

        # Get existing memorables
        existing_memorables = ""
        memorables_bytes = r.get(STM_MEMORABLES_KEY)
        if memorables_bytes:
            existing_memorables = memorables_bytes.decode("utf-8") if isinstance(memorables_bytes, bytes) else memorables_bytes

        # Load prompt template
        try:
            prompt_template = load_prompt_template()
        except FileNotFoundError:
            logfire.error("Prompt template not found", path=str(PROMPT_FILE))
            print(f"Subvox: Prompt template not found: {PROMPT_FILE}", file=sys.stderr)
            return

        # Ask OLMo
        try:
            new_memorables = ask_olmo(conversation, existing_memorables, prompt_template)
        except Exception as e:
            logfire.error("OLMo call failed", error=str(e))
            print(f"Subvox: OLMo call failed: {e}", file=sys.stderr)
            return

        # Parse memorables from OLMo's response
        parsed_memorables = parse_memorables(new_memorables)

        logfire.info(
            "OLMo returned memorables",
            response_length=len(new_memorables),
            parsed_count=len(parsed_memorables),
        )

        if parsed_memorables:
            # Found memorable content! Clear the buffer and store memorables.
            r.delete(STM_MESSAGES_KEY)
            r.set(STM_MEMORABLES_KEY, new_memorables, ex=STM_TTL)
            logfire.info(
                "Memorables found - buffer cleared",
                memorable_count=len(parsed_memorables),
            )
        else:
            # Nothing memorable yet - add exchange to buffer, keep accumulating
            r.lpush(STM_MESSAGES_KEY, json.dumps(exchange))
            r.expire(STM_MESSAGES_KEY, STM_TTL)
            # Clear old memorables since we're still accumulating
            r.delete(STM_MEMORABLES_KEY)
            logfire.info("No memorables - exchange added to buffer")


if __name__ == "__main__":
    main()
