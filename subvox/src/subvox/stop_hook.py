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
import redis

# Initialize OTel before other imports that might be instrumented
from .otel import init_otel
init_otel()

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
from .otel import finish_llm_span, llm_span


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
        print(f"[Subvox] Transcript file not found: {transcript_path}", file=sys.stderr)
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
        print("[Subvox] No user message found in transcript", file=sys.stderr)
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

    # Call Ollama API with LLM span for observability
    with llm_span(OLLAMA_MODEL, prompt, operation="memorables") as span:
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

        # Add response attributes to span
        finish_llm_span(
            span,
            response=result,
            eval_count=data.get("eval_count"),
            prompt_eval_count=data.get("prompt_eval_count"),
        )

    print(
        f"[Subvox] OLMo: eval_count={data.get('eval_count')}, "
        f"prompt_eval_count={data.get('prompt_eval_count')}, "
        f"duration_ms={data.get('total_duration', 0) / 1_000_000:.0f}",
        file=sys.stderr
    )

    return result


def main():
    """Main entry point for the stop hook."""
    # Read input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"[Subvox] Invalid JSON input: {e}", file=sys.stderr)
        return

    transcript_path = input_data.get("transcript_path")
    if not transcript_path:
        print("[Subvox] No transcript_path provided", file=sys.stderr)
        return

    # Parse transcript to get the last exchange
    exchange = parse_transcript_backwards(transcript_path)
    if not exchange:
        return

    print(
        f"[Subvox] Exchange: user={len(exchange['user_message'])} chars, "
        f"assistant={len(exchange['assistant_messages'])} msgs",
        file=sys.stderr
    )

    # Connect to Redis
    try:
        r = redis.from_url(REDIS_URL)
    except redis.RedisError as e:
        print(f"[Subvox] Redis connection failed: {e}", file=sys.stderr)
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
        print(f"[Subvox] Prompt template not found: {PROMPT_FILE}", file=sys.stderr)
        return

    # Ask OLMo
    try:
        new_memorables = ask_olmo(conversation, existing_memorables, prompt_template)
    except Exception as e:
        print(f"[Subvox] OLMo call failed: {e}", file=sys.stderr)
        return

    # Parse memorables from OLMo's response
    parsed_memorables = parse_memorables(new_memorables)

    print(
        f"[Subvox] Parsed {len(parsed_memorables)} memorables from response",
        file=sys.stderr
    )

    if parsed_memorables:
        # Found memorable content! Clear the buffer and store memorables.
        r.delete(STM_MESSAGES_KEY)
        r.set(STM_MEMORABLES_KEY, new_memorables, ex=STM_TTL)
        print(f"[Subvox] Memorables stored, buffer cleared", file=sys.stderr)
    else:
        # Nothing memorable yet - add exchange to buffer, keep accumulating
        r.lpush(STM_MESSAGES_KEY, json.dumps(exchange))
        r.expire(STM_MESSAGES_KEY, STM_TTL)
        # Clear old memorables since we're still accumulating
        r.delete(STM_MEMORABLES_KEY)
        print("[Subvox] No memorables - exchange added to buffer", file=sys.stderr)


if __name__ == "__main__":
    main()
