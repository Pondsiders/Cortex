#!/usr/bin/env python3
"""Stop hook - extracts recent exchange and asks OLMo what's memorable.

This is the heart of Subvox. When Alpha stops talking:
1. Read the transcript backwards to find the last user message
2. Extract that exchange (user message + assistant responses)
3. Add exchange to Redis STM messages list (accumulate)
4. Build conversation from all accumulated STM messages
5. Ask OLMo what's memorable (passing existing memorables for context)
6. Store OLMo's response as the new memorables (accumulate/update)

IMPORTANT: This hook only accumulates. It NEVER flushes the buffers.
Cortex handles flushing on successful store (see cortex/main.py).
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

    Finds the last user message with ACTUAL TEXT (not just tool_results),
    then collects all subsequent assistant TEXT responses (not tool_use).

    Returns a dict with:
        - user_message: str (the user's message)
        - assistant_messages: list[str] (all assistant text responses)
    """
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("parse_transcript_backwards") as span:
        path = Path(transcript_path)
        span.set_attribute("transcript_path", str(path))

        if not path.exists():
            span.set_attribute("error", "file_not_found")
            print(f"[Subvox] Transcript file not found: {transcript_path}", file=sys.stderr)
            return None

        # Read all lines
        with open(path) as f:
            lines = f.readlines()

        span.set_attribute("total_lines", len(lines))

        # Parse backwards to find last user message WITH TEXT
        user_message = None
        user_index = None
        entries_scanned = 0
        user_entries_seen = 0
        user_entries_skipped_tool_only = 0

        for i in range(len(lines) - 1, -1, -1):
            try:
                entry = json.loads(lines[i])
            except json.JSONDecodeError:
                continue

            entries_scanned += 1
            entry_type = entry.get("type")

            if entry_type == "user":
                user_entries_seen += 1
                msg = entry.get("message", {})
                content = msg.get("content")

                # Extract text content, skipping tool_results
                extracted_text = ""
                content_types_seen = []

                if isinstance(content, str):
                    extracted_text = content
                    content_types_seen.append("string")
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            item_type = item.get("type", "unknown")
                            content_types_seen.append(item_type)
                            if item_type == "text":
                                text_parts.append(item.get("text", ""))
                            # Skip tool_result entries
                        elif isinstance(item, str):
                            text_parts.append(item)
                            content_types_seen.append("bare_string")
                    extracted_text = "\n".join(text_parts)

                # Log what we found in this user entry
                span.add_event("user_entry_examined", {
                    "line_index": i,
                    "content_types": ",".join(content_types_seen),
                    "extracted_text_len": len(extracted_text),
                    "extracted_text_preview": extracted_text[:100] if extracted_text else "(empty)",
                })

                # Only accept if we got actual text
                if extracted_text.strip():
                    user_message = extracted_text
                    user_index = i
                    break
                else:
                    user_entries_skipped_tool_only += 1
                    print(f"[Subvox] Skipping user entry at line {i} (tool_results only, types: {content_types_seen})", file=sys.stderr)

        span.set_attribute("entries_scanned", entries_scanned)
        span.set_attribute("user_entries_seen", user_entries_seen)
        span.set_attribute("user_entries_skipped_tool_only", user_entries_skipped_tool_only)

        if user_message is None:
            span.set_attribute("error", "no_user_text_found")
            print("[Subvox] No user message with text found in transcript", file=sys.stderr)
            return None

        span.set_attribute("user_message_line_index", user_index)
        span.set_attribute("user_message_len", len(user_message))

        # Now collect assistant TEXT messages AFTER this user message
        assistant_messages = []
        assistant_entries_seen = 0
        assistant_text_blocks_found = 0

        for i in range(user_index + 1, len(lines)):
            try:
                entry = json.loads(lines[i])
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")

            if entry_type == "assistant":
                assistant_entries_seen += 1
                msg = entry.get("message", {})
                content = msg.get("content", [])

                # Extract only text blocks, skip tool_use
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            assistant_messages.append(text)
                            assistant_text_blocks_found += 1

        span.set_attribute("assistant_entries_seen", assistant_entries_seen)
        span.set_attribute("assistant_text_blocks_found", assistant_text_blocks_found)
        span.set_attribute("assistant_messages_count", len(assistant_messages))

        result = {
            "user_message": user_message,
            "assistant_messages": assistant_messages,
        }

        print(
            f"[Subvox] Parsed: scanned={entries_scanned}, user_seen={user_entries_seen}, "
            f"user_skipped={user_entries_skipped_tool_only}, assistant_texts={len(assistant_messages)}",
            file=sys.stderr
        )

        return result


def format_exchange(exchange: dict) -> str:
    """Format an exchange for the conversation snippet."""
    parts = [f"Jeffery: {exchange['user_message']}"]

    for msg in exchange["assistant_messages"]:
        parts.append(f"Alpha: {msg}")

    return "\n\n".join(parts)


def build_conversation_from_stm(r: redis.Redis) -> str:
    """Build full conversation from all STM messages."""
    # Get all messages from Redis (lpush adds to head, so reverse for chronological order)
    existing = r.lrange(STM_MESSAGES_KEY, 0, -1)

    parts = []
    for msg_bytes in reversed(existing):  # oldest first
        try:
            exchange = json.loads(msg_bytes)
            parts.append(format_exchange(exchange))
        except json.JSONDecodeError:
            continue

    return "\n\n---\n\n".join(parts)


def ask_olmo(conversation: str, existing_memorables: str, prompt_template: str) -> str:
    """Ask OLMo what's memorable in this conversation."""
    # Build the full prompt
    prompt = f"""{prompt_template}

<conversation>
{conversation}
</conversation>

<memorables>
{existing_memorables}
</memorables>
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
                "keep_alive": "1h",  # Keep model hot between calls
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

    # ACCUMULATE: Always add current exchange to the buffer first
    r.lpush(STM_MESSAGES_KEY, json.dumps(exchange))
    r.expire(STM_MESSAGES_KEY, STM_TTL)

    # Build conversation from ALL accumulated STM messages (including the one we just added)
    conversation = build_conversation_from_stm(r)

    # Get existing memorables (what OLMo found in previous iterations)
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

    # Ask OLMo what's memorable in the accumulated conversation
    try:
        new_memorables = ask_olmo(conversation, existing_memorables, prompt_template)
    except Exception as e:
        print(f"[Subvox] OLMo call failed: {e}", file=sys.stderr)
        return

    # Parse memorables from OLMo's response
    parsed_memorables = parse_memorables(new_memorables)

    # Get buffer size for logging
    buffer_size = r.llen(STM_MESSAGES_KEY)

    print(
        f"[Subvox] buffer_size={buffer_size}, parsed_memorables={len(parsed_memorables)}",
        file=sys.stderr
    )

    # ACCUMULATE: Store just the parsed memorables (clean bullet points)
    # NOT the full OLMo response (which includes <reasoning> tags)
    # This keeps <memory-candidates> clean for the next iteration
    # Cortex will flush both buffers when Alpha actually stores a memory
    if parsed_memorables:
        clean_memorables = "\n".join(f"- {m}" for m in parsed_memorables)
        r.set(STM_MEMORABLES_KEY, clean_memorables, ex=STM_TTL)
        print(f"[Subvox] Memorables updated ({len(parsed_memorables)} items)", file=sys.stderr)
    else:
        print("[Subvox] No memorables parsed from OLMo response", file=sys.stderr)


if __name__ == "__main__":
    main()
