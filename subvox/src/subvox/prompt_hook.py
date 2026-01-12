#!/usr/bin/env python3
"""UserPromptSubmit hook - appends memorables to context.

Reads current memorables from Redis and outputs them as a gentle reminder.
"""

import json
import sys

import redis

# Initialize OTel before other imports that might be instrumented
from .otel import init_otel
init_otel()

from .config import REDIS_URL, STM_MEMORABLES_KEY


def main():
    """Main entry point for the prompt hook."""
    try:
        # Read input from stdin (we don't actually need it, but consume it)
        _input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # No input or invalid JSON - that's fine for this hook
        pass

    # Connect to Redis
    try:
        r = redis.from_url(REDIS_URL)
        memorables = r.get(STM_MEMORABLES_KEY)
    except redis.RedisError as e:
        print(f"[Subvox] Redis error: {e}", file=sys.stderr)
        # Don't block the prompt on Redis errors
        return

    if memorables:
        memorables_str = memorables.decode("utf-8") if isinstance(memorables, bytes) else memorables

        # Only output if there's actual content
        if memorables_str.strip():
            print(f"[Subvox] Appending memorables to context: {len(memorables_str)} chars", file=sys.stderr)

            # Output to stdout - this gets added to context
            print(f"""
<subvox>
Alpha, memorable things have happened. Use Cortex **now** to store a memory.
{memorables_str}
</subvox>
""")
        else:
            print("[Subvox] Memorables key exists but is empty", file=sys.stderr)
    else:
        print("[Subvox] No memorables in STM", file=sys.stderr)


if __name__ == "__main__":
    main()
