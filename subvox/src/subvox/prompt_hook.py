#!/usr/bin/env python3
"""UserPromptSubmit hook - appends memorables to context.

Reads current memorables from Redis and outputs them as a gentle reminder.
"""

import json
import sys

import logfire
import redis

from .config import REDIS_URL, STM_MEMORABLES_KEY

# Configure logfire
logfire.configure(
    service_name="subvox-prompt",
    send_to_logfire="if-token-present",
)


def main():
    """Main entry point for the prompt hook."""
    with logfire.span("subvox-prompt-hook"):
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
            logfire.error("Redis error", error=str(e))
            # Don't block the prompt on Redis errors
            return

        if memorables:
            memorables_str = memorables.decode("utf-8") if isinstance(memorables, bytes) else memorables

            # Only output if there's actual content
            if memorables_str.strip():
                logfire.info("Appending memorables to context", length=len(memorables_str))

                # Output to stdout - this gets added to context
                print(f"""
<subvox>
Use Cortex. Suggested memories to store:
{memorables_str}
</subvox>
""")
            else:
                logfire.info("Memorables key exists but is empty")
        else:
            logfire.info("No memorables in STM")


if __name__ == "__main__":
    main()
