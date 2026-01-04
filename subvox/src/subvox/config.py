"""Configuration for Subvox.

All required environment variables - no fallbacks, fail fast.
"""

import os
import sys
from pathlib import Path

# Required environment variables
REDIS_URL = os.environ.get("REDIS_URL")
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

# Validate on import
_missing = []
if not REDIS_URL:
    _missing.append("REDIS_URL")
if not OLLAMA_URL:
    _missing.append("OLLAMA_URL")
if not OLLAMA_MODEL:
    _missing.append("OLLAMA_MODEL")

if _missing:
    print(f"Subvox: Missing required environment variables: {', '.join(_missing)}", file=sys.stderr)
    sys.exit(2)

# Redis keys
STM_MESSAGES_KEY = "stm:messages"
STM_MEMORABLES_KEY = "stm:memorables"

# TTL in seconds (1 hour)
STM_TTL = 3600

# OLMo settings
OLLAMA_CONTEXT = 24576  # 24K tokens

# Prompt template location
PROMPT_FILE = Path(__file__).parent.parent.parent / "prompt.md"
