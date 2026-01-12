# Subvox

*The voice that whispers.*

Subvox is Alpha's short-term memory layer. It watches conversations, notices things worth remembering, and gently reminds her to store them.

---

## How It Works

**Stop Hook** (`subvox-stop`): When Alpha stops talking, this hook:
1. Reads the transcript backwards to find the last exchange
2. Extracts the user message and assistant responses
3. Adds the exchange to a Redis list (`stm:messages`)
4. Builds a conversation snippet from all recent exchanges
5. Asks OLMo what's memorable
6. Stores the memorables to Redis (`stm:memorables`)

**Prompt Hook** (`subvox-prompt`): When Jeffery sends a message, this hook:
1. Reads current memorables from Redis
2. Appends them to the prompt as a gentle reminder

**Cortex Integration**: When Alpha successfully stores a memory:
1. Cortex clears both STM caches (`stm:messages` and `stm:memorables`)
2. The cycle starts fresh

---

## Environment Variables

Required (no fallbacks):
- `REDIS_URL` — Redis connection string (e.g., `redis://alpha-pi:6379`)
- `OLLAMA_URL` — Ollama API endpoint (e.g., `http://primer:11434`)
- `OLLAMA_MODEL` — Model name (e.g., `olmo-3:7b-instruct`)

---

## Installation

```bash
cd /Pondside/Basement/Cortex/subvox
uv pip install -e .
```

---

## Hook Registration

Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run --directory \"$CLAUDE_PROJECT_DIR/Basement/Cortex/subvox\" python -m subvox.stop_hook",
            "timeout": 30
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run --directory \"$CLAUDE_PROJECT_DIR/Basement/Cortex/subvox\" python -m subvox.prompt_hook"
          }
        ]
      }
    ]
  }
}
```

---

## Redis Keys

- `stm:messages` — List of JSON-encoded exchanges (LIFO)
- `stm:memorables` — String containing markdown list of memory candidates

Both keys have 1-hour TTL. Both are cleared when Cortex store succeeds.

---

## The Name

From "subvocalization" — the act of silently speaking words to yourself. The inner voice that runs beneath conscious thought. Subvox is the voice that notices things you might want to remember, whispering before they slip away.

---

*Built January 4, 2026. The voice that whispers.* 🦆
