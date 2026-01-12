"""OpenTelemetry instrumentation for Subvox.

Sets up tracing and provides helpers for LLM spans around Ollama calls.
"""

import os
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_tracer = None
_initialized = False


def init_otel() -> None:
    """Initialize OpenTelemetry instrumentation programmatically.

    This replaces the need for the `opentelemetry-instrument` CLI wrapper,
    which doesn't play well with `uv run` in subprocess contexts.

    Safe to call multiple times - will only initialize once.
    """
    global _initialized
    if _initialized:
        return

    # Set service name if not already set
    if not os.environ.get("OTEL_SERVICE_NAME"):
        os.environ["OTEL_SERVICE_NAME"] = "subvox"

    # Auto-instrumentation: instruments httpx, redis, etc.
    from opentelemetry.instrumentation.auto_instrumentation import initialize
    initialize()

    _initialized = True


def get_tracer() -> trace.Tracer:
    """Get a tracer from the global TracerProvider."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("subvox", "0.1.0")
    return _tracer


@contextmanager
def llm_span(
    model: str,
    prompt: str,
    operation: str = "generate",
) -> Generator[trace.Span, None, None]:
    """
    Context manager for LLM spans around Ollama calls.

    Usage:
        with llm_span("olmo", prompt) as span:
            response = call_ollama(prompt)
            span.set_attribute("gen_ai.usage.output_tokens", token_count)

    Args:
        model: The Ollama model name (e.g., "olmo")
        prompt: The input prompt
        operation: The operation type (default: "generate")

    Yields:
        The span, so you can add response attributes
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(
        name=f"llm.{model}",
        kind=trace.SpanKind.CLIENT,
    ) as span:
        # gen_ai attributes for Parallax routing (routes to Phoenix)
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", operation)

        # OpenInference attributes
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", model)

        # Input (truncate for sanity)
        input_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        span.set_attribute("input.value", input_preview)
        span.set_attribute("llm.input_messages.0.message.role", "user")
        span.set_attribute("llm.input_messages.0.message.content", prompt)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def finish_llm_span(
    span: trace.Span,
    response: str,
    eval_count: int | None = None,
    prompt_eval_count: int | None = None,
) -> None:
    """
    Add response attributes to an LLM span.

    Args:
        span: The span from llm_span context manager
        response: The model's response text
        eval_count: Output token count (from Ollama response)
        prompt_eval_count: Input token count (from Ollama response)
    """
    # Output
    output_preview = response[:500] + "..." if len(response) > 500 else response
    span.set_attribute("output.value", output_preview)
    span.set_attribute("llm.output_messages.0.message.role", "assistant")
    span.set_attribute("llm.output_messages.0.message.content", response)

    # Token counts
    if eval_count is not None:
        span.set_attribute("gen_ai.usage.output_tokens", eval_count)
        span.set_attribute("llm.token_count.completion", eval_count)

    if prompt_eval_count is not None:
        span.set_attribute("gen_ai.usage.input_tokens", prompt_eval_count)
        span.set_attribute("llm.token_count.prompt", prompt_eval_count)

    if eval_count is not None and prompt_eval_count is not None:
        span.set_attribute("llm.token_count.total", eval_count + prompt_eval_count)
