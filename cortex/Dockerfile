# Cortex - Semantic memory service for Alpha
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies and bootstrap OTel instrumentation
RUN uv pip install --system . && \
    opentelemetry-bootstrap --action=install

# Run the server with OTel instrumentation
# Set OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_RESOURCE_ATTRIBUTES env vars to enable tracing
EXPOSE 7867
CMD ["opentelemetry-instrument", "uvicorn", "cortex.main:app", "--host", "0.0.0.0", "--port", "7867"]
