FROM ghcr.io/astral-sh/uv:0.12.13 AS uv
FROM python:3.11.15-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    POC_DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY poc ./poc
RUN uv sync --locked --no-dev --extra telemetry --extra llm

EXPOSE 8000
CMD ["uvicorn", "poc.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
