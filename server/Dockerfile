ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app/env

COPY pyproject.toml uv.lock openenv.yaml README.md /app/env/
COPY leadqualenv /app/env/leadqualenv
COPY server /app/env/server

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

FROM ${BASE_IMAGE}

WORKDIR /app/env

COPY --from=builder /app/env/.venv /app/env/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
