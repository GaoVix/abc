FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ffmpeg git libnuma1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Keep this checkout's uv.lock in the build context to preserve its dependencies.
RUN uv sync --frozen --no-dev --no-cache

CMD ["bash"]
