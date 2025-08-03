# ---- builder stage ----
FROM python:3.12-bookworm AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential python3-dev g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry and project dependencies (including building native extensions)
COPY pyproject.toml poetry.lock* /app/
RUN pip install --no-cache-dir "poetry>=1.5.0" && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

# Copy source so that compiled extensions are in place
COPY . /app

# (Optional) Pre-build / cache models if you want them baked in:
RUN mkdir -p /cache

# ---- runtime stage ----
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/cache

WORKDIR /app

# Runtime system libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python environment from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Cache directory for insightface models
RUN mkdir -p /cache

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
