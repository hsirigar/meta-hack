# Single service: FastAPI backend + built React frontend, serving precomputed demos.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock* backend/
RUN cd backend && uv sync --no-dev --no-install-project
COPY backend/ backend/
RUN cd backend && uv sync --no-dev
COPY data/demo/ data/demo/
COPY --from=web /web/dist frontend/dist
ENV DATA_DIR=/app/data/demo DEMO_MODE=1 PORT=8000
WORKDIR /app/backend
CMD ["sh", "-c", "uv run --no-dev uvicorn ghostline.api:app --host 0.0.0.0 --port ${PORT}"]
