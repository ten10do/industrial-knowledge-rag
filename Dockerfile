FROM python:3.11.16-slim-trixie@sha256:1042b61448fef4ba92d16a8c7eb4996d027568ce64792a7877fd88511e0af7c6

ARG PIP_VERSION=26.2.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install "pip==${PIP_VERSION}" \
    && python -m pip install -r /app/backend/requirements.txt

COPY backend /app/backend
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/backend/data /app/backend/light_indexes \
        /app/backend/public_versions /app/backend/runtime_state \
    && chown -R app:app /app

USER app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
