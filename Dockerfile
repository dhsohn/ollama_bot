# Stage 1: 의존성 빌드
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock

# Stage 2: 런타임
FROM python:3.11-slim

RUN groupadd -r botuser && useradd -r -g botuser -d /app botuser

WORKDIR /app

# HuggingFace 모델 캐시 디렉터리
ENV HF_HOME=/app/data/hf_cache
ENV FASTEMBED_CACHE_PATH=/app/data/hf_cache/fastembed

COPY --from=builder /usr/local /usr/local

COPY core/ core/
COPY apps/ apps/
COPY packages/ packages/
COPY skills/ skills/
COPY auto/ auto/
COPY config/ config/
COPY scripts/ scripts/
COPY main.py .

RUN mkdir -p /app/data/conversations /app/data/memory /app/data/logs /app/data/reports \
    /app/data/hf_cache/fastembed \
    && chmod +x /app/scripts/*.sh \
    && chown -R botuser:botuser /app/data

USER botuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["bash", "scripts/healthcheck.sh"]

ENTRYPOINT ["python", "main.py"]
