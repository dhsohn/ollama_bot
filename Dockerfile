# Stage 1: 의존성 빌드
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: 런타임
FROM python:3.11-slim

RUN groupadd -r botuser && useradd -r -g botuser -d /app botuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY core/ core/
COPY skills/ skills/
COPY auto/ auto/
COPY config/ config/
COPY scripts/ scripts/
COPY main.py .

RUN mkdir -p /app/data/conversations /app/data/memory /app/data/logs /app/data/reports \
    && chmod +x /app/scripts/*.sh \
    && chown -R botuser:botuser /app/data

USER botuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["bash", "scripts/healthcheck.sh"]

ENTRYPOINT ["python", "main.py"]
