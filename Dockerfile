# Stage 1: Builder
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Create isolated virtual environment in /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install all dependencies from requirements.txt (using PyTorch CPU wheel index for lightweight download)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Runner
FROM python:3.11-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

WORKDIR /app

# Copy virtual environment dependencies from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code, tests, configuration, and trained model artifacts
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY config.yaml /app/config.yaml
COPY train.py /app/train.py
COPY model.pth /app/model.pth
COPY scaler.joblib /app/scaler.joblib
COPY anomaly_threshold.npy /app/anomaly_threshold.npy

# Expose API port
EXPOSE 8000

# Container entrypoint command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
