# ---- Build Stage ----
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip install --no-cache-dir dist/*.whl

# ---- Runtime Stage ----
FROM python:3.11-slim

# Install OpenCV system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src/approvavisa_engine/data /app/data

EXPOSE 8000

CMD ["uvicorn", "approvavisa_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
