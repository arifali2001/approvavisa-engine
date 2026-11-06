# ApprovaVisa Engine: production image (Coolify or any Docker host).
# Runtime env: API_KEY (required; the sites' PASSPORT_PHOTO_API_KEY), optional WORKERS (default 3),
# and any other setting in src/approvavisa_engine/config.py.

# ---- Build Stage ----
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app
COPY pyproject.toml constraints.txt README.md ./
COPY src/ src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip install --no-cache-dir -c constraints.txt dist/*.whl

# ---- Runtime Stage ----
FROM python:3.11-slim-bookworm

# OpenCV / MediaPipe system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Bake the background-removal model into the image so a fresh container never downloads it at
# the first customer request. (The MediaPipe models ship inside the package: data/models.)
ENV U2NET_HOME=/app/.u2net \
    PYTHONUNBUFFERED=1 \
    WORKERS=3
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=5).status == 200 else 1)"

# Each worker loads its own models (~0.5–1 GB RAM). 3 workers suit a 4-CPU / 8 GB VPS shared with
# the websites; raise WORKERS if the box is dedicated to the engine.
CMD ["sh", "-c", "exec uvicorn approvavisa_engine.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS} --proxy-headers --forwarded-allow-ips='*'"]
