# Beat Butler worker: FastAPI + librosa.
#
# Build from the service directory, not the repo root, so the image only carries
# the worker and nothing from the web app:
#
#   docker build -f services/analyze/Dockerfile services/analyze -t beat-butler-worker
#   docker run --rm -p 8010:8000 --env-file services/analyze/.env beat-butler-worker
#
# Runtime flags mirror DEPLOYMENT.md section 2. Two of them are load-bearing:
# --no-proxy-headers (uvicorn otherwise trusts X-Forwarded-For from 127.0.0.1 and
# lets a caller choose its own rate-limit bucket) and one process (the limiter
# lives in memory, so --workers 4 would mean four independent buckets).
#
# libsndfile1 decodes WAV/FLAC/AIFF via soundfile; ffmpeg covers the compressed
# formats librosa falls back to (m4a, some mp3s). Without them the worker boots
# fine and then answers 500 on the first real upload.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first: the dependency layer survives source edits.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app

# Non-root, with a real home so librosa/numba can write their caches. Uploads go
# to tempfile.gettempdir(), which stays writable by this user.
RUN useradd --create-home --uid 10001 butler \
    && chown -R butler:butler /app
USER butler

# 8000 inside the container; publish whatever you like on the host.
ENV PORT=8000
EXPOSE 8000

# No curl in the image, so probe with the interpreter that is already there.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, sys, urllib.request; \
port = os.environ.get('PORT', '8000'); \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4).status == 200 else 1)"

# PORT and UVICORN_FLAGS are honoured so a platform that injects its own port or
# wants --proxy-headers behind its own load balancer can say so without a rebuild.
# exec keeps uvicorn as PID 1, so SIGTERM stops the worker instead of a shell.
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" ${UVICORN_FLAGS:---no-proxy-headers}"]
