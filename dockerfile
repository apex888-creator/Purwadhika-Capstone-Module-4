# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HOME=/home/app \
    YOLO_CONFIG_DIR=/home/app/.config/Ultralytics \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System libs dari packages.txt (dibutuhkan OpenCV): libgl1, libglib2.0-0
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# User non-root
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Install dependency dulu (layer ini di-cache selama requirements.txt tidak berubah)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Baru copy kode + model
COPY --chown=app:app . .
USER app

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8888/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", \
    "--server.port=8888", "--server.address=0.0.0.0", "--server.headless=true"]