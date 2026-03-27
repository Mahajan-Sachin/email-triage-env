FROM ghcr.io/meta-pytorch/openenv-base:latest

LABEL maintainer="email-triage-env"
LABEL description="Email Triage & Intelligent Routing OpenEnv Environment"
LABEL version="1.0.0"

WORKDIR /app

# Copy dependency manifests first (maximise layer cache hits)
COPY requirements.txt ./

# Install dependencies — prefer uv (ships with openenv-base), fall back to pip
RUN if command -v uv > /dev/null 2>&1; then \
        uv pip install --system --no-cache -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application source
COPY models.py        ./
COPY openenv.yaml     ./
COPY README.md        ./
COPY inference.py     ./
COPY server/          ./server/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start server — sys.path includes /app so `from models import ...` works
ENV PYTHONPATH=/app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]