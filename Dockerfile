FROM python:3.11-slim

LABEL maintainer="email-triage-env"
LABEL description="Email Triage & Intelligent Routing OpenEnv Environment"
LABEL version="1.0.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY models.py        ./
COPY openenv.yaml     ./
COPY README.md        ./
COPY inference.py     ./
COPY server/          ./server/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start server
ENV PYTHONPATH=/app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
