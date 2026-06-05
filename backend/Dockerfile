# ── Backend Dockerfile — FastAPI + ML Pipeline ──────────────
FROM python:3.11-slim

# System dependencies for psycopg2, audio processing, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose port (Render sets $PORT dynamically)
EXPOSE 8000

# Run migrations then start server
CMD ["./start.sh"]
