# Production Dockerfile for Cognify
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r cognify && useradd -r -g cognify cognify

# Set work directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir --upgrade pip poetry

# Copy poetry files and README
COPY pyproject.toml poetry.lock* README.md ./

# Configure poetry
RUN poetry config virtualenvs.create false

# Install dependencies (without installing the current project)
RUN poetry install --only=main --no-root

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs && \
    chown -R cognify:cognify /app

# Switch to non-root user
USER cognify

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
