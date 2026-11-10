FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (CPU-only PyTorch for fast build and low memory)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY . .

# Start production server using shell expansion for $PORT (Render sets $PORT dynamically)
CMD gunicorn --workers=2 --threads=4 --bind 0.0.0.0:${PORT:-5000} app:app
