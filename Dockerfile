FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (CPU-only PyTorch for small image size)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY . .

# Set environment
ENV PORT=5000
EXPOSE 5000

# Start production server
CMD ["gunicorn", "--workers=2", "--threads=4", "--bind=0.0.0.0:5000", "app:app"]
