FROM python:3.11-slim

WORKDIR /app

# Install lightweight dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Hugging Face Spaces expects port 7860 by default
EXPOSE 7860

# Start production server with 300s timeout for large EDF uploads
# Compatible with both Render ($PORT) and Hugging Face Spaces (default 7860)
CMD gunicorn --workers=2 --threads=4 --timeout=300 --bind 0.0.0.0:${PORT:-7860} app:app
