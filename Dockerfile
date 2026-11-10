FROM python:3.11-slim

WORKDIR /app

# Install lightweight python dependencies (only Flask and numpy)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Start production server using shell expansion for $PORT (Render sets $PORT dynamically)
CMD gunicorn --workers=2 --threads=4 --bind 0.0.0.0:${PORT:-5000} app:app
