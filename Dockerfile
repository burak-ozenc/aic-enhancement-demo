FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Models downloaded at runtime — just ensure directory exists
RUN mkdir -p models

EXPOSE 7860

CMD ["gunicorn", "app:server", "-b", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]