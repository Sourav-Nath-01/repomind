FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose API port (HuggingFace Spaces uses 7860)
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
