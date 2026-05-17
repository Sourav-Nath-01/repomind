FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use slim API-only requirements (no torch/GPU/fine-tuning packages)
# This keeps the image ~2GB instead of ~8GB
COPY requirements-api.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements-api.txt

# Copy project code
COPY . .

# Expose API port (HuggingFace Spaces uses 7860)
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
