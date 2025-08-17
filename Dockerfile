# Use official Python base image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory inside the container
WORKDIR /app

# Install system dependencies (for matplotlib & geopy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY . .

EXPOSE 8000

# Set default command to run your script
# Run FastAPI server instead of CLI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
