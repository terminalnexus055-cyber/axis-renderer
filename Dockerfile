FROM python:3.11-slim

# Install system dependencies including FFmpeg and fonts
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "main.py"]
