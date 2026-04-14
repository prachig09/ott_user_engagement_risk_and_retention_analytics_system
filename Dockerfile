# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --default-timeout=100 \
    --retries 5 \
    -r requirements.txt
# Copy entire project
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "-m", "main"]