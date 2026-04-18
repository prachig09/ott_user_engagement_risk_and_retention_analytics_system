FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any are needed for SHAP or Scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Force a clean, no-cache install of the specific versions
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# If this fails, the build stops and tells you why.
RUN ls -R /app/model/

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONPATH=/app

# Use -u to see logs immediately in the terminal
CMD ["python", "run_app.py"]