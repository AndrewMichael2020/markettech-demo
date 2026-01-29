FROM python:3.11-slim

# Faster, quieter Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code needed by Streamlit
COPY app.py /app/app.py
COPY ai_cleaning_agent.py /app/ai_cleaning_agent.py

# Cloud Run provides PORT; Streamlit must listen on it
CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false
