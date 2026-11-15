
FROM python:3.12-slim

WORKDIR /app

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACING_ENABLED=false

# Install dependencies
COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ /app/src/

# Copy complete deployment model
COPY artifacts/model/ /app/artifacts/model/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
