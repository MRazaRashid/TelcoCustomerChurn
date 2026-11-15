
from pathlib import Path
import mlflow
import os
import logging
# Prevent the previous MLflow tracing error
os.environ.setdefault(
    "MLFLOW_TRACING_ENABLED", "false"
)
# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configure local MLflow tracking
mlflow.set_tracking_uri(
    (PROJECT_ROOT / "mlruns").as_uri()
)

# Use your existing XGBoost experiment
mlflow.set_experiment("Telco Churn - XGBoost")



import gradio as gr
from fastapi import FastAPI, HTTPException

from src.app.schema import CustomerData
from src.app.ui import create_interface
from src.serving.inference import predict

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for customer churn prediction",
    version="1.0.0",
    telemetry={
        "tracing": False,
        "metrics": False,
        "logs": False,
    }
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.model_dump())
        return {"prediction": result}

    except Exception:
        logger.exception("Churn prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

# Mount Gradio application
app = gr.mount_gradio_app(
    app,
    create_interface(),
    path="/ui"
)
