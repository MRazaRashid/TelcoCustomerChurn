import os
import pandas as pd
import mlflow
from src.serving.preprocessing import transform_customer
from pathlib import Path
import mlflow.pyfunc

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = Path(
    os.getenv(
        "CHURN_MODEL_DIR",
        str(PROJECT_ROOT / "artifacts" / "model")
    )
).resolve()

FEATURE_FILE = MODEL_DIR / "feature_columns.txt"

if not (MODEL_DIR / "MLmodel").is_file():
    raise FileNotFoundError(
        f"MLflow model missing: {MODEL_DIR}"
    )

FEATURE_COLS = [
    line.strip()
    for line in FEATURE_FILE.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

model = mlflow.pyfunc.load_model(str(MODEL_DIR))

print("Model loaded:", MODEL_DIR)
print("Feature count:", len(FEATURE_COLS))


# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
# This ensures the model receives features in the expected order
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === VERIFY FEATURES AGAINST TRAINED XGBOOST MODEL ===

# Access the underlying XGBoost model from MLflow
xgb_model = model._model_impl.xgb_model

# Retrieve the actual trained feature names
MODEL_FEATURES = xgb_model.get_booster().feature_names

if not MODEL_FEATURES:
    raise ValueError(
        "No feature names found in the trained XGBoost model"
    )

print(f"Saved feature file: {len(FEATURE_COLS)} columns")
print(f"Trained model: {len(MODEL_FEATURES)} columns")

# Temporarily use the exact model feature schema
if FEATURE_COLS != MODEL_FEATURES:
    print("WARNING: Feature schema mismatch detected")
    print("Using trained XGBoost feature names")
    FEATURE_COLS = list(MODEL_FEATURES)

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)


def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = transform_customer(df, FEATURE_COLS)    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    # Validate transformed feature structure
    if list(df_enc.columns) != FEATURE_COLS:
        raise ValueError(
            "Prediction feature schema mismatch"
        )

    # Generate prediction
    preds = model.predict(df_enc)
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service