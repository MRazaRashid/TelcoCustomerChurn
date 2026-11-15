
import pandas as pd
from fastapi.testclient import TestClient

from src.app.main import app
from src.serving.inference import FEATURE_COLS
from src.serving.preprocessing import transform_customer


client = TestClient(app)

# Example customer for automated testing
SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "No",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 20.0,
    "TotalCharges": 40.0,
}


def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_gradio_is_mounted():
    paths = [route.path for route in app.routes]
    assert "/ui" in paths


def test_model_has_25_features():
    assert len(FEATURE_COLS) == 25
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_preprocessing():
    df = pd.DataFrame([SAMPLE_CUSTOMER])

    transformed = transform_customer(
        df,
        FEATURE_COLS
    )

    assert transformed.shape == (1, 25)
    assert transformed.columns.tolist() == FEATURE_COLS
    assert transformed["SeniorCitizen"].iloc[0] == 1

    assert transformed["No_phone_service"].iloc[0] == 1
    assert transformed["No_internet_service"].iloc[0] == 1


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json=SAMPLE_CUSTOMER
    )

    assert response.status_code == 200, response.text

    result = response.json()

    assert result["prediction"] in [
        "Likely to churn",
        "Not likely to churn"
    ]


def test_invalid_customer():
    invalid_customer = SAMPLE_CUSTOMER.copy()
    invalid_customer["SeniorCitizen"] = 3

    response = client.post(
        "/predict",
        json=invalid_customer
    )

    assert response.status_code == 422
