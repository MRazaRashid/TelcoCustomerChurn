import gradio as gr
from src.app.schema import CustomerData
from src.serving.inference import predict

def gradio_interface(
    gender, Partner, Dependents,SeniorCitizen,PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    
    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string
    
    """
    # Construct data dictionary matching CustomerData schema
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "SeniorCitizen": {"Yes": 1, "No": 0}[SeniorCitizen],
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),              # Ensure integer type
        "MonthlyCharges": float(MonthlyCharges),  # Ensure float type
        "TotalCharges": float(TotalCharges),      # Ensure float type
    }
        
    # Call same inference pipeline as API endpoint
    customer = CustomerData.model_validate(data)
    result = predict(customer.model_dump())     
    return str(result)  # Return as string for Gradio display

def create_interface():
    demo = gr.Interface(
        fn=gradio_interface,
        inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Dropdown(["Yes", "No"],label="Senior Citizen"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(
            ["Yes", "No", "No phone service"],
            label="Multiple Lines"
        ),
        gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            label="Internet Service"
        ),
        *[
            gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label=label
            )
            for label in [
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies"
            ]
        ],
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract"
        ),
        gr.Dropdown(
            ["Yes", "No"],
            label="Paperless Billing"
        ),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            label="Payment Method"
        ),
        gr.Number(label="Tenure (months)", precision=0),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges")],
        outputs="text",
        title="Telco Churn Predictor",
        description="Enter customer details to predict churn."
    )

    return demo