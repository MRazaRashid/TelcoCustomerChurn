
from typing import Literal
from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
InternetOption = Literal[
    "Yes", "No", "No internet service"
]

class CustomerData(BaseModel):
    # Demographics
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo

    # Phone services
    PhoneService: YesNo
    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]

    # Internet services
    InternetService: Literal[
        "DSL", "Fiber optic", "No"
    ]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption

    # Account information
    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    # Numeric features
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
