
import pandas as pd


# Binary feature mappings
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

INTERNET_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def transform_customer(
    df: pd.DataFrame,
    feature_columns: list[str]
) -> pd.DataFrame:
    """
    Convert raw customer data into the feature schema
    expected by the trained XGBoost model.

    Feature columns are supplied by inference.py.
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    # STEP 1: Check required input columns
    required_cols = (
        NUMERIC_COLS
        + list(BINARY_MAP)
        + INTERNET_SERVICES
        + [
            "MultipleLines",
            "InternetService",
            "Contract",
            "PaymentMethod",
        ]
    )

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # STEP 2: Validate numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

        if df[col].isna().any():
            raise ValueError(
                f"Missing numeric value: {col}"
            )

    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError(
            "SeniorCitizen must be 0 or 1"
        )

    # STEP 3: Engineer special service features
    # Create indicators BEFORE replacing categories
    df["No_internet_service"] = (
        df[INTERNET_SERVICES]
        .eq("No internet service")
        .any(axis=1)
        .astype(int)
    )

    df["No_phone_service"] = (
        df["MultipleLines"] == "No phone service"
    ).astype(int)

    # STEP 4: Normalize categorical values
    df["MultipleLines"] = df["MultipleLines"].replace({
        "No phone service": "No"
    })

    for col in INTERNET_SERVICES:
        df[col] = df[col].replace({
            "No internet service": "No"
        })

    # STEP 5: Binary encoding
    for col, mapping in BINARY_MAP.items():
        df[col] = df[col].map(mapping)

        if df[col].isna().any():
            raise ValueError(
                f"Invalid category in {col}"
            )

        df[col] = df[col].astype(int)

    # STEP 6: One-hot encoding
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
        dtype=int
    )

    # STEP 7: Align to trained model feature order
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df
