import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")


def load_data(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """Load churn dataset."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw dataframe into features X and target y."""

    df = df.copy()

    # Nettoyage TotalCharges (important pour ton test)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Target
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Features
    X = df.drop(columns=["Churn", "customerID"])

    return X, y