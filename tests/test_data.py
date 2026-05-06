import pandas as pd
from churnguard.data import load_data, preprocess


def test_load_data_returns_dataframe():
    df = load_data("data/telco_churn.csv")
    assert isinstance(df, pd.DataFrame)


def test_load_data_has_expected_columns():
    df = load_data("data/telco_churn.csv")
    assert len(df.columns) == 21


def test_preprocess_returns_features_and_target():
    df = load_data("data/telco_churn.csv")
    X, y = preprocess(df)
    assert len(X) == len(y)


def test_preprocess_handles_missing_total_charges():
    df = load_data("data/telco_churn.csv")
    df.loc[0, "TotalCharges"] = " "
    X, y = preprocess(df)
    assert X.shape[0] == y.shape[0]