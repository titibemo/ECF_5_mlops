from churnguard.evaluate import compute_metrics
from churnguard.data import load_data, preprocess

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def build_test_pipeline(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])

    return pipeline


def test_compute_metrics_returns_dict():
    df = load_data()
    X, y = preprocess(df)

    X = X.head(50)
    y = y.head(50)

    model = build_test_pipeline(X)
    model.fit(X, y)

    metrics = compute_metrics(model, X, y)

    assert isinstance(metrics, dict)


def test_compute_metrics_keys():
    df = load_data()
    X, y = preprocess(df)

    X = X.head(50)
    y = y.head(50)

    model = build_test_pipeline(X)
    model.fit(X, y)

    metrics = compute_metrics(model, X, y)

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}

    assert set(metrics.keys()) == expected_keys