import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from churnguard.data import load_data, preprocess


# ----------------------------
# METRICS
# ----------------------------
def evaluate(model, X_test, y_test):
    """Compute classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


# ----------------------------
# TRAIN SINGLE MODEL
# ----------------------------
def train_model(model_name: str):
    """Train and log model to MLflow."""

    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # séparation colonnes
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    models = {
        "lr": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(),
        "gb": GradientBoostingClassifier(),
    }

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[model_name])
    ])

    with mlflow.start_run(run_name=model_name):

        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_param("model_name", model_name)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=X_train.iloc[:1]
        )

        run_id = mlflow.active_run().info.run_id

        print(f"[{model_name}] {metrics}")

    return {
        "model_name": model_name,
        "metrics": metrics,
        "run_id": run_id,
    }


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():

    mlflow.set_tracking_uri("http://mlflow:5000")

    results = []

    for m in ["lr", "rf", "gb"]:
        results.append(train_model(m))

    best = max(results, key=lambda x: x["metrics"]["roc_auc"])

    print(f"Best model: {best['model_name']}")

    model_uri = f"runs:/{best['run_id']}/model"

    mv = mlflow.register_model(
        model_uri=model_uri,
        name="churnguard"
    )

    print(f"Registered version: {mv.version}")

    client = MlflowClient()

    client.set_registered_model_alias(
        name="churnguard",
        alias="production",
        version=mv.version
    )

    print("Model promoted to alias: production")


if __name__ == "__main__":
    main()