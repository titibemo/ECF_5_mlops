from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import time
import mlflow
from mlflow.tracking import MlflowClient
from api.schemas import CustomerFeatures, PredictionResponse

app = FastAPI()

model = None


def load_model_with_retry(max_retries=20, delay=3):
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()

    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}")

            client.get_registered_model("churnguard")

            mv = client.get_model_version_by_alias(
                name="churnguard",
                alias="production"
            )

            print(f"Found version {mv.version}")

            model = mlflow.pyfunc.load_model(
                "models:/churnguard@production"
            )

            print("Model loaded successfully")
            return model

        except Exception as e:
            print(f"Retry {i+1} failed: {e}")
            time.sleep(delay)

    raise RuntimeError("Impossible de charger le modèle")


@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model_with_retry()
    except Exception as e:
        print(e)
        model = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "churnguard",
        "loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = data.model_dump()

    df = pd.DataFrame([row])

    pred = model.predict(df)[0]

    try:
        proba = model.predict_proba(df)[0][1]
    except Exception:
        proba = None

    return {
        "churn": bool(pred),
        "probability": float(proba) if proba is not None else None
    }
    
@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(data: list[CustomerFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not data:
        raise HTTPException(status_code=400, detail="Empty batch")

    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Batch size > 100")

    rows = [item.model_dump() for item in data]

    df = pd.DataFrame(rows)

    preds = model.predict(df)

    try:
        probas = model.predict_proba(df)[:, 1]
    except Exception:
        probas = [None] * len(preds)

    return [
        {
            "churn": bool(p),
            "probability": float(proba) if proba is not None else None
        }
        for p, proba in zip(preds, probas)
    ]