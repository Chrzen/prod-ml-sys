from contextlib import asynccontextmanager
from pathlib import Path
import time

import pandas as pd
from fastapi import FastAPI

from inference_service.model import ModelStore
from inference_service.schemas import PredictionRequest, PredictionResponse


APP_VERSION = "0.1.0"
MODEL_PATH = Path("artifacts/run1/model.pkl")

model_store: ModelStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_store

    max_attempts = 10
    delay = 3

    for attempt in range(max_attempts):
        try:
            model_store = ModelStore(MODEL_PATH)
            print("Model loaded successfully.")
            break
        except FileNotFoundError:
            print(f"Model not found. Retry {attempt+1}/{max_attempts}...")
            time.sleep(delay)
    else:
        raise RuntimeError("Model never became available.")

    yield  # App runs here

    # Shutdown logic could go here later


app = FastAPI(title="Inference Service", version=APP_VERSION, lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    df = pd.DataFrame([req.model_dump()])
    preds, proba = model_store.predict(df)

    return PredictionResponse(
        prediction=int(preds[0]),
        probability=float(proba[0][1]),
        model_version=APP_VERSION,
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


def run():
    import uvicorn

    uvicorn.run(
        "inference_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
