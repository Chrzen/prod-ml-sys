from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from inference_service.model import ModelStore
from inference_service.schemas import PredictionRequest, PredictionResponse

APP_VERSION = "0.1.0"
MODEL_PATH = Path("artifacts/run1/model.pkl")

app = FastAPI(title="Inference Service", version=APP_VERSION)

model_store : ModelStore | None = None

@app.on_event("startup")
def load_model():
    global model_store
    model_store = ModelStore(MODEL_PATH)

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    df = pd.DataFrame([req.model_dump()])
    preds , probs = model_store.predict(df)

    return PredictionResponse(
        prediction=int(preds[0]),   
        probability=float(probs[0][1]),
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
    