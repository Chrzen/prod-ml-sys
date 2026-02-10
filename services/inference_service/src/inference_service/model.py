from pathlib import Path
import joblib

class ModelStore:
    def __init__(self, model_path:Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, X):
        probas = self.model.predict_proba(X)
        preds = self.model.predict(X)
        return preds, probas
    
    