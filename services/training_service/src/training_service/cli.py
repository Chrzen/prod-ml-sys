import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a tiny model and write artifacts.")
    p.add_argument("--data", required=True, help="Path to CSV data file")
    p.add_argument("--out", required=True, help="Output directory for artifacts")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    target_col = "bought"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = ["city"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = LogisticRegression(max_iter=200)

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    joblib.dump(pipe, out_dir / "model.pkl")

    (out_dir / "metrics.json").write_text(
        json.dumps({"accuracy": acc}, indent=2), encoding="utf-8"
    )

    (out_dir / "params.json").write_text(
        json.dumps({"seed": args.seed, "model": "logreg", "max_iter": 200}, indent=2),
        encoding="utf-8",
    )

    print(f"✅ Trained model. Accuracy={acc:.3f}")
    print(f"📦 Wrote artifacts to: {out_dir}")
