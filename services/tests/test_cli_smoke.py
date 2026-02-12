import json
import subprocess
import sys
from pathlib import Path
from unittest import result
import pandas as pd

def test_train_model_smoke(tmp_path: Path):
    out_dir = tmp_path / "run"

    # data_path = Path("data/sample.csv")

    data_path = tmp_path / "sample.csv"
    
    df = pd.DataFrame({
        "age": [25, 40, 60],
        "income": [50000, 80000, 120000],
        "target": [0, 1, 1],
    })

    df.to_csv(data_path, index=False)

    result = subprocess.run(
        [
            "train-model",
            "--data",
            str(data_path),
            "--out",
            str(out_dir),
            "--seed",
            "123",
        ],
        capture_output=True,
        text=True,
    )

    # assert result.returncode == 0, result.stderr
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert result.returncode == 0

    assert (out_dir / "model.pkl").exists()
    assert (out_dir / "metrics.json").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "accuracy" in metrics
