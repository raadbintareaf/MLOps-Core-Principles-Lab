import json
from pathlib import Path


def test_winning_model_accuracy_is_high_enough():
    metrics_path = Path("metrics.json")
    assert metrics_path.exists(), "metrics.json does not exist. Run train.py first."

    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    winning_accuracy = metrics["winning_accuracy"]
    assert "winning_f1_score" in metrics, "metrics.json must include winning_f1_score."
    assert Path(metrics["winning_artifact"]).exists(), "Winning model artifact does not exist."
    assert winning_accuracy >= 0.85, (
        f"Winning model accuracy {winning_accuracy:.4f} is below the required 0.85"
    )
