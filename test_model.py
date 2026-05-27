import json
import os


def test_metrics_file_exists():
    assert os.path.exists("metrics.json")


def test_model_accuracy():
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    assert metrics["accuracy"] >= 0.85


def test_model_artifact_exists():
    assert (
        os.path.exists("best_model.pkl")
        or os.path.exists("best_model.h5")
    )