import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METRICS_PATH = Path("metrics.json")
MODEL_PATH = Path("best_model.pkl")
RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer()
    return train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )


def build_classical_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_deep_learning_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_predictions(y_test, predictions):
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
    }


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)
    return model, metrics


def remove_existing_artifacts():
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()


def save_metrics(metrics):
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)


def main():
    X_train, X_test, y_train, y_test = load_data()

    model_a, model_a_metrics = train_and_evaluate_model(
        build_classical_model(),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    model_b, model_b_metrics = train_and_evaluate_model(
        build_deep_learning_model(),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    remove_existing_artifacts()

    if model_a_metrics["accuracy"] >= model_b_metrics["accuracy"]:
        winning_model = "Model A - LogisticRegression"
        winning_metrics = model_a_metrics
        winning_estimator = model_a
    else:
        winning_model = "Model B - Feedforward Neural Network"
        winning_metrics = model_b_metrics
        winning_estimator = model_b

    joblib.dump(winning_estimator, MODEL_PATH)

    metrics = {
        "model_a": {
            "name": "LogisticRegression",
            "type": "Classical ML",
            **model_a_metrics,
        },
        "model_b": {
            "name": "Feedforward Neural Network",
            "type": "Deep Learning",
            "layers": 4,
            "preprocessing": "StandardScaler",
            **model_b_metrics,
        },
        "winning_model": winning_model,
        "winning_metrics": winning_metrics,
        "winning_accuracy": winning_metrics["accuracy"],
        "winning_f1_score": winning_metrics["f1_score"],
        "winning_artifact": str(MODEL_PATH),
    }
    save_metrics(metrics)

    print(f"Model A accuracy: {model_a_metrics['accuracy']:.4f}")
    print(f"Model A F1-score: {model_a_metrics['f1_score']:.4f}")
    print(f"Model B accuracy: {model_b_metrics['accuracy']:.4f}")
    print(f"Model B F1-score: {model_b_metrics['f1_score']:.4f}")
    print(f"Winning model: {winning_model}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved winning model artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()
