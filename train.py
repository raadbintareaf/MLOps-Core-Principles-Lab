import json
import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model A: Classical ML
model_a = LogisticRegression(max_iter=1000)
model_a.fit(X_train_scaled, y_train)

pred_a = model_a.predict(X_test_scaled)

accuracy_a = accuracy_score(y_test, pred_a)
f1_a = f1_score(y_test, pred_a)

# Model B: Deep Learning
model_b = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_b.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model_b.fit(
    X_train_scaled,
    y_train,
    epochs=25,
    batch_size=16,
    verbose=0
)

pred_b_prob = model_b.predict(X_test_scaled, verbose=0)
pred_b = (pred_b_prob >= 0.5).astype(int).ravel()

accuracy_b = accuracy_score(y_test, pred_b)
f1_b = f1_score(y_test, pred_b)

# Model comparison
if accuracy_a >= accuracy_b:
    best_model_name = "Logistic Regression"
    best_model_type = "Classical ML"
    best_accuracy = accuracy_a
    best_f1 = f1_a
    best_model_file = "best_model.pkl"

    joblib.dump(
        {
            "model": model_a,
            "scaler": scaler
        },
        best_model_file
    )

else:
    best_model_name = "Feedforward Neural Network"
    best_model_type = "Deep Learning"
    best_accuracy = accuracy_b
    best_f1 = f1_b
    best_model_file = "best_model.h5"

    model_b.save(best_model_file)

# Save metrics
metrics = {
    "model_a": {
        "model_name": "Logistic Regression",
        "model_type": "Classical ML",
        "accuracy": round(float(accuracy_a), 4),
        "f1_score": round(float(f1_a), 4)
    },
    "model_b": {
        "model_name": "Feedforward Neural Network",
        "model_type": "Deep Learning",
        "accuracy": round(float(accuracy_b), 4),
        "f1_score": round(float(f1_b), 4)
    },
    "winning_model": {
        "model_name": best_model_name,
        "model_type": best_model_type,
        "accuracy": round(float(best_accuracy), 4),
        "f1_score": round(float(best_f1), 4),
        "artifact": best_model_file
    },
    "accuracy": round(float(best_accuracy), 4),
    "f1_score": round(float(best_f1), 4)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


print("Training completed successfully.")
print("Classical ML Accuracy:", round(accuracy_a, 4))
print("Classical ML F1 Score:", round(f1_a, 4))
print("Deep Learning Accuracy:", round(accuracy_b, 4))
print("Deep Learning F1 Score:", round(f1_b, 4))
print("Winning model:", best_model_type, "-", best_model_name)
print("Saved artifact:", best_model_file)