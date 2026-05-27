import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
import tensorflow as tf
from tensorflow import keras

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("=" * 45)
print("Training Model A: Logistic Regression...")
model_a = LogisticRegression(max_iter=1000, random_state=42)
model_a.fit(X_train_scaled, y_train)
pred_a = model_a.predict(X_test_scaled)
acc_a  = accuracy_score(y_test, pred_a)
f1_a   = f1_score(y_test, pred_a)
print(f"  Accuracy : {acc_a:.4f}")
print(f"  F1-Score : {f1_a:.4f}")

print("=" * 45)
print("Training Model B: Neural Network...")
model_b = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])
model_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_b.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.1)

pred_b = (model_b.predict(X_test_scaled).flatten() >= 0.5).astype(int)
acc_b  = accuracy_score(y_test, pred_b)
f1_b   = f1_score(y_test, pred_b)
print(f"  Accuracy : {acc_b:.4f}")
print(f"  F1-Score : {f1_b:.4f}")

print("=" * 45)
if acc_a >= acc_b:
    winner  = "Classical ML (Logistic Regression)"
    metrics = {"model": winner, "accuracy": round(acc_a, 4), "f1_score": round(f1_a, 4)}
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model_a, f)
else:
    winner  = "Deep Learning (Neural Network)"
    metrics = {"model": winner, "accuracy": round(acc_b, 4), "f1_score": round(f1_b, 4)}
    model_b.save("best_model.h5")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"🏆 Winner : {winner}")
print(f"✅ metrics.json saved: {metrics}")