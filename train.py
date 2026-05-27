import pandas as pd
import json
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from tensorflow import keras
from tensorflow.keras import layers


# Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")  # ✔ MLOps补充


# Model A: RandomForest
model_a = RandomForestClassifier(n_estimators=200, random_state=42)
model_a.fit(X_train, y_train)

pred_a = model_a.predict(X_test)

acc_a = accuracy_score(y_test, pred_a)
f1_a = f1_score(y_test, pred_a)


# Model B: Neural Net
model_b = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_b.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_b.fit(
    X_train_scaled,
    y_train,
    epochs=30,
    batch_size=16,
    verbose=0
)

pred_b = (model_b.predict(X_test_scaled) > 0.5).astype(int).ravel()

acc_b = accuracy_score(y_test, pred_b)
f1_b = f1_score(y_test, pred_b)


# Compare
results = {
    "model_a": {"accuracy": float(acc_a), "f1": float(f1_a)},
    "model_b": {"accuracy": float(acc_b), "f1": float(f1_b)}
}

if acc_a >= acc_b:
    best_model = "model_a"
    joblib.dump(model_a, "best_model.pkl")
else:
    best_model = "model_b"
    model_b.save("best_model.h5")


# Save metrics 
results["best_model"] = best_model

with open("metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("DONE")
print(results)