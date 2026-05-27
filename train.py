import pandas as pd
import json
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# MLOps Upgrade 1: Reproducibility & Traceability
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save metrics to a JSON file
metrics = {"accuracy": accuracy, "n_estimators": 10}
with open("metrics.json", "w") as outfile:
    json.dump(metrics, outfile)

# Save Model Artifact
joblib.dump(model, 'model.pkl')
print(f"✅ Model trained! Metrics tracked and saved: {metrics}")
