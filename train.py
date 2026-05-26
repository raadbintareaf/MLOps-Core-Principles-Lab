import pandas as pd
import json
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def main():
    # 1. Load Data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scale Data (Crucial for Neural Networks and SVMs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model A: Classical ML (Support Vector Machine) ---
    print("Training Model A: Support Vector Machine...")
    model_a = SVC(probability=True, random_state=42)
    model_a.fit(X_train_scaled, y_train)
    y_pred_a = model_a.predict(X_test_scaled)
    acc_a = accuracy_score(y_test, y_pred_a)
    f1_a = f1_score(y_test, y_pred_a)
    print(f"Model A Metrics -> Accuracy: {acc_a:.4f}, F1-Score: {f1_a:.4f}\n")

    # --- Model B: Deep Learning (Feedforward Neural Network) ---
    print("Training Model B: Feedforward Neural Network...")
    model_b = Sequential([
        Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_b.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)
    
    y_pred_probs_b = model_b.predict(X_test_scaled, verbose=0)
    y_pred_b = (y_pred_probs_b > 0.5).astype(int).flatten()
    acc_b = accuracy_score(y_test, y_pred_b)
    f1_b = f1_score(y_test, y_pred_b)
    print(f"Model B Metrics -> Accuracy: {acc_b:.4f}, F1-Score: {f1_b:.4f}\n")

    # --- 4. MLOps Traceability: Compare and Save ---
    # Clear old artifacts to prevent accidentally committing both
    if os.path.exists('best_model.pkl'): os.remove('best_model.pkl')
    if os.path.exists('best_model.h5'): os.remove('best_model.h5')

    if acc_a >= acc_b:
        print("🏆 Model A (SVM) won!")
        best_name, best_acc, best_f1, best_artifact = "Classical ML (SVM)", acc_a, f1_a, "best_model.pkl"
        joblib.dump(model_a, best_artifact)
    else:
        print("🏆 Model B (Neural Network) won!")
        best_name, best_acc, best_f1, best_artifact = "Deep Learning (NN)", acc_b, f1_b, "best_model.h5"
        model_b.save(best_artifact)

    # Save Metrics to JSON
    metrics_payload = {
        "winning_model": best_name,
        "accuracy": best_acc,
        "f1_score": best_f1,
        "artifact_file": best_artifact
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics_payload, f, indent=4)
    
    print(f"✅ Saved winning artifact: {best_artifact}")
    print("✅ Saved MLOps tracking data: metrics.json")

if __name__ == "__main__":
    main()
