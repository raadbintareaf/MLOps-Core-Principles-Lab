import json

with open("metrics.json", "r") as f:
    metrics = json.load(f)

print(f"Model    : {metrics['model']}")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"F1-Score : {metrics['f1_score']:.4f}")

assert metrics["accuracy"] >= 0.85, (
    f"❌ CI FAILED: Accuracy {metrics['accuracy']:.4f} is below 0.85 threshold!"
)
print("✅ CI PASSED: Model meets the accuracy threshold.")