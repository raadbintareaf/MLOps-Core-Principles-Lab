import json

with open("metrics.json") as f:
    metrics = json.load(f)

best_model = metrics["best_model"]

accuracy = metrics[best_model]["accuracy"]
f1 = metrics[best_model]["f1"]

print("Best model:", best_model)
print("Accuracy:", accuracy)
print("F1:", f1)


assert accuracy >= 0.85, f"Accuracy too low: {accuracy}"
assert f1 >= 0.80, f"F1 too low: {f1}"

print("CI PASS ✔")