import json

def test_model_accuracy():
    # Load the tracked metrics
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    # MLOps Upgrade 2: Continuous Integration Validation
    # Enforce that the model must be at least 90% accurate
    assert metrics["accuracy"] >= 0.90, f"❌ CI Pipeline Failed! Accuracy {metrics['accuracy']} is below 90%."
    print("✅ CI Pipeline Passed: Model meets production standards.")

if __name__ == "__main__":
    test_model_accuracy()
