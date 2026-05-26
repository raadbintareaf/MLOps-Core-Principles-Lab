import json
import os

def test_accuracy_threshold():
    assert os.path.exists('metrics.json'), "metrics.json not found! You must run train.py first."

    with open('metrics.json', 'r') as f:
        metrics = json.load(f)

    accuracy = metrics.get('accuracy', 0.0)
    model_name = metrics.get('winning_model', 'Unknown')
    
    print(f"Evaluating {model_name} | Target Accuracy: >= 0.85 | Actual: {accuracy:.4f}")
    
    # CI Assertion
    assert accuracy >= 0.85, f"PIPELINE FAILED: Model accuracy ({accuracy:.4f}) is below the 0.85 threshold!"
    
    print("✅ CI PASS: Model meets the deployment threshold.")

if __name__ == "__main__":
    test_accuracy_threshold()
