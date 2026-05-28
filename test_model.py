import json

# 1) 读取 metrics.json
with open("metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

# 2) 取出获胜模型准确率（在 winning_model 里）
accuracy = metrics["winning_model"]["accuracy"]

# 3) 断言 >= 0.85，不满足会抛 AssertionError
assert accuracy >= 0.85, f"Accuracy {accuracy} is below 0.85"

print(f"CI passed: winning model accuracy = {accuracy}")