
import json, sys
with open('metrics.json', 'r') as f: metrics = json.load(f)
if metrics['accuracy'] >= 0.85: print('CI 测试通过')
else: sys.exit(1)
