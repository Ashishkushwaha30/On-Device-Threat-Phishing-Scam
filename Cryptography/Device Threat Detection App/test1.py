
import joblib
import json
from pathlib import Path

MODEL_DIR = Path(
    r"C:\Users\Lenovo\Desktop\Cryptography\Phising\Phish tank dataset\Decision Tree Brand Detection Validated"
)

model = joblib.load(
    MODEL_DIR / "decision_tree_brand_model.pkl"
)

with open(MODEL_DIR / "feature_names.json", "r") as f:
    feature_names = json.load(f)

print("\nPHISHING MODEL FEATURE IMPORTANCE")
print("=" * 55)

importance = model.feature_importances_

results = sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
)

for feature, score in results:
    print(f"{feature:25s}: {score:.6f}")

print("\nTREE DEPTH:", model.get_depth())
print("TREE LEAVES:", model.get_n_leaves())
