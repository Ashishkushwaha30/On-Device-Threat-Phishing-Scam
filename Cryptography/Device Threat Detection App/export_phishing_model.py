
import os
import json
import joblib
import numpy as np

# Existing model location
BASE_DIR = r"C:\Users\Lenovo\Desktop\Cryptography\Phising\Phish tank dataset\Decision Tree Brand Detection Validated"

MODEL_FILE = os.path.join(BASE_DIR, "decision_tree_brand_model.pkl")
BRAND_FILE = os.path.join(BASE_DIR, "brand_library.json")

# Android assets folder
ASSETS_DIR = r"C:\Users\Lenovo\AndroidStudioProjects\DeviceThreatDetection\app\src\main\assets"

os.makedirs(ASSETS_DIR, exist_ok=True)

# Load trained decision tree
model = joblib.load(MODEL_FILE)
tree = model.tree_

# Export decision tree structure
tree_data = {
    "children_left": tree.children_left.tolist(),
    "children_right": tree.children_right.tolist(),
    "feature": tree.feature.tolist(),
    "threshold": tree.threshold.tolist(),
    "prediction": np.argmax(tree.value[:, 0, :], axis=1).tolist()
}

tree_path = os.path.join(ASSETS_DIR, "phishing_tree.json")

with open(tree_path, "w", encoding="utf-8") as f:
    json.dump(tree_data, f)

# Export feature names in exact training order
feature_names = [
    "url_length",
    "hostname_length",
    "dot_count",
    "hyphen_count",
    "at_count",
    "digit_count",
    "subdomain_count",
    "path_length",
    "brand_similarity",
    "exact_brand_match",
    "brand_substring_match"
]

with open(os.path.join(ASSETS_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names, f)

# Convert brand library JSON into a compact text file
with open(BRAND_FILE, "r", encoding="utf-8") as f:
    brands = json.load(f)

brand_path = os.path.join(ASSETS_DIR, "brand_library.txt")

with open(brand_path, "w", encoding="utf-8") as f:
    for brand in brands:
        f.write(str(brand) + "\n")

print("Model export completed.")
print("Tree nodes:", tree.node_count)
print("Tree depth:", tree.max_depth)
print("Number of brands:", len(brands))
print("Android assets:", ASSETS_DIR)
