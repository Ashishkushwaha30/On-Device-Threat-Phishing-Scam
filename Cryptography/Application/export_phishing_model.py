
import os
import json
import joblib
import numpy as np

# ============================================================
# 1. PATHS
# ============================================================

MODEL_DIR = (
    r"C:\Users\Lenovo\Desktop\Cryptography\Phising"
    r"\Phish tank dataset\Decision Tree Brand Detection Model B"
)

ANDROID_ASSETS = (
    r"C:\Users\Lenovo\AndroidStudioProjects"
    r"\DeviceThreatDetection\app\src\main\assets"
)

MODEL_FILE = os.path.join(
    MODEL_DIR, "decision_tree_brand_model.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR, "feature_names.json"
)

BRAND_FILE = os.path.join(
    MODEL_DIR, "brand_library.json"
)

os.makedirs(ANDROID_ASSETS, exist_ok=True)

# ============================================================
# 2. LOAD MODEL AND FEATURE NAMES
# ============================================================

print("Loading Model B...")

model = joblib.load(MODEL_FILE)

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    feature_names = json.load(f)

expected_features = [
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

if feature_names != expected_features:
    raise ValueError(
        "Model B feature order does not match the expected "
        "10-feature order.\n"
        f"Found: {feature_names}"
    )

if model.n_features_in_ != 10:
    raise ValueError(
        f"Expected 10 model features, found {model.n_features_in_}"
    )

if list(model.classes_) != [0, 1]:
    raise ValueError(
        f"Unexpected model classes: {model.classes_}"
    )

print("Model B loaded successfully.")
print("Features:", feature_names)

# ============================================================
# 3. EXPORT DECISION TREE
# ============================================================

tree = model.tree_

children_left = tree.children_left.tolist()
children_right = tree.children_right.tolist()
features = tree.feature.tolist()
thresholds = tree.threshold.tolist()

# Convert each leaf's majority class into its actual class label.
predictions = []

for node in range(tree.node_count):
    class_index = int(np.argmax(tree.value[node][0]))
    predicted_class = int(model.classes_[class_index])
    predictions.append(predicted_class)

tree_data = {
    "children_left": children_left,
    "children_right": children_right,
    "feature": features,
    "threshold": thresholds,
    "prediction": predictions
}

tree_output = os.path.join(
    ANDROID_ASSETS, "phishing_tree.json"
)

with open(tree_output, "w", encoding="utf-8") as f:
    json.dump(tree_data, f)

# ============================================================
# 4. EXPORT FEATURE NAMES
# ============================================================

feature_output = os.path.join(
    ANDROID_ASSETS, "feature_names.json"
)

with open(feature_output, "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=4)

# ============================================================
# 5. EXPORT BRAND LIBRARY
# ============================================================

print("\nLoading brand library...")

with open(BRAND_FILE, "r", encoding="utf-8") as f:
    brands = json.load(f)

brand_output = os.path.join(
    ANDROID_ASSETS, "brand_library.txt"
)

with open(brand_output, "w", encoding="utf-8") as f:
    f.write("\n".join(brands))

# ============================================================
# 6. EXPORT SUMMARY
# ============================================================

print("\n====================================")
print("MODEL B ANDROID EXPORT COMPLETED")
print("====================================")

print("Model:", MODEL_FILE)
print("Tree nodes:", tree.node_count)
print("Tree depth:", model.get_depth())
print("Tree leaves:", model.get_n_leaves())
print("Number of features:", len(feature_names))
print("Number of brands:", len(brands))

print("\nAndroid assets saved:")

print(tree_output)
print(feature_output)
print(brand_output)

print("\nFeature order:")
for i, name in enumerate(feature_names):
    print(f"{i}: {name}")
