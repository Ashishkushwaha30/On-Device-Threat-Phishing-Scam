
import os
import json
import joblib
import pandas as pd
from urllib.parse import urlparse

# ==========================================
# 1. Directory configuration
# ==========================================

base_dir = r"C:\Users\Lenovo\Desktop\Cryptography\Phising\Phish tank dataset"

model_dir = os.path.join(base_dir, "Decision Tree")

model_path = os.path.join(model_dir, "decision_tree_model.pkl")
feature_path = os.path.join(model_dir, "feature_names.json")

# ==========================================
# 2. Load trained Decision Tree model
# ==========================================

dt_model = joblib.load(model_path)

with open(feature_path, "r") as f:
    feature_names = json.load(f)

print("Decision Tree model loaded successfully!")

# ==========================================
# 3. URL preprocessing
# ==========================================

def preprocess_url(url):

    url = str(url).strip()

    # Remove HTTP/HTTPS
    url = pd.Series([url]).str.replace(
        r"^https?://", "",
        regex=True,
        case=False
    ).iloc[0]

    # Remove www
    url = pd.Series([url]).str.replace(
        r"^www\.", "",
        regex=True,
        case=False
    ).iloc[0]

    return url


# ==========================================
# 4. Extract URL features
# ==========================================

def extract_features(url):

    url = str(url)

    parsed = urlparse("http://" + url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "digit_count": sum(c.isdigit() for c in url),
        "subdomain_count": max(
            0, len(hostname.split(".")) - 2
        ),
        "path_length": len(path)
    }


# ==========================================
# 5. Take URL input
# ==========================================

url = input("\nEnter URL to check: ").strip()

if not url:
    raise ValueError("URL cannot be empty.")

processed_url = preprocess_url(url)

# ==========================================
# 6. Extract numerical features
# ==========================================

extracted_features = extract_features(processed_url)

input_df = pd.DataFrame([extracted_features])

# Maintain original training feature order
input_df = input_df.reindex(
    columns=feature_names,
    fill_value=0
)

print("\nExtracted URL Features:")
print(input_df.to_string(index=False))

# ==========================================
# 7. Predict URL class
# ==========================================

prediction = dt_model.predict(input_df)[0]

probabilities = dt_model.predict_proba(input_df)[0]

# ==========================================
# 8. Display prediction
# ==========================================

print("\n==========================================")
print("       PHISHING URL DETECTION RESULT")
print("==========================================")

print("\nInput URL:", url)

if prediction == 1:
    print("\nPrediction: PHISHING")
else:
    print("\nPrediction: LEGITIMATE")

print("\nPrediction Probabilities:")

for class_label, probability in zip(
    dt_model.classes_, probabilities
):

    if class_label == 1:
        label = "Phishing"
    else:
        label = "Legitimate"

    print(f"{label}: {probability * 100:.2f}%")

print("\n==========================================")