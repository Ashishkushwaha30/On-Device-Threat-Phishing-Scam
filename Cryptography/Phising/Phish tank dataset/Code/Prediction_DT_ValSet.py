
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import tldextract
import matplotlib.pyplot as plt

from urllib.parse import urlparse
from rapidfuzz import process, fuzz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = (
    r"C:\Users\Lenovo\Desktop\Cryptography"
    r"\Phising\Phish tank dataset"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "Decision Tree Brand Detection Fast"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "decision_tree_brand_model.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)

BRAND_FILE = os.path.join(
    MODEL_DIR,
    "brand_library.json"
)

# Validation dataset
VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "Validation_data.csv"
)

# Output directory
OUTPUT_DIR = os.path.join(
    MODEL_DIR,
    "Validation Results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Offline public suffix list
extractor = tldextract.TLDExtract(
    suffix_list_urls=None
)


# ============================================================
# 2. LOAD MODEL AND REQUIRED FILES
# ============================================================

print("\nLoading trained Decision Tree model...")

model = joblib.load(MODEL_FILE)

with open(FEATURE_FILE, "r") as file:
    FEATURE_COLUMNS = json.load(file)

with open(BRAND_FILE, "r") as file:
    brands = json.load(file)

brand_set = set(brands)

print("Model loaded successfully.")
print("Total brands:", len(brands))
print("Total features:", len(FEATURE_COLUMNS))


# ============================================================
# 3. CREATE BRAND INDEX
# ============================================================

brand_index = {}

for brand in brands:

    if not brand:
        continue

    first_char = brand[0]

    brand_index.setdefault(
        first_char, []
    ).append(brand)


# ============================================================
# 4. NORMALIZATION
# ============================================================

def normalize_brand(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower().strip()
    )


# ============================================================
# 5. URL CLEANING
# ============================================================

def clean_url(url):

    if pd.isna(url):
        return ""

    url = str(url).strip().lower()

    # Remove HTTP/HTTPS
    url = re.sub(
        r"^https?://",
        "",
        url
    )

    # Remove www
    url = re.sub(
        r"^www\.",
        "",
        url
    )

    # Remove trailing slash
    url = url.rstrip("/")

    return url


# ============================================================
# 6. DOMAIN EXTRACTION
# ============================================================

def extract_domain_parts(url):

    cleaned_url = clean_url(url)

    if not cleaned_url:
        return "", ""

    try:

        parsed = urlparse(
            "http://" + cleaned_url
        )

        hostname = parsed.hostname or ""

        hostname = hostname.lower().strip(".")

        extracted = extractor(hostname)

        domain_name = extracted.domain or ""

        return hostname, domain_name

    except Exception:

        return "", ""


# ============================================================
# 7. BRAND SIMILARITY
# ============================================================

def calculate_brand_similarity(domain_name):

    domain_name = normalize_brand(domain_name)

    if not domain_name:
        return 0.0

    candidates = brand_index.get(
        domain_name[0], []
    )

    if not candidates:
        return 0.0

    result = process.extractOne(
        domain_name,
        candidates,
        scorer=fuzz.ratio,
        score_cutoff=0
    )

    if result is None:
        return 0.0

    best_brand, similarity_score, _ = result

    return similarity_score / 100.0


# ============================================================
# 8. FEATURE EXTRACTION
# ============================================================

def extract_features(url):

    raw_url = clean_url(url)

    hostname, domain_name = extract_domain_parts(
        raw_url
    )

    try:

        parsed = urlparse(
            "http://" + raw_url
        )

        path = parsed.path or ""

    except Exception:

        path = ""

    # Basic URL features

    url_length = len(raw_url)

    hostname_length = len(hostname)

    dot_count = raw_url.count(".")

    hyphen_count = raw_url.count("-")

    at_count = raw_url.count("@")

    digit_count = sum(
        character.isdigit()
        for character in raw_url
    )

    subdomain_count = (
        max(0, len(hostname.split(".")) - 2)
        if hostname else 0
    )

    path_length = len(path)

    # Brand features

    normalized_domain = normalize_brand(
        domain_name
    )

    brand_similarity = calculate_brand_similarity(
        normalized_domain
    )

    exact_brand_match = int(
        normalized_domain in brand_set
        if normalized_domain else False
    )

    candidates = brand_index.get(
        normalized_domain[0], []
    ) if normalized_domain else []

    brand_substring_match = int(
        any(
            len(brand) >= 4 and
            brand in normalized_domain
            for brand in candidates
        )
    )

    return {

        "url_length": url_length,

        "hostname_length": hostname_length,

        "dot_count": dot_count,

        "hyphen_count": hyphen_count,

        "at_count": at_count,

        "digit_count": digit_count,

        "subdomain_count": subdomain_count,

        "path_length": path_length,

        "brand_similarity": brand_similarity,

        "exact_brand_match": exact_brand_match,

        "brand_substring_match": brand_substring_match
    }


# ============================================================
# 9. LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

df = pd.read_csv(VALIDATION_FILE)

if "url" not in df.columns:
    raise ValueError("CSV must contain a 'url' column.")

if "target" not in df.columns:
    raise ValueError("CSV must contain a 'target' column.")

print("Total validation records:", len(df))

# Remove rows having missing URL or target

df = df.dropna(
    subset=["url", "target"]
).copy()

# Ensure target is numeric

df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce"
)

df = df.dropna(
    subset=["target"]
).copy()

df["target"] = df["target"].astype(int)

if not set(df["target"].unique()).issubset({0, 1}):
    raise ValueError(
        "Target column must contain only 0 and 1."
    )

print("Records after cleaning:", len(df))

print("\nActual target distribution:")
print(df["target"].value_counts().sort_index())


# ============================================================
# 10. EXTRACT FEATURES FOR ALL URLS
# ============================================================

print("\nExtracting features from validation URLs...")

feature_rows = []

total = len(df)

for i, url in enumerate(df["url"], start=1):

    try:

        features = extract_features(url)

    except Exception as error:

        print(
            f"\nFeature extraction error at row {i}:",
            error
        )

        features = {
            column: 0
            for column in FEATURE_COLUMNS
        }

    feature_rows.append(features)

    if i % 1000 == 0 or i == total:

        print(
            f"Processed {i}/{total} URLs"
        )


# Convert extracted features into DataFrame

X_validation = pd.DataFrame(
    feature_rows
).reindex(
    columns=FEATURE_COLUMNS,
    fill_value=0
)

print("\nFeature extraction completed.")

print(
    "Validation feature shape:",
    X_validation.shape
)


# ============================================================
# 11. PREDICTION
# ============================================================

print("\nPredicting validation URLs...")

y_true = df["target"].values

y_pred = model.predict(
    X_validation
)

probabilities = model.predict_proba(
    X_validation
)

# Find correct probability columns using model classes

classes = list(model.classes_)

if 0 not in classes or 1 not in classes:
    raise ValueError(
        "Model must support both classes 0 and 1."
    )

legitimate_probability = probabilities[
    :, classes.index(0)
]

phishing_probability = probabilities[
    :, classes.index(1)
]

print("Prediction completed.")


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n========================================")
print("             CONFUSION MATRIX")
print("========================================")

print(cm)

print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)


# ============================================================
# 13. EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else 0
)

fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0 else 0
)

fnr = (
    fn / (fn + tp)
    if (fn + tp) > 0 else 0
)

roc_auc = roc_auc_score(
    y_true,
    phishing_probability
)

print("\n========================================")
print("           EVALUATION RESULTS")
print("========================================")

print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-Score:       {f1:.4f}")
print(f"Specificity:    {specificity:.4f}")
print(f"FPR:            {fpr:.4f}")
print(f"FNR:            {fnr:.4f}")
print(f"ROC-AUC:        {roc_auc:.4f}")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=[
        "Legitimate",
        "Phishing"
    ],
    digits=4,
    zero_division=0
)

print("\n========================================")
print("          CLASSIFICATION REPORT")
print("========================================")

print(report)


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================

df["predicted_target"] = y_pred

df["legitimate_probability"] = legitimate_probability

df["phishing_probability"] = phishing_probability

prediction_file = os.path.join(
    OUTPUT_DIR,
    "validation_predictions.csv"
)

df.to_csv(
    prediction_file,
    index=False
)

print("\nPredictions saved:")
print(prediction_file)


# ============================================================
# 16. SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Decision Tree Confusion Matrix")

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Legitimate", "Phishing"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Phishing"]
)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_file = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_file,
    dpi=300
)

plt.close()

print("\nConfusion matrix image saved:")
print(cm_file)


# ============================================================
# 17. SAVE EVALUATION REPORT
# ============================================================

metrics = {

    "Total Samples": len(y_true),

    "Accuracy": accuracy,

    "Precision": precision,

    "Recall": recall,

    "F1 Score": f1,

    "Specificity": specificity,

    "False Positive Rate": fpr,

    "False Negative Rate": fnr,

    "ROC AUC": roc_auc,

    "True Negative": int(tn),

    "False Positive": int(fp),

    "False Negative": int(fn),

    "True Positive": int(tp)
}

metrics_df = pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Value"]
)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "evaluation_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

report_file = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(report_file, "w") as file:

    file.write("DECISION TREE VALIDATION RESULTS\n")

    file.write("=" * 50 + "\n\n")

    file.write(
        metrics_df.to_string(index=False)
    )

    file.write("\n\nCONFUSION MATRIX\n")

    file.write(
        np.array2string(cm)
    )

    file.write("\n\nCLASSIFICATION REPORT\n")

    file.write(report)

print("\nEvaluation report saved:")
print(report_file)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("          VALIDATION COMPLETED")
print("========================================")

print("Total URLs evaluated:", len(y_true))

print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall: {recall * 100:.2f}%")

print(f"F1 Score: {f1 * 100:.2f}%")

print("\nAll results saved inside:")

print(OUTPUT_DIR)

print("\n========================================")