import os

import re

import json

import joblib

import warnings



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tldextract



from urllib.parse import urlparse, urlunparse

from rapidfuzz import process, fuzz



from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    classification_report,

    confusion_matrix,

    roc_auc_score

)



warnings.filterwarnings("ignore", category=UserWarning)





# ============================================================

# 1. PATHS AND CONFIGURATION

# ============================================================



BASE_DIR = r"C:\Users\Lenovo\Desktop\Cryptography\Phising\Phish tank dataset"



DATA_FILE = os.path.join(BASE_DIR, "Final_phishing_dataset.csv")

BRAND_FILE = os.path.join(BASE_DIR, "brand_lib.csv")



OUTPUT_DIR = os.path.join(

    BASE_DIR, "Decision Tree Brand Detection No URL Length"

)

os.makedirs(OUTPUT_DIR, exist_ok=True)



URL_COLUMN = "url"

TARGET_COLUMN = "target"

BRAND_COLUMN = "domain_name"



TEST_FOLDS = 5                 # One fold is approximately 20% test data

CV_FOLDS = 5

RANDOM_STATE = 42



# Offline public suffix list. No online download is performed.

extractor = tldextract.TLDExtract(suffix_list_urls=None)





# ============================================================

# 2. NORMALIZATION AND URL HELPERS

# ============================================================



def normalize_brand(value):

    """Keep only lowercase letters and digits for brand comparison."""

    return re.sub(r"[^a-z0-9]", "", str(value).lower().strip())





def add_scheme(url):

    """Add http:// only when the URL has no scheme."""

    url = str(url).strip()

    if url and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):

        url = "http://" + url

    return url





def canonicalize_url(value):

    """

    Normalize a URL for duplicate detection only.

    Host and scheme are lowercased; fragments are removed.

    Path and query are otherwise preserved because they can be meaningful.

    """

    if pd.isna(value):

        return ""



    raw = str(value).strip()

    if not raw:

        return ""



    try:

        parsed = urlparse(add_scheme(raw))

        scheme = parsed.scheme.lower()

        hostname = (parsed.hostname or "").lower().rstrip(".")



        if not hostname:

            return raw.lower()



        # Preserve a non-default port.

        port = parsed.port

        netloc = hostname

        if port and not (

            (scheme == "http" and port == 80) or

            (scheme == "https" and port == 443)

        ):

            netloc = f"{hostname}:{port}"



        path = parsed.path or "/"

        # Treat an empty root path and "/" as the same URL.

        if path == "":

            path = "/"



        return urlunparse((

            scheme,

            netloc,

            path,

            parsed.params,

            parsed.query,

            ""                         # Ignore fragment

        ))

    except Exception:

        return raw.lower()





def extract_domain_parts(url):

    """Return hostname and registrable domain label."""

    if pd.isna(url):

        return "", ""



    try:

        parsed = urlparse(add_scheme(str(url).strip()))

        hostname = (parsed.hostname or "").lower().strip(".")

        if not hostname:

            return "", ""



        extracted = extractor(hostname)

        return hostname, (extracted.domain or "").lower()

    except Exception:

        return "", ""





def normalize_root_trailing_slash(url):

    """

    Remove a final slash only for a bare domain/root URL.

    Examples:

      https://www.kaggle.com/ -> https://www.kaggle.com

      https://site.com/login/  -> unchanged

      https://site.com/?a=1    -> unchanged

    """

    raw = str(url).strip()

    try:

        parsed = urlparse(add_scheme(raw))

        if parsed.path in ("", "/") and not parsed.query and not parsed.fragment:

            # Preserve whether the original URL had a scheme.

            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):

                return raw.rstrip("/")

            return raw.rstrip("/")

    except Exception:

        pass

    return raw





# ============================================================

# 3. LOAD AND INDEX THE INDEPENDENT BRAND LIBRARY

# ============================================================



print("\nLoading brand library...")



brand_df = pd.read_csv(

    BRAND_FILE,

    usecols=[BRAND_COLUMN],

    dtype=str

)



brands = sorted(set(

    brand_df[BRAND_COLUMN]

    .dropna()

    .map(normalize_brand)

    .loc[lambda s: s.str.len() > 0]

))



if not brands:

    raise ValueError("No valid brand names found in brand_lib.csv.")



brand_set = set(brands)

brand_lengths = sorted({len(brand) for brand in brands if len(brand) >= 4})



# Index brands by their first character for faster similarity search.

brand_index = {}

for brand in brands:

    brand_index.setdefault(brand[0], []).append(brand)



print("Total unique brands:", len(brands))

print("Brand index groups:", len(brand_index))





# ============================================================

# 4. BRAND SIMILARITY AND BRAND MATCHING

# ============================================================



def calculate_brand_similarity(domain_name):

    """

    Compare the extracted domain label against candidate brands.

    Candidates are selected using first-character indexing.

    """

    domain_name = normalize_brand(domain_name)

    if not domain_name:

        return 0.0



    candidates = brand_index.get(domain_name[0], [])

    if not candidates:

        return 0.0



    result = process.extractOne(

        domain_name,

        candidates,

        scorer=fuzz.ratio

    )



    return (result[1] / 100.0) if result else 0.0





def calculate_brand_substring_match(domain_name):

    """

    Detect a brand anywhere inside the normalized domain label.

    Uses substring hash lookups, avoiding a scan through the full

    brand library for every URL.

    """

    domain_name = normalize_brand(domain_name)

    if not domain_name:

        return 0



    domain_length = len(domain_name)



    for start in range(domain_length):

        for brand_length in brand_lengths:

            end = start + brand_length

            if end > domain_length:

                break



            if domain_name[start:end] in brand_set:

                return 1



    return 0





# ============================================================

# 5. EXTRACT THE 10 URL AND BRAND FEATURES

# ============================================================



def extract_features(url):

    raw_url = "" if pd.isna(url) else str(url).strip()

    feature_url = normalize_root_trailing_slash(raw_url)



    hostname, domain_name = extract_domain_parts(feature_url)



    try:

        parsed = urlparse(add_scheme(feature_url))

        path = parsed.path or ""

    except Exception:

        path = ""



    normalized_domain = normalize_brand(domain_name)



    # The three requested brand features are retained.

    brand_similarity = calculate_brand_similarity(normalized_domain)



    exact_brand_match = int(

        normalized_domain in brand_set if normalized_domain else False

    )



    brand_substring_match = calculate_brand_substring_match(

        normalized_domain

    )



    return {

        "url_length": len(feature_url),

        "hostname_length": len(hostname),

        "dot_count": feature_url.count("."),

        "hyphen_count": feature_url.count("-"),

        "at_count": feature_url.count("@"),

        "digit_count": sum(ch.isdigit() for ch in feature_url),

        "subdomain_count": (

            max(0, len(hostname.split(".")) - 2) if hostname else 0

        ),

        "path_length": len(path),

        "brand_similarity": brand_similarity,

        "exact_brand_match": exact_brand_match,

        "brand_substring_match": brand_substring_match

    }





# ============================================================

# 6. LOAD AND VALIDATE THE DATASET

# ============================================================



print("\nLoading phishing dataset...")

df = pd.read_csv(DATA_FILE)



for column in [URL_COLUMN, TARGET_COLUMN]:

    if column not in df.columns:

        raise ValueError(f"Required column '{column}' not found.")



df = df.dropna(subset=[URL_COLUMN, TARGET_COLUMN]).copy()



df[TARGET_COLUMN] = pd.to_numeric(

    df[TARGET_COLUMN], errors="coerce"

)

df = df.dropna(subset=[TARGET_COLUMN]).copy()

df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)



if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):

    raise ValueError("Target column must contain only 0 and 1.")



df[URL_COLUMN] = df[URL_COLUMN].astype(str).str.strip()

df = df[df[URL_COLUMN] != ""].copy()



print("Original usable dataset shape:", df.shape)

print("\nOriginal class distribution:")

print(df[TARGET_COLUMN].value_counts().sort_index())





# ============================================================

# 7. DUPLICATE AND CONFLICTING-LABEL HANDLING

# ============================================================



print("\nChecking duplicate URLs and conflicting labels...")



df["_url_group"] = df[URL_COLUMN].map(canonicalize_url)



# Remove rows whose URL could not be normalized.

df = df[df["_url_group"] != ""].copy()



# If the same canonical URL has both labels, its ground truth is ambiguous.

labels_per_url = df.groupby("_url_group")[TARGET_COLUMN].nunique()

conflicting_urls = labels_per_url[labels_per_url > 1].index



conflict_rows = df[df["_url_group"].isin(conflicting_urls)].copy()

conflict_rows.to_csv(

    os.path.join(OUTPUT_DIR, "conflicting_url_labels_excluded.csv"),

    index=False

)



print("Conflicting URL groups:", len(conflicting_urls))

print("Rows excluded due to conflicting labels:", len(conflict_rows))



df = df[~df["_url_group"].isin(conflicting_urls)].copy()



# Keep one row per canonical URL to avoid repeated URLs dominating results.

before_dedup = len(df)

df = df.drop_duplicates(subset="_url_group", keep="first").copy()



print("Duplicate rows removed:", before_dedup - len(df))

print("Dataset after URL deduplication:", df.shape)



if df[TARGET_COLUMN].nunique() != 2:

    raise ValueError("Both target classes must remain after duplicate cleaning.")



if df[TARGET_COLUMN].value_counts().min() < TEST_FOLDS:

    raise ValueError(

        "Not enough samples in the minority class for the selected "

        "StratifiedGroupKFold split."

    )





# ============================================================

# 8. EXTRACT FEATURES

# ============================================================



print("\nExtracting URL and brand features...")



feature_records = []

total_urls = len(df)



for i, url in enumerate(df[URL_COLUMN], start=1):

    feature_records.append(extract_features(url))



    if i % 1000 == 0 or i == total_urls:

        print(f"Processed {i}/{total_urls} URLs")



features_df = pd.DataFrame(feature_records)



FEATURE_COLUMNS = [


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



# Enforce the intended feature order and ensure numeric values.

features_df = features_df[FEATURE_COLUMNS].astype(float)



final_df = pd.concat(

    [

        df[[URL_COLUMN]].reset_index(drop=True),

        df[["_url_group", TARGET_COLUMN]].reset_index(drop=True),

        features_df.reset_index(drop=True)

    ],

    axis=1

)



final_df.to_csv(

    os.path.join(OUTPUT_DIR, "engineered_phishing_dataset.csv"),

    index=False

)



print("\nFeature extraction completed.")

print("Number of features:", len(FEATURE_COLUMNS))

print("Features:", FEATURE_COLUMNS)





# ============================================================

# 9. STRATIFIED GROUP TRAIN-TEST SPLIT

# ============================================================



X = final_df[FEATURE_COLUMNS]

y = final_df[TARGET_COLUMN]

groups = final_df["_url_group"]



# Group-aware splitting ensures that the same canonical URL group

# cannot occur in both train and test sets.

outer_splitter = StratifiedGroupKFold(

    n_splits=TEST_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE

)



train_idx, test_idx = next(

    outer_splitter.split(X, y, groups=groups)

)



X_train = X.iloc[train_idx].copy()

X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()

y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx].copy()

groups_test = groups.iloc[test_idx].copy()



print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))

print("Training class distribution:")

print(y_train.value_counts().sort_index())

print("Testing class distribution:")

print(y_test.value_counts().sort_index())



# Explicitly verify no URL group is shared across the split.

group_overlap = set(groups_train).intersection(set(groups_test))

print("Train-test URL group overlap:", len(group_overlap))



if group_overlap:

    raise RuntimeError("URL group leakage detected between train and test.")





# ============================================================

# 10. HYPERPARAMETER TUNING WITH GROUP-AWARE CROSS-VALIDATION

# ============================================================



print("\nTuning Decision Tree using 5-fold grouped cross-validation...")



inner_cv = StratifiedGroupKFold(

    n_splits=CV_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE

)



base_model = DecisionTreeClassifier(

    criterion="gini",

    random_state=RANDOM_STATE

)



# The grid includes both restricted and unrestricted depths.

# Cross-validation selects the best combination using F1-score.

param_grid = {

    "max_depth": [5, 10, 15, None],

    "min_samples_split": [2, 10],

    "min_samples_leaf": [1, 5]

}



grid_search = GridSearchCV(

    estimator=base_model,

    param_grid=param_grid,

    scoring="f1",

    cv=inner_cv,

    n_jobs=-1,

    refit=True,

    verbose=1,

    return_train_score=True

)



grid_search.fit(

    X_train,

    y_train,

    groups=groups_train

)



model = grid_search.best_estimator_



print("\nBest parameters:", grid_search.best_params_)

print("Best cross-validation F1-score:", grid_search.best_score_)





# ============================================================

# 11. TRAINING AND TEST PREDICTIONS

# ============================================================



train_pred = model.predict(X_train)

y_pred = model.predict(X_test)



train_prob = model.predict_proba(X_train)[:, 1]

y_prob = model.predict_proba(X_test)[:, 1]





# ============================================================

# 12. EVALUATION

# ============================================================



def calculate_metrics(y_true, y_prediction, y_probability):

    return {

        "accuracy": float(accuracy_score(y_true, y_prediction)),

        "precision": float(precision_score(

            y_true, y_prediction, zero_division=0

        )),

        "recall": float(recall_score(

            y_true, y_prediction, zero_division=0

        )),

        "f1_score": float(f1_score(

            y_true, y_prediction, zero_division=0

        )),

        "roc_auc": float(roc_auc_score(y_true, y_probability))

    }





train_metrics = calculate_metrics(y_train, train_pred, train_prob)

test_metrics = calculate_metrics(y_test, y_pred, y_prob)



print("\n========== TRAINING PERFORMANCE ==========")

for name, value in train_metrics.items():

    print(f"{name:12s}: {value:.4f}")



print("\n========== CROSS-VALIDATION PERFORMANCE ==========")

print(f"Best CV F1 : {grid_search.best_score_:.4f}")



print("\n========== TESTING PERFORMANCE ==========")

for name, value in test_metrics.items():

    print(f"{name:12s}: {value:.4f}")



print("\n========== GENERALIZATION GAP ==========")

print(

    "Accuracy gap:",

    round(train_metrics["accuracy"] - test_metrics["accuracy"], 4)

)

print(

    "F1-score gap:",

    round(train_metrics["f1_score"] - test_metrics["f1_score"], 4)

)



report = classification_report(

    y_test,

    y_pred,

    labels=[0, 1],

    target_names=["Legitimate", "Phishing"],

    zero_division=0

)



print("\nClassification Report:")

print(report)





# ============================================================

# 13. SAVE MODEL, BRAND LIBRARY, AND METRICS

# ============================================================



print("\nSaving model and related files...")



joblib.dump(

    model,

    os.path.join(OUTPUT_DIR, "decision_tree_brand_model.pkl")

)



with open(

    os.path.join(OUTPUT_DIR, "feature_names.json"), "w"

) as file:

    json.dump(FEATURE_COLUMNS, file, indent=4)



with open(

    os.path.join(OUTPUT_DIR, "brand_library.json"), "w"

) as file:

    json.dump(brands, file, indent=4)



metrics = {

    "training": train_metrics,

    "testing": test_metrics,

    "best_cross_validation_f1": float(grid_search.best_score_),

    "best_parameters": grid_search.best_params_,

    "training_samples": int(len(X_train)),

    "testing_samples": int(len(X_test)),

    "number_of_features": int(len(FEATURE_COLUMNS)),

    "number_of_brands": int(len(brands)),

    "tree_depth": int(model.get_depth()),

    "tree_leaves": int(model.get_n_leaves()),

    "train_test_url_group_overlap": int(len(group_overlap)),

    "conflicting_url_groups_excluded": int(len(conflicting_urls)),

    "duplicate_url_rows_removed": int(before_dedup - len(df))

}



with open(

    os.path.join(OUTPUT_DIR, "metrics.json"), "w"

) as file:

    json.dump(metrics, file, indent=4)



with open(

    os.path.join(OUTPUT_DIR, "classification_report.txt"), "w"

) as file:

    file.write(report)



with open(

    os.path.join(OUTPUT_DIR, "model_configuration.json"), "w"

) as file:

    json.dump({

        "model": "DecisionTreeClassifier",

        "criterion": "gini",

        "test_split": "StratifiedGroupKFold",

        "cross_validation": "StratifiedGroupKFold",

        "selection_metric": "f1",

        "random_state": RANDOM_STATE,

        "feature_columns": FEATURE_COLUMNS,

        "brand_features_retained": [

            "brand_similarity",

            "exact_brand_match",

            "brand_substring_match"

        ],

        "brand_matching": (

            "First-character indexed RapidFuzz similarity, "

            "exact match, and indexed substring match"

        ),

        "brand_library_file": "brand_lib.csv",

        "brand_column": BRAND_COLUMN,

        "url_column": URL_COLUMN,

        "target_column": TARGET_COLUMN,

        "best_parameters": grid_search.best_params_

    }, file, indent=4)





# ============================================================

# 14. SAVE CONFUSION MATRIX

# ============================================================



cm = confusion_matrix(y_test, y_pred, labels=[0, 1])



plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Model B Decision Tree without URL Length")

plt.colorbar()



plt.xticks([0, 1], ["Legitimate", "Phishing"])

plt.yticks([0, 1], ["Legitimate", "Phishing"])

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")



for i in range(2):

    for j in range(2):

        plt.text(j, i, str(cm[i, j]), ha="center", va="center")



plt.tight_layout()

plt.savefig(

    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),

    dpi=300

)

plt.close()





# ============================================================

# 15. FEATURE IMPORTANCE

# ============================================================



importance_df = pd.DataFrame({

    "Feature": FEATURE_COLUMNS,

    "Importance": model.feature_importances_

}).sort_values("Importance", ascending=False)



importance_df.to_csv(

    os.path.join(OUTPUT_DIR, "feature_importance.csv"),

    index=False

)



plt.figure(figsize=(10, 6))

plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]

)

plt.xlabel("Feature Importance")

plt.ylabel("Features")

plt.title("Decision Tree Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(

    os.path.join(OUTPUT_DIR, "feature_importance.png"),

    dpi=300

)

plt.close()





# ============================================================

# 16. SAVE TRAINING AND TESTING DATA

# ============================================================



train_output = final_df.iloc[train_idx][

    [URL_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

].copy()



train_output.to_csv(

    os.path.join(OUTPUT_DIR, "train_data.csv"),

    index=False

)



test_output = final_df.iloc[test_idx][

    [URL_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

].copy()



test_output["predicted_target"] = y_pred

test_output["phishing_probability"] = y_prob



test_output.to_csv(

    os.path.join(OUTPUT_DIR, "test_predictions.csv"),

    index=False

)





# ============================================================

# 17. SAVE CROSS-VALIDATION RESULTS

# ============================================================



cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results.to_csv(

    os.path.join(OUTPUT_DIR, "cross_validation_results.csv"),

    index=False

)





# ============================================================

# 18. FINAL SUMMARY

# ============================================================



print("\n==============================================")

print("MODEL B - DECISION TREE WITHOUT URL LENGTH COMPLETED")

print("==============================================")

print("\nOutput folder:")

print(OUTPUT_DIR)



print("\nSelected model parameters:")

print(grid_search.best_params_)



print("\nFinal testing performance:")

print(f"Accuracy : {test_metrics['accuracy']:.4f}")

print(f"Precision: {test_metrics['precision']:.4f}")

print(f"Recall   : {test_metrics['recall']:.4f}")

print(f"F1 Score : {test_metrics['f1_score']:.4f}")

print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")



print("\nTree depth:", model.get_depth())

print("Tree leaves:", model.get_n_leaves())

print("Total brands:", len(brands))

print("Total unique URLs used:", len(final_df))

print("Model B files and evaluation results saved (10 features; url_length excluded).")
