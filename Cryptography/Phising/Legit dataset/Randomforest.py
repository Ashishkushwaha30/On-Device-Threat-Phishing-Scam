
# ============================================================
# RANDOM FOREST CLASSIFIER - LEGITPHISH DATASET
# Phishing URL Detection
# ============================================================

# PART 1: IMPORT LIBRARIES

import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ============================================================
# PART 2: LOAD DATASET
# ============================================================

# Change this path according to your dataset location

file_path = "C:\\Users\\Lenovo\\Desktop\\Cryptography\\Phising\\url_features_extracted1.csv"

data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.strip()

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst five records:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nDataset information:")
print(data.info())

# ============================================================
# PART 3: DATASET INSPECTION
# ============================================================

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate records:", data.duplicated().sum())

print("\nStatistical summary:")
print(data.describe(include="all"))

# ============================================================
# PART 4: REMOVE DUPLICATES AND MISSING TARGET VALUES
# ============================================================

# Detect target column
target_candidates = ["ClassLabel", "classlabel", "label", "Label"]

target_col = next(
    (col for col in target_candidates if col in data.columns),
    None
)

if target_col is None:
    raise ValueError(
        "Target column not found. Check your dataset column names."
    )

# Remove rows without target labels
data = data.dropna(subset=[target_col]).copy()

# Remove duplicate records
data = data.drop_duplicates().reset_index(drop=True)

print("\nTarget column:", target_col)
print("Dataset shape after cleaning:", data.shape)

print("\nClass distribution:")
print(data[target_col].value_counts())

# ============================================================
# PART 5: PREPARE TARGET VARIABLE
# ============================================================

# Expected dataset convention:
# 0 = Phishing
# 1 = Legitimate

# Handle text-based labels if present
if not pd.api.types.is_numeric_dtype(data[target_col]):

    label_mapping = {
        "phishing": 0,
        "phish": 0,
        "malicious": 0,
        "legitimate": 1,
        "legit": 1,
        "benign": 1,
        "safe": 1
    }

    normalized_labels = (
        data[target_col].astype(str).str.strip().str.lower()
    )

    unknown_labels = set(normalized_labels.unique()) - set(label_mapping)

    if unknown_labels:
        raise ValueError(
            f"Unknown target labels: {unknown_labels}. "
            "Please define their mapping."
        )

    y = normalized_labels.map(label_mapping)

else:
    y = data[target_col]

# Check binary classification
if y.nunique() != 2:
    raise ValueError(
        f"Expected two classes, but found {y.nunique()} classes."
    )

# ============================================================
# PART 6: PREPARE INPUT FEATURES
# ============================================================

# Exclude target column
X = data.drop(columns=[target_col])

# Remove raw URL column because we are using
# already extracted numerical URL features.

url_columns = [
    col for col in X.columns
    if col.strip().lower() in ["url", "urls"]
]

X = X.drop(columns=url_columns)

# Remove columns that contain no useful information
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

X = X.drop(columns=constant_columns)

# Convert infinite values into missing values
X = X.replace([np.inf, -np.inf], np.nan)

print("\nInput features:")
print(X.columns.tolist())

print("\nNumber of input features:", X.shape[1])

if X.shape[1] == 0:
    raise ValueError("No usable input features found.")

# ============================================================
# PART 7: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================================
# PART 8: HANDLE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number", "bool"]
).columns.tolist()

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ============================================================
# PART 9: CREATE RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Combine preprocessing and classifier
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])

# ============================================================
# PART 10: TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

# ============================================================
# PART 11: MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)

# Find the probability column corresponding to class 1
classes = model.named_steps["classifier"].classes_
positive_index = list(classes).index(1)

positive_prob = y_prob[:, positive_index]

# ============================================================
# PART 12: MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

recall = recall_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    positive_prob
)

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Phishing (0)", "Legitimate (1)"],
    zero_division=0
))

# ============================================================
# PART 13: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Legitimate"],
    yticklabels=["Phishing", "Legitimate"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

# ============================================================
# PART 14: ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    positive_prob,
    pos_label=1
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ============================================================
# PART 15: FEATURE IMPORTANCE
# ============================================================

# Get feature names after preprocessing
feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)

importances = (
    model.named_steps["classifier"].feature_importances_
)

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15).to_string(index=False))

plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ============================================================
# PART 16: SAVE MODEL
# ============================================================

output_folder = "RandomForest_Results"

os.makedirs(output_folder, exist_ok=True)

# Save complete pipeline
joblib.dump(
    model,
    os.path.join(output_folder, "random_forest_model.pkl")
)

# Save feature importance
feature_importance.to_csv(
    os.path.join(output_folder, "feature_importance.csv"),
    index=False
)

# Save classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=["Phishing (0)", "Legitimate (1)"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(output_folder, "classification_report.csv")
)

# Save confusion matrix
pd.DataFrame(
    cm,
    index=["Actual_Phishing", "Actual_Legitimate"],
    columns=["Predicted_Phishing", "Predicted_Legitimate"]
).to_csv(
    os.path.join(output_folder, "confusion_matrix.csv")
)

# Save predictions
predictions = pd.DataFrame({
    "Actual_Label": y_test.to_numpy(),
    "Predicted_Label": y_pred,
    "Phishing_Probability": y_prob[:, list(classes).index(0)],
    "Legitimate_Probability": positive_prob
})

predictions.to_csv(
    os.path.join(output_folder, "predictions.csv"),
    index=False
)

print("\nAll results saved in:", output_folder)

print("\nRandom Forest pipeline saved successfully!")