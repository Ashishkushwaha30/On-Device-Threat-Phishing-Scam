
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. Create Output Folder
# ==========================================

base_folder = r"C:\Users\Lenovo\Desktop\Cryptography\SMS Spam"

output_folder = os.path.join(base_folder, "Decision Tree")

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# 2. Load Dataset
# ==========================================

data_path = os.path.join(
    base_folder,
    "SMSSpamCollection_Final.csv"
)

df = pd.read_csv(data_path)

df = df.dropna(subset=["text", "target"])

df["text"] = df["text"].astype(str)
df["target"] = df["target"].astype(int)

X = df["text"]
y = df["target"]

print("Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(y.value_counts())

# ==========================================
# 3. Train-Test Split (80:20)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# 4. TF-IDF Feature Extraction
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Training Shape:", X_train_tfidf.shape)
print("TF-IDF Testing Shape:", X_test_tfidf.shape)

# ==========================================
# 5. Decision Tree Hyperparameter Tuning
# ==========================================

dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],

    "max_depth": [10, 20, 30, 50, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "class_weight": [None, "balanced"]
}

# Stratified 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

print("\nStarting Decision Tree Hyperparameter Tuning...")

grid_search.fit(X_train_tfidf, y_train)

# ==========================================
# 6. Select Best Model
# ==========================================

best_model = grid_search.best_estimator_

print("\n========== BEST PARAMETERS ==========")

print(grid_search.best_params_)

print(
    "\nBest Cross-Validation Macro F1:",
    round(grid_search.best_score_, 4)
)

# ==========================================
# 7. Final Predictions on Test Set
# ==========================================

y_pred = best_model.predict(X_test_tfidf)

y_prob = best_model.predict_proba(X_test_tfidf)[:, 1]

# ==========================================
# 8. Evaluation Metrics
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

macro_f1 = f1_score(
    y_test, y_pred, average="macro"
)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== FINAL MODEL PERFORMANCE ==========")

print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")
print(f"Macro F1-Score : {macro_f1:.4f}")
print(f"ROC-AUC        : {roc_auc:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

# ==========================================
# 9. Save Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Save confusion matrix as CSV
cm_df = pd.DataFrame(
    cm,
    index=["Actual_Not_Spam", "Actual_Spam"],
    columns=["Predicted_Not_Spam", "Predicted_Spam"]
)

cm_df.to_csv(
    os.path.join(output_folder, "confusion_matrix.csv")
)

# Save confusion matrix as PNG
fig, ax = plt.subplots(figsize=(7, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Spam", "Spam"]
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Decision Tree Confusion Matrix")

fig.tight_layout()

fig.savefig(
    os.path.join(output_folder, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nConfusion Matrix saved successfully!")

# ==========================================
# 10. Save Only the Best Model and Vectorizer
# ==========================================

joblib.dump(
    best_model,
    os.path.join(output_folder, "best_decision_tree_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(output_folder, "tfidf_vectorizer.pkl")
)

# ==========================================
# 11. Save Evaluation Results
# ==========================================

results = pd.DataFrame({
    "Metric": [
        "Best CV Macro F1",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Macro F1-Score",
        "ROC-AUC"
    ],

    "Score": [
        grid_search.best_score_,
        accuracy,
        precision,
        recall,
        f1,
        macro_f1,
        roc_auc
    ]
})

results.to_csv(
    os.path.join(output_folder, "evaluation_results.csv"),
    index=False
)

# ==========================================
# 12. Save Best Hyperparameters
# ==========================================

best_params = pd.DataFrame(
    list(grid_search.best_params_.items()),
    columns=["Parameter", "Best Value"]
)

best_params.to_csv(
    os.path.join(output_folder, "best_parameters.csv"),
    index=False
)

# ==========================================
# 13. Final Confirmation
# ==========================================

print("\n====================================")
print("DECISION TREE TRAINING COMPLETED")
print("====================================")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Macro F1:", round(grid_search.best_score_, 4))

print("\nFinal Test Macro F1:", round(macro_f1, 4))

print("\nAll files saved successfully!")

print("\nOutput Folder:")
print(output_folder)