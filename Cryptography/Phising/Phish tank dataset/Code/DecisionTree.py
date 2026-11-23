
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. Directory configuration
# ==========================================

base_dir = r"C:\Users\Lenovo\Desktop\Cryptography\Phising\Phish tank dataset"

output_dir = os.path.join(base_dir, "Decision Tree")

os.makedirs(output_dir, exist_ok=True)

data_path = r"C:\Users\Lenovo\Desktop\Cryptography\Phising\Phish tank dataset\Final_phishing_dataset_features.csv"

# ==========================================
# 2. Load dataset
# ==========================================

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["target"].value_counts())

# ==========================================
# 3. Separate features and target
# ==========================================

X = df.drop(columns=["url", "target"])
y = df["target"]

# ==========================================
# 4. Train-test split (80:20)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 5. Decision Tree model
# ==========================================

dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# ==========================================
# 6. Train model
# ==========================================

print("\nTraining Decision Tree...")

dt_model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 7. Predictions
# ==========================================

y_pred = dt_model.predict(X_test)

# ==========================================
# 8. Evaluation metrics
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

print("\n========== DECISION TREE RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# ==========================================
# 9. Classification report
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    os.path.join(output_dir, "classification_report.txt"),
    "w"
) as f:
    f.write(report)

# ==========================================
# 10. Confusion matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Phishing"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "confusion_matrix.png"),
    dpi=300
)

plt.show()

# ==========================================
# 11. Feature importance
# ==========================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

importance_df.to_csv(
    os.path.join(output_dir, "feature_importance.csv"),
    index=False
)

# Feature importance plot
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Decision Tree Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "feature_importance.png"),
    dpi=300
)

plt.show()

# ==========================================
# 12. Save trained model
# ==========================================

joblib.dump(
    dt_model,
    os.path.join(output_dir, "decision_tree_model.pkl")
)

# Save feature names
with open(
    os.path.join(output_dir, "feature_names.json"),
    "w"
) as f:
    json.dump(list(X.columns), f, indent=4)

# ==========================================
# 13. Save evaluation metrics
# ==========================================

metrics = {
    "Model": "Decision Tree",
    "Train_Test_Split": "80:20",
    "Training_Samples": len(X_train),
    "Testing_Samples": len(X_test),
    "Number_of_Features": X.shape[1],
    "Criterion": "Gini",
    "Tree_Depth": dt_model.get_depth(),
    "Number_of_Leaves": dt_model.get_n_leaves(),
    "Accuracy": float(accuracy),
    "Precision": float(precision),
    "Recall": float(recall),
    "F1_Score": float(f1)
}

with open(
    os.path.join(output_dir, "metrics.txt"),
    "w"
) as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

# ==========================================
# 14. Save train and test datasets
# ==========================================

train_data = df.loc[X_train.index].copy()
test_data = df.loc[X_test.index].copy()

train_data.to_csv(
    os.path.join(output_dir, "train_data.csv"),
    index=False
)

test_data.to_csv(
    os.path.join(output_dir, "test_data.csv"),
    index=False
)

# ==========================================
# 15. Save test predictions
# ==========================================

predictions = pd.DataFrame({
    "url": df.loc[X_test.index, "url"].values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

predictions.to_csv(
    os.path.join(output_dir, "test_predictions.csv"),
    index=False
)

# ==========================================
# 16. Final summary
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print("Model: Decision Tree")
print("Train-test split: 80:20")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features used:", X.shape[1])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nTree depth:", dt_model.get_depth())
print("Number of leaves:", dt_model.get_n_leaves())

print("\nAll files saved successfully in:")
print(output_dir)