
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from decimal import Decimal, ROUND_DOWN
from functools import partial

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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

base_dir = r"C:\Users\Lenovo\Desktop\Cryptography\Network Intrusion Detection"

output_dir = os.path.join(base_dir, "Decision tree with Feature Selection")

os.makedirs(output_dir, exist_ok=True)

data_path = os.path.join(
    base_dir,
    "Balanced_Network_Intrusion_20000.csv"
)

# ==========================================
# 2. Load dataset
# ==========================================

df = pd.read_csv(data_path)

print("Original Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["Label"].value_counts())

# ==========================================
# 3. Separate features and target
# ==========================================

X = df.drop(columns=["Label"]).copy()
y = df["Label"]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinity with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Remove completely empty columns
empty_columns = X.columns[X.isna().all()].tolist()

if empty_columns:
    print("\nRemoving completely empty features:", empty_columns)
    X = X.drop(columns=empty_columns)

print("\nFeatures before selection:", X.shape[1])

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

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# 5. Feature Selection + Decision Tree
# ==========================================

mi_score_function = partial(
    mutual_info_classif,
    random_state=42
)

pipeline = Pipeline([

    # Handle missing values
    ("imputer", SimpleImputer(strategy="median")),

    # Mutual Information Feature Selection
    ("feature_selection", SelectKBest(
        score_func=mi_score_function
    )),

    # Decision Tree Classifier
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ))
])

# ==========================================
# 6. Find the best number of features
# ==========================================

# Compare different feature counts
param_grid = {
    "feature_selection__k": [
        10, 15, 20, 25, 30, "all"
    ]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("\nFinding the best number of features...")

grid_search.fit(X_train, y_train)

# Best trained model
model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)

print(
    "Best Cross-Validation F1 Score:",
    Decimal(str(grid_search.best_score_)).quantize(
        Decimal("0.0001"),
        rounding=ROUND_DOWN
    )
)

# ==========================================
# 7. Get selected features
# ==========================================

selector = model.named_steps["feature_selection"]

selected_features = X.columns[
    selector.get_support()
].tolist()

print("\n==========================================")
print("FEATURE SELECTION RESULTS")
print("==========================================")

print("Original Features:", X.shape[1])
print("Selected Features:", len(selected_features))
print("Removed Features:", X.shape[1] - len(selected_features))

print("\nSelected Feature Names:")

for i, feature in enumerate(selected_features, 1):
    print(i, feature)

# ==========================================
# 8. Mutual Information scores
# ==========================================

mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MI_Score": selector.scores_
})

mi_df = mi_df.sort_values(
    by="MI_Score",
    ascending=False
).reset_index(drop=True)

print("\nMutual Information Ranking:")

print(mi_df.to_string(
    index=False,
    formatters={
        "MI_Score": lambda x: str(
            Decimal(str(x)).quantize(
                Decimal("0.0001"),
                rounding=ROUND_DOWN
            )
        )
    }
))

mi_df.to_csv(
    os.path.join(output_dir, "mutual_information_scores.csv"),
    index=False
)

# ==========================================
# 9. Save selected feature names
# ==========================================

with open(
    os.path.join(output_dir, "selected_features.json"),
    "w"
) as f:
    json.dump(selected_features, f, indent=4)

# ==========================================
# 10. Create reduced dataset
# ==========================================

# Apply the fitted imputer and feature selector
X_reduced = selector.transform(
    model.named_steps["imputer"].transform(X)
)

reduced_df = pd.DataFrame(
    X_reduced,
    columns=selected_features,
    index=X.index
)

reduced_df["Label"] = y

reduced_df.to_csv(
    os.path.join(output_dir, "Reduced_Network_Intrusion_Dataset.csv"),
    index=False
)

print("\nReduced Dataset Shape:", reduced_df.shape)

# ==========================================
# 11. Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 12. Evaluation metrics
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

# Truncate to 4 decimal places
def truncate(value):
    return Decimal(str(value)).quantize(
        Decimal("0.0001"),
        rounding=ROUND_DOWN
    )

print("\n========== DECISION TREE RESULTS ==========")

print("Accuracy  :", truncate(accuracy))
print("Precision :", truncate(precision))
print("Recall    :", truncate(recall))
print("F1-Score  :", truncate(f1))

# ==========================================
# 13. Classification report
# ==========================================

report_dict = classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Attack"],
    output_dict=True,
    zero_division=0
)

print("\nClassification Report:\n")

print(
    f"{'':>14}{'precision':>12}{'recall':>12}"
    f"{'f1-score':>12}{'support':>12}"
)

for label in ["Benign", "Attack"]:

    values = report_dict[label]

    print(
        f"{label:>14}"
        f"{str(truncate(values['precision'])):>12}"
        f"{str(truncate(values['recall'])):>12}"
        f"{str(truncate(values['f1-score'])):>12}"
        f"{int(values['support']):>12}"
    )

print(
    f"{'accuracy':>14}"
    f"{'':>12}{'':>12}"
    f"{str(truncate(report_dict['accuracy'])):>12}"
    f"{len(y_test):>12}"
)

for label in ["macro avg", "weighted avg"]:

    values = report_dict[label]

    print(
        f"{label:>14}"
        f"{str(truncate(values['precision'])):>12}"
        f"{str(truncate(values['recall'])):>12}"
        f"{str(truncate(values['f1-score'])):>12}"
        f"{int(values['support']):>12}"
    )

# Save report
report = classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Attack"],
    zero_division=0
)

with open(
    os.path.join(output_dir, "classification_report.txt"),
    "w"
) as f:
    f.write(report)

# ==========================================
# 14. Confusion matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Attack"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Decision Tree with MI Feature Selection")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "confusion_matrix.png"),
    dpi=300
)

plt.show()

# ==========================================
# 15. Decision Tree feature importance
# ==========================================

dt_model = model.named_steps["classifier"]

importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": dt_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nDecision Tree Feature Importance:")

print(importance_df.to_string(
    index=False,
    formatters={
        "Importance": lambda x: str(truncate(x))
    }
))

importance_df.to_csv(
    os.path.join(output_dir, "feature_importance.csv"),
    index=False
)

# ==========================================
# 16. Feature importance visualization
# ==========================================

top_features = importance_df.head(20)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")

plt.title("Feature Importance - Decision Tree")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "feature_importance.png"),
    dpi=300
)

plt.show()

# ==========================================
# 17. Save trained model
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": list(X.columns),
        "selected_features": selected_features
    },
    os.path.join(
        output_dir,
        "decision_tree_MI_model.pkl"
    )
)

# ==========================================
# 18. Save evaluation metrics
# ==========================================

metrics = {
    "Model": "Decision Tree + Mutual Information",
    "Feature_Selection": "SelectKBest",
    "Train_Test_Split": "80:20",
    "Cross_Validation": "5-Fold",
    "Training_Samples": len(X_train),
    "Testing_Samples": len(X_test),
    "Original_Features": X.shape[1],
    "Selected_Features": len(selected_features),
    "Best_CV_F1_Score": str(truncate(grid_search.best_score_)),
    "Criterion": "Gini",
    "Tree_Depth": dt_model.get_depth(),
    "Number_of_Leaves": dt_model.get_n_leaves(),
    "Accuracy": str(truncate(accuracy)),
    "Precision": str(truncate(precision)),
    "Recall": str(truncate(recall)),
    "F1_Score": str(truncate(f1))
}

with open(
    os.path.join(output_dir, "metrics.txt"),
    "w"
) as f:

    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

# ==========================================
# 19. Save train and test datasets
# ==========================================

X_train[selected_features].assign(
    Label=y_train
).to_csv(
    os.path.join(output_dir, "train_data.csv"),
    index=False
)

X_test[selected_features].assign(
    Label=y_test
).to_csv(
    os.path.join(output_dir, "test_data.csv"),
    index=False
)

# ==========================================
# 20. Save test predictions
# ==========================================

predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

predictions.to_csv(
    os.path.join(output_dir, "test_predictions.csv"),
    index=False
)

# ==========================================
# 21. Final summary
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print("Model: Decision Tree")
print("Feature Selection: Mutual Information")
print("Train-Test Split: 80:20")

print("Original Features:", X.shape[1])
print("Selected Features:", len(selected_features))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("Accuracy:", truncate(accuracy))
print("Precision:", truncate(precision))
print("Recall:", truncate(recall))
print("F1-Score:", truncate(f1))

print("\nTree Depth:", dt_model.get_depth())
print("Number of Leaves:", dt_model.get_n_leaves())

print("\nAll files saved successfully in:")
print(output_dir)
