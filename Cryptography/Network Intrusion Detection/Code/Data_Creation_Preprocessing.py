
# ============================================
# CREATE BALANCED BINARY CLASSIFICATION DATASET
# CIC-IDS2017
# ============================================

import pandas as pd
import numpy as np
import glob
import os

# Folder containing all 8 CSV files
folder_path = r"C:\Users\Lenovo\Desktop\Cryptography\Network Intrusion Detection"

# Find all CSV files
files = glob.glob(os.path.join(folder_path, "*.csv"))

# Read and combine all files
dataframes = []

for file in files:

    print("Reading:", os.path.basename(file))

    df = pd.read_csv(file, low_memory=False)

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Standardize label column name
    label_col = next(
        (c for c in df.columns if c.lower() == "label"),
        None
    )

    if label_col is None:
        raise ValueError(f"Label column missing in {file}")

    df = df.rename(columns={label_col: "Label"})

    # Check duplicate column names
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in {file}")

    dataframes.append(df)

# Combine all datasets
data = pd.concat(dataframes, ignore_index=True)

print("\nOriginal Dataset Shape:", data.shape)

# ============================================
# DATA CLEANING
# ============================================

# Remove extra spaces and standardize labels
data["Label"] = data["Label"].astype(str).str.strip().str.upper()

# Remove rows having missing labels
data = data[data["Label"] != "NAN"]

# Convert labels into binary classes
data["Label"] = np.where(
    data["Label"] == "BENIGN",
    0,
    1
)

# Separate classes
benign = data[data["Label"] == 0]
attack = data[data["Label"] == 1]

print("\nAvailable Benign Samples:", len(benign))
print("Available Attack Samples:", len(attack))

# ============================================
# SELECT 10000 SAMPLES FROM EACH CLASS
# ============================================

if len(benign) < 10000 or len(attack) < 10000:
    raise ValueError(
        "Not enough samples to create a balanced dataset."
    )

benign_sample = benign.sample(
    n=10000,
    random_state=42
)

attack_sample = attack.sample(
    n=10000,
    random_state=42
)

# Combine both classes
balanced_data = pd.concat(
    [benign_sample, attack_sample],
    ignore_index=True
)

# Shuffle dataset
balanced_data = balanced_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ============================================
# HANDLE INVALID FEATURE VALUES
# ============================================

# Replace infinity with NaN
balanced_data.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Convert feature columns to numeric
feature_cols = [
    col for col in balanced_data.columns
    if col != "Label"
]

for col in feature_cols:
    balanced_data[col] = pd.to_numeric(
        balanced_data[col],
        errors="coerce"
    )

# Remove columns containing no valid values
empty_cols = balanced_data[feature_cols].columns[
    balanced_data[feature_cols].isna().all()
]

balanced_data.drop(columns=empty_cols, inplace=True)

# ============================================
# SAVE FINAL DATASET
# ============================================

output_path = os.path.join(
    folder_path,
    "Balanced_Network_Intrusion_20000.csv"
)

balanced_data.to_csv(output_path, index=False)

print("\nFinal Dataset Shape:", balanced_data.shape)

print("\nClass Distribution:")
print(balanced_data["Label"].value_counts())

print("\nDataset saved successfully!")
print(output_path)