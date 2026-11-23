import pandas as pd
import os

folder = r"C:\Users\Lenovo\Desktop\Cryptography\SMS Spam"

input_file = os.path.join(folder, "SMSSpamCollection")
output_file = os.path.join(folder, "SMSSpamCollection_Final.csv")

# Read CSV file
df = pd.read_csv(input_file, encoding="utf-8")

# Keep required columns
df = df[["text", "target"]]

# Remove missing values
df = df.dropna(subset=["text", "target"])

# Convert target to integer
df["target"] = df["target"].astype(int)

# Save final CSV
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Dataset saved successfully!")
print("Shape:", df.shape)
print("\nClass distribution:")
print(df["target"].value_counts())
print("\nFirst 5 rows:")
print(df.head())

print("\nSaved at:", output_file)