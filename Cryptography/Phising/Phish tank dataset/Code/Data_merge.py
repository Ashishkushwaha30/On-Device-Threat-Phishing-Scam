import pandas as pd

# Load datasets
df1 = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Cryptography\\Phising\\Phish tank dataset\\tranco_746WX.csv")
df2 = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Cryptography\\Phising\\Phish tank dataset\\verified_online.csv")

print("Original dataset shapes:")
print("df1:", df1.shape)
print("df2:", df2.shape)

# Assign target labels
df1["target"] = 0
df2["target"] = 1

# Take equal number of samples from df1 as df2
n = len(df2)

df1 = df1.sample(n=n, random_state=42)

# Merge both datasets
final_df = pd.concat([df1, df2], ignore_index=True)

# Shuffle final dataset
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final dataset
final_df.to_csv("final_dataset.csv", index=False)

# Print final dataset information
print("\nFinal dataset shape:", final_df.shape)

print("\nTarget distribution:")
print(final_df["target"].value_counts().sort_index())

print("\nClass-wise sample count:")
print("Class 0:", (final_df["target"] == 0).sum())
print("Class 1:", (final_df["target"] == 1).sum())

print("\nFinal dataset saved as final_dataset.csv")