
import pandas as pd
import numpy as np
from urllib.parse import urlparse

# Load dataset
df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Cryptography\\Phising\\Phish tank dataset\\Final_phishing_dataset.csv")

# Remove missing URLs
df = df.dropna(subset=["url"]).copy()

# Convert URL to string
df["url"] = df["url"].astype(str).str.strip()

# Remove HTTP/HTTPS and www
df["url"] = df["url"].str.replace(
    r"^https?://", "", regex=True, case=False
)

df["url"] = df["url"].str.replace(
    r"^www\.", "", regex=True, case=False
)


# Feature extraction
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
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "path_length": len(path)
    }


# Extract features
features_df = pd.DataFrame(
    df["url"].apply(extract_features).tolist()
)

# Combine URL, features and target
final_df = pd.concat(
    [
        df[["url", "target"]].reset_index(drop=True),
        features_df.reset_index(drop=True)
    ],
    axis=1
)

# Handle missing/infinite values
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.fillna(0, inplace=True)

# Save processed dataset
final_df.to_csv("C:\\Users\\Lenovo\\Desktop\\Cryptography\\Phising\\Phish tank dataset\\phishing_features.csv", index=False)

# Display results
print("Final dataset shape:", final_df.shape)

print("\nClass distribution:")
print(final_df["target"].value_counts())

print("\nColumns:")
print(final_df.columns.tolist())

print("\nFirst five samples:")
print(final_df.head())

print("\nSaved as phishing_features.csv")