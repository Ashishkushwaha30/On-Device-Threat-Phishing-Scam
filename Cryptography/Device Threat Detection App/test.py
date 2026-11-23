from app import load_assets, phishing_features_for_url

# Load the existing trained assets
assets = load_assets()

test_urls = [
    "https://www.kaggle.com/",
    "https://www.kaggle.com/login/",
    "https://www.g00gle.com/",
    "https://google.com/",
    "https://example.com/path?id=123"
]

print("\nPHISHING FEATURE EXTRACTION TEST")
print("=" * 70)

for url in test_urls:
    features = phishing_features_for_url(url, assets)

    print("\nURL:", url)
    print("Extracted features:", features)