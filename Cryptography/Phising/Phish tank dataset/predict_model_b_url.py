import json
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import tldextract
from rapidfuzz import process, fuzz

# Model B artifacts created by train_model_b_no_url_length.py
MODEL_DIR = Path(
    r"C:\Users\Lenovo\Desktop\Cryptography\Phising\Phish tank dataset"
) / "Decision Tree Brand Detection No URL Length"

MODEL_FILE = MODEL_DIR / "decision_tree_brand_model.pkl"
FEATURE_FILE = MODEL_DIR / "feature_names.json"
BRAND_FILE = MODEL_DIR / "brand_library.json"

extractor = tldextract.TLDExtract(suffix_list_urls=None)


def normalize_brand(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower().strip())


def add_scheme(url):
    url = str(url).strip()
    if url and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def extract_domain_parts(url):
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
    raw = str(url).strip()
    try:
        parsed = urlparse(add_scheme(raw))
        if parsed.path in ("", "/") and not parsed.query and not parsed.fragment:
            return raw.rstrip("/")
    except Exception:
        pass
    return raw


# Load the exact sorted brand library saved during Model B training.
with open(BRAND_FILE, "r", encoding="utf-8") as f:
    brands = json.load(f)

brand_set = set(brands)
brand_lengths = sorted({len(brand) for brand in brands if len(brand) >= 4})
brand_index = {}
for brand in brands:
    if brand:
        brand_index.setdefault(brand[0], []).append(brand)


def calculate_brand_similarity(domain_name):
    domain_name = normalize_brand(domain_name)
    if not domain_name:
        return 0.0
    candidates = brand_index.get(domain_name[0], [])
    if not candidates:
        return 0.0
    result = process.extractOne(domain_name, candidates, scorer=fuzz.ratio)
    return (result[1] / 100.0) if result else 0.0


def calculate_brand_substring_match(domain_name):
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


def extract_model_b_features(url):
    """Create the same 10 features and order used when training Model B."""
    raw_url = str(url).strip()
    feature_url = normalize_root_trailing_slash(raw_url)
    hostname, domain_name = extract_domain_parts(feature_url)

    try:
        parsed = urlparse(add_scheme(feature_url))
        path = parsed.path or ""
    except Exception:
        path = ""

    normalized_domain = normalize_brand(domain_name)
    return {
        "hostname_length": len(hostname),
        "dot_count": feature_url.count("."),
        "hyphen_count": feature_url.count("-"),
        "at_count": feature_url.count("@"),
        "digit_count": sum(ch.isdigit() for ch in feature_url),
        "subdomain_count": max(0, len(hostname.split(".")) - 2) if hostname else 0,
        "path_length": len(path),
        "brand_similarity": calculate_brand_similarity(normalized_domain),
        "exact_brand_match": int(normalized_domain in brand_set if normalized_domain else False),
        "brand_substring_match": calculate_brand_substring_match(normalized_domain),
    }


def main():
    if not MODEL_FILE.exists() or not FEATURE_FILE.exists() or not BRAND_FILE.exists():
        raise FileNotFoundError(
            "Model B files not found. Run train_model_b_no_url_length.py first."
        )

    model = joblib.load(MODEL_FILE)
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    print("MODEL B - PHISHING URL DETECTION (WITHOUT url_length)")
    print("Type 'exit' to stop.\n")

    while True:
        url = input("Enter URL: ").strip()
        if url.lower() in {"exit", "quit", "q"}:
            break
        if not url:
            print("Please enter a URL.\n")
            continue

        feature_values = extract_model_b_features(url)
        # Use the saved feature order, not dictionary order assumptions.
        row = [[feature_values[name] for name in feature_names]]
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        class_to_probability = {
            int(cls): float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }

        if prediction == 1:
            label = "PHISHING"
        elif prediction == 0:
            label = "LEGITIMATE"
        else:
            label = f"UNKNOWN CLASS ({prediction})"

        print("Prediction:", label)
        print(f"Phishing probability: {class_to_probability.get(1, 0.0):.4f}")
        print(f"Legitimate probability: {class_to_probability.get(0, 0.0):.4f}")
        print("Features used:")
        for name in feature_names:
            print(f"  {name}: {feature_values[name]}")
        print()


if __name__ == "__main__":
    main()
