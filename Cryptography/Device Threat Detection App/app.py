from __future__ import annotations

import json
import re
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tldextract
from rapidfuzz import fuzz, process
from scipy import sparse

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DB_PATH = APP_DIR / "prediction_history.db"

# Existing trained artifacts: paths are relative to the Cryptography folder.
PATHS = {
    "nids": PROJECT_ROOT / "Network Intrusion Detection" / "Decision tree with Feature Selection" / "decision_tree_MI_model.pkl",
    "phishing_model": PROJECT_ROOT / "Phising" / "Phish tank dataset" / "Decision Tree Brand Detection Validated" / "decision_tree_brand_model.pkl",
    "phishing_features": PROJECT_ROOT / "Phising" / "Phish tank dataset" / "Decision Tree Brand Detection Validated" / "feature_names.json",
    "brand_library": PROJECT_ROOT / "Phising" / "Phish tank dataset" / "Decision Tree Brand Detection Validated" / "brand_library.json",
    "sms_model": PROJECT_ROOT / "SMS Spam" / "Decision Tree" / "best_decision_tree_model.pkl",
    "sms_vectorizer": PROJECT_ROOT / "SMS Spam" / "Decision Tree" / "tfidf_vectorizer.pkl",
}
LLAMA_PATH = APP_DIR / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

st.set_page_config(page_title="Device Threat Detection", page_icon="🛡️", layout="wide")

@st.cache_resource(show_spinner="Loading trained models and feature library...")
def load_assets():
    missing = [str(p) for p in PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing))

    nids_bundle = joblib.load(PATHS["nids"])
    phishing_model = joblib.load(PATHS["phishing_model"])
    sms_model = joblib.load(PATHS["sms_model"])
    sms_vectorizer = joblib.load(PATHS["sms_vectorizer"])

    with open(PATHS["phishing_features"], "r", encoding="utf-8") as f:
        phishing_features = json.load(f)
    with open(PATHS["brand_library"], "r", encoding="utf-8") as f:
        brands = json.load(f)

    if not isinstance(nids_bundle, dict) or "model" not in nids_bundle or "features" not in nids_bundle:
        raise ValueError("Unexpected NIDS pickle structure.")
    if len(phishing_features) != int(phishing_model.n_features_in_):
        raise ValueError("Phishing feature_names.json does not match the model input size.")
    if int(sms_vectorizer.transform(["startup check"]).shape[1]) != int(sms_model.n_features_in_):
        raise ValueError("SMS vectorizer and model feature dimensions do not match.")
    if not isinstance(brands, list):
        raise ValueError("brand_library.json must contain a list.")

    # Keep the exact normalized entries from the validated training artifact.
    brand_set = set(brands)
    brand_lengths = sorted({len(x) for x in brand_set if len(x) >= 4})
    brand_index = {}
    # Preserve the sorted candidate order used by the original training script.
    # This also makes RapidFuzz tie handling deterministic and consistent.
    for brand in brands:
        if brand:
            brand_index.setdefault(brand[0], []).append(brand)

    return {
        "nids_bundle": nids_bundle,
        "nids_model": nids_bundle["model"],
        "nids_features": list(nids_bundle["features"]),
        "phishing_model": phishing_model,
        "phishing_features": phishing_features,
        "brands": brand_set,
        "brand_lengths": brand_lengths,
        "brand_index": brand_index,
        "sms_model": sms_model,
        "sms_vectorizer": sms_vectorizer,
    }

def normalize_brand(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower().strip())

def add_scheme(url: str) -> str:
    value = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return "http://" + value
    return value

def normalize_root_trailing_slash(url: str) -> str:
    value = url.strip()
    parsed = urlparse(add_scheme(value))
    if parsed.path in ("", "/") and not parsed.query and not parsed.fragment:
        return value.rstrip("/")
    return value


def shorten_url_for_prediction(url: str) -> str:
    """
    Demonstration-only preprocessing:
    - removes http:// or https://
    - removes leading www.
    - truncates to a randomly selected length between 13 and 17 characters

    The shortened value is intentionally used for phishing feature extraction
    and prediction in this prototype.
    """
    value = str(url).strip()
    value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^www\\.", "", value, flags=re.IGNORECASE)

    target_length = random.randint(13, 17)
    if len(value) > target_length:
        value = value[:target_length]

    return value

def phishing_features_for_url(url: str, assets: dict) -> dict:
    clean_url = normalize_root_trailing_slash(url)
    parsed = urlparse(add_scheme(clean_url))
    # Match the original training script: remove leading/trailing hostname dots.
    hostname = (parsed.hostname or "").lower().strip(".")
    extracted = tldextract.TLDExtract(suffix_list_urls=None)(hostname)
    domain_name = normalize_brand(extracted.domain)

    brand_similarity = 0.0
    candidates = assets["brand_index"].get(domain_name[:1], []) if domain_name else []
    if domain_name and candidates:
        best = process.extractOne(domain_name, candidates, scorer=fuzz.ratio)
        if best:
            brand_similarity = float(best[1]) / 100.0

    exact_brand_match = int(bool(domain_name) and domain_name in assets["brands"])

    substring_match = 0
    if domain_name:
        for length in assets["brand_lengths"]:
            if length > len(domain_name):
                break
            if any(domain_name[i:i + length] in assets["brands"]
                   for i in range(len(domain_name) - length + 1)):
                substring_match = 1
                break

    # Feature formulas mirror the validated training pipeline.
    host = hostname
    path = parsed.path or ""
    feature_values = {
        "url_length": len(clean_url),
        "hostname_length": len(host),
        "dot_count": clean_url.count("."),
        "hyphen_count": clean_url.count("-"),
        "at_count": clean_url.count("@"),
        "digit_count": sum(ch.isdigit() for ch in clean_url),
        "subdomain_count": max(0, len(host.split(".")) - 2) if host else 0,
        "path_length": len(path),
        "brand_similarity": brand_similarity,
        "exact_brand_match": exact_brand_match,
        "brand_substring_match": substring_match,
    }
    return feature_values

def decision_path_explanation(tree_model, X, feature_names, max_items=8):
    """Return the actual branch conditions traversed for one row."""
    row = X
    if sparse.issparse(row):
        row = row.toarray()
    row = np.asarray(row, dtype=float)
    if row.ndim == 1:
        row = row.reshape(1, -1)

    tree = tree_model.tree_
    node_indicator = tree_model.decision_path(row)
    leaf_id = int(tree_model.apply(row)[0])
    node_ids = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    conditions = []
    for node_id in node_ids:
        if node_id == leaf_id:
            continue
        feature_idx = int(tree.feature[node_id])
        if feature_idx < 0 or feature_idx >= len(feature_names):
            continue
        threshold = float(tree.threshold[node_id])
        value = float(row[0, feature_idx])
        operator = "<=" if value <= threshold else ">"
        conditions.append(
            f"{feature_names[feature_idx]} = {value:.4g} {operator} {threshold:.4g}"
        )
    return conditions[:max_items]

def save_history(module, item, label, confidence, details):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                module TEXT NOT NULL,
                input_summary TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                details TEXT
            )
        """)
        conn.execute(
            "INSERT INTO predictions (created_at,module,input_summary,prediction,confidence,details) VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), module, item[:500], label,
             float(confidence), json.dumps(details, ensure_ascii=False))
        )

def load_history(limit=100):
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT created_at,module,input_summary,prediction,confidence FROM predictions ORDER BY id DESC LIMIT ?",
            conn, params=(limit,)
        )

@st.cache_resource(show_spinner=False)
def load_local_llama(model_path: str):
    from llama_cpp import Llama
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=max(2, min(4, (os_cpu_count() or 4) - 1)),
        n_batch=128,
        n_gpu_layers=0,
        verbose=False,
    )

def os_cpu_count():
    import os
    return os.cpu_count()

def explain_with_llama(module: str, verdict: str, probability: float, evidence: list[str]) -> str:
    if not LLAMA_PATH.exists():
        return (
            "Local Llama is not configured yet. The model prediction above is from your trained "
            "Decision Tree. Evidence: " + ("; ".join(evidence[:5]) if evidence else "No path details available.")
        )
    try:
        llm = load_local_llama(str(LLAMA_PATH))
        prompt = f"""
            You are a cybersecurity assistant explaining machine learning predictions.

            Explain the model's prediction in simple, professional language.

            Module: {module}
            Input: {user_input}
            Prediction: {label}
            Probability: {confidence:.2f}%
            Evidence: {evidence}

            Instructions:
            1. Never refuse to explain a harmless cybersecurity classification.
            2. Explain only what the machine learning model has predicted.
            3. Do not claim that a device is compromised or secure based only on a URL.
            4. Do not invent security evidence or verification results.
            5. Clearly distinguish model predictions from confirmed facts.
            6. Give a concise explanation in 2-3 sentences.

            Generate the explanation:
            """
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Explain only the supplied classifier output and evidence."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=160,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"Local Llama could not start ({type(exc).__name__}). The classifier result remains valid; use the rule-based evidence below."

def show_result(module, item, prediction, probabilities, class_labels, evidence):
    pred = int(prediction)
    verdict = class_labels[pred]
    confidence = float(probabilities[pred])
    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error(f"**{verdict}**")
        else:
            st.success(f"**{verdict}**")
    with col2:
        st.metric("Model confidence for predicted class", f"{confidence:.1%}")
    st.progress(max(0.0, min(1.0, confidence)))
    st.caption("The confidence is the model's predicted class probability, not a guarantee.")
    st.subheader("Decision-tree evidence")
    if evidence:
        for line in evidence:
            st.write("• " + line)
    else:
        st.write("No branch conditions were available to display.")
    st.subheader("Plain-language explanation")
    with st.spinner("Preparing explanation..."):
        st.write(explain_with_llama(module, verdict, confidence, evidence))
    save_history(module, item, verdict, confidence, {"evidence": evidence})

def nids_page(assets):
    st.header("Network Intrusion Detection")
    st.write("Upload a CSV containing one or more flow records with the exact 78 CICFlowMeter feature columns used during training.")
    st.info("The app does not capture live network packets. This prototype classifies uploaded flow-feature records.")
    uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"], key="nids_csv")
    if uploaded is None:
        st.caption("Required columns are taken from the saved NIDS model bundle.")
        with st.expander("View required NIDS columns"):
            st.code("\n".join(assets["nids_features"]))
        return
    try:
        df = pd.read_csv(uploaded)
        required = assets["nids_features"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"CSV is missing {len(missing)} required columns.")
            st.code("\n".join(missing))
            return
        X = df.loc[:, required].apply(pd.to_numeric, errors="coerce")
        model = assets["nids_model"]
        preds = model.predict(X)
        probs = model.predict_proba(X)
        result = df.copy()
        result["prediction"] = ["Attack / Intrusion" if int(p) == 1 else "Benign / Normal" for p in preds]
        result["predicted_class_probability"] = probs.max(axis=1)
        st.dataframe(result.head(100), use_container_width=True)
        st.download_button("Download predictions CSV", result.to_csv(index=False).encode("utf-8"),
                           "nids_predictions.csv", "text/csv")
        if len(df) == 1:
            pipeline = model
            imputed = pipeline.named_steps["imputer"].transform(X)
            selected = pipeline.named_steps["feature_selection"].transform(imputed)
            tree_model = pipeline.named_steps["classifier"]
            evidence = decision_path_explanation(tree_model, selected, required)
            show_result("NIDS", "Single uploaded network flow", preds[0], probs[0],
                        {0: "Benign / Normal", 1: "Attack / Intrusion"}, evidence)
        else:
            st.success(f"Processed {len(df)} flow records.")
            st.caption("For a multi-row file, detailed branch explanations are shown only for a single-row upload.")
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                save_history("NIDS", f"CSV row {i + 1}", "Attack / Intrusion" if int(pred) else "Benign / Normal",
                             float(prob[int(pred)]), {"source": "uploaded CSV"})
    except Exception as exc:
        st.exception(exc)

def phishing_page(assets):
    st.header("Phishing URL Detection")
    url = st.text_input("Enter a URL", placeholder="https://example.com/login")
    if st.button("Analyze URL", type="primary", disabled=not url.strip()):
        try:
            # The shortened URL is used for both feature extraction and prediction.
            prediction_url = shorten_url_for_prediction(url)
            # st.caption(f"URL used for analysis: {prediction_url}")

            features = phishing_features_for_url(prediction_url, assets)
            X = pd.DataFrame([[features[name] for name in assets["phishing_features"]]],
                             columns=assets["phishing_features"])
            model = assets["phishing_model"]
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]

            # Do not expose extracted feature values or decision-tree feature rules
            # in the Streamlit phishing interface.
            show_result("Phishing URL", url, pred, probs,
                        {0: "Legitimate URL", 1: "Phishing URL"}, [])
        except Exception as exc:
            st.exception(exc)

def sms_page(assets):
    st.header("SMS Spam Detection")
    sms = st.text_area("Enter SMS message", height=150, placeholder="Paste the message here...")
    if st.button("Analyze SMS", type="primary", disabled=not sms.strip()):
        try:
            X = assets["sms_vectorizer"].transform([sms])
            model = assets["sms_model"]
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            feature_names = assets["sms_vectorizer"].get_feature_names_out()
            evidence = decision_path_explanation(model, X, feature_names)
            show_result("SMS Spam", sms, pred, probs,
                        {0: "Not Spam (Ham)", 1: "Spam"}, evidence)
        except Exception as exc:
            st.exception(exc)

def main():
    st.title("🛡️ Device Threat Detection")
    st.caption("A local prototype using your three previously trained Decision Tree models.")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose detection module",
                            ["Dashboard", "Network Intrusion Detection", "Phishing URL Detection",
                             "SMS Spam Detection", "Prediction History", "Model Status"])
    try:
        assets = load_assets()
    except Exception as exc:
        st.error("The application could not load one or more trained artifacts.")
        st.exception(exc)
        st.info("Check that this app folder is located directly inside your Cryptography folder, as described in README.md.")
        st.stop()

    if page == "Dashboard":
        st.subheader("Security analysis modules")
        c1, c2, c3 = st.columns(3)
        c1.metric("NIDS input features", len(assets["nids_features"]))
        c2.metric("Phishing input features", len(assets["phishing_features"]))
        c3.metric("SMS TF-IDF vocabulary", len(assets["sms_vectorizer"].vocabulary_))
        st.markdown(
            "- **Network Intrusion Detection:** classifies uploaded network-flow CSV rows.\n"
            "- **Phishing URL Detection:** removes the URL scheme and leading www, then randomly truncates the URL before extracting features for the saved 11-feature model. This is a demonstration-only preprocessing experiment.\n"
            "- **SMS Spam Detection:** classifies text using the saved TF-IDF vectorizer and Decision Tree.\n\n"
            "All classification is performed locally. The optional Llama explanation also runs locally once its GGUF file is installed."
        )
        st.warning("This is a research prototype, not a replacement for endpoint protection, a firewall, or professional incident response.")
    elif page == "Network Intrusion Detection":
        nids_page(assets)
    elif page == "Phishing URL Detection":
        phishing_page(assets)
    elif page == "SMS Spam Detection":
        sms_page(assets)
    elif page == "Prediction History":
        st.header("Local Prediction History")
        history = load_history()
        if history.empty:
            st.info("No predictions have been saved yet.")
        else:
            st.dataframe(history, use_container_width=True)
            st.download_button("Download history CSV", history.to_csv(index=False).encode("utf-8"),
                               "prediction_history.csv", "text/csv")
    elif page == "Model Status":
        st.header("Loaded model status")
        st.write("NIDS pipeline steps:", [name for name, _ in assets["nids_model"].steps])
        st.write("NIDS original input features:", len(assets["nids_features"]))
        st.write("NIDS tree depth:", assets["nids_model"].named_steps["classifier"].tree_.max_depth)
        st.write("Phishing model input features:", assets["phishing_model"].n_features_in_)
        st.write("Brand library entries:", len(assets["brands"]))
        st.write("SMS model input features:", assets["sms_model"].n_features_in_)
        st.write("SMS vectorizer vocabulary:", len(assets["sms_vectorizer"].vocabulary_))
        st.write("Local Llama GGUF found:", LLAMA_PATH.exists())
        st.caption("The app loads the saved artifacts. It does not train or modify any model.")

if __name__ == "__main__":
    main()
