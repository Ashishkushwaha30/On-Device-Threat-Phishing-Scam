
import os
import json
import joblib
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = r"C:\Users\Lenovo\Desktop\Cryptography\SMS Spam\Decision Tree"

MODEL_PATH = os.path.join(
    BASE_DIR, "best_decision_tree_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR, "tfidf_vectorizer.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "sms_model_android.json"
)

# --------------------------------------------------
# Load trained model and TF-IDF vectorizer
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --------------------------------------------------
# Verify model and vectorizer
# --------------------------------------------------

if not hasattr(model, "tree_"):
    raise TypeError(
        "The loaded model is not a DecisionTreeClassifier."
    )

if not hasattr(vectorizer, "vocabulary_"):
    raise TypeError(
        "The loaded object is not a fitted TfidfVectorizer."
    )

tree = model.tree_

if tree.n_features != len(vectorizer.vocabulary_):
    raise ValueError(
        "Model and vectorizer feature counts do not match."
    )

# --------------------------------------------------
# Export Decision Tree
# --------------------------------------------------

tree_predictions = np.argmax(
    tree.value[:, 0, :],
    axis=1
)

classes = model.classes_.tolist()

predictions = [
    classes[index].item()
    if isinstance(classes[index], np.generic)
    else classes[index]
    for index in tree_predictions
]

# --------------------------------------------------
# Export TF-IDF configuration
# --------------------------------------------------

params = vectorizer.get_params()

stop_words = params["stop_words"]

if isinstance(stop_words, np.ndarray):
    stop_words = stop_words.tolist()

tfidf_config = {
    "analyzer": params["analyzer"],
    "lowercase": params["lowercase"],
    "token_pattern": params["token_pattern"],
    "ngram_range": list(params["ngram_range"]),
    "norm": params["norm"],
    "use_idf": params["use_idf"],
    "smooth_idf": params["smooth_idf"],
    "sublinear_tf": params["sublinear_tf"],
    "binary": params["binary"],
    "stop_words": stop_words
}

# --------------------------------------------------
# Create Android JSON
# --------------------------------------------------

data = {
    "vocabulary": vectorizer.vocabulary_,
    "idf": (
        vectorizer.idf_.tolist()
        if params["use_idf"]
        else []
    ),
    "tfidf_config": tfidf_config,

    "classes": [
        value.item() if isinstance(value, np.generic)
        else value
        for value in model.classes_
    ],

    "children_left": tree.children_left.tolist(),
    "children_right": tree.children_right.tolist(),
    "feature": tree.feature.tolist(),
    "threshold": tree.threshold.tolist(),
    "prediction": predictions
}

# --------------------------------------------------
# Save JSON
# --------------------------------------------------

def convert_numpy_types(obj):

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, set):
        return list(obj)

    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        data,
        file,
        ensure_ascii=False,
        separators=(",", ":"),
        default=convert_numpy_types
    )

print("\nSMS model exported successfully.")

print("Output:", OUTPUT_PATH)
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Tree nodes:", tree.node_count)
print("Tree features:", tree.n_features)
print("Classes:", model.classes_)
print("\nTF-IDF configuration:")
print(json.dumps(tfidf_config, indent=4))
