# Device Threat Detection — Streamlit prototype

This project folder is designed to sit **directly inside** your existing `Cryptography` folder.

Expected layout:

```text
Cryptography/
├── Device Threat Detection App/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── models/                  # optional local Llama GGUF goes here
├── Network Intrusion Detection/
│   └── Decision tree with Feature Selection/
│       └── decision_tree_MI_model.pkl
├── Phising/
│   └── Phish tank dataset/
│       └── Decision Tree Brand Detection Validated/
│           ├── decision_tree_brand_model.pkl
│           ├── feature_names.json
│           └── brand_library.json
└── SMS Spam/
    └── Decision Tree/
        ├── best_decision_tree_model.pkl
        └── tfidf_vectorizer.pkl
```

## 1. Create the project folder

Extract this ZIP into:

`C:\Users\Lenovo\Desktop\Cryptography\Device Threat Detection App`

Do not move or rename your existing model folders.

## 2. Open VS Code terminal

```powershell
cd "C:\Users\Lenovo\Desktop\Cryptography\Device Threat Detection App"

python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the same Python version/environment you used to run the inspection script, where scikit-learn 1.9.0 is installed. If `python --version` shows a different interpreter, select the matching interpreter in VS Code before creating the virtual environment. The pinned scikit-learn version is intentional because pickle compatibility can be version-sensitive.

If PowerShell blocks environment activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Start Streamlit

```powershell
streamlit run app.py
```

Streamlit should open the app in your browser at `http://localhost:8501`.

## 4. Local Llama explanation (optional add-on)

Your PC has 8 GB RAM. Start with a **Llama 3.2 1B Instruct Q4_K_M GGUF** model, not the 3B model. Meta describes its quantized 1B/3B models as intended for constrained-memory and edge-device use. A Q4_K_M 1B GGUF is roughly 0.8 GB on disk; runtime memory is higher.

1. Obtain a Llama 3.2 1B Instruct Q4_K_M GGUF file from a trusted model publisher and review/accept the Llama Community License.
2. Put the `.gguf` file in this folder:
   `models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`
3. Install the optional Python runtime in the activated environment:

```powershell
pip install llama-cpp-python
```

If a prebuilt wheel is unavailable for your Python version, use the official llama.cpp Windows installation/build guidance rather than an unknown binary.

The app works without Llama installed: it displays a transparent evidence-based fallback explanation. Once the GGUF file and `llama-cpp-python` are installed, the explanation is generated locally. Model predictions are always produced by your saved Decision Trees, not by Llama.

## 5. How to use the three modules

- **NIDS:** upload a CSV with the exact 78 original feature columns. For one row, the app also shows the decision-tree branch conditions. The app does not sniff live packets or generate CICFlowMeter features.
- **Phishing:** enter a URL. The app extracts the 11 validated features using the saved 867,570-entry brand library.
- **SMS:** paste an SMS. The saved TF-IDF vectorizer and Decision Tree are used; the vectorizer is not refitted.

## Important limitations / validation

- This is a local research prototype, not a certified security product.
- The NIDS output assumes `0 = Benign / Normal` and `1 = Attack / Intrusion`, as confirmed for this project.
- Phishing assumes `0 = Legitimate` and `1 = Phishing`.
- SMS assumes `0 = Ham / Not Spam` and `1 = Spam`.
- The phishing feature extractor has been reconstructed from the training-pipeline description shared in the conversation. Before relying on results, compare its 11 feature values against the exact `extract_features()` function from your validated training script on several known test URLs. If the training script differs in any detail, copy that exact function into `app.py`.
- The app reports Decision Tree probabilities as model confidence estimates; they are not calibrated guarantees.
- Local history is stored in `prediction_history.db` inside this app folder.
- Do not expose this Streamlit development server to the public internet.

## Troubleshooting

- **Missing artifact:** verify the folder names and paths above exactly, including `Phising` spelling.
- **scikit-learn warning/error:** activate the environment with the compatible scikit-learn version used for serialization (reported as 1.9.0).
- **Llama not available:** model inference still works; only natural-language generation falls back to the evidence summary.
- **Memory pressure:** close other heavy apps, keep Llama context at 2048, and use the 1B Q4 quantization.
