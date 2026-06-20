# ❤️ Heart Disease Prediction & Health Report System

An AI-powered cardiovascular risk assessment dashboard built with **Streamlit**, **Scikit-learn**, and **Google Gemini**. It collects clinical patient parameters, predicts heart disease risk using a trained Logistic Regression model, visualises results with interactive Plotly charts, and generates a downloadable PDF health report.

---

## 📸 Features

- 🔬 **ML-Powered Prediction** — Logistic Regression model trained on the UCI Heart Disease dataset (85% accuracy)
- 📊 **Interactive Dashboard** — Gauge chart, vitals comparison bar chart, and health indicator chart powered by Plotly
- 🩺 **Clinical Analysis Table** — Structured table comparing user values against medical reference ranges with status badges
- 🧠 **AI Model Explainability** — Feature importance chart showing which clinical metrics most influence the prediction
- 💡 **Personalised Health Recommendations** — Actionable health tips tailored to the patient's risk level
- 💬 **AI Cardiovascular Assistant** — Gemini-powered chat assistant that reads the full patient profile and answers health questions
- 📄 **Downloadable PDF Report** — Full clinical report including patient info, prediction, risk table, and recommendations
- 🎨 **Premium Dark Theme** — Glassmorphism UI with animated hover effects and cyan accent colours

---

## 🗂️ Project Structure

```
Heart-Disease-Prediction-ML/
│
├── app.py                          # Main Streamlit application
├── heart_model.pkl                 # Trained Logistic Regression model
├── heart_scaler.pkl                # Fitted StandardScaler
├── heart.csv                       # UCI Heart Disease dataset
├── heart-disease-prediction.ipynb  # Training & evaluation notebook
├── requirements.txt                # Python dependencies
├── .streamlit/
│   ├── config.toml                 # Streamlit theme config
│   └── secrets.toml                # API keys (not committed to Git)
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/shehara2002/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

The AI Chat Assistant requires a **Google Gemini API key**. Set it via one of these methods:

**Option A — Streamlit Secrets (recommended for local dev):**

Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

**Option B — Environment Variable:**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"

# macOS / Linux
export GEMINI_API_KEY="your-api-key-here"
```

> Get your free API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🧬 Clinical Input Parameters

| Parameter | Description |
|---|---|
| **Age** | Patient age in years |
| **Sex** | Biological gender (Male / Female) |
| **Chest Pain Type (cp)** | Typical Angina / Atypical Angina / Non-anginal / Asymptomatic |
| **Resting Blood Pressure (trestbps)** | Measured in mmHg at rest |
| **Cholesterol (chol)** | Serum cholesterol in mg/dL |
| **Fasting Blood Sugar (fbs)** | Whether fasting blood sugar > 120 mg/dL |
| **Resting ECG (restecg)** | Normal / ST-T Wave Abnormality / Left Ventricular Hypertrophy |
| **Max Heart Rate (thalach)** | Maximum heart rate achieved during stress test |
| **Exercise Induced Angina (exang)** | Whether exercise triggered chest pain |
| **ST Depression (oldpeak)** | ST segment depression relative to rest |
| **Slope of ST Segment** | Upsloping / Flat / Downsloping |
| **Major Vessels (ca)** | Number of major vessels coloured by fluoroscopy (0–4) |
| **Thalassemia (thal)** | Normal / Fixed Defect / Reversible Defect |

---

## 🤖 Model Details

| Metric | Score |
|---|---|
| Algorithm | Logistic Regression |
| Dataset | UCI Heart Disease (Cleveland) |
| Accuracy | 85% |
| Precision | 85% |
| Recall | 85% |
| F1 Score | 85% |

- **Label 0** → No Heart Disease (Low Risk)
- **Label 1** → Heart Disease Detected (High Risk)

The model artifact (`heart_model.pkl`) is expected to be a `dict` with keys:
```python
{
  "model": <LogisticRegression>,
  "feature_names": ["age", "sex", "cp", ...]
}
```
The scaler (`heart_scaler.pkl`) is a fitted `StandardScaler` instance.

---

## 📦 Dependencies

```
streamlit
scikit-learn
joblib
numpy
pandas
plotly
google-generativeai
reportlab
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

> This application is built for **educational and informational purposes only**.  
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
> Always consult a licensed cardiologist or healthcare professional for medical decisions.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).