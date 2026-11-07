# Cardiovascular Disease Risk Predictor (Streamlit ML Web App)

**InternCircle | Data Science & AI Project 05**

An end-to-end Machine Learning web application deployed using Streamlit. The application accepts patient clinical biomarkers and vital signs, passes them through a Scikit-Learn Random Forest Pipeline cached via `@st.cache_resource`, and provides real-time risk predictions with probabilistic metrics and visual indicators.

---

## 🚀 Key Features

- **Interactive UI**: Sliders, radio buttons, and select boxes with strict physical validation boundaries.
- **Pre-configured Profiles**: Quick preset scenarios (Healthy Young Adult, Active Middle-Aged, High-Risk) for instant evaluation.
- **Cached Inference**: Utilizes `@st.cache_resource` for low-latency predictions without reloading the model on every click.
- **Defensive Error Handling**: Gracefully handles missing model files and unexpected inputs without exposing raw tracebacks.
- **Educational Disclaimer**: Clearly states the dataset scope, model architecture, and compliance with non-medical advice guidelines.

---

## 📁 Project Structure

```
interncircle-ml-app/
│
├── train_model.py       # Model training, validation, and serialization pipeline
├── model.joblib         # Exported model bundle (classifier + feature schema)
├── app.py               # Streamlit interactive web application
├── requirements.txt     # Python dependencies for local and cloud deployment
└── README.md            # Project documentation and submission guide
```

---

## 🛠️ Getting Started Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to generate `model.joblib`:
```bash
python train_model.py
```

### 4. Run the Streamlit Application
Launch the web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud (Free Hosting)

Follow these steps to deploy your project publicly:

1. **Initialize Git & Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Streamlit interactive ML web app"
   git branch -M main
   git remote add origin https://github.com/moryesoham4-SR/Streamlit-Gradio-Interactive-ML-Web-App.git
   git push -u origin main
   ```
   *(Ensure `model.joblib` is committed so the cloud app can load it).*

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"New app"**.
   - Select your repository (`moryesoham4-SR/Streamlit-Gradio-Interactive-ML-Web-App`), branch (`main`), and set the main file path to `app.py`.
   - Click **"Deploy!"**.

3. **Verify Your Public URL**:
   - Once deployed, test the link in desktop and mobile browsers.

---

## 📋 InternCircle Submission Checklist

- [x] **Model loads from a saved file**: Uses `joblib.load()` inside `@st.cache_resource`.
- [x] **Inputs have sensible limits**: Sliders and options correspond to realistic clinical ranges.
- [x] **Prediction is shown clearly with a caption**: Displays percentage with `st.metric` and educational disclaimer with `st.caption`.
- [x] **Missing files and bad input show friendly messages**: Handled with `try...except` and `st.error()`.
- [x] **requirements.txt lists every package**: Includes `streamlit`, `scikit-learn`, `pandas`, `joblib`, and `numpy`.
- [x] **Ready for GitHub & Cloud Deployment**: Tested and self-contained.
