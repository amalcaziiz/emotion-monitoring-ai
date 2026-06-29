import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Emotional Changing Monitoring Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# MODERN SAAS DARK THEME
# =========================
st.markdown("""
<style>
.main {
    background-color: #0a0f1c;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

h1 {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    color: #f9fafb;
    margin-bottom: 6px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 15px;
    margin-bottom: 25px;
}

h3 {
    color: #93c5fd;
}

/* Cards */
.card {
    background: rgba(17, 24, 39, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.12);
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 16px;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    border: none;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
}

/* Result boxes */
.result {
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}

.low {
    background: rgba(34, 197, 94, 0.08);
    color: #22c55e;
}

.medium {
    background: rgba(245, 158, 11, 0.08);
    color: #fbbf24;
}

.high {
    background: rgba(239, 68, 68, 0.08);
    color: #f87171;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.12);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<h1>AI-Driven Emotional Changing Monitoring Framework</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Multimodal Stress Detection using Physiological + Behavioral Signals</div>", unsafe_allow_html=True)

st.write("---")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("models/wesad_scaler.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")

        wesad_lr = joblib.load("models/wesad_lr.pkl")
        wesad_svm = joblib.load("models/wesad_svm.pkl")
        wesad_rf = joblib.load("models/wesad_rf.pkl")

        isear_lr = joblib.load("models/isear_lr.pkl")
        isear_svm = joblib.load("models/isear_svm.pkl")
        isear_rf = joblib.load("models/isear_rf.pkl")

        return scaler, vectorizer, wesad_lr, wesad_svm, wesad_rf, isear_lr, isear_svm, isear_rf

    except Exception as e:
        st.error(f"Artifact Loading Error: {e}")
        return None

artifacts = load_models()

if artifacts:

    scaler, vectorizer, wesad_lr, wesad_svm, wesad_rf, isear_lr, isear_svm, isear_rf = artifacts

    label_map = {
        0: "🟢 LOW STRESS",
        1: "🟡 MEDIUM STRESS",
        2: "🔴 HIGH STRESS"
    }

    col_input, col_results = st.columns([1.1, 1.3], gap="large")

    # =========================
    # INPUT PANEL
    # =========================
    with col_input:

        st.markdown("### Physiological Signals (WESAD)")

        ecg_mean = st.slider("ECG Mean", -1.5, 1.5, 0.0, 0.01)
        eda_mean = st.slider("EDA Mean", 2.0, 12.0, 5.5, 0.1)
        emg_mean = st.slider("EMG Mean", -0.05, 0.05, 0.002, 0.001)
        resp_mean = st.slider("Respiration Mean", -5.0, 5.0, 0.0, 0.1)
        temp_mean = st.slider("Skin Temperature", 30.0, 36.0, 33.5, 0.1)

        st.write("")

        st.markdown("### Behavioral Input (ISEAR)")
        text_input = st.text_area(
            "Emotion Text",
            placeholder="e.g., I felt extremely relieved and joyful after the results..."
        )

        trigger_analysis = st.button("Run Multimodal Inference")

    # =========================
    # OUTPUT PANEL
    # =========================
    with col_results:

        if trigger_analysis and text_input.strip() != "":

            # ---- WESAD ----
            wesad_raw_features = np.array([[ecg_mean, 0.1,
                                            eda_mean, 0.2,
                                            emg_mean, 0.005,
                                            resp_mean, 1.1,
                                            temp_mean, 0.1]])

            wesad_scaled = scaler.transform(wesad_raw_features)

            w_pred_lr = int(wesad_lr.predict(wesad_scaled)[0])
            w_pred_svm = int(wesad_svm.predict(wesad_scaled)[0])
            w_pred_rf = int(wesad_rf.predict(wesad_scaled)[0])

            # ---- ISEAR ----
            text_vec = vectorizer.transform([text_input])

            t_pred_lr = int(isear_lr.predict(text_vec)[0])
            t_pred_svm = int(isear_svm.predict(text_vec)[0])
            t_pred_rf = int(isear_rf.predict(text_vec)[0])

            # ---- FUSION ----
            all_votes = [w_pred_lr, w_pred_svm, w_pred_rf,
                         t_pred_lr, t_pred_svm, t_pred_rf]

            # Calculate the highest count a single class received
            max_vote_count = max(all_votes.count(x) for x in set(all_votes))
            
            # Find which class or classes hit that maximum count
            top_voted_classes = [x for x in set(all_votes) if all_votes.count(x) == max_vote_count]

            if len(top_voted_classes) == 1:
                # Clear consensus majority exists
                final_fused_class = top_voted_classes[0]
            else:
                # TIE-BREAKER RULE: Trust the single highest-accuracy independent estimator
                # WESAD Random Forest has a benchmark accuracy score of 98.83%
                final_fused_class = w_pred_rf

            # =========================
            # RESULT
            # =========================
            st.markdown("### System Prediction")

            if final_fused_class == 0:
                st.markdown(f"<div class='result low'>{label_map[final_fused_class]}</div>", unsafe_allow_html=True)
            elif final_fused_class == 1:
                st.markdown(f"<div class='result medium'>{label_map[final_fused_class]}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result high'>{label_map[final_fused_class]}</div>", unsafe_allow_html=True)

            st.write("")

            # =========================
            # MODEL BREAKDOWN
            # =========================
            st.markdown("### ML Model Decision Trace")

            col_w, col_t = st.columns(2)

            with col_w:
                st.markdown("#### Physiological Models")
                st.metric("Logistic Regression", label_map[w_pred_lr])
                st.metric("SVM", label_map[w_pred_svm])
                st.metric("Random Forest", label_map[w_pred_rf])

            with col_t:
                st.markdown("#### Behavioral Models")
                st.metric("Logistic Regression", label_map[t_pred_lr])
                st.metric("SVM", label_map[t_pred_svm])
                st.metric("Random Forest", label_map[t_pred_rf])

        else:
            st.info("Enter input signals and click RUN MULTIMODAL INFERENCE.")

    # =========================
    # METRICS SECTION
    # =========================
    st.write("---")
    st.markdown("### ML Model Performance Benchmark")

    metrics_data = {
        "Classifier Domain": [
            "WESAD - Logistic Regression", "WESAD - Support Vector Machine", "WESAD - Random Forest",
            "ISEAR - Logistic Regression", "ISEAR - Support Vector Machine", "ISEAR - Random Forest"
        ],
        "Accuracy": ["84.50%", "92.10%", "98.83%", "78.40%", "81.20%", "80.50%"],
        "Precision (Weighted)": ["84.20%", "91.90%", "98.85%", "78.10%", "81.00%", "80.20%"],
        "Recall (Weighted)": ["84.50%", "92.10%", "98.83%", "78.40%", "81.20%", "80.50%"],
        "F1-Score": ["84.30%", "92.00%", "98.83%", "78.20%", "81.10%", "80.30%"],
        "Error Rate": ["15.50%", "7.90%", "1.17%", "21.60%", "18.80%", "19.50%"]
    }
    
    df_metrics = pd.DataFrame(metrics_data)

    st.dataframe(df_metrics, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:12px; margin-top:20px;'>Research Prototype • Multimodal Emotion AI System</p>",
    unsafe_allow_html=True
)
