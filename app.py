import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Emotional Changing Monitoring Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM UI
# ============================================================
st.markdown("""
<style>
/* ---------- GLOBAL ---------- */
.stApp {
    background: #0a0f1c;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

* {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ---------- HEADER ---------- */
.hero {
    text-align: center;
    padding: 8px 0 20px 0;
}

.hero h1 {
    color: #f8fafc;
    font-size: 34px;
    font-weight: 750;
    letter-spacing: -0.7px;
    margin: 0;
}

.hero p {
    color: #94a3b8;
    font-size: 15px;
    margin: 8px 0 0 0;
}

/* ---------- SECTION TITLES ---------- */
.section-title {
    color: #f1f5f9;
    font-size: 19px;
    font-weight: 700;
    margin: 0 0 14px 0;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-top: -8px;
    margin-bottom: 16px;
}

/* ---------- CARDS ---------- */
.panel {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 22px;
    height: 100%;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

/* ---------- INPUT LABELS ---------- */
label {
    color: #cbd5e1 !important;
}

/* ---------- SLIDERS ---------- */
div[data-baseweb="slider"] {
    margin-bottom: 8px;
}

/* ---------- TEXT AREA ---------- */
textarea {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* ---------- BUTTON ---------- */
.stButton > button {
    width: 100%;
    height: 46px;
    border-radius: 10px;
    border: 1px solid #3b82f6;
    background: #2563eb;
    color: white;
    font-size: 15px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #1d4ed8;
    border-color: #60a5fa;
    transform: translateY(-1px);
}

/* ---------- RESULT ---------- */
.prediction-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 8px;
}

.result {
    width: 100%;
    box-sizing: border-box;
    padding: 18px 12px;
    border-radius: 13px;
    text-align: center;
    font-size: 21px;
    font-weight: 800;
    letter-spacing: 0.3px;
    white-space: nowrap;
}

.low {
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.30);
    color: #4ade80;
}

.medium {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.30);
    color: #fbbf24;
}

.high {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(239, 68, 68, 0.30);
    color: #f87171;
}

/* ---------- TRACE ---------- */
.trace-title {
    color: #e2e8f0;
    font-size: 15px;
    font-weight: 700;
    margin: 0 0 10px 0;
}

.trace-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 11px;
    padding: 12px 14px;
    margin-bottom: 8px;
}

.trace-name {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 3px;
}

.trace-value {
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 700;
}

/* ---------- INFO MESSAGE ---------- */
div[data-testid="stAlert"] {
    border-radius: 11px;
}

/* ---------- METRICS ---------- */
div[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 10px 12px;
    min-height: 74px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 11px !important;
}

div[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 14px !important;
    white-space: normal !important;
    line-height: 1.25 !important;
}

/* ---------- TABLE ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- DIVIDER ---------- */
hr {
    border-color: #1e293b;
    margin: 24px 0;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 22px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>AI-Driven Emotional Changing Monitoring Framework</h1>
    <p>ISED Multimodal Stress Detection using Physiological + Behavioral Signals</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
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

        return (
            scaler,
            vectorizer,
            wesad_lr,
            wesad_svm,
            wesad_rf,
            isear_lr,
            isear_svm,
            isear_rf
        )

    except Exception as e:
        st.error(f"Artifact Loading Error: {e}")
        return None


artifacts = load_models()

if artifacts:

    (
        scaler,
        vectorizer,
        wesad_lr,
        wesad_svm,
        wesad_rf,
        isear_lr,
        isear_svm,
        isear_rf
    ) = artifacts

    label_map = {
        0: "LOW STRESS",
        1: "MEDIUM STRESS",
        2: "HIGH STRESS"
    }

    # ========================================================
    # MAIN INPUT / OUTPUT AREA
    # ========================================================
    col_input, col_results = st.columns(
        [1, 1.15],
        gap="large"
    )

    # ========================================================
    # INPUT PANEL
    # ========================================================
    with col_input:

        st.markdown("""
        <div class="panel">
            <div class="section-title">Physiological Signals</div>
            <div class="section-subtitle">
                WESAD physiological features
            </div>
        """, unsafe_allow_html=True)

        ecg_mean = st.slider(
            "ECG Mean",
            -1.5,
            1.5,
            0.0,
            0.01
        )

        eda_mean = st.slider(
            "EDA Mean",
            2.0,
            12.0,
            5.5,
            0.1
        )

        emg_mean = st.slider(
            "EMG Mean",
            -0.05,
            0.05,
            0.002,
            0.001
        )

        resp_mean = st.slider(
            "Respiration Mean",
            -5.0,
            5.0,
            0.0,
            0.1
        )

        temp_mean = st.slider(
            "Skin Temperature",
            30.0,
            36.0,
            33.5,
            0.1
        )

        st.markdown("""
        <div style="height:10px;"></div>
        <div class="section-title">Behavioral Input</div>
        <div class="section-subtitle">
            ISEAR emotion-related text
        </div>
        """, unsafe_allow_html=True)

        text_input = st.text_area(
            "Emotion Text",
            placeholder="Example: I felt extremely relieved and joyful after the results...",
            height=115
        )

        trigger_analysis = st.button(
            "RUN MULTIMODAL INFERENCE",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # OUTPUT PANEL
    # ========================================================
    with col_results:

        if trigger_analysis and text_input.strip() != "":

            # ------------------------------------------------
            # WESAD
            # ------------------------------------------------
            wesad_raw_features = np.array([[
                ecg_mean,
                0.1,
                eda_mean,
                0.2,
                emg_mean,
                0.005,
                resp_mean,
                1.1,
                temp_mean,
                0.1
            ]])

            wesad_scaled = scaler.transform(wesad_raw_features)

            w_pred_lr = int(
                wesad_lr.predict(wesad_scaled)[0]
            )

            w_pred_svm = int(
                wesad_svm.predict(wesad_scaled)[0]
            )

            w_pred_rf = int(
                wesad_rf.predict(wesad_scaled)[0]
            )

            # ------------------------------------------------
            # ISEAR
            # ------------------------------------------------
            text_vec = vectorizer.transform([text_input])

            t_pred_lr = int(
                isear_lr.predict(text_vec)[0]
            )

            t_pred_svm = int(
                isear_svm.predict(text_vec)[0]
            )

            t_pred_rf = int(
                isear_rf.predict(text_vec)[0]
            )

            # ------------------------------------------------
            # FUSION
            # ------------------------------------------------
            all_votes = [
                w_pred_lr,
                w_pred_svm,
                w_pred_rf,
                t_pred_lr,
                t_pred_svm,
                t_pred_rf
            ]

            max_vote_count = max(
                all_votes.count(x)
                for x in set(all_votes)
            )

            top_voted_classes = [
                x for x in set(all_votes)
                if all_votes.count(x) == max_vote_count
            ]

            if len(top_voted_classes) == 1:
                final_fused_class = top_voted_classes[0]
            else:
                final_fused_class = w_pred_rf

            # =================================================
            # SYSTEM PREDICTION
            # =================================================
            st.markdown("""
            <div class="panel">
                <div class="prediction-label">System Prediction</div>
            """, unsafe_allow_html=True)

            if final_fused_class == 0:
                result_class = "low"
            elif final_fused_class == 1:
                result_class = "medium"
            else:
                result_class = "high"

            st.markdown(
                f"""
                <div class="result {result_class}">
                    {label_map[final_fused_class]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("""
                <div style="height:20px;"></div>
                <div class="section-title">ML Model Decision Trace</div>
            """, unsafe_allow_html=True)

            # =================================================
            # DECISION TRACE
            # =================================================
            col_w, col_t = st.columns(2, gap="medium")

            def trace_card(name, prediction):
                return f"""
                <div class="trace-card">
                    <div class="trace-name">{name}</div>
                    <div class="trace-value">{label_map[prediction]}</div>
                </div>
                """

            with col_w:
                st.markdown(
                    '<div class="trace-title">Physiological Models</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    trace_card(
                        "Logistic Regression",
                        w_pred_lr
                    ),
                    unsafe_allow_html=True
                )

                st.markdown(
                    trace_card(
                        "Support Vector Machine",
                        w_pred_svm
                    ),
                    unsafe_allow_html=True
                )

                st.markdown(
                    trace_card(
                        "Random Forest",
                        w_pred_rf
                    ),
                    unsafe_allow_html=True
                )

            with col_t:
                st.markdown(
                    '<div class="trace-title">Behavioral Models</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    trace_card(
                        "Logistic Regression",
                        t_pred_lr
                    ),
                    unsafe_allow_html=True
                )

                st.markdown(
                    trace_card(
                        "Support Vector Machine",
                        t_pred_svm
                    ),
                    unsafe_allow_html=True
                )

                st.markdown(
                    trace_card(
                        "Random Forest",
                        t_pred_rf
                    ),
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="panel">
                <div class="section-title">System Prediction</div>
                <div class="section-subtitle">
                    Multimodal inference result
                </div>
            """, unsafe_allow_html=True)

            st.info(
                "Enter physiological inputs and emotion text, "
                "then run multimodal inference."
            )

            st.markdown("""
            <div style="
                margin-top:20px;
                padding:18px;
                border-radius:12px;
                background:#0f172a;
                border:1px solid #1e293b;
                color:#94a3b8;
                font-size:13px;
                line-height:1.6;
            ">
                The system combines predictions from six trained
                machine learning models to produce the final
                stress classification.
            </div>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================
    # PERFORMANCE BENCHMARK
    # ========================================================
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">ML Model Performance Benchmark</div>
    <div class="section-subtitle">
        Evaluation results from the trained WESAD and ISEAR classifiers
    </div>
    """, unsafe_allow_html=True)

    metrics_data = {
        "Classifier Domain": [
            "WESAD - Logistic Regression",
            "WESAD - Support Vector Machine",
            "WESAD - Random Forest",
            "ISEAR - Logistic Regression",
            "ISEAR - Support Vector Machine",
            "ISEAR - Random Forest"
        ],
        "Accuracy": [
            "84.50%",
            "92.10%",
            "98.83%",
            "78.40%",
            "81.20%",
            "80.50%"
        ],
        "Precision (Weighted)": [
            "84.20%",
            "91.90%",
            "98.85%",
            "78.10%",
            "81.00%",
            "80.20%"
        ],
        "Recall (Weighted)": [
            "84.50%",
            "92.10%",
            "98.83%",
            "78.40%",
            "81.20%",
            "80.50%"
        ],
        "F1-Score": [
            "84.30%",
            "92.00%",
            "98.83%",
            "78.20%",
            "81.10%",
            "80.30%"
        ],
        "Error Rate": [
            "15.50%",
            "7.90%",
            "1.17%",
            "21.60%",
            "18.80%",
            "19.50%"
        ]
    }

    df_metrics = pd.DataFrame(metrics_data)

    st.dataframe(
        df_metrics,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    Research Prototype • Multimodal Emotion AI System
</div>
""", unsafe_allow_html=True)
