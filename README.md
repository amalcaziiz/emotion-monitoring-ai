# ISED: Integrated Multi-Modal Stress Detection System

> **AI-Driven Emotional Changing Monitoring Framework**

An AI-driven multimodal stress detection framework developed as a senior thesis project. ISED combines physiological signals and emotion-related text data to classify stress into **Low, Medium, and High** levels.

---

#  Overview

ISED uses two complementary data sources:

- **WESAD** — physiological signals including ECG, EDA, EMG, respiration, and temperature.
- **ISEAR** — emotion-related text used for behavioral/text-based analysis.

Each modality has its own machine learning pipeline. The resulting model predictions are then combined through a multimodal decision process.

---

#  Machine Learning Pipeline

## WESAD — Physiological Signals

The WESAD pipeline processes five physiological signals:

- ECG
- EDA
- EMG
- Respiration
- Temperature

For each signal, the pipeline extracts:

- Mean
- Standard deviation

This produces **10 statistical features** for each physiological segment.

The features are standardized using `StandardScaler` and used to train:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

---

## ISEAR — Text Classification

The ISEAR pipeline processes emotion-related text using **TF-IDF vectorization** with up to 5,000 features.

Three classification models are trained:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

Both pipelines use an **80/20 stratified train-test split** for evaluation.

---

# Multimodal Prediction

ISED combines predictions from **six trained models**:

The six predictions are combined using a **majority-vote approach** to determine the final stress classification.

If the predictions result in a tie, the **WESAD Random Forest** prediction is used as the tie-breaker.

---

# Interactive Application

The trained models are integrated into a **Streamlit application** for interactive multimodal inference.

The application allows users to:

- Enter physiological signal features
- Enter emotion-related text
- Run multimodal inference
- View the final stress classification
- Inspect individual model predictions
- Compare model performance

---

#  Tech Stack

### Language
- Python

### Machine Learning
- Scikit-learn
- Logistic Regression
- Support Vector Machine
- Random Forest

### Data Processing
- Pandas
- NumPy
- StandardScaler
- TF-IDF

### Application
- Streamlit

### Model Persistence
- Joblib

### Datasets
- WESAD
- ISEAR

---

#  Project Context

**Senior Capstone / Thesis Project**

**Research Focus:** Multimodal machine learning, physiological signal analysis, natural language processing, and stress classification.
