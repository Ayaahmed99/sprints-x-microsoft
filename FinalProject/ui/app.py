# ui/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path("models/final_pipeline.pkl")  # or random_forest_best.pkl / random_forest_baseline
if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}. Please train and save the model first.")
else:
    model = joblib.load(MODEL_PATH)

st.title("Heart Disease Risk Predictor")
st.write("Fill in patient data to get a prediction (binary classification).")

# Load sample features from training data to present inputs
df_sample = pd.read_csv("data/heart_disease_cleaned.csv")
TARGET_COL = "target"
X_columns = df_sample.drop(columns=[TARGET_COL]).columns.tolist()

# Create inputs dynamically
inputs = {}
st.sidebar.header("Patient information")
for col in X_columns:
    # guess input type by dtype / name
    if df_sample[col].dtype in [int, float] or np.issubdtype(df_sample[col].dtype, np.number):
        minv = float(df_sample[col].min())
        maxv = float(df_sample[col].max())
        meanv = float(df_sample[col].median())
        inputs[col] = st.sidebar.number_input(f"{col}", value=meanv, min_value=minv, max_value=maxv, step=(maxv-minv)/100 if maxv!=minv else 1.0)
    else:
        unique_vals = df_sample[col].unique().tolist()
        inputs[col] = st.sidebar.selectbox(f"{col}", unique_vals)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([inputs])
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1] if hasattr(model.named_steps['clf'], "predict_proba") else None
    st.subheader("Prediction")
    if pred == 1:
        st.error(f"Model predicts POSITIVE for heart disease (probability: {proba:.3f})" if proba is not None else "Model predicts POSITIVE for heart disease")
    else:
        st.success(f"Model predicts NEGATIVE for heart disease (probability: {proba:.3f})" if proba is not None else "Model predicts NEGATIVE for heart disease")

    st.write("Input values used for prediction:")
    st.table(input_df.T)

# Optionally show feature importance (if RF)
if "RandomForest" in str(type(model.named_steps.get('clf', model))):
    try:
        importances = model.named_steps['clf'].feature_importances_
        feat = pd.Series(importances, index=X_columns).sort_values(ascending=False)
        st.subheader("Feature importances")
        st.bar_chart(feat.head(10))
    except Exception:
        pass
