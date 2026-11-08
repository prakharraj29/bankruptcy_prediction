import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Bankruptcy Prediction", page_icon="📊", layout="wide")

st.title("💼 Bankruptcy Prediction - Multi-Model Evaluation")
st.markdown("Evaluate 5 classical ML models on uploaded test data without retraining.")

target_col = "Bankrupt?"
model_map = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Support Vector Machine": "models/support_vector_machine.pkl"
}

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose Model", list(model_map.keys()))

uploaded_file = st.file_uploader("Upload test CSV dataset (e.g. test_data.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        df_test.columns = df_test.columns.str.strip()

        if target_col not in df_test.columns:
            st.error(f"Missing required target column '{target_col}' in uploaded file.")
        else:
            X_test = df_test.drop(columns=[target_col])
            y_test = df_test[target_col]

            scaler = joblib.load("models/scaler.pkl")
            model = joblib.load(model_map[selected_model_name])

            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if (y_prob is not None and len(np.unique(y_test)) > 1) else 0.0
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted")
            mcc = matthews_corrcoef(y_test, y_pred)

            st.markdown(f"### 📈 Performance Metrics: {selected_model_name}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("AUC", f"{auc:.4f}")
            c3.metric("Precision", f"{prec:.4f}")
            c4.metric("Recall", f"{rec:.4f}")
            c5.metric("F1 Score", f"{f1:.4f}")
            c6.metric("MCC", f"{mcc:.4f}")

            st.markdown("---")
            col_cm, col_cr = st.columns([1, 1])

            with col_cm:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Non-Bankrupt (0)", "Bankrupt (1)"],
                            yticklabels=["Non-Bankrupt (0)", "Bankrupt (1)"],
                            cbar=False, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col_cr:
                st.subheader("Classification Report")
                report = classification_report(
                    y_test, y_pred,
                    target_names=["Non-Bankrupt (0)", "Bankrupt (1)"],
                    zero_division=0
                )
                st.code(report, language="text")

    except Exception as e:
        st.error(f"Error evaluating model: {e}")
else:
    st.info("Upload `test_data.csv` to preview evaluation metrics and charts.")
