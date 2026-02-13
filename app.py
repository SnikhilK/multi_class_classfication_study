import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Label Mapping
# -----------------------------
LABEL_MAP = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

# -----------------------------
# Mark best metric values (Streamlit-safe)
# -----------------------------
def mark_best_values(df):
    df_marked = df.copy()

    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        max_val = pd.to_numeric(df[col]).max()
        df_marked[col] = df[col].apply(
            lambda v: f"{float(v):.4f} ★"
            if abs(float(v) - max_val) < 1e-6
            else f"{float(v):.4f}"
        )

    return df_marked

# -----------------------------
# UI Layout
# -----------------------------
st.set_page_config(
    page_title="Classification Models - Study",
    layout="wide"
)

st.title("Multi-Class Classification Model Evaluation")
st.subheader("Demonstrated on Human Activity Recognition (HAR) dataset from UCI Repositories.")
st.caption("Developed by Sai Nikhil Kandagiri, as part of WILP Machine Learning Course (reg. 2025AA05555)")

# -----------------------------
# Load Offline Metrics
# -----------------------------
@st.cache_data
def load_metrics():
    return pd.read_csv("results/metrics.csv")

metrics_df = load_metrics()

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    for file in os.listdir("models"):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "").replace("_", " ").title()
            models[model_name] = joblib.load(os.path.join("models", file))
    return models

models = load_models()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Sample Test Dataset")

if os.path.exists("data/har_test.csv"):
    with open("data/har_test.csv", "rb") as f:
        st.sidebar.download_button(
            label="Download har_test.csv",
            data=f,
            file_name="har_test.csv",
            mime="text/csv"
        )
else:
    st.sidebar.warning(
        "Test file har_test.csv not found. Please download it from the GitHub repository."
    )

# -----------------------------
# Overall Model Metrics
# -----------------------------
st.header("Overall Model Comparison at a Glance")

marked_metrics = mark_best_values(metrics_df)

st.dataframe(
    marked_metrics,
    width="stretch"
)

st.caption("★ indicates the best-performing model for each metric.")

# -----------------------------
# Prediction & Evaluation
# -----------------------------
if uploaded_file is not None:
    st.header("Evaluation on Uploaded Test Data")

    df_test = pd.read_csv(uploaded_file)

    if "label" not in df_test.columns:
        st.error("Uploaded CSV must contain a 'label' column.")
    else:
        X_test = df_test.drop(columns=["label"])
        y_true = df_test["label"] - 1  # HAR labels (1–6) → encoded (0–5)

        model = models[selected_model_name]
        y_pred = model.predict(X_test)

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.rename(index={str(k): v for k, v in LABEL_MAP.items()})

        class_rows = list(LABEL_MAP.values())
        class_df = report_df.loc[class_rows][
            ["precision", "recall", "f1-score", "support"]
        ]

        summary_df = pd.DataFrame({
            "Metric": ["Accuracy", "Macro Avg F1", "Weighted Avg F1"],
            "Value": [
                report["accuracy"],
                report["macro avg"]["f1-score"],
                report["weighted avg"]["f1-score"],
            ]
        })

        # -----------------------------
        # Tables
        # -----------------------------
        st.subheader("Evaluation Tables")
        tcol1, tcol2 = st.columns(2)

        with tcol1:
            st.markdown("Class-wise Performance")
            st.dataframe(
                class_df.style.format({
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1-score": "{:.3f}",
                }),
                width="stretch"
            )

        with tcol2:
            st.markdown("Overall Evaluation Metrics")
            st.dataframe(
                summary_df.style.format({"Value": "{:.4f}"}),
                width="stretch"
            )

        # -----------------------------
        # Visuals
        # -----------------------------
        st.subheader("Visual Analysis")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.markdown("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            labels = list(LABEL_MAP.values())

            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax1
            )
            ax1.set_xlabel("Predicted Activity")
            ax1.set_ylabel("True Activity")
            st.pyplot(fig1)

        with vcol2:
            st.markdown("Class-wise Metric Heatmap")
            heatmap_df = class_df[["precision", "recall", "f1-score"]]

            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                heatmap_df,
                annot=True,
                fmt=".3f",
                cmap="YlGnBu",
                cbar=True,
                ax=ax2
            )
            ax2.set_xlabel("Metric")
            ax2.set_ylabel("Activity")
            st.pyplot(fig2)

else:
    st.info(
        "Upload a small test CSV (with features and a 'label' column) "
        "to evaluate the selected model."
    )
