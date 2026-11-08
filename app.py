"""Credit Card Default Risk — model evaluation dashboard (ML Assignment 2)."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.features import FEATURES, TARGET
from src.metrics import evaluate

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Credit Default Risk Lab", page_icon="💳", layout="wide")


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def score(model, X, threshold):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int), proba


# ---------------- Sidebar ----------------
st.sidebar.title("💳 Credit Default Risk Lab")
st.sidebar.caption("UCI *Default of Credit Card Clients* · Taiwan, 2005")

uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
use_sample = st.sidebar.toggle("Use bundled test_data.csv", value=uploaded is None)
model_name = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES))
threshold = st.sidebar.slider("Decision threshold P(default)", 0.05, 0.95, 0.50, 0.05,
                              help="Clients with predicted default probability above this are flagged.")

with open(ROOT / "test_data.csv", "rb") as fh:
    st.sidebar.download_button("⬇️ Download sample test_data.csv", fh, "test_data.csv", "text/csv")

# ---------------- Data ----------------
st.title("Will this client default next month?")
st.write("Upload held-out client records and evaluate the five pre-trained classifiers. "
         "Models are **loaded from disk** — nothing is retrained on the uploaded data.")

if uploaded is not None and not use_sample:
    df = read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    df = read_csv(ROOT / "test_data.csv")
    source = "test_data.csv (bundled 20% hold-out)"
else:
    st.info("👈 Upload a CSV or switch on the bundled test data to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

df = df.dropna(subset=FEATURES + [TARGET])
X, y = df[FEATURES], df[TARGET].astype(int)

c1, c2, c3 = st.columns(3)
c1.metric("Source", source.split(" (")[0])
c2.metric("Rows", f"{len(df):,}")
c3.metric("Actual default rate", f"{y.mean():.1%}")

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(50), width="stretch")

tab_single, tab_compare = st.tabs([f"🔍 {model_name}", "📊 Compare all models"])

# ---------------- Single model ----------------
with tab_single:
    model = load_model(model_name)
    y_pred, y_proba = score(model, X, threshold)
    m = evaluate(y, y_pred, y_proba)

    cols = st.columns(3) + st.columns(3)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}", border=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No default", "Default"], yticklabels=["No default", "Default"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig, width="content")
    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y, y_proba)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
        st.pyplot(fig, width="content")

    st.subheader("Classification report")
    report = classification_report(y, y_pred, target_names=["No default (0)", "Default (1)"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    out = df.assign(predicted=y_pred, p_default=np.round(y_proba, 4))
    st.download_button("⬇️ Download predictions", out.to_csv(index=False), f"predictions_{MODEL_FILES[model_name][:-7]}.csv")

# ---------------- Compare ----------------
with tab_compare:
    rows = []
    for name in MODEL_FILES:
        p, pr = score(load_model(name), X, threshold)
        rows.append({"ML Model Name": name, **evaluate(y, p, pr)})
    table = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"), width="stretch")

    long = table.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", ax=ax)
    ax.set_ylim(0, 1); ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    st.pyplot(fig)
    best = table["MCC"].idxmax()
    st.success(f"Best model by MCC at threshold {threshold:.2f}: **{best}** (MCC = {table.loc[best, 'MCC']:.4f})")
