"""Train the five classifiers on the UCI Credit Card Default dataset.

Usage:  python -m src.train
Outputs: model/*.joblib, model/metrics.csv, test_data.csv
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features import CATEGORICAL, ENGINEERED_NUMERIC, FEATURES, TARGET, prepare
from src.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "credit_default.csv"
MODEL_DIR = ROOT / "model"
SEED = 26


def make_preprocessor() -> Pipeline:
    encode = ColumnTransformer([
        ("num", StandardScaler(), ENGINEERED_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("prepare", FunctionTransformer(prepare)), ("encode", encode)])


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance"),
    "Naive Bayes": GaussianNB(),
    # Platt-scaled RBF SVM so that predict_proba (needed for AUC) is available
    "Support Vector Machine": CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED), method="sigmoid", cv=5, ensemble=False
    ),
}


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Held-out test split, exported for upload into the Streamlit app
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for name, clf in CLASSIFIERS.items():
        pipe = Pipeline([("pre", make_preprocessor()), ("clf", clf)])
        print(f"Training {name} ...", flush=True)
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / f"{slug(name)}.joblib", compress=3)
        m = evaluate(y_test, pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1])
        rows.append({"ML Model Name": name, **m})
        print("   ", json.dumps({k: round(v, 4) for k, v in m.items()}))

    table = pd.DataFrame(rows).round(4)
    table.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
