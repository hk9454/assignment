"""Feature definitions and cleaning shared by training and the Streamlit app."""
import numpy as np
import pandas as pd

TARGET = "default"

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented category codes into an 'other' bucket.

    The UCI documentation lists EDUCATION in {1,2,3,4} and MARRIAGE in {1,2,3},
    but the raw file also contains 0, 5 and 6. These are mapped to 'other'.
    """
    X = X.copy()
    X["EDUCATION"] = X["EDUCATION"].where(X["EDUCATION"].isin([1, 2, 3]), 4)
    X["MARRIAGE"] = X["MARRIAGE"].where(X["MARRIAGE"].isin([1, 2]), 3)
    return X


def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """Engineered features: credit utilisation and repayment ratio for the latest month."""
    X = X.copy()
    limit = X["LIMIT_BAL"].replace(0, np.nan)
    X["UTIL_1"] = (X["BILL_AMT1"] / limit).fillna(0).clip(-1, 5)
    bill = X["BILL_AMT2"].where(X["BILL_AMT2"] > 0, np.nan)
    X["REPAY_RATIO_1"] = (X["PAY_AMT1"] / bill).fillna(1).clip(0, 5)
    X["MAX_DELAY"] = X[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].max(axis=1)
    return X


def prepare(X: pd.DataFrame) -> pd.DataFrame:
    return add_ratio_features(clean_features(X[FEATURES]))


ENGINEERED_NUMERIC = NUMERIC + ["UTIL_1", "REPAY_RATIO_1", "MAX_DELAY"]
