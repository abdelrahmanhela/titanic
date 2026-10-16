"""Data loading and preprocessing (notebook cells 0-3).

The preprocessing is written as an explicit fit / transform pair so the exact
same transformations that were fitted on the training data (e.g. the Age mean
used to fill missing values) are replayed on the test data — no leakage, no
drift between what the model saw in the notebook and what the test pipeline
produces.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Columns used by the model (same selection as the notebook)
SELECTED_FEATURES = [
    "Sex", "Upper_Class", "Middle_Class", "Lower_Class", "Age", "SibSp", "Parch",
]

# Fixed encodings (deterministic, so they are safe to hard-code for train & test)
SEX_MAP = {"male": 1, "female": 0}
CLASS_MAP = {"Upper_Class": 1, "Middle_Class": 2, "Lower_Class": 3}


@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    """`source` can be an uploaded file object or a path."""
    return pd.read_csv(source)


def fit_preprocessing(train_df: pd.DataFrame) -> dict:
    """Learn everything the transformation needs from the TRAINING data only.

    Currently that is just the Age mean used to impute missing ages — the same
    value the notebook filled NaNs with, but now recorded so the test pipeline
    reuses it instead of recomputing its own.
    """
    return {"age_mean": float(train_df["Age"].mean())}


def build_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply the fitted preprocessing and return the model feature matrix.

    Works identically on train and test: same imputation value, same class
    dummies, same Sex encoding, same column order.
    """
    X = df.copy()

    # Filling the NaN values of Age with the TRAINING mean (fitted)
    X["Age"] = X["Age"].fillna(params["age_mean"])

    # One class dummy per passenger class (the Upper/Middle/Lower_Class step)
    for name, pclass in CLASS_MAP.items():
        X[name] = (X["Pclass"] == pclass).astype(int)

    X["Sex"] = X["Sex"].map(SEX_MAP)
    return X[SELECTED_FEATURES]


def load_train():
    """Return (raw train df, features X, target y)."""
    train_df = load_csv(TRAIN_CSV)
    params = fit_preprocessing(train_df)
    X = build_features(train_df, params)
    y = train_df["Survived"].astype(int)
    return train_df, params, X, y


def load_test_raw() -> pd.DataFrame:
    """Raw (unprocessed) test dataframe."""
    return load_csv(TEST_CSV)
