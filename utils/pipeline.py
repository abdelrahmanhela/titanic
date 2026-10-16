"""Test-data pipeline (the step missing from the notebook).

Everything that was applied to the training data is replayed on the test data:

1. The preprocessing is *fitted on train only* (Age mean imputation) and then
   applied to test — the same `build_features` function, the same columns, the
   same dtypes.
2. A model is trained on the FULL training set with the chosen hyper-params.
3. The model predicts `Survived` for every test passenger.
4. A Kaggle-style `submission.csv` (PassengerId, Survived) is returned, ready
   to download.
"""
import pandas as pd

from utils.data import TEST_CSV, TRAIN_CSV, build_features, fit_preprocessing, load_csv
from utils.model import fit_final_model


def make_submission(test_raw: pd.DataFrame, predictions) -> pd.DataFrame:
    return pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Survived": predictions.astype(int),
    })


def run_test_pipeline(train_raw: pd.DataFrame, test_raw: pd.DataFrame, model_name: str,
                      random_state: int, n_estimators: int, max_depth: int,
                      max_iter: int = 1000) -> dict:
    """Fit on train, transform test identically, predict, build the submission."""
    # 1. Fit preprocessing on TRAIN, apply the same transform to both
    params = fit_preprocessing(train_raw)
    X_train = build_features(train_raw, params)
    y_train = train_raw["Survived"].astype(int)
    X_test = build_features(test_raw, params)

    # 2. Train the final model on the full training set
    model = fit_final_model(
        X_train, y_train, model_name, random_state, n_estimators, max_depth, max_iter
    )

    # 3./4. Predict and package
    predictions = model.predict(X_test)
    submission = make_submission(test_raw, predictions)

    return {
        "params": params,
        "X_test": X_test,
        "submission": submission,
        "survival_rate": float(predictions.mean()),
        "n_passengers": len(predictions),
        "train_accuracy": float(model.score(X_train, y_train)),
    }


def run_test_pipeline_from_paths(model_name: str, random_state: int, n_estimators: int,
                                 max_depth: int, max_iter: int = 1000) -> dict:
    """Convenience wrapper that reads the CSVs from disk (for CLI / scripts)."""
    return run_test_pipeline(
        load_csv(TRAIN_CSV), load_csv(TEST_CSV),
        model_name, random_state, n_estimators, max_depth, max_iter,
    )
