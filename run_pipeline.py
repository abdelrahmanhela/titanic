"""CLI entry point for the test-data pipeline (no Streamlit needed).

Fits the preprocessing on train, applies the identical transform to test,
trains on the full training set, and writes Data/submission.csv.

Usage:
    python run_pipeline.py
    python run_pipeline.py --model "Logistic Regression" --random-state 1
"""
import argparse

from utils.data import TEST_CSV, TRAIN_CSV, build_features, fit_preprocessing, load_csv
from utils.model import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    MODEL_NAMES,
    fit_final_model,
)
from utils.pipeline import make_submission


def main():
    parser = argparse.ArgumentParser(description="Run the Titanic test-set pipeline.")
    parser.add_argument("--model", choices=MODEL_NAMES, default=MODEL_NAMES[0])
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help="0 = unlimited")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    train_raw = load_csv(TRAIN_CSV)
    test_raw = load_csv(TEST_CSV)

    # 1. Fit on train, apply the same transform to both
    params = fit_preprocessing(train_raw)
    X_train = build_features(train_raw, params)
    y_train = train_raw["Survived"].astype(int)
    X_test = build_features(test_raw, params)

    # 2. Train on the full training set
    model = fit_final_model(
        X_train, y_train, args.model, args.random_state,
        args.n_estimators, args.max_depth,
    )

    # 3./4. Predict and write the submission
    predictions = model.predict(X_test)
    submission = make_submission(test_raw, predictions)

    out = args.output or str(TRAIN_CSV.parent / "submission.csv")
    submission.to_csv(out, index=False)

    print(f"Model:              {args.model}")
    print(f"Train accuracy:     {model.score(X_train, y_train):.4f}")
    print(f"Predicted survivors:{submission['Survived'].sum():>6} "
          f"({submission['Survived'].mean():.1%})")
    print(f"Submission written: {out}")


if __name__ == "__main__":
    main()
