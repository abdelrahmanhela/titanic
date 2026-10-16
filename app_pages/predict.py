"""Predict page: run the fitted preprocessing + trained model on the test set."""
import streamlit as st

from utils.data import load_test_raw, load_train
from utils.model import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    MODEL_NAMES,
)
from utils.pipeline import run_test_pipeline

st.title("Foretell their fate", icon=":material/upload:")
st.caption(
    "The pipeline replays **every training transformation** on the test manifest — "
    "the train-fitted Age imputation, the class dummies, the Sex encoding — then "
    "trains on the full training set and predicts `Survived` for each passenger."
)

train_df, params, X, y = load_train()
test_df = load_test_raw()

# --- Pipeline settings (sidebar form) ----------------------------------------
with st.sidebar:
    st.header("Pipeline settings", icon=":material/settings:")
    with st.form("pipeline_settings"):
        model_name = st.segmented_control("Model", MODEL_NAMES, default=MODEL_NAMES[0])
        random_state = st.number_input("Random state", value=DEFAULT_RANDOM_STATE, step=1,
                                       key="pipe_rs")
        if model_name == "Logistic Regression":
            max_depth = 0
            max_iter = int(st.number_input("Max iterations", 100, 5000, 1000, 100,
                                           key="pipe_iter"))
        else:
            max_depth = int(st.number_input("Max depth (0 = unlimited)", 0, 50,
                                            DEFAULT_MAX_DEPTH, 1, key="pipe_depth"))
            max_iter = 1000
        n_estimators = int(st.number_input(
            "Trees (random forest)", 50, 1000, DEFAULT_N_ESTIMATORS, 50, key="pipe_trees"
        ))
        run = st.form_submit_button("Run pipeline", type="primary", width="stretch")

model_name = model_name or MODEL_NAMES[0]

with st.spinner("Running the test pipeline..."):
    result = run_test_pipeline(
        train_df, test_df, model_name, int(random_state), n_estimators, max_depth, max_iter
    )

submission = result["submission"]

# --- The voyage (pipeline steps) ---------------------------------------------
st.subheader("The voyage", icon=":material/steps:")
st.markdown(
    f"""
1. **Fit on train** — Age mean `= {result['params']['age_mean']:.2f}` recorded from the training manifest.
2. **Transform** — the same `build_features` transform applied to train *and* test.
3. **Train** — `{model_name}` fitted on the full training set ({len(train_df):,} rows).
4. **Predict** — `Survived` foretold for each of the {result['n_passengers']:,} test passengers.
"""
)

k1, k2, k3 = st.columns(3)
k1.metric("Predicted survivors", f"{int(submission['Survived'].sum()):,}",
          border=True, height="stretch")
k2.metric("Predicted survival rate", f"{result['survival_rate']:.1%}",
          border=True, height="stretch")
k3.metric("Train accuracy (resubstitution)", f"{result['train_accuracy']:.4f}",
          border=True, height="stretch")

# --- Submission --------------------------------------------------------------
st.subheader("Manifest of fates", icon=":material/download:")
st.dataframe(submission.head(20), width="stretch", hide_index=True,
             alt="First 20 rows of the submission file")
st.download_button(
    "Download submission.csv",
    data=submission.to_csv(index=False).encode("utf-8"),
    file_name="submission.csv",
    mime="text/csv",
    type="primary",
    width="stretch",
)
