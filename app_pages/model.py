"""Model page: training settings, evaluation metrics, ROC, feature importance."""
import pandas as pd
import streamlit as st

from utils.data import load_train
from utils.model import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    MODEL_NAMES,
    train_and_evaluate,
)
from utils.plots import confusion_matrix_fig, roc_curve_fig

st.title("The forecaster", icon=":material/model_training:")
st.caption("A classifier reads seven traits of each passenger and guesses their fate.")

_, _, X, y = load_train()

# --- Training settings (sidebar form: retrains only on submit) ---------------
with st.sidebar:
    st.header("Training settings", icon=":material/tune:")
    with st.form("training_settings"):
        model_name = st.segmented_control("Model", MODEL_NAMES, default=MODEL_NAMES[0])
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", value=DEFAULT_RANDOM_STATE, step=1)
        if model_name == "Logistic Regression":
            max_depth = 0
            max_iter = int(st.number_input("Max iterations", 100, 5000, 1000, 100))
        else:
            max_depth = int(st.number_input("Max depth (0 = unlimited)", 0, 50,
                                            DEFAULT_MAX_DEPTH, 1))
            max_iter = 1000
        n_estimators = int(st.number_input(
            "Trees (random forest)", 50, 1000, DEFAULT_N_ESTIMATORS, 50
        ))
        st.form_submit_button("Train model", type="primary", width="stretch")

model_name = model_name or MODEL_NAMES[0]

res = train_and_evaluate(
    X, y, model_name, test_size, int(random_state), n_estimators, max_depth, max_iter
)

# --- Scoreboard ----------------------------------------------------------------
st.subheader("Scoreboard", icon=":material/trophy:")
st.caption(
    f"{model_name} · {test_size:.0%} held out · seed {int(random_state)}"
)

k1, k2, k3 = st.columns(3)
k1.metric("Accuracy", f"{res['accuracy']:.4f}", border=True, height="stretch")
k2.metric("ROC AUC", f"{res['roc_auc']:.4f}", border=True, height="stretch")
k3.metric("Held-out passengers", f"{res['shapes']['X_test'][0]:,}",
          border=True, height="stretch")

# --- Errors --------------------------------------------------------------------
st.subheader("Where it errs", icon=":material/crisis_alert:")
c1, c2 = st.columns(2)
c1.plotly_chart(
    confusion_matrix_fig(res["y_test"], res["y_pred"]),
    width="stretch",
    alt="Confusion matrix of actual versus predicted survival",
)
c2.plotly_chart(
    roc_curve_fig(res["y_test"], res["y_proba"], res["roc_auc"]),
    width="stretch",
    alt="ROC curve of the trained model",
)

st.subheader("Classification report", icon=":material/receipt_long:")
st.dataframe(
    res["report"].round(3),
    width="stretch",
    alt="Precision, recall and F1 score per class",
)


def _split_caption(dist: pd.Series) -> str:
    """Format a 0/1 class distribution as 'Did not survive 61.6% · Survived 38.4%'."""
    labels = {0: "Did not survive", 1: "Survived"}
    return " · ".join(f"{labels.get(int(k), k)} {v:.1%}" for k, v in dist.items())


with st.container(border=True):
    st.subheader("Train / test split", icon=":material/call_split:")
    left, right = st.columns(2)
    with left:
        st.markdown("**Training set**")
        st.metric("Rows", f"{res['shapes']['X_train'][0]:,}")
        st.caption(_split_caption(res["train_dist"]))
    with right:
        st.markdown("**Testing set**")
        st.metric("Rows", f"{res['shapes']['X_test'][0]:,}")
        st.caption(_split_caption(res["test_dist"]))

st.subheader("What the model weighs", icon=":material/bar_chart:")
imp = pd.Series(res["importance"], index=res["feature_names"]).sort_values()
st.bar_chart(imp, horizontal=True)
