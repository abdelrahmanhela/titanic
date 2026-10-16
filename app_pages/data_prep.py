"""Data preparation page: the transformation fitted on train, replayed on test."""
import streamlit as st

from utils.data import (
    SELECTED_FEATURES,
    build_features,
    load_test_raw,
    load_train,
)

st.title("Data preparation", icon=":material/filter_alt:")
st.caption(
    "Everything the model learned about *how to clean* was learned from the "
    "training manifest alone — then replayed, step for step, on the test manifest."
)

train_df, params, X, y = load_train()
test_df = load_test_raw()

# --- Step 1 -------------------------------------------------------------------
st.subheader("1 · Age imputation", icon=":material/water_drop:")
st.markdown(
    "Ages lost to history are filled with the **training** mean. The value is "
    "*fitted once* on train and reused for test — the test manifest never "
    "influences it."
)
m1, m2, m3 = st.columns(3)
m1.metric("Fitted train Age mean", f"{params['age_mean']:.2f}")
m2.metric("Train ages recovered", int(train_df["Age"].isna().sum()))
m3.metric("Test ages recovered", int(test_df["Age"].isna().sum()))

# --- Step 2 -------------------------------------------------------------------
st.subheader("2 · Passenger class dummies", icon=":material/grid_view:")
st.markdown("`Pclass` becomes three 0/1 columns — Upper (1st), Middle (2nd), Lower (3rd).")
st.dataframe(
    train_df[["Pclass"]].join(X[["Upper_Class", "Middle_Class", "Lower_Class"]]).head(),
    width="stretch",
    hide_index=True,
    alt="Pclass mapped to Upper, Middle and Lower class dummy columns",
)

# --- Step 3 -------------------------------------------------------------------
st.subheader("3 · Feature matrix", icon=":material/table_chart:")
st.caption(f"Seven features carried to the model: {', '.join(SELECTED_FEATURES)}")

m1, m2 = st.columns(2)
m1.metric("Train shape", f"{X.shape[0]} × {X.shape[1]}")
m2.metric("Missing values in X", int(X.isna().sum().sum()))

tab_train, tab_test = st.tabs(["Train features", "Test features · same transform"])
with tab_train:
    st.dataframe(X.head(), width="stretch", hide_index=True,
                 alt="First rows of the training feature matrix")
with tab_test:
    X_test = build_features(test_df, params)
    st.dataframe(X_test.head(), width="stretch", hide_index=True,
                 alt="First rows of the test feature matrix")
    st.badge("Identical columns, dtypes & imputation value", icon=":material/check:",
             color="green")
