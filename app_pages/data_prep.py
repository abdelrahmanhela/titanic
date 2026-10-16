"""Data preparation page: imputation, class dummies, feature matrix."""
import streamlit as st

from utils.data import (
    SELECTED_FEATURES,
    build_features,
    fit_preprocessing,
    load_train,
    load_test_raw,
)

st.title("Data preparation", icon=":material/filter_alt:")
st.caption(
    "The preprocessing is fitted on the **training** data only, then the exact "
    "same transform is applied to the test data."
)

train_df, params, X, y = load_train()
test_df = load_test_raw()

# --- Step 1: fitted imputation ----------------------------------------------
with st.container(border=True):
    st.subheader("1. Fitted on train: Age imputation", icon=":material/water_drop:")
    st.caption(
        "Missing `Age` values are filled with the **training** mean — recorded here "
        "so the test pipeline reuses the same value instead of its own."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Train Age mean (fitted)", f"{params['age_mean']:.2f}")
    m2.metric("Train ages filled", int(train_df["Age"].isna().sum()))
    m3.metric("Test ages filled", int(test_df["Age"].isna().sum()))

# --- Step 2: class dummies ---------------------------------------------------
with st.container(border=True):
    st.subheader("2. Passenger class dummies", icon=":material/grid_view:")
    st.caption("`Pclass` becomes three 0/1 columns: Upper (1), Middle (2), Lower (3).")
    st.dataframe(
        train_df[["Pclass"]].join(X[["Upper_Class", "Middle_Class", "Lower_Class"]]).head(),
        width="stretch",
        hide_index=True,
        alt="Pclass mapped to Upper, Middle and Lower class dummy columns",
    )

# --- Step 3: feature matrix --------------------------------------------------
with st.container(border=True):
    st.subheader("3. Feature matrix", icon=":material/table_chart:")
    st.caption(f"Selected features: {', '.join(SELECTED_FEATURES)}")
    m1, m2 = st.columns(2)
    m1.metric("Shape", f"{X.shape[0]} × {X.shape[1]}")
    m2.metric("Missing values in X", int(X.isna().sum().sum()))

    tab_train, test_tab = st.tabs(["Train features", "Test features (same transform)"])
    with tab_train:
        st.dataframe(X.head(), width="stretch", hide_index=True,
                     alt="First rows of the training feature matrix")
    with test_tab:
        X_test = build_features(test_df, params)
        st.dataframe(X_test.head(), width="stretch", hide_index=True,
                     alt="First rows of the test feature matrix")
        st.caption("Identical columns, identical dtypes, identical imputation value.")
