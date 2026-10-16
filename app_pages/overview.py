"""Overview page: data source, dataset summary, and preview."""
import pandas as pd
import streamlit as st

from utils.data import TEST_CSV, TRAIN_CSV, load_csv

st.title("Overview", icon=":material/home:")
st.caption(
    "Explore the Titanic dataset, review the model, and generate Kaggle-style "
    "predictions for the test set — **Data prep → EDA → Model → Predict**."
)

train_df = load_csv(TRAIN_CSV)
test_df = load_csv(TEST_CSV)

with st.container(border=True):
    st.subheader("Data source", icon=":material/database:")
    c1, c2 = st.columns(2)
    c1.metric("Training passengers", f"{len(train_df):,}", border=True, height="stretch")
    c2.metric("Test passengers", f"{len(test_df):,}", border=True, height="stretch")
    st.caption(
        f"`{TRAIN_CSV.name}` and `{TEST_CSV.name}` are loaded automatically from the "
        "`Data/` folder."
    )

# --- KPI row -----------------------------------------------------------------
survival_rate = train_df["Survived"].mean()
avg_age = train_df["Age"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Survival rate", f"{survival_rate:.1%}", border=True, height="stretch")
k2.metric("Avg age", f"{avg_age:.1f}", border=True, height="stretch")
k3.metric("Avg fare", f"${train_df['Fare'].mean():.2f}", border=True, height="stretch")
k4.metric(
    "Missing ages",
    f"{train_df['Age'].isna().sum():,}",
    border=True,
    height="stretch",
    delta=f"test: {test_df['Age'].isna().sum():,}",
)

# --- Previews -----------------------------------------------------------------
tab_train, test_tab = st.tabs(["Training set", "Test set"])

with tab_train:
    st.dataframe(train_df.head(20), width="stretch", alt="First 20 rows of the training set")

with test_tab:
    st.dataframe(test_df.head(20), width="stretch", alt="First 20 rows of the test set")
    st.caption("No `Survived` column — the Predict page fills it in.")

# --- Schema -------------------------------------------------------------------
st.subheader("Columns", icon=":material/table_rows:")
schema = pd.DataFrame({
    "column": train_df.columns,
    "dtype": train_df.dtypes.astype(str).values,
    "train missing": train_df.isna().sum().values,
    "test missing": [int(test_df[c].isna().sum()) if c in test_df.columns else None
                     for c in train_df.columns],
})
st.dataframe(schema, width="stretch", hide_index=True, alt="Column types and missing values")
