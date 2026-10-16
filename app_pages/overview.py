"""Overview page: the two manifests at a glance, previews, and schema."""
import pandas as pd
import streamlit as st

from utils.data import TEST_CSV, TRAIN_CSV, load_csv

st.title("The passenger manifests", icon=":material/directions_boat:")
st.caption(
    "891 souls aboard the training manifest, 418 more whose fate the model must "
    "foretell. Explore the data, then move through **Data prep → Survival "
    "patterns → Model → Predict**."
)

train_df = load_csv(TRAIN_CSV)
test_df = load_csv(TEST_CSV)

# --- The manifests ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training passengers", f"{len(train_df):,}", border=True, height="stretch")
c2.metric("Test passengers", f"{len(test_df):,}", border=True, height="stretch")
c3.metric("Survival rate (train)", f"{train_df['Survived'].mean():.1%}",
          border=True, height="stretch")
c4.metric(
    "Ages never recorded",
    f"{train_df['Age'].isna().sum()} / {test_df['Age'].isna().sum()}",
    border=True, height="stretch",
    help="Missing Age values in train / test — imputed in Data prep",
)

st.caption(
    f"Loaded from `{TRAIN_CSV.name}` and `{TEST_CSV.name}` in the `Data/` folder."
)

# --- Previews -----------------------------------------------------------------
tab_train, tab_test = st.tabs(["Training manifest", "Test manifest"])

with tab_train:
    st.dataframe(train_df.head(20), width="stretch", alt="First 20 rows of the training set")

with tab_test:
    st.dataframe(test_df.head(20), width="stretch", alt="First 20 rows of the test set")
    st.badge("No `Survived` column", icon=":material/help:", color="orange")

# --- Schema -------------------------------------------------------------------
with st.expander("Columns, types & missing values"):
    schema = pd.DataFrame({
        "column": train_df.columns,
        "dtype": train_df.dtypes.astype(str).values,
        "train missing": train_df.isna().sum().values,
        "test missing": [int(test_df[c].isna().sum()) if c in test_df.columns else None
                         for c in train_df.columns],
    })
    st.dataframe(schema, width="stretch", hide_index=True,
                 alt="Column types and missing values for train and test")
