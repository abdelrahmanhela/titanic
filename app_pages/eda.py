"""Survival patterns page: who lived, who died, and along which dimensions."""
import streamlit as st

from utils.data import load_train
from utils.plots import (
    fare_by_survival,
    survival_by,
    survival_pie,
    survived_distribution,
)

st.title("Survival patterns", icon=":material/query_stats:")
st.caption("Who lived, who died — and along which dimensions the fate turned.")

train_df, _, _, _ = load_train()

# --- Overall rate --------------------------------------------------------------
st.plotly_chart(survival_pie(train_df), width="stretch",
                alt="Pie chart of overall survival rate")

# --- Breakdown explorer ---------------------------------------------------------
st.subheader("Break it down", icon=":material/stacked_bar_chart:")
dimension = st.pills(
    "Dimension",
    ["Passenger class", "Sex", "Fare", "Family size"],
    default="Passenger class",
    selection_mode="single",
)

if dimension == "Passenger class":
    st.plotly_chart(
        survival_by(train_df, "Pclass", title="Survival count by passenger class",
                    labels={"Pclass": "Passenger class", "count": "Number of passengers"}),
        width="stretch",
        alt="Grouped bar chart of survival count per passenger class",
    )
elif dimension == "Sex":
    st.plotly_chart(
        survival_by(train_df, "Sex", title="Survival count by sex",
                    labels={"Sex": "Sex", "count": "Number of passengers"}),
        width="stretch",
        alt="Grouped bar chart of survival count per sex",
    )
elif dimension == "Fare":
    st.plotly_chart(fare_by_survival(train_df), width="stretch",
                    alt="Histogram of fare coloured by survival")
else:
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        survived_distribution(train_df, "Parch", "Survivors by Parch (parents / children)"),
        width="stretch",
        alt="Distribution of Parch among survivors",
    )
    c2.plotly_chart(
        survived_distribution(train_df, "SibSp", "Survivors by SibSp (siblings / spouses)"),
        width="stretch",
        alt="Distribution of SibSp among survivors",
    )
