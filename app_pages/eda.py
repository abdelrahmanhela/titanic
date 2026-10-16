"""EDA page: survival distributions (notebook plotting cells as tabs)."""
import streamlit as st

from utils.data import load_train
from utils.plots import (
    fare_by_survival,
    survival_by,
    survival_pie,
    survived_distribution,
)

st.title("Exploratory data analysis", icon=":material/query_stats:")
st.caption("Survival patterns across class, sex, fare, and family size.")

train_df, _, _, _ = load_train()

tab_overview, tab_class, tab_sex, tab_fare, tab_family = st.tabs(
    ["Overview", "Passenger class", "Sex", "Fare", "Family"],
    on_change="rerun",
    key="eda_tabs",
)

with tab_overview:
    if tab_overview.open:
        st.plotly_chart(survival_pie(train_df), width="stretch",
                        alt="Pie chart of overall survival rate")

with tab_class:
    if tab_class.open:
        st.plotly_chart(
            survival_by(train_df, "Pclass", title="Survival Count by Passenger Class",
                        labels={"Pclass": "Passenger Class", "count": "Number of Passengers"}),
            width="stretch",
            alt="Grouped bar chart of survival count per passenger class",
        )

with tab_sex:
    if tab_sex.open:
        st.plotly_chart(
            survival_by(train_df, "Sex", title="Survival Count by Sex",
                        labels={"Sex": "Sex", "count": "Number of Passengers"}),
            width="stretch",
            alt="Grouped bar chart of survival count per sex",
        )

with tab_fare:
    if tab_fare.open:
        st.plotly_chart(fare_by_survival(train_df), width="stretch",
                        alt="Histogram of fare coloured by survival")

with tab_family:
    if tab_family.open:
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            survived_distribution(train_df, "Parch", "Survivors by Parch (parents/children)"),
            width="stretch",
            alt="Distribution of Parch among survivors",
        )
        c2.plotly_chart(
            survived_distribution(train_df, "SibSp", "Survivors by SibSp (siblings/spouses)"),
            width="stretch",
            alt="Distribution of SibSp among survivors",
        )
