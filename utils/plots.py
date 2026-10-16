"""Plotly figure builders (replaces the copy-pasted notebook plotting cells)."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

SURVIVAL_LABELS = {0: "Did not Survive", 1: "Survived"}


def survival_pie(train_df: pd.DataFrame):
    counts = train_df["Survived"].value_counts()
    return px.pie(
        names=[SURVIVAL_LABELS[i] for i in counts.index],
        values=counts.values,
        title="Titanic Survival Rate",
    )


def survival_by(train_df: pd.DataFrame, x: str, title: str | None = None,
                labels: dict | None = None):
    """Grouped bar: survival counts for each value of `x` (Pclass, Sex, ...)."""
    counts = (
        train_df.groupby("Survived")[x]
        .value_counts()
        .reset_index()
    )
    counts["Survived"] = counts["Survived"].map(SURVIVAL_LABELS)
    return px.bar(
        counts,
        x=x,
        y="count",
        color="Survived",
        barmode="group",
        title=title or f"Survival Count by {x}",
        labels=labels or {x: x, "count": "Number of Passengers"},
    )


def fare_by_survival(train_df: pd.DataFrame):
    counts = train_df.value_counts().reset_index()
    return px.histogram(
        counts,
        x="Fare",
        y="count",
        color="Survived",
        title="Survival Count by Fare",
        labels={"count": "Number of Passengers"},
    )


def survived_distribution(train_df: pd.DataFrame, x: str, title: str):
    """Distribution of `x` among survivors (the notebook's KDE cells)."""
    survived = train_df[train_df["Survived"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=survived[x], name="Survived", histnorm="probability density",
        opacity=0.75, nbinsx=20,
    ))
    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title="Density",
        bargap=0.1, showlegend=False,
    )
    return fig


def confusion_matrix_fig(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    return px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted: Did not Survive", "Predicted: Survived"],
        y=["Actual: Did not Survive", "Actual: Survived"],
        title="Confusion Matrix",
    )


def roc_curve_fig(y_true, y_proba, auc: float):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC = {auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    return fig
