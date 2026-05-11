# evaluation/visualization.py

import plotly.graph_objects as go


def create_anomaly_figure(df):

    df = df.sort_values("timestamp")

    fig = go.Figure()

    # ------------------------------------------------
    # Main Signal
    # ------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["load"],
            mode="lines",
            name="Load"
        )
    )

    # ------------------------------------------------
    # True Positives
    # ------------------------------------------------

    tp = df[
        (df["anomaly_flag"] == 1) &
        (df["is_true_anomaly"] == 1)
    ]

    fig.add_trace(
        go.Scatter(
            x=tp["timestamp"],
            y=tp["load"],
            mode="markers",
            name="True Positive"
        )
    )

    # ------------------------------------------------
    # False Positives
    # ------------------------------------------------

    fp = df[
        (df["anomaly_flag"] == 1) &
        (df["is_true_anomaly"] == 0)
    ]

    fig.add_trace(
        go.Scatter(
            x=fp["timestamp"],
            y=fp["load"],
            mode="markers",
            name="False Positive"
        )
    )

    # ------------------------------------------------
    # False Negatives
    # ------------------------------------------------

    fn = df[
        (df["anomaly_flag"] == 0) &
        (df["is_true_anomaly"] == 1)
    ]

    fig.add_trace(
        go.Scatter(
            x=fn["timestamp"],
            y=fn["load"],
            mode="markers",
            name="False Negative"
        )
    )

    # ------------------------------------------------
    # Layout
    # ------------------------------------------------

    fig.update_layout(
        title="Grid Anomaly Detection",
        xaxis_title="Time",
        yaxis_title="Load"
    )

    return fig