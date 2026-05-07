# evaluation/visualization.py

import plotly.graph_objects as go


def plot_anomalies(df):
    fig = go.Figure()

    # Normal signal
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["load"],
            mode="lines",
            name="Load"
        )
    )

    # Anomalies
    anomalies = df[df["anomaly_flag"] == 1]

    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["load"],
            mode="markers",
            name="Anomalies"
        )
    )

    fig.update_layout(
        title="Grid Load Anomaly Detection",
        xaxis_title="Time",
        yaxis_title="Load"
    )

    fig.show()