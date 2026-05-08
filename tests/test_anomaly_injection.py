from datetime import datetime

from ingestion.entsoe_client import EntsoeClient
from processing.parser import parse_entsoe_xml
from processing.cleaner import clean_time_series

from anomalies.inject import inject_random_anomalies

import plotly.graph_objects as go


client = EntsoeClient()

xml_data = client.fetch_load_actual(
    zone="10Y1001A1001A83F",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 7)
)

df_raw = parse_entsoe_xml(xml_data)

df_clean = clean_time_series(df_raw)

df_anom = inject_random_anomalies(df_clean)

print(df_anom["is_true_anomaly"].value_counts())


# ----------------------------------------
# Visualization
# ----------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_anom["timestamp"],
        y=df_anom["value"],
        mode="lines",
        name="Signal"
    )
)

anoms = df_anom[df_anom["is_true_anomaly"] == 1]

fig.add_trace(
    go.Scatter(
        x=anoms["timestamp"],
        y=anoms["value"],
        mode="markers",
        name="Injected Anomalies"
    )
)

fig.update_layout(
    title="Injected Grid Anomalies"
)

fig.show()