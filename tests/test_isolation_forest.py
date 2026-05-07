from datetime import datetime

from ingestion.entsoe_client import EntsoeClient
from processing.parser import parse_entsoe_xml
from processing.cleaner import clean_time_series
from features.build_features import build_features
from models.isolation_forest import IsolationForestDetector
from evaluation.visualization import plot_anomalies


FEATURE_COLUMNS = [
    "load",
    "rolling_mean",
    "rolling_std",
    "ramp_rate",
    "z_score",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

client = EntsoeClient()

xml_data = client.fetch_load_actual(
    zone="10Y1001A1001A83F",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 7)
)

df_raw = parse_entsoe_xml(xml_data)

df_clean = clean_time_series(df_raw)

df_features = build_features(df_clean)

detector = IsolationForestDetector(
    contamination=0.02
)

detector.fit(df_features, FEATURE_COLUMNS)

results = detector.predict(df_features)

print(results.head())

print("\nAnomaly count:")
print(results["anomaly_flag"].value_counts())

print("\nMost anomalous points:")
print(
    results.sort_values("anomaly_score")
    .head(10)[
        ["timestamp", "load", "anomaly_score", "anomaly_flag"]
    ]
)

# Visualization
plot_anomalies(results)