from datetime import datetime

from ingestion.entsoe_client import EntsoeClient
from processing.parser import parse_entsoe_xml
from processing.cleaner import clean_time_series

from anomalies.inject import inject_random_anomalies

from features.build_features import build_features

from models.lstm_autoencoder import (
    LSTMAutoencoderDetector
)

from evaluation.metrics import compute_metrics
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
    end=datetime(2024, 1, 14)
)

# ----------------------------------------
# Pipeline
# ----------------------------------------

df_raw = parse_entsoe_xml(xml_data)

df_clean = clean_time_series(df_raw)

df_anom = inject_random_anomalies(df_clean)

df_features = build_features(df_anom)

# ----------------------------------------
# LSTM
# ----------------------------------------

detector = LSTMAutoencoderDetector(
    seq_len=24,
    hidden_size=32,
    epochs=100
)

train_df = df_features[
    df_features["is_true_anomaly"] == 0
]
detector.fit(train_df, FEATURE_COLUMNS)

results = detector.predict(df_features)

# ----------------------------------------
# Metrics
# ----------------------------------------

metrics = compute_metrics(results)

# ----------------------------------------
# Visualization
# ----------------------------------------

plot_anomalies(results)