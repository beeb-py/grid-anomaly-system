# inference/predictor.py

from datetime import datetime
import time

from ingestion.entsoe_client import EntsoeClient

from processing.parser import parse_entsoe_xml
from processing.cleaner import clean_time_series

from anomalies.inject import inject_random_anomalies

from features.build_features import build_features

from models.isolation_forest import (
    IsolationForestDetector
)

from models.lstm_autoencoder import (
    LSTMAutoencoderDetector
)

from evaluation.metrics import compute_metrics

from evaluation.visualization import (
    create_anomaly_figure
)


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


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def run_prediction_pipeline(
    model_name="iforest",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 14),
    zone="10Y1001A1001A83F"
):

    # --------------------------------------------
    # Ingestion
    # --------------------------------------------

    client = EntsoeClient()

    xml_data = client.fetch_load_actual(
        zone=zone,
        start=start,
        end=end
    )

    # --------------------------------------------
    # Processing
    # --------------------------------------------

    df_raw = parse_entsoe_xml(xml_data)

    df_clean = clean_time_series(df_raw)

    # --------------------------------------------
    # Inject anomalies
    # --------------------------------------------

    df_anom = inject_random_anomalies(df_clean)

    # --------------------------------------------
    # Features
    # --------------------------------------------

    df_features = build_features(df_anom)

    # --------------------------------------------
    # Model Selection
    # --------------------------------------------

    if model_name == "iforest":

        detector = IsolationForestDetector(
            contamination=0.05
        )

        # ----------------------------------------
        # Training Timer
        # ----------------------------------------

        train_start = time.time()

        detector.fit(
            df_features,
            FEATURE_COLUMNS
        )

        train_end = time.time()

        # ----------------------------------------
        # Inference Timer
        # ----------------------------------------

        inference_start = time.time()

        results = detector.predict(df_features)

        inference_end = time.time()

    elif model_name == "lstm":

        detector = LSTMAutoencoderDetector(
            seq_len=48,
            hidden_size=32,
            epochs=100,
            threshold_percentile=95
        )

        train_df = df_features[
            df_features["is_true_anomaly"] == 0
        ]

        # ----------------------------------------
        # Training Timer
        # ----------------------------------------

        train_start = time.time()

        detector.fit(
            train_df,
            FEATURE_COLUMNS
        )

        train_end = time.time()

        # ----------------------------------------
        # Inference Timer
        # ----------------------------------------

        inference_start = time.time()

        results = detector.predict(df_features)

        inference_end = time.time()

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # --------------------------------------------
    # Metrics
    # --------------------------------------------

    metrics = compute_metrics(
        results,
        verbose=False
    )

    # --------------------------------------------
    # Visualization
    # --------------------------------------------

    figure = create_anomaly_figure(results)

    # --------------------------------------------
    # Unified Output
    # --------------------------------------------

    return {
        "model_name": model_name,
        "results_df": results,
        "metrics": metrics,
        "figure": figure,
        "training_time": round(
            train_end - train_start,
            4
        ),
        "inference_time": round(
            inference_end - inference_start,
            4
        )
    }