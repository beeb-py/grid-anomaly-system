from inference.predictor import (
    run_prediction_pipeline
)

# ----------------------------------------
# Isolation Forest
# ----------------------------------------

iforest_results = run_prediction_pipeline(
    model_name="iforest"
)

print("\n=== ISOLATION FOREST ===")

print(iforest_results["metrics"])

print(
    f"Training Time: "
    f"{iforest_results['training_time']} sec"
)

print(
    f"Inference Time: "
    f"{iforest_results['inference_time']} sec"
)

iforest_results["figure"].show()

# ----------------------------------------
# LSTM
# ----------------------------------------

lstm_results = run_prediction_pipeline(
    model_name="lstm"
)

print("\n=== LSTM ===")

print(lstm_results["metrics"])

print(
    f"Training Time: "
    f"{lstm_results['training_time']} sec"
)

print(
    f"Inference Time: "
    f"{lstm_results['inference_time']} sec"
)

lstm_results["figure"].show()