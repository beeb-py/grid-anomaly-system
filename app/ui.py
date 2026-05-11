# app/ui.py

import gradio as gr
from datetime import datetime

from inference.predictor import (
    run_prediction_pipeline
)


# ---------------------------------------------------
# Gradio Backend Wrapper
# ---------------------------------------------------

def run_dashboard(
    model_name,
    start_date,
    end_date
):

    start_dt = datetime.strptime(
        start_date,
        "%Y-%m-%d"
    )

    end_dt = datetime.strptime(
        end_date,
        "%Y-%m-%d"
    )

    results = run_prediction_pipeline(
        model_name=model_name,
        start=start_dt,
        end=end_dt
    )

    metrics = results["metrics"]

    metrics_text = f"""
Precision: {metrics['precision']:.4f}

Recall: {metrics['recall']:.4f}

F1 Score: {metrics['f1_score']:.4f}

Training Time: {results['training_time']} sec

Inference Time: {results['inference_time']} sec
"""

    return (
        results["figure"],
        metrics_text
    )


# ---------------------------------------------------
# Gradio Interface
# ---------------------------------------------------

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Grid Anomaly Detection System

        Compare Isolation Forest and LSTM Autoencoder
        models on real ENTSO-E grid load data.
        """
    )

    with gr.Row():

        model_dropdown = gr.Dropdown(
            choices=[
                "iforest",
                "lstm"
            ],
            value="iforest",
            label="Model"
        )

        start_date = gr.Textbox(
            value="2024-01-01",
            label="Start Date"
        )

        end_date = gr.Textbox(
            value="2024-01-14",
            label="End Date"
        )

    run_button = gr.Button(
        "Run Detection"
    )

    output_plot = gr.Plot(
        label="Anomaly Visualization"
    )

    output_metrics = gr.Textbox(
        label="Metrics"
    )

    run_button.click(
        fn=run_dashboard,
        inputs=[
            model_dropdown,
            start_date,
            end_date
        ],
        outputs=[
            output_plot,
            output_metrics
        ]
    )


# ---------------------------------------------------
# Launch
# ---------------------------------------------------

demo.launch()