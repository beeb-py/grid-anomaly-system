# Grid Anomaly Detection System

An end-to-end machine learning system for detecting anomalies in electrical grid load data using real ENTSO-E power system telemetry.

This project implements:
- Real-world ENTSO-E data ingestion
- Time-series preprocessing and feature engineering
- Synthetic anomaly injection
- Isolation Forest anomaly detection
- LSTM Autoencoder sequence modeling
- Interactive Gradio dashboard
- Quantitative model evaluation and visualization

---

# Dashboard Preview

TBD

---

# Problem Statement

Modern power systems generate massive streams of telemetry data. Detecting anomalous load behavior is critical for:
- grid stability
- outage detection
- operational monitoring
- cyber-physical security
- predictive maintenance

This project explores both:
- classical machine learning methods
- deep sequence modeling approaches

for anomaly detection on real grid load data.

---

# Features

## Real ENTSO-E Data Ingestion
- Fetches historical grid load data directly from the ENTSO-E Transparency Platform API
- XML parsing and normalization
- Time-consistent preprocessing

## Feature Engineering
Generated features include:
- rolling statistics
- ramp rate
- z-score normalization
- cyclical temporal encodings
- weekday/hour embeddings

## Synthetic Anomaly Injection
Supports:
- spikes
- dropouts
- drift anomalies

Used for controlled benchmarking and evaluation.

## Isolation Forest
Classical unsupervised anomaly detection baseline:
- lightweight
- fast inference
- strong point anomaly detection

## LSTM Autoencoder
Sequence-based anomaly detector:
- temporal modeling
- reconstruction error scoring
- contextual anomaly detection

## Interactive Dashboard
Built with Gradio:
- model selection
- date-range querying
- anomaly visualization
- performance metrics

---

# System Architecture

TBD

Pipeline overview:

```text
ENTSO-E API
    ↓
XML Parsing
    ↓
Cleaning & Validation
    ↓
Feature Engineering
    ↓
Anomaly Injection
    ↓
Model Inference
    ↓
Metrics + Visualization
    ↓
Gradio Dashboard
```

---

# Model Comparison

| Model | Strengths | Weaknesses |
|---|---|---|
| Isolation Forest | Fast, interpretable, strong point anomaly detection | Limited temporal awareness |
| LSTM Autoencoder | Temporal sequence modeling, contextual anomaly detection | Higher computational cost, more false positives |

---

# Example Results

## Isolation Forest

TBD

---

## LSTM Autoencoder

TBD

---

# Repository Structure

```text
grid-anomaly-system/
│
├── app/                # Gradio application
├── ingestion/          # ENTSO-E API ingestion
├── processing/         # Cleaning and parsing
├── features/           # Feature engineering
├── anomalies/          # Synthetic anomaly injection
├── models/             # Isolation Forest + LSTM
├── inference/          # Unified inference pipeline
├── evaluation/         # Metrics and visualization
├── tests/              # Pipeline tests
├── assets/             # README images
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/beeb-py/grid-anomaly-system.git

cd grid-anomaly-system
```

Create virtual environment:

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Dashboard

```bash
python -m app.ui
```

Then open:

```text
http://127.0.0.1:7860
```

---

# Future Improvements

Potential extensions:
- streaming inference
- Kafka integration
- transformer-based sequence models
- online anomaly adaptation
- multivariate telemetry ingestion
- FastAPI deployment layer

---

# Tech Stack

- Python
- Pandas
- Scikit-learn
- PyTorch
- Plotly
- Gradio
- ENTSO-E API

---

# License

MIT License
