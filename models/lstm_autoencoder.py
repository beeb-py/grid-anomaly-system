# models/lstm_autoencoder.py

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------
# Sequence Builder
# ---------------------------------------------------

def create_sequences(data, seq_len=24):

    sequences = []

    for i in range(len(data) - seq_len):
        seq = data[i:i + seq_len]
        sequences.append(seq)

    return np.array(sequences)


# ---------------------------------------------------
# LSTM Autoencoder
# ---------------------------------------------------

class LSTMAutoencoder(nn.Module):

    def __init__(self, n_features, hidden_size=32):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            batch_first=True
        )

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)

        # Repeat hidden state across sequence length
        repeated = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)

        reconstructed, _ = self.decoder(repeated)

        return reconstructed


# ---------------------------------------------------
# Detector Wrapper
# ---------------------------------------------------

class LSTMAutoencoderDetector:

    def __init__(
        self,
        seq_len=24,
        hidden_size=32,
        epochs=50,
        lr=1e-3,
        threshold_percentile=95,
        device=None
    ):

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.threshold_percentile = threshold_percentile

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.scaler = StandardScaler()

        self.model = None

        self.threshold = None

        self.feature_columns = None

    # ------------------------------------------------

    def fit(self, df, feature_columns):

        self.feature_columns = feature_columns

        X = df[feature_columns].values

        X_scaled = self.scaler.fit_transform(X)

        sequences = create_sequences(
            X_scaled,
            seq_len=self.seq_len
        )

        X_tensor = torch.tensor(
            sequences,
            dtype=torch.float32
        ).to(self.device)

        n_features = X_tensor.shape[2]

        self.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_size=self.hidden_size
        ).to(self.device)

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        self.model.train()

        for epoch in range(self.epochs):

            optimizer.zero_grad()

            reconstructed = self.model(X_tensor)

            loss = criterion(reconstructed, X_tensor)

            loss.backward()

            optimizer.step()

            print(
                f"Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.6f}"
            )

        # --------------------------------------------
        # Reconstruction errors for threshold
        # --------------------------------------------

        self.model.eval()

        with torch.no_grad():

            reconstructed = self.model(X_tensor)

            errors = torch.mean(
                (X_tensor[:, -1, :] - reconstructed[:, -1, :]) ** 2,
                dim=(1)
            ).cpu().numpy()

        self.threshold = np.percentile(
            errors,
            self.threshold_percentile
        )

    # ------------------------------------------------

    def predict(self, df):

        X = df[self.feature_columns].values

        X_scaled = self.scaler.transform(X)

        sequences = create_sequences(
            X_scaled,
            seq_len=self.seq_len
        )

        X_tensor = torch.tensor(
            sequences,
            dtype=torch.float32
        ).to(self.device)

        self.model.eval()

        with torch.no_grad():

            reconstructed = self.model(X_tensor)

            errors = torch.mean(
                (X_tensor[:, -1, :] - reconstructed[:, -1, :]) ** 2,
                dim=(1)
            ).cpu().numpy()

        preds = (errors > self.threshold).astype(int)

        result = df.iloc[self.seq_len:].copy()

        result["anomaly_score"] = errors

        result["anomaly_flag"] = preds

        return result