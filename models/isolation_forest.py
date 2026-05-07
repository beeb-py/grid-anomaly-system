# models/isolation_forest.py

from sklearn.ensemble import IsolationForest
import pandas as pd


class IsolationForestDetector:
    def __init__(
        self,
        contamination: float = 0.01,
        random_state: int = 42
    ):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

        self.feature_columns = None

    def fit(self, df: pd.DataFrame, feature_columns: list):
        self.feature_columns = feature_columns

        X = df[feature_columns]

        self.model.fit(X)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            raise ValueError("Model has not been fitted")

        X = df[self.feature_columns]

        # sklearn output:
        #  1  = normal
        # -1  = anomaly
        preds = self.model.predict(X)

        scores = self.model.decision_function(X)

        result = df.copy()

        result["anomaly_flag"] = (preds == -1).astype(int)

        # Lower score = more anomalous
        result["anomaly_score"] = scores

        return result