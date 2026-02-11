from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    def __init__(self):
        # Initialize the Isolation Forest model
        self.model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        self.fitted = False

    def fit(self, data):
        # Fit the model to the data (e.g., crowd counts)
        if len(data) > 1:  # Ensure we have more than one data point
            self.model.fit(data)
            self.fitted = True

    def predict(self, data):
        # Predict anomalies (-1 for anomalies, 1 for normal data)
        if self.fitted:
            return self.model.predict(data)
        else:
            return [1] * len(data)  # If not yet fitted, assume normal data
