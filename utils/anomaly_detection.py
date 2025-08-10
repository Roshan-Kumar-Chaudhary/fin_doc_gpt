

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Union

def detect_anomalies(data: Union[pd.Series, list], contamination: float = 0.05) -> np.ndarray:
    """Enhanced anomaly detection for financial metrics"""
    if len(data) < 10:
        return np.array([])
    
    X = np.array(data).reshape(-1, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        max_samples='auto',
        behaviour='new'
    )
    
    model.fit(X_scaled)
    anomalies = model.predict(X_scaled)
    
    return np.where(anomalies == -1)[0]

def detect_tabular_anomalies(df: pd.DataFrame, numerical_cols: list = None) -> dict:
    """Detect anomalies in financial tables"""
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    results = {}
    for col in numerical_cols:
        if len(df[col].dropna()) >= 10:
            anomalies = detect_anomalies(df[col].dropna())
            if len(anomalies) > 0:
                results[col] = {
                    'anomaly_indices': anomalies,
                    'anomaly_values': df[col].iloc[anomalies].values,
                    'percentage': len(anomalies)/len(df)*100
                }
    
    return results