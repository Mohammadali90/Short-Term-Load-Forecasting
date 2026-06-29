"""Shared regression evaluation metrics used by every model and experiment."""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute MAE, RMSE, R^2, and MAPE for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def print_metrics(metrics: Dict[str, float]) -> None:
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R^2: {metrics['R2']:.4f}")
    print(f"MAPE: {metrics['MAPE'] * 100:.4f}%")
