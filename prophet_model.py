"""Prophet model: definition, hyperparameter grid search, training, and prediction."""

import itertools

from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

from src import config


def build_model(params: dict) -> Prophet:
    """Instantiate a Prophet model with the given seasonality/changepoint
    settings and attach the lag/calendar regressors used by this project."""
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
    )
    for regressor in config.PROPHET_REGRESSOR_COLUMNS:
        model.add_regressor(regressor)
    return model


def train_model(train_data, val_data, param_grid: dict = None):
    """Grid-search Prophet hyperparameters using validation MAPE, fitting one
    model per candidate parameter set.

    Returns (best_model, best_params, best_val_mape).
    """
    param_grid = param_grid or config.PROPHET_PARAM_GRID
    all_params = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    best_mape = float("inf")
    best_params = None
    best_model = None

    val_future = val_data[["ds"] + config.PROPHET_REGRESSOR_COLUMNS]
    val_actual = val_data["y"].values

    for params in all_params:
        try:
            model = build_model(params)
            model.fit(train_data)

            val_forecast = model.predict(val_future)
            mape = mean_absolute_percentage_error(val_actual, val_forecast["yhat"].values)

            if mape < best_mape:
                best_mape = mape
                best_params = params
                best_model = model
        except Exception as exc:  # pragma: no cover - mirrors original notebook behavior
            print(f"Error with params {params}: {exc}")
            continue

    return best_model, best_params, best_mape


def predict(model: Prophet, future_data):
    """Generate predictions (returns the 'yhat' column) from a trained
    Prophet model."""
    forecast = model.predict(future_data)
    return forecast["yhat"].values
