"""XGBoost model: definition, hyperparameter search, training, and prediction."""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from src import config


def build_model(seed: int = config.DEFAULT_SEED) -> xgb.XGBRegressor:
    """Instantiate the base XGBoost regressor."""
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=seed)


def train_model(
    X_train,
    y_train,
    seed: int = config.DEFAULT_SEED,
    param_grid: dict = None,
):
    """Grid-search and fit an XGBoost regressor on the training set.

    Returns the fitted GridSearchCV object (`.best_estimator_` is the trained
    model, `.best_params_` the selected hyperparameters).
    """
    param_grid = param_grid or config.XGBOOST_PARAM_GRID
    model = build_model(seed)
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error", verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def predict(model, X_test):
    """Generate predictions from a trained XGBoost model."""
    return model.predict(X_test)
