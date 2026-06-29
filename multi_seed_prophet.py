"""
Multi-seed robustness experiment: re-run Prophet (with a fixed best-known
hyperparameter set) across several random seeds and report Mean +/- Std
performance. Prophet's optimizer is largely deterministic given fixed data
and hyperparameters, so this mainly verifies run-to-run stability.

Usage:
    python -m src.experiments.multi_seed_prophet
"""

import numpy as np

from src import config, metrics, preprocessing
from src.experiments._common import save_multi_seed_results
from src.models import prophet_model

# Fixed, previously-tuned hyperparameters (single combination) used for every
# seed in this robustness check.
FIXED_PARAM_GRID = {
    "changepoint_prior_scale": [0.001],
    "seasonality_prior_scale": [10.0],
    "seasonality_mode": ["multiplicative"],
}


def main():
    print("Loading and preprocessing data (common for all seeds)...")
    dataset = preprocessing.build_prophet_dataset()
    prophet_data = preprocessing.prepare_prophet_frame(dataset)
    train_data, val_data, test_data = preprocessing.train_val_test_split_prophet(prophet_data)

    all_results = []
    for seed in config.MULTI_SEED_LIST:
        print(f"\n{'=' * 60}\nRunning with SEED = {seed}\n{'=' * 60}")
        np.random.seed(seed)

        best_model, best_params, best_val_mape = prophet_model.train_model(
            train_data, val_data, param_grid=FIXED_PARAM_GRID
        )
        print(f"Best params: {best_params}, Validation MAPE: {best_val_mape * 100:.2f}%")

        test_future = test_data.drop(columns=["y"])
        y_pred = prophet_model.predict(best_model, test_future)
        y_test = test_data["y"].values

        result = metrics.compute_regression_metrics(y_test, y_pred)
        metrics.print_metrics(result)

        all_results.append(
            {"seed": seed, **result, "best_params": str(best_params), "best_val_mape": best_val_mape}
        )

    save_multi_seed_results("Prophet", all_results, "prophet_multi_seed")


if __name__ == "__main__":
    main()
