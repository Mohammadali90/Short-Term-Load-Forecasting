"""
Multi-seed robustness experiment: re-run the XGBoost baseline across several
random seeds and report Mean +/- Std performance.

Usage:
    python -m src.experiments.multi_seed_xgboost
"""

from src import config, metrics, preprocessing
from src.experiments._common import save_multi_seed_results
from src.models import xgboost_model


def main():
    print("Loading and preprocessing data (common for all seeds)...")
    dataset = preprocessing.build_xgboost_dataset()
    X, y = preprocessing.get_feature_target_arrays(dataset)

    all_results = []
    for seed in config.MULTI_SEED_LIST:
        print(f"\n{'=' * 60}\nRunning with SEED = {seed}\n{'=' * 60}")

        X_train, X_test, y_train, y_test = preprocessing.train_test_split_tabular(X, y)
        grid_search = xgboost_model.train_model(X_train, y_train, seed=seed)
        best_model = grid_search.best_estimator_
        print(f"Best Parameters for seed {seed}: {grid_search.best_params_}")

        y_pred = xgboost_model.predict(best_model, X_test)
        result = metrics.compute_regression_metrics(y_test, y_pred)
        metrics.print_metrics(result)

        all_results.append({"seed": seed, **result, "best_params": str(grid_search.best_params_)})

    save_multi_seed_results("XGBoost", all_results, "xgboost_multi_seed")


if __name__ == "__main__":
    main()
