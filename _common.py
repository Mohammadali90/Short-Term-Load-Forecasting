"""
Shared orchestration helper for the standard single-run sequence-model
baseline experiments (TE-BiLSTM, CNN-LSTM, TCN, Transformer). These four
experiments differ only in which model module they train, so the common
load -> scale -> sequence -> split -> train -> evaluate -> plot -> save
pipeline lives here once instead of being copy-pasted four times.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config, io_utils, metrics, plotting, preprocessing
from src.datasets import to_tensor_dataset


def run_sequence_baseline(
    model_module,
    display_name: str,
    result_prefix: str,
    row_slice=config.MAIN_SUBSTATION_SLICE,
    clean_invalid: bool = False,
    csv_path=config.MAIN_SUBSTATION_FILE,
    seed: int = config.DEFAULT_SEED,
    dataset: pd.DataFrame = None,
):
    """Run the standard single-seed baseline pipeline for any sequence model
    module exposing `train_model(train_loader, val_loader, input_size,
    seed=...)` and `predict(model, X_test_tensor)`.

    If `dataset` is provided (an already feature-engineered DataFrame), it
    is used directly instead of loading from `csv_path`/`row_slice` -- this
    is how the Quran Gate substation (which is loaded from a different file
    format) reuses this same pipeline.

    Returns a dict with the trained model, metrics, and test-set arrays, in
    case a caller (e.g. an additional-substations or peak-demand experiment)
    needs to continue working with them.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset is None:
        print(f"[{display_name}] Loading and preprocessing data...")
        dataset = preprocessing.build_sequence_dataset(
            csv_path=csv_path, row_slice=row_slice, clean_invalid=clean_invalid
        )
    features_scaled, target_scaled, _, scaler_target = preprocessing.scale_features_and_target(dataset)
    X_seq, y_seq = preprocessing.create_sequences(features_scaled, target_scaled)
    X_train, X_val, X_test, y_train, y_val, y_test, X_temp = preprocessing.train_val_test_split_sequences(
        X_seq, y_seq
    )
    test_dates = preprocessing.get_test_set_timestamps(
        dataset, len(X_temp), config.SEQUENCE_LENGTH, len(X_test)
    )

    input_size = features_scaled.shape[1]
    train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=32, shuffle=False)
    X_test_tensor = torch.FloatTensor(X_test)

    print(f"[{display_name}] Training...")
    model = model_module.train_model(train_loader, val_loader, input_size, seed=seed)

    y_pred = scaler_target.inverse_transform(model_module.predict(model, X_test_tensor))
    y_test_actual = scaler_target.inverse_transform(y_test)

    results = metrics.compute_regression_metrics(y_test_actual, y_pred)
    metrics.print_metrics(results)

    fig1 = plotting.plot_first_n_samples(y_test_actual, y_pred)
    plotting.save_figure(fig1, f"{result_prefix}_first_100_samples")

    fig2 = plotting.plot_full_test_set(test_dates, y_test_actual, y_pred)
    plotting.save_figure(fig2, f"{result_prefix}_full_test_set")

    random_idx = np.random.randint(0, len(X_test) - 1)
    actual_next_hour = float(y_test_actual[random_idx + 1])
    predicted_next_hour = float(
        scaler_target.inverse_transform(
            model_module.predict(model, X_test_tensor[random_idx : random_idx + 1])
        )[0][0]
    )
    print(f"Actual next-hour value: {actual_next_hour:.4f}")
    print(f"Predicted next-hour value: {predicted_next_hour:.4f}")

    summary = pd.DataFrame(
        [
            {
                "Model": display_name,
                **results,
                "Best Epoch": getattr(model, "best_epoch_", None),
                "Actual Next Hour": actual_next_hour,
                "Predicted Next Hour": predicted_next_hour,
            }
        ]
    )
    io_utils.save_table(summary, f"{result_prefix}_results")

    return {
        "model": model,
        "metrics": results,
        "y_pred": y_pred,
        "y_test_actual": y_test_actual,
        "test_dates": test_dates,
        "scaler_target": scaler_target,
        "dataset": dataset,
    }


def save_multi_seed_results(model_name: str, all_results: list, result_prefix: str) -> pd.DataFrame:
    """
    Aggregate a list of per-seed result dicts (each must contain at least
    'seed', 'MAE', 'RMSE', 'R2', 'MAPE') into:
      - results/<prefix>_per_seed.csv|.xlsx  (one row per seed)
      - results/<prefix>_summary.csv|.xlsx   (mean +/- std across seeds)

    Returns the per-seed DataFrame, which downstream experiments (e.g.
    statistical_analysis.py) can reload to run real significance tests
    instead of hard-coding numbers.
    """
    per_seed_df = pd.DataFrame(all_results)
    io_utils.save_table(per_seed_df, f"{result_prefix}_per_seed")

    metric_cols = [c for c in ["MAE", "RMSE", "R2", "MAPE"] if c in per_seed_df.columns]
    summary_row = {"Model": model_name, "Number of Seeds": len(all_results)}
    for col in metric_cols:
        summary_row[f"{col} Mean"] = per_seed_df[col].mean()
        summary_row[f"{col} Std"] = per_seed_df[col].std(ddof=0)
    summary_df = pd.DataFrame([summary_row])
    io_utils.save_table(summary_df, f"{result_prefix}_summary")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY STATISTICS ACROSS ALL SEEDS ({model_name})")
    print(f"{'=' * 60}")
    for col in metric_cols:
        print(f"{col} - Mean: {summary_row[f'{col} Mean']:.4f}, Std: {summary_row[f'{col} Std']:.4f}")

    return per_seed_df


def run_multi_seed_sequence_experiment(
    model_module,
    display_name: str,
    result_prefix: str,
    seeds=config.MULTI_SEED_LIST,
    row_slice=config.MAIN_SUBSTATION_SLICE,
    clean_invalid: bool = False,
    csv_path=config.MAIN_SUBSTATION_FILE,
) -> pd.DataFrame:
    """
    Shared multi-seed robustness runner for any sequence model module.

    The data is loaded and split once (a fixed chronological split, matching
    the original notebooks), and only the model's weight initialization /
    training stochasticity varies across seeds.
    """
    print(f"[{display_name}] Loading and preprocessing data (common for all seeds)...")
    dataset = preprocessing.build_sequence_dataset(
        csv_path=csv_path, row_slice=row_slice, clean_invalid=clean_invalid
    )
    features_scaled, target_scaled, _, scaler_target = preprocessing.scale_features_and_target(dataset)
    X_seq, y_seq = preprocessing.create_sequences(features_scaled, target_scaled)
    X_train, X_val, X_test, y_train, y_val, y_test, X_temp = preprocessing.train_val_test_split_sequences(
        X_seq, y_seq
    )
    input_size = features_scaled.shape[1]
    y_test_actual = scaler_target.inverse_transform(y_test)
    X_test_tensor = torch.FloatTensor(X_test)

    all_results = []
    for seed in seeds:
        print(f"\n{'=' * 60}\nRunning with SEED = {seed}\n{'=' * 60}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=32, shuffle=False)

        checkpoint_path = config.CHECKPOINTS_DIR / f"{result_prefix}_seed_{seed}.pth"
        model = model_module.train_model(
            train_loader, val_loader, input_size, seed=seed, checkpoint_path=checkpoint_path
        )

        y_pred = scaler_target.inverse_transform(model_module.predict(model, X_test_tensor))
        result = metrics.compute_regression_metrics(y_test_actual, y_pred)
        metrics.print_metrics(result)

        all_results.append({"seed": seed, **result, "best_epoch": getattr(model, "best_epoch_", None)})

    return save_multi_seed_results(display_name, all_results, result_prefix)
