"""
Statistical analysis utilities.

Consolidates every statistical computation previously scattered across the
notebooks (multi-seed summary statistics, Welch's t-tests, pairwise
significance matrices, confidence intervals) into one place so that
experiment scripts only need to call these functions on real per-seed
results.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def summary_stats(values: Sequence[float]) -> Tuple[float, float]:
    """Return (mean, population std) of a sequence of metric values."""
    return float(np.mean(values)), float(np.std(values))


def confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Two-sided confidence interval for the mean of `values`, using the
    t-distribution (appropriate for the small samples produced by a handful
    of random seeds).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        mean = float(values.mean()) if n else float("nan")
        return mean, mean

    mean = values.mean()
    sem = stats.sem(values)
    half_width = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(mean - half_width), float(mean + half_width)


def paired_ttest(values_a: Sequence[float], values_b: Sequence[float]):
    """Paired t-test between two equal-length sequences of metric values
    (e.g. the same seeds evaluated with two different models)."""
    return stats.ttest_rel(values_a, values_b)


def welch_ttest(values_a: Sequence[float], values_b: Sequence[float]):
    """Welch's t-test (unequal variance assumed) between two independent
    samples of metric values."""
    return stats.ttest_ind(values_a, values_b, equal_var=False)


def significance_stars(p_value: float) -> str:
    """Format a p-value with conventional significance markers."""
    if p_value is None or np.isnan(p_value):
        return "N/A"
    if p_value < 0.001:
        return f"{p_value:.6f}***"
    if p_value < 0.01:
        return f"{p_value:.6f}**"
    if p_value < 0.05:
        return f"{p_value:.6f}*"
    return f"{p_value:.6f}n.s."


def build_model_performance_table(model_metrics: Dict[str, Dict[str, Sequence[float]]]) -> pd.DataFrame:
    """
    Build a Mean ± Std performance summary table.

    Parameters
    ----------
    model_metrics : mapping of model name -> {metric_name: [values across seeds]}
    """
    rows = []
    for model_name, metrics in model_metrics.items():
        row = {"Model": model_name}
        for metric_name, values in metrics.items():
            mean, std = summary_stats(values)
            row[metric_name] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_ttest_table(
    model_metrics: Dict[str, Dict[str, Sequence[float]]],
    reference_model: str,
    test: str = "welch",
) -> pd.DataFrame:
    """
    Build a table of p-values (with significance stars) comparing every
    other model's per-seed metrics against a reference model.
    """
    test_fn = welch_ttest if test == "welch" else paired_ttest
    reference = model_metrics[reference_model]
    metric_names = list(reference.keys())

    rows = []
    for model_name, metrics in model_metrics.items():
        if model_name == reference_model:
            continue
        row = {"Model": model_name}
        for metric_name in metric_names:
            try:
                _, p_value = test_fn(metrics[metric_name], reference[metric_name])
            except Exception:
                p_value = np.nan
            row[f"{metric_name} (p-value)"] = significance_stars(p_value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_matrix(
    model_metrics: Dict[str, Dict[str, Sequence[float]]],
    metric_name: str,
    test: str = "welch",
) -> pd.DataFrame:
    """Build a full pairwise model x model matrix of significance-annotated
    p-values for a single metric."""
    test_fn = welch_ttest if test == "welch" else paired_ttest
    model_names = list(model_metrics.keys())

    matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=object)
    for model_a in model_names:
        for model_b in model_names:
            if model_a == model_b:
                matrix.loc[model_a, model_b] = "1.000"
                continue
            try:
                _, p_value = test_fn(
                    model_metrics[model_a][metric_name],
                    model_metrics[model_b][metric_name],
                )
                matrix.loc[model_a, model_b] = significance_stars(p_value)
            except Exception:
                matrix.loc[model_a, model_b] = "N/A"
    return matrix.reset_index().rename(columns={"index": "Model"})


def best_model_per_metric(
    model_metrics: Dict[str, Dict[str, Sequence[float]]],
    higher_is_better: Dict[str, bool],
) -> pd.DataFrame:
    """For each metric, find the model with the best mean value."""
    metric_names = next(iter(model_metrics.values())).keys()
    rows = []
    for metric_name in metric_names:
        best_model_name, best_value = None, None
        better = higher_is_better.get(metric_name, False)
        for model_name, metrics in model_metrics.items():
            mean_val = float(np.mean(metrics[metric_name]))
            if best_value is None or (mean_val > best_value if better else mean_val < best_value):
                best_value = mean_val
                best_model_name = model_name
        rows.append({"Metric": metric_name, "Best Model": best_model_name, "Best Value": best_value})
    return pd.DataFrame(rows)


def peak_magnitude_error(y_true, y_pred):
    """Absolute and percentage error between the true and predicted peak
    (maximum) demand."""
    true_peak = float(np.max(y_true))
    pred_peak = float(np.max(y_pred))
    abs_error = abs(true_peak - pred_peak)
    pct_error = (abs_error / true_peak) * 100
    return abs_error, pct_error


def max_absolute_error(y_true, y_pred) -> float:
    return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
