"""
Shared plotting utilities.

Every figure produced anywhere in this project should be created through
`new_figure()` / saved through `save_figure()` so that every plot
consistently lands in figures/ as both PNG and PDF (replacing the old
docx-embedded-image workflow).
"""

import matplotlib.pyplot as plt

from src import config


def save_figure(fig, name: str, dpi: int = 300) -> None:
    """Save a matplotlib Figure as figures/<name>.png and figures/<name>.pdf."""
    config.ensure_output_dirs()
    png_path = config.FIGURES_DIR / f"{name}.png"
    pdf_path = config.FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {png_path} and {pdf_path}")


def plot_first_n_samples(y_actual, y_pred, n: int = 100, scale: float = 1e6):
    """Line plot comparing actual vs. predicted values for the first n test
    samples."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_actual[:n] / scale, label="Actual", color="blue")
    ax.plot(y_pred[:n] / scale, label="Predicted", color="red", linestyle="--")
    ax.set_title(f"Comparison of Actual and Predicted Values (First {n} Samples)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Power Consumption (MW)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_full_test_set(test_dates, y_actual, y_pred, scale: float = 1e6):
    """Line plot comparing actual vs. predicted values across the entire
    test set."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates, y_actual / scale, label="Actual", color="blue")
    ax.plot(test_dates, y_pred / scale, label="Predicted", color="orange", linestyle="--")
    ax.set_title("Comparison of Actual and Predicted Values (Entire Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Power Consumption (MW)")
    ax.legend()
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    return fig


def plot_residual_scatter(y_pred, residuals):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, s=10, alpha=0.6, color="steelblue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted Power Consumption")
    ax.set_ylabel("Residual Error")
    ax.set_title("Residual Plot")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_residual_histogram(residuals, bins: int = 35):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=bins, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Residual Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Residuals")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_scenario_bar(scenarios, values, ylabel: str, title: str, highlight: str = "Peak"):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tomato" if s == highlight else "steelblue" for s in scenarios]
    ax.bar(scenarios, values, color=colors, width=0.40)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scenario")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    return fig


def plot_complexity_summary(df):
    """2x2 grid of bar charts summarizing trainable parameters, model size,
    training time, and inference time across models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        ("Trainable Parameters", "Number of Parameters", "{:,}"),
        ("Model Size (MB)", "Size (MB)", "{:.3f}"),
        ("Training Time (s)", "Time (seconds)", "{:.3f}"),
        ("Inference Time (ms/sample)", "Time (ms/sample)", "{:.4f}"),
    ]

    for ax, (column, ylabel, fmt) in zip(axes.flat, panels):
        ax.bar(df["Model"], df[column])
        ax.set_title(column)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(df[column]):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom")

    fig.tight_layout()
    return fig
