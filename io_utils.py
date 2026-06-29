"""
Output-handling utilities.

Replaces the manuscript-era Microsoft Word (.docx) export logic from the
original notebooks. All tabular results are now saved as CSV and XLSX under
results/, and all figures are saved as PNG and PDF under figures/ (see
src/plotting.py for figure saving).
"""

from pathlib import Path

import pandas as pd

from src import config


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    """Save a DataFrame as both CSV and XLSX under results/<name>.{csv,xlsx}."""
    config.ensure_output_dirs()
    csv_path = config.RESULTS_DIR / f"{name}.csv"
    xlsx_path = config.RESULTS_DIR / f"{name}.xlsx"
    df.to_csv(csv_path, index=index)
    df.to_excel(xlsx_path, index=index)
    print(f"Saved table to {csv_path} and {xlsx_path}")


def results_path(name: str, suffix: str = ".csv") -> Path:
    """Resolve a path under results/ for ad-hoc (non-table) outputs."""
    config.ensure_output_dirs()
    return config.RESULTS_DIR / f"{name}{suffix}"
