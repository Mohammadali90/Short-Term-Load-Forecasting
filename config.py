"""
Central configuration for the project: filesystem paths, reproducibility
constants, feature definitions, and per-model hyperparameters.

All paths are resolved relative to the repository root so that scripts can be
run from any working directory without hard-coded absolute paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# Raw data files expected under data/raw/ (see README for dataset preparation)
MAIN_SUBSTATION_FILE = RAW_DATA_DIR / "Dataset.csv"
QURAN_GATE_FILE = RAW_DATA_DIR / "Quran Gate.xlsx"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
MULTI_SEED_LIST = [42, 52, 62, 72, 82]

# ---------------------------------------------------------------------------
# Dataset row ranges (the raw CSV stacks multiple substations sequentially)
# ---------------------------------------------------------------------------
MAIN_SUBSTATION_SLICE = (0, 12703)
AMIRKABIR_SUBSTATION_SLICE = (12703, 25406)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
LAG_HOURS = [1, 2, 3, 24, 168]
PROPHET_LAG_HOURS = [1, 24, 168]
WEEKEND_DAYS = [4, 5]  # Thursday/Friday, after the Sat=0 day-of-week shift
SEQUENCE_LENGTH = 24  # hours of history used to predict the next hour

FEATURE_COLUMNS = [
    "Value",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_168",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

XGBOOST_FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_168",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

PROPHET_REGRESSOR_COLUMNS = [
    "lag_1",
    "lag_24",
    "lag_168",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

# ---------------------------------------------------------------------------
# Train / validation / test split ratios
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
VAL_SIZE_OF_REMAINDER = 0.25  # 0.25 of the 80% train+val -> 60/20/20 overall

# ---------------------------------------------------------------------------
# Training defaults shared by all PyTorch sequence models
# ---------------------------------------------------------------------------
TRAINING_DEFAULTS = {
    "num_epochs": 200,
    "patience": 15,
    "batch_size": 32,
    "scheduler_factor": 0.1,
    "scheduler_patience": 10,
}

# ---------------------------------------------------------------------------
# Per-model hyperparameters (as tuned in the original notebooks)
# ---------------------------------------------------------------------------
XGBOOST_PARAM_GRID = {
    "n_estimators": [100],
    "learning_rate": [0.1],
    "max_depth": [5],
}

PROPHET_PARAM_GRID = {
    "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    "seasonality_prior_scale": [0.1, 1.0, 10.0],
    "seasonality_mode": ["additive", "multiplicative"],
}

LSTM_HYPERPARAM_SEARCH = {
    "hidden_sizes": [32, 50, 64],
    "num_layers": [1, 2],
    "learning_rates": [0.01, 0.001, 0.0005],
}

TE_BILSTM_PARAMS = {
    "hidden_size": 19,
    "num_layers": 3,
    "learning_rate": 0.0005,
}

CNN_LSTM_PARAMS = {
    "cnn_channels": 32,
    "hidden_size": 32,
    "num_layers": 2,
    "learning_rate": 0.0005,
}

TCN_PARAMS = {
    "num_channels": [32, 32, 32],
    "kernel_size": 3,
    "learning_rate": 0.0005,
}

TRANSFORMER_PARAMS = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "learning_rate": 0.0005,
}


def ensure_output_dirs() -> None:
    """Create the results/, figures/, and checkpoints/ directories if they do
    not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
