# In the Name of Allah

"""
Power Consumption Forecasting Package
=====================================

Official implementation for hourly power consumption prediction models.
"""

__version__ = "1.0.0"
__author__ = "Mohammad Ali Nematollahi"
__description__ = "Advanced Time Series Forecasting for Power Consumption using TE-BiLSTM and other models"

# Core utilities
from ._common import (
    load_and_preprocess_data,
    add_features,
    create_sequences
)

from .config import (
    DEVICE,
    SEEDS,
    TIME_STEPS,
    HIDDEN_SIZE_BILSTM,
    NUM_LAYERS_BILSTM
)

from .datasets import (
    load_quran_gate_data,
    prepare_data_for_model
)

from .metrics import calculate_metrics
from .training import train_model
from .plotting import create_comparison_plots

# Optional: expose main runners
# from .run_te_bilstm import main as run_te_bilstm

__all__ = [
    # Core
    "load_and_preprocess_data",
    "add_features",
    "create_sequences",
    "prepare_data_for_model",
    
    # Config
    "DEVICE",
    "SEEDS",
    "TIME_STEPS",
    
    # Training & Evaluation
    "train_model",
    "calculate_metrics",
    "create_comparison_plots",
]