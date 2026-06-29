"""
Multi-seed robustness experiment: re-train CNN-LSTM across several random
seeds (fixed data split, fixed hyperparameters) and report Mean +/- Std
performance.

Usage:
    python -m src.experiments.multi_seed_cnn_lstm
"""

from src.experiments._common import run_multi_seed_sequence_experiment
from src.models import cnn_lstm_model


def main():
    run_multi_seed_sequence_experiment(cnn_lstm_model, "CNN-LSTM", "cnn_lstm_multi_seed")


if __name__ == "__main__":
    main()
