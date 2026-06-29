"""
Multi-seed robustness experiment: re-train TE-BiLSTM across several random
seeds (fixed data split, fixed hyperparameters) and report Mean +/- Std
performance.

Usage:
    python -m src.experiments.multi_seed_te_bilstm
"""

from src.experiments._common import run_multi_seed_sequence_experiment
from src.models import te_bilstm_model


def main():
    run_multi_seed_sequence_experiment(te_bilstm_model, "TE-BiLSTM", "te_bilstm_multi_seed")


if __name__ == "__main__":
    main()
