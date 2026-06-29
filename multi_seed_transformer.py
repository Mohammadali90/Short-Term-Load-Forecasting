"""
Multi-seed robustness experiment: re-train the Transformer across several
random seeds (fixed data split, fixed hyperparameters) and report Mean +/-
Std performance.

Usage:
    python -m src.experiments.multi_seed_transformer
"""

from src.experiments._common import run_multi_seed_sequence_experiment
from src.models import transformer_model


def main():
    run_multi_seed_sequence_experiment(transformer_model, "Transformer", "transformer_multi_seed")


if __name__ == "__main__":
    main()
