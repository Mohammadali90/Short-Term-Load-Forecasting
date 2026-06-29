"""
Baseline experiment: forecast main-substation power consumption one hour
ahead with TE-BiLSTM.

Usage:
    python -m src.experiments.run_te_bilstm
"""

from src.experiments._common import run_sequence_baseline
from src.models import te_bilstm_model


def main():
    run_sequence_baseline(te_bilstm_model, "TE-BiLSTM", "te_bilstm")


if __name__ == "__main__":
    main()
