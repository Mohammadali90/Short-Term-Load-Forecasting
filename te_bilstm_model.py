"""
TE-BiLSTM model (bidirectional LSTM): architecture, compilation, training,
and prediction.

This is the project's primary forecasting model, reused unchanged across the
main substation, the Amirkabir and Quran Gate substations (see
src/experiments/additional_substations.py), the multi-seed robustness run,
and the peak-demand / robustness experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src import config, training


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def build_model(input_size, output_size=1, params: dict = None) -> BiLSTMModel:
    params = params or config.TE_BILSTM_PARAMS
    return BiLSTMModel(input_size, params["hidden_size"], params["num_layers"], output_size)


def train_model(
    train_loader,
    val_loader,
    input_size: int,
    output_size: int = 1,
    params: dict = None,
    checkpoint_path=None,
    seed: int = config.DEFAULT_SEED,
):
    """Build, compile, and train a TE-BiLSTM model with early stopping.

    Returns the trained model (with `.best_epoch_` / `.best_val_loss_`
    attributes set by the shared training loop).
    """
    params = params or config.TE_BILSTM_PARAMS
    torch.manual_seed(seed)

    model = build_model(input_size, output_size, params)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    checkpoint_path = checkpoint_path or (config.CHECKPOINTS_DIR / "best_te_bilstm_model.pth")
    model = training.train_with_early_stopping(
        model, train_loader, val_loader, optimizer, scheduler, checkpoint_path=checkpoint_path
    )
    return model


def predict(model: BiLSTMModel, X_test_tensor):
    return training.predict(model, X_test_tensor)
