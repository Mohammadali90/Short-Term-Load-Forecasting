"""LSTM model: architecture, hyperparameter search, training, and prediction."""

import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader

from src import config, training
from src.datasets import to_tensor_dataset


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def build_model(input_size, hidden_size, num_layers, output_size=1) -> LSTMModel:
    return LSTMModel(input_size, hidden_size, num_layers, output_size)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    scaler_target,
    input_size: int,
    seed: int = config.DEFAULT_SEED,
    search_space: dict = None,
):
    """
    Grid-search over (hidden_size, num_layers, learning_rate), training a
    fresh LSTM for each combination and keeping the one with the lowest test
    MAPE (matching the original notebook's model-selection criterion).

    X_train/X_val/X_test and y_train/y_val/y_test are the *scaled* sequence
    arrays produced by src.preprocessing; scaler_target is used to invert
    predictions back to physical units for MAPE comparison.

    Returns (best_model, best_config).
    """
    torch.manual_seed(seed)
    search_space = search_space or config.LSTM_HYPERPARAM_SEARCH
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=32, shuffle=False)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_actual = scaler_target.inverse_transform(y_test)

    hyperparams = list(
        itertools.product(
            search_space["hidden_sizes"],
            search_space["num_layers"],
            search_space["learning_rates"],
        )
    )

    best_mape = float("inf")
    best_config = None
    best_model_state = None

    for hidden_size, num_layers, lr in hyperparams:
        print(f"Training with hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}")
        model = build_model(input_size, hidden_size, num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

        checkpoint_path = config.CHECKPOINTS_DIR / "lstm_search_candidate.pth"
        model = training.train_with_early_stopping(
            model, train_loader, val_loader, optimizer, scheduler, checkpoint_path=checkpoint_path
        )

        y_pred_scaled = training.predict(model, X_test_tensor, device=device)
        y_pred = scaler_target.inverse_transform(y_pred_scaled)
        mape = mean_absolute_percentage_error(y_test_actual, y_pred)
        print(f"MAPE for config (h={hidden_size}, l={num_layers}, lr={lr}): {mape * 100:.2f}%")

        if mape < best_mape:
            best_mape = mape
            best_config = {"hidden_size": hidden_size, "num_layers": num_layers, "lr": lr}
            best_model_state = model.state_dict()

    best_model = build_model(input_size, best_config["hidden_size"], best_config["num_layers"]).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    return best_model, best_config


def predict(model: LSTMModel, X_test_tensor):
    return training.predict(model, X_test_tensor)
