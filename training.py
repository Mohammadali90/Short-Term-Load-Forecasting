"""
Generic PyTorch training loop with early stopping, shared by every sequence
model (LSTM, TE-BiLSTM, CNN-LSTM, TCN, Transformer). Each model module is
responsible for building its own architecture, optimizer, and scheduler;
this function only implements the (identical) train/validate/early-stop
mechanics that were previously copy-pasted into every notebook.
"""

import torch
import torch.nn as nn

from src import config


def train_with_early_stopping(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    checkpoint_path=None,
    num_epochs: int = config.TRAINING_DEFAULTS["num_epochs"],
    patience: int = config.TRAINING_DEFAULTS["patience"],
    device: str = None,
    verbose_every: int = 10,
):
    """Train `model` with MSE loss and early stopping on validation loss.

    Returns the trained model with its best-validation-loss weights loaded.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()

    checkpoint_path = checkpoint_path or (config.CHECKPOINTS_DIR / "temp_best_model.pth")
    config.ensure_output_dirs()

    best_val_loss = float("inf")
    early_stop_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(val_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best at epoch {best_epoch})")
                break

        if (epoch + 1) % verbose_every == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    model.load_state_dict(torch.load(checkpoint_path))
    model.best_epoch_ = best_epoch
    model.best_val_loss_ = best_val_loss
    return model


def predict(model: nn.Module, X_test_tensor, device: str = None):
    """Run inference on a trained PyTorch model and return a numpy array."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor.to(device)).cpu().numpy()
    return y_pred_scaled
