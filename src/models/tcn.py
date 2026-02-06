import numpy as np
import optuna
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Dilated causal conv block with residual connection. Left-padded to stay causal."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 1x1 conv if channels don't match for residual
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

        self.padding = padding

    def forward(self, x):
        """Two dilated convs with residual connection, trimmed to stay causal."""
        out = self.conv1(x)
        # trim right side to stay causal
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class RouteDelayTCN(nn.Module):
    """
    Stacked dilated causal convolutions.
    Dilations [1, 2, 4] with kernel 3 gives a receptive field of 29 timesteps.
    """

    def __init__(self, input_size, num_channels=None, kernel_size=3, dropout=0.2):
        super().__init__()

        if num_channels is None:
            num_channels = [32, 64, 64]

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i

            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))

        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # RF = 1 + 2*(k-1)*sum(dilations), should be >= sequence length
        rf = 1
        for i in range(num_levels):
            rf += 2 * (kernel_size - 1) * (2 ** i)
        self.receptive_field = rf

    def forward(self, x):
        # Conv1d wants (batch, channels, seq_len)
        x = x.transpose(1, 2)

        out = self.tcn(x)
        out = self.global_pool(out).squeeze(-1)
        out = self.fc(out)

        return out.squeeze(-1)


class TCNTrainer:

    def __init__(self, model, learning_rate=0.001, device=None):
        if device is None:
            from src.config import get_device
            device = get_device()
        self.device = device
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()

        return total_loss / n_batches

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(x_batch)
                loss = self.criterion(predictions, y_batch)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def fit(self, train_loader, val_loader, epochs=50, early_stopping_patience=10,
            verbose=True, trial=None):
        """Trains with early stopping, restores best weights when done.
        Optionally accepts an Optuna trial for epoch-level pruning.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return self.history

    def predict(self, data_loader):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                preds = self.model(x_batch)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def save(self, path):
        # persists weights, optimizer state, history, and receptive field
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "receptive_field": self.model.receptive_field
        }, path)

    def load(self, path):
        # loads weights and optimizer from disk
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]


if __name__ == "__main__":
    input_size = 22
    model = RouteDelayTCN(input_size=input_size, num_channels=[32, 64, 64])
    print(f"Receptive field: {model.receptive_field}")

    x = torch.randn(16, 28, input_size)
    output = model(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
