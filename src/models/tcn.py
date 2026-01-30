"""
TCN (Temporal Convolutional Network) for flight delay prediction.
Uses dilated causal convolutions instead of recurrence, which makes it
fully parallelizable. The receptive field grows exponentially with depth.
"""

import numpy as np
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    One block: dilated causal conv -> batch norm -> residual connection.
    Left-padded and right-trimmed so the output at time t only sees inputs <= t.
    Higher dilation = wider gaps between kernel positions = sees further back.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        # left-padding to maintain sequence length
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

        # 1x1 conv to match channel dimensions for the residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

        self.padding = padding

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        """
        out = self.conv1(x)
        # trim the right side to remove "future" information from right-padding
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


class FlightDelayTCN(nn.Module):
    """
    Temporal Convolutional Network for delay prediction.

    Stacks temporal blocks with exponentially increasing dilation (1, 2, 4, ...),
    so the receptive field grows exponentially with depth. With 3 blocks and
    kernel_size=3, the receptive field is 29 days -- just enough for 28-day sequences.
    Global average pooling collapses the sequence, then a small FC head predicts.
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

        self.receptive_field = self._calculate_receptive_field(
            num_levels, kernel_size
        )

    def _calculate_receptive_field(self, num_levels, kernel_size):
        """
        For 3 levels with kernel=3: RF = 1 + 2*(3-1)*(1+2+4) = 29 days.
        If RF < sequence_length, the model can't use the full input history.
        """
        rf = 1
        for i in range(num_levels):
            dilation = 2 ** i
            rf += 2 * (kernel_size - 1) * dilation
        return rf

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features) - same input format as the LSTM
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        out = self.tcn(x)
        out = self.global_pool(out).squeeze(-1)
        out = self.fc(out)

        return out.squeeze(-1)

    def get_intermediate_outputs(self, x):
        """Get outputs from each temporal block for debugging."""
        x = x.transpose(1, 2)
        outputs = []

        current = x
        for layer in self.tcn:
            current = layer(current)
            outputs.append(current)

        return outputs


class TCNTrainer:
    """
    Training wrapper for TCN. Same idea as LSTMTrainer but uses cosine
    annealing with warm restarts for the LR schedule.
    """

    def __init__(self, model, learning_rate=0.001, device=None):
        self.device = device or torch.device("mps")
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
        """Train for one epoch."""
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
        """Evaluate on validation set."""
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
            verbose=True):
        """Training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
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
        """Generate predictions."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                preds = self.model(x_batch)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "receptive_field": self.model.receptive_field
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]


if __name__ == "__main__":
    input_size = 30
    model = FlightDelayTCN(input_size=input_size, num_channels=[32, 64, 64])

    print(f"Receptive field: {model.receptive_field} time steps")

    batch_size = 16
    seq_length = 28
    x = torch.randn(batch_size, seq_length, input_size)

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    intermediates = model.get_intermediate_outputs(x)
    for i, out in enumerate(intermediates):
        print(f"Block {i + 1} output shape: {out.shape}")
