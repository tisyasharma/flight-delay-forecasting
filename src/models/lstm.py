import numpy as np
import optuna
import torch
import torch.nn as nn


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention over the temporal dimension."""

    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.head_dim),
                nn.Tanh(),
                nn.Linear(self.head_dim, 1)
            ) for _ in range(num_heads)
        ])

        self.output_projection = nn.Linear(num_heads * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_output):
        """Computes attention-weighted context vector from LSTM hidden states."""
        head_contexts = []
        all_weights = []

        for head in self.heads:
            attn_scores = head(lstm_output)
            attn_weights = torch.softmax(attn_scores, dim=1)
            all_weights.append(attn_weights.squeeze(-1))

            head_context = torch.sum(attn_weights * lstm_output, dim=1)
            head_contexts.append(head_context)

        combined = torch.cat(head_contexts, dim=-1)
        context = self.output_projection(combined)
        context = self.dropout(context)

        return context, torch.stack(all_weights, dim=1)

    def get_attention_weights(self, lstm_output):
        """Returns the attention weight matrix without computing gradients."""
        with torch.no_grad():
            _, weights = self.forward(lstm_output)
        return weights


class RouteDelayLSTM(nn.Module):
    """
    Takes 28-day sequence, attention collapses to context vector, FC head outputs forecast.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3,
                 num_attention_heads=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.use_multi_head = num_attention_heads > 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        if self.use_multi_head:
            self.attention = MultiHeadTemporalAttention(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                dropout=0.1
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        if self.use_multi_head:
            context, _ = self.attention(lstm_out)
        else:
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)

        output = self.fc(context)
        return output.squeeze(-1)

    def get_attention_weights(self, x):
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            if self.use_multi_head:
                return self.attention.get_attention_weights(lstm_out)
            else:
                attn_weights = self.attention(lstm_out)
                attn_weights = torch.softmax(attn_weights, dim=1)
                return attn_weights.squeeze(-1)


class LSTMTrainer:

    def __init__(self, model, learning_rate=0.001, device=None):
        if device is None:
            from src.config import get_device
            device = get_device()
        self.device = device
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
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

            self.scheduler.step(val_loss)

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
        # checkpoint includes model weights, optimizer, and loss history
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }, path)

    def load(self, path):
        # restores model and optimizer from checkpoint file
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]


if __name__ == "__main__":
    input_size = 22
    batch_size = 16
    seq_length = 28
    x = torch.randn(batch_size, seq_length, input_size)

    model_single = RouteDelayLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    output = model_single(x)
    attn = model_single.get_attention_weights(x)
    print(f"Single-head: input {x.shape}, output {output.shape}, attn {attn.shape}")

    model_multi = RouteDelayLSTM(
        input_size=input_size, hidden_size=64, num_layers=2, num_attention_heads=4
    )
    output = model_multi(x)
    attn = model_multi.get_attention_weights(x)
    print(f"Multi-head:  input {x.shape}, output {output.shape}, attn {attn.shape}")
