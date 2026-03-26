"""
GNNHAR1L model for volatility forecasting with spillover effects.

Architecture (Paper 2, Equation 8):
    HAR branch:  y_har = V @ beta + alpha        (linear, self-information)
    GNN branch:  H = ReLU(W @ V @ Theta)          (nonlinear, spillover)
                 y_gnn = H @ gamma
    Prediction:  RV_hat = y_har + y_gnn

Trained with QLIKE loss by default (Paper 2, Equation 12).
"""

import numpy as np
import torch
import torch.nn as nn

from gnn_vol.config import (
    GNN_HIDDEN_DIM,
    GNN_LEARNING_RATE,
    GNN_BATCH_SIZE,
    GNN_MAX_EPOCHS,
    GNN_EARLY_STOP_PATIENCE,
    GNN_PRED_FLOOR,
    GNN_N_ENSEMBLE,
)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

def qlike_loss(y_pred, y_true):
    """
    Quasi-likelihood loss (Paper 2, Equation 12).
    Asymmetric — penalises under-prediction more than over-prediction.
    """
    y_pred = torch.clamp(y_pred, min=GNN_PRED_FLOOR)
    ratio = y_true / y_pred
    return torch.mean(ratio - torch.log(ratio) - 1)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class GNNHAR(nn.Module):
    """
    1-layer GNN-enhanced HAR model.

    Forward pass:
        1. GNN branch: aggregate neighbour features through W,
           pass through learned weights and ReLU
        2. HAR branch: linear projection of own past volatility
        3. Sum both branches for final prediction
    """

    def __init__(self, n_stocks, n_features=3, hidden_dim=GNN_HIDDEN_DIM):
        super().__init__()

        self.n_stocks = n_stocks

        # HAR branch: V @ beta + alpha (per-stock intercept)
        self.beta = nn.Parameter(torch.randn(n_features, 1) * 0.01)
        self.alpha = nn.Parameter(torch.zeros(n_stocks))

        # GNN branch: ReLU(W @ V @ Theta) @ gamma
        self.theta = nn.Parameter(torch.randn(n_features, hidden_dim) * 0.01)
        self.gamma = nn.Parameter(torch.randn(hidden_dim, 1) * 0.01)

    def forward(self, V, W):
        """
        Args:
            V: (batch, N, 3) — HAR features [daily, weekly, monthly]
            W: (N, N) — normalised adjacency matrix (no self-connections)

        Returns:
            (batch, N) — predicted sqrt(RV) for each stock
        """
        # HAR branch: linear, each stock uses its own features
        y_har = (V @ self.beta).squeeze(-1) + self.alpha  # (batch, N)

        # GNN branch: aggregate neighbour info, apply nonlinearity
        neighbour_features = W @ V         # (batch, N, 3)
        # (batch, N, hidden_dim)
        H = torch.relu(neighbour_features @ self.theta)
        y_gnn = (H @ self.gamma).squeeze(-1)              # (batch, N)

        return y_har + y_gnn


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def _make_batches(V, targets, batch_size):
    """Split data into mini-batches."""
    n = V.shape[0]
    indices = torch.randperm(n)
    batches = []
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        batches.append((V[idx], targets[idx]))
    return batches


def train_single_model(V_train, y_train, V_val, y_val, W, seed=0):
    """
    Train one GNNHAR model with early stopping using QLIKE loss.

    Args:
        V_train:  (T_train, N, 3) numpy array of HAR features
        y_train:  (T_train, N) numpy array of targets
        V_val:    (T_val, N, 3) numpy array of validation features
        y_val:    (T_val, N) numpy array of validation targets
        W:        (N, N) numpy array — normalised adjacency matrix
        seed:     random seed for reproducibility

    Returns:
        Trained GNNHAR model (best weights from early stopping).
    """
    torch.manual_seed(seed)
    n_stocks = V_train.shape[1]

    # Convert to tensors
    V_train_t = torch.tensor(V_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    V_val_t = torch.tensor(V_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    W_t = torch.tensor(W, dtype=torch.float32)

    # Remove NaN rows from training and validation
    train_valid = ~torch.isnan(y_train_t).any(
        dim=1) & ~torch.isnan(V_train_t).any(dim=2).any(dim=1)
    V_train_t = V_train_t[train_valid]
    y_train_t = y_train_t[train_valid]

    val_valid = ~torch.isnan(y_val_t).any(
        dim=1) & ~torch.isnan(V_val_t).any(dim=2).any(dim=1)
    V_val_t = V_val_t[val_valid]
    y_val_t = y_val_t[val_valid]

    # Replace any remaining inf values
    V_train_t = torch.nan_to_num(V_train_t, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_t = torch.nan_to_num(y_train_t, nan=0.0, posinf=0.0, neginf=0.0)
    V_val_t = torch.nan_to_num(V_val_t, nan=0.0, posinf=0.0, neginf=0.0)
    y_val_t = torch.nan_to_num(y_val_t, nan=0.0, posinf=0.0, neginf=0.0)

    # Clamp targets to positive (QLIKE needs positive values)
    y_train_t = torch.clamp(y_train_t, min=GNN_PRED_FLOOR)
    y_val_t = torch.clamp(y_val_t, min=GNN_PRED_FLOOR)

    model = GNNHAR(n_stocks)
    optimiser = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(GNN_MAX_EPOCHS):
        # Training
        model.train()
        batches = _make_batches(V_train_t, y_train_t, GNN_BATCH_SIZE)

        for V_batch, y_batch in batches:
            y_pred = model(V_batch, W_t)
            loss = qlike_loss(y_pred, y_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(V_val_t, W_t)
            val_loss = qlike_loss(y_val_pred, y_val_t).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= GNN_EARLY_STOP_PATIENCE:
                break

    # Load best weights (if training never improved, use final weights)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_ensemble(V_train, y_train, V_val, y_val, W):
    """
    Train an ensemble of GNNHAR models with different random seeds.

    Args:
        V_train, y_train, V_val, y_val, W: same as train_single_model

    Returns:
        List of trained GNNHAR models.
    """
    models = []

    for i in range(GNN_N_ENSEMBLE):
        print(f"  Training ensemble member {i + 1}/{GNN_N_ENSEMBLE}...")
        model = train_single_model(V_train, y_train, V_val, y_val, W, seed=i)
        models.append(model)

    return models


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict(models, V, W):
    """
    Generate predictions by averaging across ensemble members.

    Args:
        models: list of trained GNNHAR models
        V:      (T, N, 3) numpy array of HAR features
        W:      (N, N) numpy array — normalised adjacency matrix

    Returns:
        (T, N) numpy array of predicted sqrt(RV).
    """
    V_t = torch.tensor(V, dtype=torch.float32)
    W_t = torch.tensor(W, dtype=torch.float32)

    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(V_t, W_t).numpy()
        preds.append(pred)

    # Average across ensemble
    avg_pred = np.mean(preds, axis=0)

    # Floor predictions to avoid zero/negative values
    avg_pred = np.maximum(avg_pred, GNN_PRED_FLOOR)

    return avg_pred


if __name__ == "__main__":
    from gnn_vol.rv_compute import RVComputer
    from gnn_vol.har_features import build_har_features, build_targets
    from gnn_vol.graph_builder import GraphBuilder
    from gnn_vol.config import VALIDATION_SPLIT

    # Load data
    print("Loading RV data...")
    rv_df = RVComputer([]).load_rv()

    print("Building HAR features...")
    features = build_har_features(rv_df)
    V = features["V"]
    targets = build_targets(rv_df, horizon=1, dates=features["dates"])

    print("Loading graph...")
    W = GraphBuilder.load("latest")["normalised"]

    print(f"  V: {V.shape}, targets: {targets.shape}, W: {W.shape}")

    # Train/val split on last 500 days (quick test)
    V_small = V[-500:]
    y_small = targets[-500:]
    split = int(len(V_small) * (1 - VALIDATION_SPLIT))

    V_train, y_train = V_small[:split], y_small[:split]
    V_val, y_val = V_small[split:], y_small[split:]
    print(f"  Train: {V_train.shape[0]} days, Val: {V_val.shape[0]} days")

    # Train 2 ensemble members (fast)
    import gnn_vol.config as cfg
    cfg.GNN_N_ENSEMBLE = 2
    models = train_ensemble(V_train, y_train, V_val, y_val, W)

    # Predict
    preds = predict(models, V_val, W)
    print(f"\n  Predictions: {preds.shape}")
    print(f"  Range: {preds.min():.6f} to {preds.max():.6f}")
    print(f"  Mean:  {preds.mean():.6f}")

    # MAE
    valid = ~np.isnan(y_val).any(axis=1)
    mae = np.mean(np.abs(preds[valid] - y_val[valid]))
    print(f"  MAE:   {mae:.6f}")
    print("\nDone.")
