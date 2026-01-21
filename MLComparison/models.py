"""Neural network models for cell age prediction.

This module provides several neural network architectures for predicting
cell age from single-cell mRNA count profiles.

The task is regression: given mRNA counts for G genes, predict cell age t ∈ [0, T_div].

Models:
1. SimpleMLP: Basic feedforward network
2. DeepMLP: Deeper network with more layers
3. GeneAttention: Attention-based model that learns gene importance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PyTorch is required for ML models. Install torch to use these models."
    ) from exc


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    n_genes: int
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    activation: str = "relu"
    output_type: str = "regression"  # "regression" or "classification"
    n_bins: int = 20  # For classification output
    normalize_input: bool = True


def _activation(name: str) -> nn.Module:
    """Get activation module by name."""
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class _TorchRegressor:
    """Base wrapper providing fit/predict and normalization for PyTorch models."""

    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._output_mean: Optional[float] = None
        self._output_std: Optional[float] = None

    def _normalize_input(self, X: np.ndarray) -> np.ndarray:
        if self.config.normalize_input:
            if self._input_mean is None or self._input_std is None:
                self._input_mean = np.mean(X, axis=0)
                self._input_std = np.std(X, axis=0) + 1e-8
            return (X - self._input_mean) / self._input_std
        return X

    def _normalize_output(self, y: np.ndarray) -> np.ndarray:
        if self._output_mean is None or self._output_std is None:
            self._output_mean = float(np.mean(y))
            self._output_std = float(np.std(y) + 1e-8)
        return (y - self._output_mean) / self._output_std

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
        verbose: bool = False,
        device: Optional[str] = None,
    ) -> dict:
        """Train the model.

        Args:
            X: Training data (n_samples, n_genes)
            y: Training labels - cell ages (n_samples,)
            n_epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Learning rate for Adam optimizer
            seed: Random seed
            verbose: Print training progress
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.

        Returns:
            Training history dictionary with loss per epoch
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(seed)

        Xn = self._normalize_input(X)
        yn = self._normalize_output(y)

        X_t = torch.tensor(Xn, dtype=torch.float32)
        y_t = torch.tensor(yn, dtype=torch.float32).view(-1, 1)

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.model.to(device)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        history = {"loss": []}

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item())

            epoch_loss /= max(1, len(loader))
            history["loss"].append(epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.6f}")

        return history

    def predict(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """Predict cell ages.

        Args:
            X: Input data (n_samples, n_genes)
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.

        Returns:
            Predicted ages (n_samples,)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        Xn = self._normalize_input(X)
        X_t = torch.tensor(Xn, dtype=torch.float32).to(device)

        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_t).cpu().numpy().ravel()

        if self._output_mean is None or self._output_std is None:
            return pred
        return pred * self._output_std + self._output_mean


class SimpleMLP(_TorchRegressor):
    """Simple multi-layer perceptron for age prediction.

    Architecture:
    - Input: mRNA counts (n_genes,)
    - Hidden layers with configurable activation and dropout
    - Output: predicted age (scalar)

    This serves as a baseline for comparison.
    """

    def __init__(self, config: ModelConfig):
        layers = []
        dims = (config.n_genes,) + config.hidden_dims + (1,)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_activation(config.activation))
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
        super().__init__(nn.Sequential(*layers), config)


class DeepMLP(_TorchRegressor):
    """Deeper MLP with more layers.

    Uses a deeper architecture (512, 256, 256, 128, 64) for potentially
    better representation learning on complex data.
    """

    def __init__(self, config: ModelConfig):
        deep_config = ModelConfig(
            n_genes=config.n_genes,
            hidden_dims=(512, 256, 256, 128, 64),
            dropout=config.dropout,
            activation=config.activation,
            output_type=config.output_type,
            n_bins=config.n_bins,
            normalize_input=config.normalize_input,
        )
        layers = []
        dims = (deep_config.n_genes,) + deep_config.hidden_dims + (1,)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_activation(deep_config.activation))
                if deep_config.dropout > 0:
                    layers.append(nn.Dropout(deep_config.dropout))
        super().__init__(nn.Sequential(*layers), deep_config)


class GeneAttention(_TorchRegressor):
    """Attention-based model that learns gene importance.

    Uses self-attention to weight different genes based on their
    relevance for age prediction. This can help identify which
    genes are most informative about cell cycle position.

    Architecture:
    - Gene embedding layer
    - Expression projection
    - Multi-head self-attention
    - Mean pooling across genes
    - MLP head for prediction
    """

    def __init__(self, config: ModelConfig, embed_dim: int = 32, n_heads: int = 4):
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        class _GeneAttentionNet(nn.Module):
            def __init__(self, n_genes: int, d_model: int, heads: int, dropout: float):
                super().__init__()
                self.n_genes = n_genes
                self.gene_embed = nn.Embedding(n_genes, d_model)
                self.expr_proj = nn.Linear(1, d_model)
                self.attn = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (batch, n_genes)
                bsz, n_genes = x.shape
                gene_ids = torch.arange(n_genes, device=x.device)
                gene_embed = self.gene_embed(gene_ids)[None, :, :]  # (1, n_genes, d)

                x_expr = x.unsqueeze(-1)  # (batch, n_genes, 1)
                expr_embed = self.expr_proj(x_expr)
                x_embed = gene_embed + expr_embed

                attn_out, _ = self.attn(x_embed, x_embed, x_embed)
                pooled = attn_out.mean(dim=1)
                return self.head(pooled)

        model = _GeneAttentionNet(config.n_genes, embed_dim, n_heads, config.dropout)
        super().__init__(model, config)

    def get_gene_importance(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """Get gene importance scores based on attention weights.

        Args:
            X: Input data (n_samples, n_genes)
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.

        Returns:
            importance: Gene importance scores (n_genes,)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        Xn = self._normalize_input(X)
        X_t = torch.tensor(Xn, dtype=torch.float32).to(device)

        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            bsz, n_genes = X_t.shape
            gene_ids = torch.arange(n_genes, device=X_t.device)
            gene_embed = self.model.gene_embed(gene_ids)[None, :, :]
            x_expr = X_t.unsqueeze(-1)
            expr_embed = self.model.expr_proj(x_expr)
            x_embed = gene_embed + expr_embed

            _, attn_w = self.model.attn(x_embed, x_embed, x_embed)
            importance = attn_w.mean(dim=0).sum(dim=0)

        return importance.cpu().numpy()
