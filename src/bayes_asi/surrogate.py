from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from bayes_asi.config import SurrogateConfig


class TanhMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(last, int(width)))
            layers.append(nn.Tanh())
            last = int(width)
        layers.append(nn.Linear(last, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(slots=True)
class Normalization:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


class SurrogateModel:
    def __init__(
        self,
        network: TanhMLP,
        normalization: Normalization,
        hidden_layers: list[int],
        version: int,
        val_loss: float,
        device: str,
    ) -> None:
        self.network = network.to(device).eval()
        self.normalization = normalization
        self.hidden_layers = list(hidden_layers)
        self.version = int(version)
        self.val_loss = float(val_loss)
        self.device = device

    @property
    def parameter_dim(self) -> int:
        return int(self.normalization.x_mean.numel())

    @property
    def output_dim(self) -> int:
        return int(self.normalization.y_mean.numel())

    def forward_torch(self, theta: torch.Tensor) -> torch.Tensor:
        theta = theta.to(dtype=torch.float32, device=self.device)
        original_shape = theta.shape
        theta_2d = theta.reshape(-1, self.parameter_dim)
        x_norm = (theta_2d - self.normalization.x_mean) / self.normalization.x_std
        y_norm = self.network(x_norm)
        y = y_norm * self.normalization.y_std + self.normalization.y_mean
        return y.reshape(*original_shape[:-1], self.output_dim)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return self.forward_torch(theta)

    def predict_numpy(self, theta: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            theta_tensor = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
            return self.forward_torch(theta_tensor).detach().cpu().numpy()

    def jacobian_numpy(self, theta: np.ndarray) -> np.ndarray:
        theta_tensor = torch.as_tensor(theta, dtype=torch.float32, device=self.device).reshape(1, -1)
        theta_tensor.requires_grad_(True)
        y = self.forward_torch(theta_tensor).reshape(-1)
        rows: list[torch.Tensor] = []
        for i in range(y.numel()):
            grad = torch.autograd.grad(y[i], theta_tensor, retain_graph=True)[0]
            rows.append(grad.reshape(-1).detach().cpu())
        return torch.stack(rows, dim=0).numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "val_loss": self.val_loss,
            "hidden_layers": self.hidden_layers,
            "parameter_dim": self.parameter_dim,
            "output_dim": self.output_dim,
            "state_dict": self.network.state_dict(),
            "x_mean": self.normalization.x_mean.detach().cpu(),
            "x_std": self.normalization.x_std.detach().cpu(),
            "y_mean": self.normalization.y_mean.detach().cpu(),
            "y_std": self.normalization.y_std.detach().cpu(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "SurrogateModel":
        payload = torch.load(path, map_location=device)
        network = TanhMLP(
            int(payload["parameter_dim"]),
            int(payload["output_dim"]),
            [int(value) for value in payload["hidden_layers"]],
        )
        network.load_state_dict(payload["state_dict"])
        normalization = Normalization(
            x_mean=payload["x_mean"].to(device=device, dtype=torch.float32),
            x_std=payload["x_std"].to(device=device, dtype=torch.float32),
            y_mean=payload["y_mean"].to(device=device, dtype=torch.float32),
            y_std=payload["y_std"].to(device=device, dtype=torch.float32),
        )
        return cls(
            network=network,
            normalization=normalization,
            hidden_layers=[int(value) for value in payload["hidden_layers"]],
            version=int(payload["version"]),
            val_loss=float(payload["val_loss"]),
            device=device,
        )


@dataclass(slots=True)
class TrainResult:
    model: SurrogateModel
    train_loss: float
    val_loss: float
    epochs: int
    trained_from_scratch: bool


class SurrogateTrainer:
    def __init__(self, config: SurrogateConfig, parameter_dim: int, output_dim: int, seed: int) -> None:
        self.config = config
        self.parameter_dim = int(parameter_dim)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        if config.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = config.device

    def train(
        self,
        theta: np.ndarray,
        outputs: np.ndarray,
        *,
        version: int,
        previous: SurrogateModel | None = None,
        force_scratch: bool = False,
    ) -> TrainResult:
        theta = np.asarray(theta, dtype=np.float32).reshape(-1, self.parameter_dim)
        outputs = np.asarray(outputs, dtype=np.float32).reshape(-1, self.output_dim)
        if theta.shape[0] < 2:
            raise ValueError("at least two completed samples are required to train a surrogate")

        seed = self.seed + int(version) * 1009 + (17 if force_scratch else 0)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        x_mean = torch.as_tensor(theta.mean(axis=0), dtype=torch.float32, device=self.device)
        x_std_np = theta.std(axis=0)
        x_std_np[x_std_np < self.config.gradient_epsilon] = 1.0
        x_std = torch.as_tensor(x_std_np, dtype=torch.float32, device=self.device)
        y_mean = torch.as_tensor(outputs.mean(axis=0), dtype=torch.float32, device=self.device)
        y_std_np = outputs.std(axis=0)
        y_std_np[y_std_np < self.config.gradient_epsilon] = 1.0
        y_std = torch.as_tensor(y_std_np, dtype=torch.float32, device=self.device)
        normalization = Normalization(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

        network = TanhMLP(self.parameter_dim, self.output_dim, self.config.hidden_layers)
        if previous is not None and not force_scratch and previous.hidden_layers == self.config.hidden_layers:
            network.load_state_dict(previous.network.state_dict())
        network.to(self.device).train()

        x = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(outputs, dtype=torch.float32, device=self.device)
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        train_idx, val_idx = self._split_indices(theta.shape[0], rng)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
        )
        loss_fn = nn.MSELoss()

        best_metric = float("inf")
        best_state: dict[str, Any] | None = None
        best_epoch = 0
        stale_epochs = 0
        train_loss = float("inf")
        val_loss = float("inf")
        batch_size = max(1, int(self.config.batch_size))

        for epoch in range(1, self.config.max_epochs + 1):
            order = train_idx[torch.randperm(train_idx.numel(), device=self.device)]
            network.train()
            for start in range(0, order.numel(), batch_size):
                batch_idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred = network(x_norm[batch_idx])
                loss = loss_fn(pred, y_norm[batch_idx])
                loss.backward()
                optimizer.step()

            network.eval()
            with torch.no_grad():
                train_loss = float(loss_fn(network(x_norm[train_idx]), y_norm[train_idx]).detach().cpu())
                if val_idx.numel() > 0:
                    val_loss = float(loss_fn(network(x_norm[val_idx]), y_norm[val_idx]).detach().cpu())
                    metric = val_loss
                else:
                    val_loss = train_loss
                    metric = train_loss
            scheduler.step(metric)
            if metric < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(network.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.patience:
                break

        if best_state is not None:
            network.load_state_dict(best_state)
        model = SurrogateModel(
            network=network,
            normalization=normalization,
            hidden_layers=self.config.hidden_layers,
            version=version,
            val_loss=best_metric,
            device=self.device,
        )
        return TrainResult(
            model=model,
            train_loss=float(train_loss),
            val_loss=float(best_metric),
            epochs=int(best_epoch or self.config.max_epochs),
            trained_from_scratch=force_scratch or previous is None,
        )

    def _split_indices(self, n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        order = rng.permutation(n)
        val_n = int(round(n * self.config.validation_fraction))
        if n < 5:
            val_n = 0
        val_n = max(0, min(val_n, n - 1))
        val_idx_np = order[:val_n]
        train_idx_np = order[val_n:]
        return (
            torch.as_tensor(train_idx_np, dtype=torch.long, device=self.device),
            torch.as_tensor(val_idx_np, dtype=torch.long, device=self.device),
        )

