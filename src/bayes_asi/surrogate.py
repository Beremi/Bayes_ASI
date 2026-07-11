from __future__ import annotations

import copy
import os
import time
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
        training_samples: int | None = None,
        input_space: str = "standard_normal",
    ) -> None:
        self.network = network.to(device).eval()
        self.normalization = normalization
        self.hidden_layers = list(hidden_layers)
        self.version = int(version)
        self.val_loss = float(val_loss)
        self.device = device
        self.training_samples = None if training_samples is None else int(training_samples)
        self.input_space = str(input_space)

    @property
    def input_dim(self) -> int:
        return int(self.normalization.x_mean.numel())

    @property
    def parameter_dim(self) -> int:
        """Compatibility alias for checkpoints created with physical inputs."""

        return self.input_dim

    @property
    def output_dim(self) -> int:
        return int(self.normalization.y_mean.numel())

    def forward_torch(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(dtype=torch.float32, device=self.device)
        original_shape = z.shape
        z_2d = z.reshape(-1, self.input_dim)
        x_norm = (z_2d - self.normalization.x_mean) / self.normalization.x_std
        y_norm = self.network(x_norm)
        y = y_norm * self.normalization.y_std + self.normalization.y_mean
        return y.reshape(*original_shape[:-1], self.output_dim)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward_torch(z)

    def predict_numpy(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z_tensor = torch.as_tensor(z, dtype=torch.float32, device=self.device)
            return self.forward_torch(z_tensor).detach().cpu().numpy()

    def jacobian_numpy(self, z: np.ndarray) -> np.ndarray:
        z_tensor = torch.as_tensor(z, dtype=torch.float32, device=self.device).reshape(1, -1)
        z_tensor.requires_grad_(True)
        y = self.forward_torch(z_tensor).reshape(-1)
        rows: list[torch.Tensor] = []
        for i in range(y.numel()):
            grad = torch.autograd.grad(y[i], z_tensor, retain_graph=True)[0]
            rows.append(grad.reshape(-1).detach().cpu())
        return torch.stack(rows, dim=0).numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 3,
            "version": self.version,
            "val_loss": self.val_loss,
            "training_samples": self.training_samples,
            "hidden_layers": self.hidden_layers,
            "input_dim": self.input_dim,
            "input_space": self.input_space,
            "output_dim": self.output_dim,
            "state_dict": self.network.state_dict(),
            "x_mean": self.normalization.x_mean.detach().cpu(),
            "x_std": self.normalization.x_std.detach().cpu(),
            "y_mean": self.normalization.y_mean.detach().cpu(),
            "y_std": self.normalization.y_std.detach().cpu(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> SurrogateModel:
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # PyTorch versions before weights_only support.
            payload = torch.load(path, map_location=device)
        required = {
            "version",
            "val_loss",
            "hidden_layers",
            "output_dim",
            "state_dict",
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"checkpoint {path} is missing: {', '.join(missing)}")
        input_dim = payload.get("input_dim", payload.get("parameter_dim"))
        if input_dim is None:
            raise ValueError(f"checkpoint {path} is missing: input_dim")
        network = TanhMLP(
            int(input_dim),
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
            training_samples=(
                int(payload["training_samples"])
                if payload.get("training_samples") is not None
                else None
            ),
            input_space=str(payload.get("input_space", "physical_parameters")),
        )


@dataclass(slots=True)
class TrainResult:
    model: SurrogateModel
    train_loss: float
    val_loss: float
    epochs: int
    best_epoch: int
    trained_from_scratch: bool
    stopped_by_deadline: bool


class SurrogateTrainer:
    def __init__(self, config: SurrogateConfig, input_dim: int, output_dim: int, seed: int) -> None:
        self.config = config
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        requested = torch.device(config.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = str(requested)

    def train(
        self,
        coordinates: np.ndarray,
        outputs: np.ndarray,
        *,
        version: int,
        previous: SurrogateModel | None = None,
        force_scratch: bool = False,
        deadline_monotonic: float | None = None,
    ) -> TrainResult:
        coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, self.input_dim)
        outputs = np.asarray(outputs, dtype=np.float32).reshape(-1, self.output_dim)
        if coordinates.shape[0] != outputs.shape[0]:
            raise ValueError("coordinates and outputs row counts differ")
        if coordinates.shape[0] < 2:
            raise ValueError("at least two completed samples are required to train a surrogate")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(outputs)):
            raise ValueError("coordinates and outputs must contain only finite values")

        seed = self.seed + int(version) * 1009 + (17 if force_scratch else 0)
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        x_mean = torch.as_tensor(coordinates.mean(axis=0), dtype=torch.float32, device=self.device)
        x_std_np = coordinates.std(axis=0)
        x_std_np[x_std_np < self.config.gradient_epsilon] = 1.0
        x_std = torch.as_tensor(x_std_np, dtype=torch.float32, device=self.device)
        y_mean = torch.as_tensor(outputs.mean(axis=0), dtype=torch.float32, device=self.device)
        y_std_np = outputs.std(axis=0)
        y_std_np[y_std_np < self.config.gradient_epsilon] = 1.0
        y_std = torch.as_tensor(y_std_np, dtype=torch.float32, device=self.device)
        normalization = Normalization(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            network = TanhMLP(self.input_dim, self.output_dim, self.config.hidden_layers)
        warm_started = False
        if previous is not None and not force_scratch:
            warm_started = True
            if previous.input_dim != self.input_dim or previous.output_dim != self.output_dim:
                raise ValueError("previous surrogate dimensions do not match the trainer")
            if previous.input_space != "standard_normal":
                raise ValueError("previous surrogate was not trained in standard-normal coordinates")
            if previous.hidden_layers != self.config.hidden_layers:
                warm_started = False
            else:
                network.load_state_dict(previous.network.state_dict())
                self._adapt_normalization(network, previous.normalization, normalization)
        network.to(self.device).train()

        x = torch.as_tensor(coordinates, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(outputs, dtype=torch.float32, device=self.device)
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        train_idx, val_idx = self._split_indices(coordinates.shape[0], rng)
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
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
        batch_size = max(1, int(self.config.batch_size))
        epochs_run = 0
        stopped_by_deadline = False

        for epoch in range(1, self.config.max_epochs + 1):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                stopped_by_deadline = True
                break
            order = train_idx[torch.randperm(train_idx.numel(), generator=generator, device=self.device)]
            network.train()
            interrupted = False
            for start in range(0, order.numel(), batch_size):
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    interrupted = True
                    stopped_by_deadline = True
                    break
                batch_idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred = network(x_norm[batch_idx])
                loss = loss_fn(pred, y_norm[batch_idx])
                loss.backward()
                optimizer.step()

            if interrupted:
                break

            network.eval()
            with torch.no_grad():
                train_loss = float(loss_fn(network(x_norm[train_idx]), y_norm[train_idx]).detach().cpu())
                if val_idx.numel() > 0:
                    metric = float(loss_fn(network(x_norm[val_idx]), y_norm[val_idx]).detach().cpu())
                else:
                    metric = train_loss
            epochs_run = epoch
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

        if best_state is None:
            if stopped_by_deadline:
                raise TimeoutError("surrogate training reached the run deadline before completing an epoch")
            raise RuntimeError("surrogate training did not produce a valid model state")
        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        with torch.no_grad():
            train_loss = float(loss_fn(network(x_norm[train_idx]), y_norm[train_idx]).detach().cpu())
            if val_idx.numel() > 0:
                best_metric = float(loss_fn(network(x_norm[val_idx]), y_norm[val_idx]).detach().cpu())
            else:
                best_metric = train_loss
        model = SurrogateModel(
            network=network,
            normalization=normalization,
            hidden_layers=self.config.hidden_layers,
            version=version,
            val_loss=best_metric,
            device=self.device,
            training_samples=coordinates.shape[0],
            input_space="standard_normal",
        )
        return TrainResult(
            model=model,
            train_loss=float(train_loss),
            val_loss=float(best_metric),
            epochs=epochs_run,
            best_epoch=best_epoch,
            trained_from_scratch=not warm_started,
            stopped_by_deadline=stopped_by_deadline,
        )

    @staticmethod
    def _adapt_normalization(
        network: TanhMLP,
        previous: Normalization,
        current: Normalization,
    ) -> None:
        """Preserve physical predictions while changing input/output scaling."""

        linear_layers = [layer for layer in network.network if isinstance(layer, nn.Linear)]
        first = linear_layers[0]
        last = linear_layers[-1]
        device = first.weight.device
        dtype = first.weight.dtype
        old_x_mean = previous.x_mean.to(device=device, dtype=dtype)
        old_x_std = previous.x_std.to(device=device, dtype=dtype)
        new_x_mean = current.x_mean.to(device=device, dtype=dtype)
        new_x_std = current.x_std.to(device=device, dtype=dtype)
        old_y_mean = previous.y_mean.to(device=device, dtype=dtype)
        old_y_std = previous.y_std.to(device=device, dtype=dtype)
        new_y_mean = current.y_mean.to(device=device, dtype=dtype)
        new_y_std = current.y_std.to(device=device, dtype=dtype)

        with torch.no_grad():
            first_weight = first.weight.detach().clone()
            input_scale = new_x_std / old_x_std
            input_shift = (new_x_mean - old_x_mean) / old_x_std
            first.weight.copy_(first_weight * input_scale.reshape(1, -1))
            first.bias.add_(first_weight @ input_shift)

            last_weight = last.weight.detach().clone()
            last_bias = last.bias.detach().clone()
            output_scale = old_y_std / new_y_std
            output_shift = (old_y_mean - new_y_mean) / new_y_std
            last.weight.copy_(last_weight * output_scale.reshape(-1, 1))
            last.bias.copy_(last_bias * output_scale + output_shift)

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
