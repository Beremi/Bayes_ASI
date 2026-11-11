"""Run from the repository: PYTHONPATH=src python examples/parallel_tempering.py."""
from __future__ import annotations

import numpy as np
import torch

from bayes_asi import ParallelTemperedHMCConfig, ParallelTemperedHMCSampler


def log_density(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard-normal reference and a symmetric two-mode likelihood."""
    log_reference = -0.5 * z.square().sum(dim=-1)
    coordinate = z[:, 0]
    modes = torch.stack(
        (-0.5 * ((coordinate + 2.2) / 0.28).square(),
         -0.5 * ((coordinate - 2.2) / 0.28).square()),
        dim=-1,
    )
    return log_reference, torch.logsumexp(modes, dim=-1)


def main() -> None:
    config = ParallelTemperedHMCConfig(warmup_steps=220, samples=700)
    initial_z = np.linspace(-2.2, 2.2, len(config.inverse_temperatures))[:, None]
    sampler = ParallelTemperedHMCSampler(config, log_density, seed=9182, dtype=torch.float64)
    result = sampler.sample(initial_z)
    print("Cold samples:", result.cold_z.shape)
    print("Cold negative-mode fraction:", np.mean(result.cold_z[:, 0] < 0))
    print("Neighbor swap acceptance:", result.swap_accept_rate)


if __name__ == "__main__":
    main()
