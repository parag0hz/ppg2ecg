"""Fixed (t, z) validation banks for a deterministic validation CFM loss (A0-b pre-registration §3)."""
from __future__ import annotations

import hashlib

import torch


def make_banks(n: int, T: int, n_banks: int = 4, seed: int = 1000) -> list[tuple[torch.Tensor, torch.Tensor]]:
    banks = []
    for b in range(n_banks):
        g = torch.Generator().manual_seed(seed + b)
        t = torch.rand(n, 1, generator=g)
        z = torch.randn(n, 1, T, generator=g)
        banks.append((t, z))
    return banks


def bank_hash(banks) -> str:
    h = hashlib.sha256()
    for t, z in banks:
        h.update(t.numpy().astype("float32").tobytes())
        h.update(z.numpy().astype("float32").tobytes())
    return h.hexdigest()


@torch.no_grad()
def fixed_cfm_loss(model, x_val: torch.Tensor, y_val: torch.Tensor, banks, batch_size: int = 64) -> tuple[float, list[float]]:
    """Window-weighted MSE(v_theta(x_t, ppg, t), x1 - z) on each bank; returns (mean over banks, per-bank)."""
    from ppg2ecg.flow.cfm import cfm_targets

    device = x_val.device
    per_bank = []
    for t_b, z_b in banks:
        se, n = 0.0, 0
        for i in range(0, len(x_val), batch_size):
            ppg, ecg = x_val[i : i + batch_size], y_val[i : i + batch_size]
            t, z = t_b[i : i + batch_size].to(device), z_b[i : i + batch_size].to(device)
            x_t, v_star, _, _ = cfm_targets(ecg.unsqueeze(1), t, z)
            v = model.forward_step(x_t, ppg.unsqueeze(1), t)
            se += ((v - v_star) ** 2).mean(dim=(1, 2)).sum().item()
            n += len(ppg)
        per_bank.append(se / n)
    return float(sum(per_bank) / len(per_bank)), per_bank
