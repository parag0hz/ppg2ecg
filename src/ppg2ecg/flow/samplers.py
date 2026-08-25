"""ODE samplers for velocity fields with explicit NFE accounting.

`heun_sample` reproduces upstream PENGUIN.sample/heun_step arithmetic exactly (tests/test_samplers_match_upstream.py):
    time grid  = linspace(0, 1, n_steps + 1)
    Heun step  = 2 velocity evaluations  ->  NFE = 2 * n_steps   (upstream n_step=25  =>  50 NFE)
`euler_sample` gives NFE = n_steps, so the pre-registered NFE axis {25,10,5,2,1} is measured with Euler,
and Heun is reported as {50,20,10,4,2} NFE.

A "velocity function" here is v(x_t: [B,1,T], t: [B,1]) -> [B,1,T]; for upstream use
`lambda x, t: model.forward_step(x, ppg, t)`.
"""
from __future__ import annotations

from typing import Callable

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class NFECounter:
    """Wrap a velocity function and count how many times it is evaluated."""

    def __init__(self, fn: VelocityFn):
        self.fn, self.nfe = fn, 0

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.fn(x, t)

    def reset(self) -> None:
        self.nfe = 0


def _time_grid(n_steps: int, B: int, device) -> torch.Tensor:
    return torch.linspace(0, 1.0, n_steps + 1).repeat(B, 1).to(device)  # identical to upstream L246


@torch.no_grad()
def heun_sample(v: VelocityFn, x0: torch.Tensor, n_steps: int, last_step_euler: bool = False):
    """Upstream-exact Heun. Returns (x1, nfe)."""
    cnt = NFECounter(v)
    B = x0.shape[0]
    ts = _time_grid(n_steps, B, x0.device)
    x_t = x0
    for i in range(n_steps):
        t_start, t_end = ts[:, i].unsqueeze(-1), ts[:, i + 1].unsqueeze(-1)  # [B,1]
        dx_t = cnt(x_t, t_start)
        if last_step_euler and i == n_steps - 1:
            x_t = x_t + (t_end - t_start).unsqueeze(-1) * dx_t
            break
        pre_x_t1 = x_t + (t_end - t_start).unsqueeze(-1) * dx_t
        x_t = x_t + (t_end - t_start).unsqueeze(-1) / 2 * (dx_t + cnt(pre_x_t1, t_end))
    return x_t, cnt.nfe


@torch.no_grad()
def euler_sample(v: VelocityFn, x0: torch.Tensor, n_steps: int):
    """Forward Euler on the same uniform grid. NFE = n_steps. Returns (x1, nfe)."""
    cnt = NFECounter(v)
    B = x0.shape[0]
    ts = _time_grid(n_steps, B, x0.device)
    x_t = x0
    for i in range(n_steps):
        t_start, t_end = ts[:, i].unsqueeze(-1), ts[:, i + 1].unsqueeze(-1)
        x_t = x_t + (t_end - t_start).unsqueeze(-1) * cnt(x_t, t_start)
    return x_t, cnt.nfe


SAMPLERS = {"heun": heun_sample, "euler": euler_sample}


def nfe_of(solver: str, n_steps: int, last_step_euler: bool = False) -> int:
    if solver == "euler":
        return n_steps
    if solver == "heun":
        return 2 * n_steps - (1 if last_step_euler else 0)
    raise ValueError(solver)
