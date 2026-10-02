"""DP0 models: specialists, the shared conditioning encoder, the three sharing patterns and round-robin training.

SPECIALIST P   SF0 / AF0 WW-L1 (C0-A WWDet 72 x 5): [C0 encoder(PPG), event raster] -> 512-sample ECG, L1.
SPECIALIST G   SF0 SCALEFLOW-COUPLED (ScaleFM(50, coupled)): flow matching on the whole ECG in the fixed Haar space.

DualReadout    H = E_shared(PPG, raster); mu = D_point(H); v = D_flow(x_t, t, H). The two waveform outputs are never added.
  encoder      the WW-L1 encoder family with the event raster moved into its input: stem 1x1 (2 -> 64) and eight C0
               residual blocks (64 channels, kernel 5, dilations 1, 2, 4, 8, 16, 32, 1, 2). Natural stages:
               E0 stem | E1 blocks 1-6 (first dilation cycle, receptive field 1 -> 505 samples) | E2 blocks 7-8 (second
               cycle, refinement). Multiscale taps by the fixed parameter-free Haar low-pass of the 512-sample features:
               H_256 = low1(H_512), H_128 = low1(H_256).
  sharing      S0 FULL: E0 + E1 + E2 shared. S1 MIDDLE: E0 + E1 shared, E2 duplicated per task. S2 STEM-ONLY: E0 shared,
               E1 + E2 duplicated per task. Every readout path has the same architecture in all three patterns.
  adapters     one point and one flow adapter (1x1 conv 64 -> 64 + GELU) at the output of the last shared stage.
  D_point      1x1 (64 -> w_p) -> five C0 residual blocks (dilations 1, 2, 4, 8, 16) -> 1x1 -> mu (512).
  D_flow       SCALEFLOW-COUPLED branches (six time-conditioned residual blocks each, width w_f) whose inputs are
               [Haar_s(x_t), H_s] (coarse, mid: H_128; fine: H_256) plus the SF0 coarse -> mid, coarse / mid -> fine
               projections; inverse Haar gives ONE whole-window velocity. x_t never enters the shared encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ppg2ecg.coherentbeat import ablation as AB
from ppg2ecg.coherentbeat import model as CM
from ppg2ecg.scaleflow import model as SM

T = 512
CH, KERNEL = CM.CH, CM.KERNEL
ENC_DIL = tuple(int(d) for d in CM.ENC_DILATIONS)                  # (1, 2, 4, 8, 16, 32, 1, 2)
STAGES = {"E0": "stem 1x1 (2 -> 64)", "E1": "blocks 1-6 (dilations 1..32)", "E2": "blocks 7-8 (dilations 1, 2)"}
N_SHARED_BLOCKS = {"S0": 8, "S1": 6, "S2": 0}
SHARED_STAGES = {"S0": ("E0", "E1", "E2"), "S1": ("E0", "E1"), "S2": ("E0",)}
POINT_DEPTH = 5
TASKS = ("point", "flow")
M_CORR, M_FP, M_RECALL, M_FD, E1_RATIO = 0.02, 0.05, 0.01, 1.0, 0.85
WIDTH_GRID = tuple(range(16, 161))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------- specialists
def specialist_point() -> nn.Module:
    return SM.ww_l1()


def specialist_gen() -> nn.Module:
    return SM.ScaleFM(50, True)


# ----------------------------------------------------------------------------------------------- dual components
def enc_blocks(dils) -> nn.ModuleList:
    return nn.ModuleList([CM.ResBlock(CH, KERNEL, int(d)) for d in dils])


class Adapter(nn.Module):
    """Task adapter: 1x1 conv (same channel count) + GELU. No attention."""

    def __init__(self, ch: int = CH):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 1)
        self.act = nn.GELU()

    def forward(self, h):
        return self.act(self.conv(h))


class PointDecoder(nn.Module):
    """The WW-L1 decoder family on shared features: 1x1 -> residual blocks -> 1x1 (no noise, no time input)."""

    def __init__(self, w: int, cin: int = CH):
        super().__init__()
        self.dec_in = nn.Conv1d(cin, w, 1)
        self.dec = nn.ModuleList([CM.ResBlock(w, KERNEL, int(AB.WW_DEC_DILATIONS[i])) for i in range(POINT_DEPTH)])
        self.head = nn.Conv1d(w, 1, 1)

    def forward(self, f):
        z = self.dec_in(f)
        for b in self.dec:
            z = b(z)
        return self.head(z)[:, 0]


def pyramid(f):
    """[B, C, 512] -> (H_256, H_128) by the fixed orthonormal Haar low-pass (no parameters)."""
    f256 = SM.haar1(f)[0]
    return f256, SM.haar1(f256)[0]


class FlowDecoder(nn.Module):
    """SCALEFLOW-COUPLED vector field whose condition inputs are the shared features instead of Haar(PPG), Haar(raster)."""

    def __init__(self, w: int, cin: int = CH):
        super().__init__()
        p = SM.PROJ
        self.temb = SM.TimeEmbed()
        self.coarse = SM.Branch(1 + cin, w)
        self.mid = SM.Branch(1 + cin + p, w)
        self.fine = SM.Branch(1 + cin + 2 * p, w)
        self.proj_cm = nn.Conv1d(w, p, 1)
        self.proj_cf = nn.Conv1d(w, p, 1)
        self.proj_mf = nn.Conv1d(w, p, 1)

    def forward(self, xt, t, cond):
        f256, f128 = cond
        xc, xm, xf = SM.haar(xt)
        e = self.temb(t)
        hc, vc = self.coarse(torch.cat([xc[:, None], f128], dim=1), e)
        hm, vm = self.mid(torch.cat([xm[:, None], f128, self.proj_cm(hc)], dim=1), e)
        _, vf = self.fine(torch.cat([xf[:, None], f256, SM.up2(self.proj_mf(hm)), SM.up2(self.proj_cf(hc))], dim=1), e)
        return SM.ihaar(vc, vm, vf)


class DualReadout(nn.Module):
    def __init__(self, share: str, w_point: int, w_flow: int):
        super().__init__()
        if share not in N_SHARED_BLOCKS:
            raise ValueError(share)
        self.share = share
        k = N_SHARED_BLOCKS[share]
        self.stem = nn.Conv1d(2, CH, 1)                                    # condition inputs only: [PPG, event raster]
        self.shared_blocks = enc_blocks(ENC_DIL[:k])
        self.point_adapter = Adapter()
        self.flow_adapter = Adapter()
        self.point_tower = enc_blocks(ENC_DIL[k:])
        self.flow_tower = enc_blocks(ENC_DIL[k:])
        self.point_dec = PointDecoder(w_point)
        self.flow_dec = FlowDecoder(w_flow)

    # -- shared condition encoding (PPG and raster only; never x_t)
    def trunk(self, ppg, raster):
        h = self.stem(torch.stack([ppg, raster], dim=1))
        for b in self.shared_blocks:
            h = b(h)
        return h

    def point_features(self, h):
        z = self.point_adapter(h)
        for b in self.point_tower:
            z = b(z)
        return z

    def flow_features(self, h):
        z = self.flow_adapter(h)
        for b in self.flow_tower:
            z = b(z)
        return pyramid(z)

    # -- readouts
    def point(self, ppg, raster):
        return self.point_dec(self.point_features(self.trunk(ppg, raster)))

    def encode_flow(self, ppg, raster):
        return self.flow_features(self.trunk(ppg, raster))

    def velocity(self, xt, t, cond):
        return self.flow_dec(xt, t, cond)

    def flow(self, xt, t, ppg, raster):
        return self.velocity(xt, t, self.encode_flow(ppg, raster))

    def both(self, ppg, raster):
        """Point estimate and the cached flow condition from ONE shared trunk pass."""
        h = self.trunk(ppg, raster)
        return self.point_dec(self.point_features(h)), self.flow_features(h)

    # -- ownership
    def ownership(self) -> dict:
        own = {}
        for n, _ in self.named_parameters():
            top = n.split(".")[0]
            own[n] = ("shared" if top in ("stem", "shared_blocks") else "point" if top.startswith("point_") else
                      "flow" if top.startswith("flow_") else None)
            if own[n] is None:
                raise ValueError(n)
        return own

    def groups(self) -> dict:
        own = self.ownership()
        g = {"shared": [], "point": [], "flow": []}
        for n, p in self.named_parameters():
            g[own[n]].append(p)
        return g

    def accounting(self) -> dict:
        g = self.groups()
        c = {k: sum(p.numel() for p in v) for k, v in g.items()}
        adapters = n_params(self.point_adapter) + n_params(self.flow_adapter)
        return {"shared": c["shared"], "point_private": c["point"], "flow_private": c["flow"], "total": sum(c.values()),
                "adapters": adapters, "adapter_fraction": adapters / sum(c.values()),
                "point_path": c["shared"] + c["point"], "flow_path": c["shared"] + c["flow"]}


# ----------------------------------------------------------------------------------------------- width rule (parameter count only)
def path_params(w_point: int | None = None, w_flow: int | None = None) -> dict:
    """Readout-path parameter counts (identical for S0 / S1 / S2): encoder + adapter + decoder."""
    enc = n_params(nn.Conv1d(2, CH, 1)) + n_params(enc_blocks(ENC_DIL))
    ad = n_params(Adapter())
    out = {}
    if w_point is not None:
        out["point"] = enc + ad + n_params(PointDecoder(w_point))
    if w_flow is not None:
        out["flow"] = enc + ad + n_params(FlowDecoder(w_flow))
    return out


def choose_widths(p_point: int, p_gen: int) -> dict:
    """w_p: point path closest to SPECIALIST P's count; w_f: flow path closest to SPECIALIST G's count (ties -> smaller)."""
    cp = {w: path_params(w_point=w)["point"] for w in WIDTH_GRID}
    cf = {w: path_params(w_flow=w)["flow"] for w in WIDTH_GRID}
    wp = min(cp, key=lambda w: (abs(cp[w] - p_point), w))
    wf = min(cf, key=lambda w: (abs(cf[w] - p_gen), w))
    return {"w_point": wp, "w_flow": wf, "point_path": cp[wp], "flow_path": cf[wf],
            "point_path_vs_P": (cp[wp] - p_point) / p_point, "flow_path_vs_G": (cf[wf] - p_gen) / p_gen}


def saving(p_dual: int, p_sep: int) -> float:
    return 1.0 - p_dual / p_sep


def e1_pass(p_dual: int, p_sep: int) -> bool:
    return p_dual <= E1_RATIO * p_sep


# ----------------------------------------------------------------------------------------------- flow matching
def fm_loss(net: DualReadout, x1, ppg, raster, gen):
    """SF0 objective: x0 ~ N(0, I), t ~ U(0, 1) (same draw order as scaleflow.fm_loss), x_t = (1 - t) x0 + t x1,
    target u = x1 - x0, MSE of the flow readout."""
    x0 = torch.randn(x1.shape, generator=gen, device=x1.device)
    t = torch.rand(x1.shape[0], generator=gen, device=x1.device)
    xt, u = SM.fm_path(x1, x0, t)
    return ((net.flow(xt, t, ppg, raster) - u) ** 2).mean()


@torch.no_grad()
def euler(net: DualReadout, x0, ppg, raster, nfe: int = 8, cond=None):
    """Euler NFE steps of the flow readout; the shared condition is encoded once and reused by every step."""
    cond = net.encode_flow(ppg, raster) if cond is None else cond
    x = x0.clone()
    dt = 1.0 / nfe
    for i in range(nfe):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * net.velocity(x, t, cond)
    return x


# ----------------------------------------------------------------------------------------------- round-robin training
def cycle_order(cycle: int) -> tuple:
    """Cycle numbering starts at 1: odd cycles POINT -> FLOW, even cycles FLOW -> POINT."""
    return ("point", "flow") if cycle % 2 == 1 else ("flow", "point")


def substep(net: DualReadout, opt, groups: dict, task: str, loss_fn, clip: float):
    """One masked update: gradients exist only for the shared parameters and this task's private parameters; the other
    task's private parameters have grad None, so AdamW skips them entirely (no moment, no weight-decay update)."""
    other = "flow" if task == "point" else "point"
    opt.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    for p in groups[other]:
        if p.grad is not None:
            raise RuntimeError(f"{task} substep produced a gradient on a {other}-private parameter")
    torch.nn.utils.clip_grad_norm_(groups["shared"] + groups[task], clip)
    opt.step()
    return loss


def train_round_robin(net: DualReadout, opt, n_cycles: int, point_loss, flow_loss, clip: float, on_cycle=None) -> dict:
    """n_cycles cycles of two masked substeps each (n_cycles POINT and n_cycles FLOW updates)."""
    groups = net.groups()
    counts = {"point": 0, "flow": 0}
    order_log = []
    for c in range(1, n_cycles + 1):
        order = cycle_order(c)
        order_log.append(order[0])
        for task in order:
            loss = substep(net, opt, groups, task, point_loss if task == "point" else flow_loss, clip)
            counts[task] += 1
            if on_cycle is not None:
                on_cycle(c, task, loss)
    return {"counts": counts, "first_task_by_cycle": order_log}


# ----------------------------------------------------------------------------------------------- frozen gates / selection
def gates(d: dict, p_dual: int, p_sep: int) -> dict:
    """d: [point, CI low, CI high] triples of dual - specialist (corr, fp, recall, fd) and shuffled - conditioned
    (fd_shuf, corr_shuf)."""
    g = {"P1": d["corr"][1] > -M_CORR, "P2": d["fp"][2] < M_FP and d["recall"][1] > -M_RECALL, "G1": d["fd"][2] < M_FD,
         "CONDITION": d["fd_shuf"][1] > 0 and d["corr_shuf"][2] < 0, "E1": e1_pass(p_dual, p_sep)}
    g = {k: bool(v) for k, v in g.items()}
    g["QUALIFIED"] = all(g.values())
    return g


def select_winner(cands: dict):
    """cands: id -> {"qualified", "saving", "fd", "corr", "latency_both_ms"}. Frozen lexicographic rule over qualifying
    candidates: largest saving, lower FD, higher point corr, lower combined latency."""
    q = [c for c, v in cands.items() if v["qualified"]]
    if not q:
        return None
    return sorted(q, key=lambda c: (-cands[c]["saving"], cands[c]["fd"], -cands[c]["corr"], cands[c]["latency_both_ms"]))[0]
