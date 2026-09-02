"""V1 group figures and the static dashboard. Reads committed CSV artefacts only."""
from __future__ import annotations

import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppg2ecg.evaluation import v1_timing as V

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/v1_stepwise_visualization"
F = A / "figures"; D = A / "dashboard"; D.mkdir(parents=True, exist_ok=True)
NFES = (1, 2, 4, 8, 50)
rows = lambda n: list(csv.DictReader(open(A / n)))  # noqa: E731
SUBS = V.TRAIN + V.VAL
SPLIT = {s: ("train" if s in V.TRAIN else "val") for s in SUBS}
SC = {"train": "tab:gray", "val": "tab:green"}

msub, msite = rows("metrics_by_subject.csv"), rows("metrics_by_site.csv")
dly, dsum = rows("r_to_ppg_peak_delays.csv"), rows("delay_summary.csv")
tp, tps = rows("timing_prior_validation.csv"), rows("timing_prior_summary.csv")
delay_ms = np.array([float(r["delay_ms"]) for r in dly])
d_sub = np.array([r["subject"] for r in dly]); d_site = np.array([r["site"] for r in dly])
d_split = np.array([r["split"] for r in dly]); d_hr = np.array([float(r["estimated_HR"]) for r in dly])

def get(lst, **kw):
    return [r for r in lst if all(str(r[k]) == str(v) for k, v in kw.items())]

# 1 all-subject NFE metrics (train/val separated)
KEYS = ["raw_rmse", "raw_corr", "qrs_rmse_core", "qrs_energy_dev", "qrs_deriv_rmse", "f1_excess"]
fig, axes = plt.subplots(2, 3, figsize=(17, 8))
for ax, k in zip(axes.ravel(), KEYS):
    for s in SUBS:
        v = [float(get(msub, subject=s, nfe=n)[0][k]) for n in NFES]
        ax.plot(NFES, v, "o-", lw=1.0, ms=3.5, alpha=0.65, color=SC[SPLIT[s]])
    for sp in ("train", "val"):
        v = [float(get(msite, split=sp, site="ALL", nfe=n)[0][k]) for n in NFES]
        ax.plot(NFES, v, "s-", lw=2.6, ms=7, color=SC[sp], label=f"{sp} mean")
    ax.set_xscale("log", base=2); ax.set_xticks(NFES); ax.set_xticklabels(NFES)
    ax.set_title(k, fontsize=10); ax.set_xlabel("NFE"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("V1 — stepwise NFE behaviour, one thin line per subject; train and val never pooled")
fig.tight_layout(); fig.savefig(F / "all_subject_nfe_metrics.png", dpi=115); plt.close(fig)

# 9 val-only summary
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, k in zip(axes, ["raw_rmse", "qrs_rmse_core", "qrs_deriv_rmse", "f1_excess"]):
    for s in V.VAL:
        ax.plot(NFES, [float(get(msub, subject=s, nfe=n)[0][k]) for n in NFES], "o-", lw=2, label=s)
    ax.set_xscale("log", base=2); ax.set_xticks(NFES); ax.set_xticklabels(NFES)
    ax.set_title(k, fontsize=10); ax.set_xlabel("NFE"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("V1 — validation subjects only (an0, k2s)")
fig.tight_layout(); fig.savefig(F / "val_only_nfe_summary.png", dpi=115); plt.close(fig)

# 6/7/8 per-subject NFE bars
for key, fname in (("qrs_rmse_core", "nfe_qrs_rmse_by_subject.png"),
                   ("qrs_deriv_rmse", "nfe_derivative_error_by_subject.png"),
                   ("f1_excess", "nfe_f1_excess_by_subject.png")):
    fig, ax = plt.subplots(figsize=(15, 4.6))
    x = np.arange(len(SUBS)); w = 0.16
    for i, n in enumerate(NFES):
        ax.bar(x + (i - 2) * w, [float(get(msub, subject=s, nfe=n)[0][key]) for s in SUBS], w, label=f"NFE {n}")
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n[{SPLIT[s]}]" for s in SUBS], fontsize=8)
    ax.set_title(f"V1 — {key} by subject and NFE (split labelled on each tick)")
    ax.legend(fontsize=8, ncol=5); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(F / fname, dpi=115); plt.close(fig)

# 2 delay by subject
fig, ax = plt.subplots(figsize=(15, 4.6))
data = [delay_ms[d_sub == s] for s in SUBS]
bp = ax.boxplot(data, labels=[f"{s}\n[{SPLIT[s]}]" for s in SUBS], showfliers=False, patch_artist=True)
for patch, s in zip(bp["boxes"], SUBS):
    patch.set_facecolor(SC[SPLIT[s]]); patch.set_alpha(0.55)
ax.set_ylabel("ECG-R → PPG-peak delay (ms)"); ax.grid(alpha=0.3, axis="y")
ax.set_title("V1 — R→PPG-peak delay by subject (not pulse transit time; includes electromechanical delay and rise time)")
fig.tight_layout(); fig.savefig(F / "all_subject_r_ppg_delay.png", dpi=115); plt.close(fig)

# 3 site boxplot
fig, ax = plt.subplots(figsize=(9, 4.6))
pos, labels, dat, cols = [], [], [], []
for i, site in enumerate(V.SITES):
    for j, sp in enumerate(("train", "val")):
        pos.append(i * 3 + j); labels.append(f"{site}\n{sp}")
        dat.append(delay_ms[(d_site == site) & (d_split == sp)]); cols.append(SC[sp])
bp = ax.boxplot(dat, positions=pos, labels=labels, showfliers=False, patch_artist=True, widths=0.8)
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.55)
ax.set_ylabel("delay (ms)"); ax.grid(alpha=0.3, axis="y")
ax.set_title("V1 — R→PPG-peak delay by site and split")
fig.tight_layout(); fig.savefig(F / "site_delay_boxplot.png", dpi=115); plt.close(fig)

# 4 subject x site heatmaps
for stat, fname in (("median", "subject_site_delay_heatmap.png"), ("iqr", "subject_site_delay_iqr_heatmap.png")):
    Mx = np.array([[float(get(dsum, subject=s, site=si)[0][stat]) for si in V.SITES] for s in SUBS])
    fig, ax = plt.subplots(figsize=(7.5, 8))
    im = ax.imshow(Mx, aspect="auto", cmap="viridis")
    ax.set_xticks(range(4)); ax.set_xticklabels(V.SITES); ax.set_yticks(range(len(SUBS)))
    ax.set_yticklabels([f"{s} [{SPLIT[s]}]" for s in SUBS], fontsize=8)
    for i in range(len(SUBS)):
        for j in range(4):
            ax.text(j, i, f"{Mx[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="w" if Mx[i, j] < Mx.mean() else "k")
    fig.colorbar(im, ax=ax, label=f"{stat} delay (ms)")
    ax.set_title(f"V1 — subject × site {stat} R→PPG-peak delay (ms)")
    fig.tight_layout(); fig.savefig(F / fname, dpi=115); plt.close(fig)

# 5 delay vs HR
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
ok = np.isfinite(d_hr)
ax[0].hexbin(d_hr[ok], delay_ms[ok], gridsize=45, cmap="magma", bins="log")
ax[0].set_xlabel("estimated HR (bpm)"); ax[0].set_ylabel("delay (ms)"); ax[0].set_title("delay vs HR (all non-test)")
rr = np.array([float(r["preceding_RR_ms"]) for r in dly]); ok2 = np.isfinite(rr)
ax[1].hexbin(rr[ok2], delay_ms[ok2], gridsize=45, cmap="magma", bins="log")
ax[1].set_xlabel("preceding RR (ms)"); ax[1].set_ylabel("delay (ms)"); ax[1].set_title("delay vs preceding RR")
fig.tight_layout(); fig.savefig(F / "delay_vs_hr.png", dpi=115); plt.close(fig)

# timing prior
fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(3); names = ["A_global", "B_site", "C_hr"]
for i, t_ in enumerate((25, 50, 100, 150)):
    ax.bar(x + (i - 1.5) * 0.2, [float(get(tps, predictor=n)[0][f"cov_{t_}ms"]) for n in names], 0.2, label=f"≤{t_} ms")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("fraction of validation GT beats covered")
ax.set_title("V1 — train-only PPG→R timing prior, validated on an0/k2s"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(F / "timing_prior_coverage.png", dpi=115); plt.close(fig)

# ---------------- dashboard ----------------
CSS = ("body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}"
       "h1,h2,h3{margin:.4em 0}.grid{display:flex;flex-wrap:wrap;gap:14px}"
       ".card{border:1px solid #ccc;border-radius:8px;padding:12px;background:#fff;min-width:260px}"
       ".val{border-left:6px solid #2a9d3f}.train{border-left:6px solid #888}"
       "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:3px 7px}"
       "img{max-width:100%;border:1px solid #ddd;margin:6px 0}a{color:#1a5fb4}")
cards = []
for s in SUBS:
    sp = SPLIT[s]
    med = " · ".join(f"{si}: {float(get(dsum, subject=s, site=si)[0]['median']):.0f} ms" for si in V.SITES)
    nwin = sum(int(get(msub, subject=s, nfe=1)[0]["n_windows"]) for _ in [0])
    cards.append(f'<div class="card {sp}"><h3><a href="subject_{s}.html">{s}</a></h3>'
                 f'<b>split:</b> {sp}<br><b>metric windows:</b> {nwin}<br>'
                 f'<b>sites:</b> {", ".join(V.SITES)}<br><b>median R→PPG delay:</b><br>{med}</div>')
(D / "index.html").write_text(
    f"<html><head><meta charset='utf-8'><title>V1 dashboard</title><style>{CSS}</style></head><body>"
    f"<h1>V1 — all-subject stepwise visualization</h1>"
    f"<p>Frozen iMeanFlow baseline replay (round 45). Source seed 0, identical across NFE. "
    f"No training, no test access; <b>kjd/ssx never loaded</b>. Green = validation, grey = train — "
    f"train results are <b>not</b> generalization evidence.</p>"
    f"<h2>Group figures</h2>" +
    "".join(f'<p><b>{p.name}</b><br><img src="../figures/{p.name}"></p>' for p in sorted(F.glob("*.png"))) +
    f"<h2>Subjects</h2><div class='grid'>{''.join(cards)}</div></body></html>")

for s in SUBS:
    sp = SPLIT[s]
    sec = []
    for site in V.SITES:
        ds = get(dsum, subject=s, site=site)[0]
        tbl = ("<table><tr><th>NFE</th>" + "".join(f"<th>{k}</th>" for k in KEYS) + "</tr>" +
               "".join("<tr><td>%d</td>%s</tr>" % (n, "".join(
                   f"<td>{float(get(msub, subject=s, nfe=n)[0][k]):.4f}</td>" for k in KEYS)) for n in NFES) +
               "</table>")
        figs = sorted(F.parent.joinpath("figures").glob(f"{s}_{site}_w*.png"))
        zs = sorted(F.parent.joinpath("beat_zooms").glob(f"{s}_{site}_w*_zoom.png"))
        imgs = "".join(f'<p><img src="../figures/{p.name}"></p>' for p in figs)
        imgz = "".join(f'<p><img src="../beat_zooms/{p.name}"></p>' for p in zs)
        sec.append(f"<h2>{site}</h2><p><b>R→PPG delay</b> n={ds['n']} median {float(ds['median']):.1f} ms "
                   f"IQR {float(ds['iqr']):.1f} p5–p95 {float(ds['p5']):.0f}–{float(ds['p95']):.0f} "
                   f"CV {float(ds['cv']):.3f}</p>{tbl}<h3>Stepwise windows</h3>{imgs}"
                   f"<h3>R-centred zooms</h3>{imgz}")
    (D / f"subject_{s}.html").write_text(
        f"<html><head><meta charset='utf-8'><title>{s}</title><style>{CSS}</style></head><body>"
        f"<p><a href='index.html'>← all subjects</a></p><h1>{s} <small>[{sp}]</small></h1>"
        f"{''.join(sec)}</body></html>")
print(f"wrote {len(list(F.glob('*.png')))} group figures and a {len(SUBS)+1}-page dashboard")
