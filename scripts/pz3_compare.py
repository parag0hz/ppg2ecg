"""PZ3 — per-patient comparison against PENGUIN on the personalised axes
(docs/PZ3_PER_PATIENT_COMPARISON_PREREGISTRATION.md). No training.

HR and R-peak F1 come from the SR1 seed-42 arrays (same windows, same noise seeds); only `morph_corr@GT` — beats cut
from the GENERATED waveform at the TARGET's R peaks — is computed here, plus the two template arms and SBV."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from pz1_personalisation import RI, beats_of, case_path, ci, train_beats  # noqa: E402
from ppg2ecg.evaluation.rpeaks import detect_rpeaks  # noqa: E402
from ppg2ecg.evaluation.stamping import build_template_a  # noqa: E402
from ppg2ecg.models.imf_dit import ImfDiT1d  # noqa: E402

FS, DRAWS, N_CAL, N_TRAIN_CASES = 128, 4, 8, 200
OUT = ROOT / "artifacts/pz3_per_patient"
GEN = {"C50": ("C", 50), "C1": ("C", 1), "I1": ("I", 1), "S1": ("S", 1)}
RUN = {"C": "outputs/v1_vitaldb_armC_seed42", "I": "outputs/v1_vitaldb_armI_seed42", "S": "outputs/vm1_S_seed42"}


def peaks_of(sig):
    return np.asarray(detect_rpeaks(sig, FS), int)


def morph_at(args):
    """mean per-beat Pearson between the generated beats and the target beats, both cut at the TARGET's R peaks."""
    pred, tgt, pk = args
    pb, tb = beats_of(pred, pk), beats_of(tgt, pk)
    n = min(len(pb), len(tb))
    if n == 0:
        return np.nan
    cc = [np.corrcoef(p, t)[0, 1] if p.std() > 0 and t.std() > 0 else np.nan for p, t in zip(pb[:n], tb[:n])]
    return float(np.nanmean(cc))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    split = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]
    X, Y, pid = VM.load("test")
    # position of each window inside its case, and the case id per window
    pos, case_of = [], []
    for c in split["test"]:
        d = np.load(case_path(c))
        pos += list(np.argsort(np.argsort(d["window_index"])))
        case_of += [c] * len(d["x"])
    pos, case_of = np.array(pos), np.array(case_of)
    aligned = pos >= N_CAL                                  # windows a calibrated template may be scored on
    ex = ProcessPoolExecutor(8)
    pk_list = list(ex.map(peaks_of, list(Y), chunksize=64))
    print(f"[pz3] windows {len(X)} aligned {int(aligned.sum())}", flush=True)

    res = {}
    for name, (arm, nfe) in GEN.items():
        ck = ROOT / RUN[arm] / "checkpoint_last.pt"
        if arm == "S":
            c = torch.load(ck, map_location="cpu", weights_only=False)
            net = ImfDiT1d(**c["model_cfg"]).to(dev).eval(); net.load_state_dict(c["state_dict"])
            cfg = tuple(float(v) for v in np.load(ROOT / "outputs/vm1_eval/arm_S.npz")["cfg"])
            gen = lambda seed: VM.generate(net, X, seed, nfe, cfg, dev)[0]  # noqa: E731
        else:
            net = U2.build(ck, dev)[0]
            def gen(seed, _net=net, _arm=arm, _nfe=nfe):
                e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(seed))
                return U2.generate(_net, X, e0, _nfe, dev, _arm)[0]
        per_draw = []
        for s in range(DRAWS):
            pred = gen(s)
            per_draw.append(np.array(list(ex.map(morph_at, zip(pred, Y, pk_list), chunksize=64))))
            print(f"[pz3] {name} draw {s} done", flush=True)
        res[name] = np.nanmean(np.vstack(per_draw), 0)
        del net; torch.cuda.empty_cache()

    # template arms (GT timing, diagnostic) and SBV, on the aligned windows
    g = build_template_a(train_beats(sorted(split["train"])[:N_TRAIN_CASES]))[0]
    tmpl_global, tmpl_k8, sbv = np.full(len(X), np.nan), np.full(len(X), np.nan), np.full(len(X), np.nan)
    for c in split["test"]:
        m = case_of == c
        idx = np.flatnonzero(m)
        cal = idx[pos[m] < N_CAL]
        beats = [b for i in cal for b in beats_of(Y[i], pk_list[i])]
        t8 = build_template_a(np.stack(beats))[0] if len(beats) >= 5 else g
        s_val = float(np.median([1 - np.corrcoef(b, t8)[0, 1] for b in beats if b.std() > 0])) if len(beats) >= 5 else np.nan
        for i in idx[pos[m] >= N_CAL]:
            tb = beats_of(Y[i], pk_list[i])
            if not tb:
                continue
            cc = lambda t: float(np.nanmean([np.corrcoef(t, b)[0, 1] for b in tb if b.std() > 0]))  # noqa: E731
            tmpl_global[i], tmpl_k8[i], sbv[i] = cc(g), cc(t8), s_val
    res["GLOBAL_tmpl"], res["CAL32s_tmpl"] = tmpl_global, tmpl_k8

    sr1 = {a: dict(np.load(ROOT / f"outputs/sr1_eval/arm_{a}_seed42.npz")) for a in ("C", "I", "S")}
    assert all(np.array_equal(d["pid"], pid) for d in sr1.values())
    scal = {"HR": {"C50": sr1["C"]["nfe50_HR"], "C1": sr1["C"]["nfe1_HR"], "I1": sr1["I"]["nfe1_HR"], "S1": sr1["S"]["nfe1_HR"],
                   "I1_K16": sr1["I"]["cons16_nfe1_hr_err"]},
            "Rpeak_F1": {k: sr1[k[0]][f"nfe{v[1]}_Rpeak_F1"] for k, v in GEN.items()},
            "morph_corr@GT": res}

    subs = np.unique(pid)
    tert_edges = np.nanpercentile(np.array([np.nanmean(sbv[pid == s]) for s in subs]), [33.33, 66.67])

    def per_patient(v, mask=None):
        m = np.ones(len(pid), bool) if mask is None else mask
        return np.array([np.nanmean(v[(pid == s) & m]) for s in subs])

    def winrate(diff_better, n=2000, seed=20260911):
        rng = np.random.default_rng(seed)
        ok = np.isfinite(diff_better)
        d = diff_better[ok]
        draws = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
        return float(np.mean(d)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    rows, verdict = [], {}
    sbv_pp = per_patient(sbv)
    groups = {"all": np.ones(len(subs), bool), "SBV low": sbv_pp <= tert_edges[0],
              "SBV mid": (sbv_pp > tert_edges[0]) & (sbv_pp <= tert_edges[1]), "SBV high": sbv_pp > tert_edges[1]}
    for metric, arms in scal.items():
        lower = metric == "HR"
        mask = aligned if metric == "morph_corr@GT" else None
        ref = per_patient(arms["C50"], mask)
        for name, v in arms.items():
            if name == "C50":
                continue
            x = per_patient(v, mask)
            better = (x < ref) if lower else (x > ref)
            for gname, gm in groups.items():
                wr = winrate(better[gm].astype(float))
                mean_diff = ci((v - arms["C50"])[mask if mask is not None else slice(None)],
                               pid[mask] if mask is not None else pid) if v.shape == arms["C50"].shape else (np.nan,) * 3
                rows.append(dict(metric=metric, arm=name, group=gname, n_patients=int(gm.sum()),
                                 value=float(np.nanmean(x[gm])), ref=float(np.nanmean(ref[gm])),
                                 win_rate=wr[0], wr_lo=wr[1], wr_hi=wr[2],
                                 mean_diff=mean_diff[0], md_lo=mean_diff[1], md_hi=mean_diff[2]))
                if gname == "all":
                    verdict[f"{metric}|{name}"] = ("WINS FOR MOST PATIENTS" if wr[1] > 0.5 else
                                                   "LOSES FOR MOST PATIENTS" if wr[2] < 0.5 else "SPLIT")
    import csv
    with open(OUT / "per_patient.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "verdict.json").write_text(json.dumps({"verdict": verdict, "sbv_tertile_edges": [float(x) for x in tert_edges],
                                                  "n_patients": len(subs), "n_aligned_windows": int(aligned.sum())}, indent=1))
    np.savez_compressed(OUT / "per_window.npz", pid=pid, aligned=aligned, sbv=sbv, **{f"morph_{k}": v for k, v in res.items()})
    print(json.dumps(verdict, indent=1))
    for r in rows:
        if r["group"] == "all":
            print(f"{r['metric']:<14} {r['arm']:<8} win {r['win_rate']:.3f} [{r['wr_lo']:.3f}, {r['wr_hi']:.3f}]  value {r['value']:.4f} vs C50 {r['ref']:.4f}")


if __name__ == "__main__":
    main()
