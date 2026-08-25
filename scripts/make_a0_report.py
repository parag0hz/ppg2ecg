"""Generate docs/A0_PENGUIN_REPRODUCTION_REPORT.md from the A0 output directory (provenance, training log, metrics, NFE curve).
Narrative sections are spliced from <out>/analysis.md if present (headings must match the report's '## ' titles)."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_HR = 15.64
MARGINS = {"rpeak_f1": ("drop", 0.02), "hr_abs_err_bpm": ("rise", 1.0), "morph_corr": ("drop", 0.05), "qrs_width_err_ms": ("rise", 10.0)}


def f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def verdict(hr):
    return "PASS" if hr <= 17.2 else "BORDERLINE" if hr <= 20 else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/a0_penguin_otcfm_ppgdalia_8s_seed42")
    ap.add_argument("--report", default="docs/A0_PENGUIN_REPRODUCTION_REPORT.md")
    args = ap.parse_args()
    out = ROOT / args.out_dir
    prov = json.loads((out / "provenance.json").read_text())
    cfg = prov["config"]
    log = list(csv.DictReader(open(out / "training_log.csv")))
    summ = json.loads((out / "training_summary.json").read_text()) if (out / "training_summary.json").exists() else {}
    met = json.loads((out / "metrics.json").read_text())
    rows = list(csv.DictReader(open(out / "nfe_curve.csv")))
    analysis = {}
    if (out / "analysis.md").exists():
        for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", (out / "analysis.md").read_text(), re.S | re.M):
            analysis[m.group(1).strip()] = m.group(2).strip()

    def narrative(title, default="_(to be written after results are inspected)_"):
        return analysis.get(title, default)

    ref = max(rows, key=lambda r: int(r["actual_NFE"]))  # reference = highest-NFE arm (Heun 25 steps = 50 NFE in the full run)
    ref_hr = float(ref["hr_abs_err_bpm"])
    best = int(summ.get("best_epoch", -1)) + 1
    L = []
    L.append("# A0 PENGUIN Reproduction Report\n")
    L.append(f"Experiment `{cfg['exp_name']}` — generated {met['created']} from `{args.out_dir}/`. Single seed (42). No new method; baseline gate only.\n")
    L.append("## Frozen protocol")
    L.append(f"- Model: {cfg['model']}; cfg `{cfg['model_cfg']}`; params total {prov['model_params']['total']:,} (effective {prov['model_params']['effective']:,}; dead `cross_attn`/`revin` excluded)")
    L.append("- Objective: OT-CFM (Lipman conditional-OT path, σ=0, independent coupling), t ~ U(0,1), MSE on velocity — upstream `train_flow`/`optimize` unchanged")
    L.append(f"- Train/val sampler: Heun {cfg['sampler_train_val']['steps']} steps = **{cfg['sampler_train_val']['nfe']} NFE**")
    L.append(f"- Optimiser {cfg['optimizer']}, lr {cfg['lr']}, weight decay {cfg['weight_decay']}, batch {cfg['batch_size']}, ≤ {cfg['epochs_max']} epochs, early stopping on {cfg['early_stopping']['metric']} (patience {cfg['early_stopping']['patience']}), checkpoint = {cfg['checkpoint']}")
    L.append(f"- Precision: {cfg['precision']['dtype']}, AMP {cfg['precision']['amp']}, BF16 {cfg['precision']['bf16']}, TF32 matmul {cfg['precision']['tf32_matmul']}; deterministic: {cfg['deterministic']}")
    L.append("- Pre-registration: `docs/PREREGISTRATION_V0.md` (§6 window/split/seed decisions fixed before training)\n")
    L.append("## Git / environment provenance")
    L.append(f"- Repo commit `{prov['git']['commit']}` (branch {prov['git']['branch']}, {len(prov['git']['dirty_files'])} dirty files at preflight)")
    L.append(f"- Upstream PENGUIN `{prov['upstream']['commit']}` (expected `{prov['upstream']['expected']}`, dirty {len(prov['upstream']['dirty_files'])}) — {prov['upstream']['url']}")
    hw = prov["hardware"]
    L.append(f"- GPU {hw['gpu']['name']} ({hw['gpu']['total_MiB']/1024:.1f} GiB, sm_{hw['gpu']['capability'][0]}{hw['gpu']['capability'][1]}), torch {hw['torch']} / CUDA {hw['cuda']} / cuDNN {hw['cudnn']}, Python {hw['python']}")
    L.append(f"- Full provenance: `{args.out_dir}/provenance.json`\n")
    L.append("## Dataset")
    pm = prov["dataset"]["processed_manifest"]
    L.append(f"- PPG-DaLiA (UCI #495, CC BY 4.0); raw zip sha256 `{prov['dataset']['raw_checksums_sha256'][0].split()[0][:16]}…` (`data/raw/CHECKSUMS.sha256`)")
    L.append(f"- Processed `{cfg['window_s']} s` windows @ {cfg['sample_rate']} Hz = {pm['samples_per_window']} samples, {pm['total_windows']} windows total, built {pm['built']}; per-file sha256 in provenance")
    L.append(f"- Preprocessing (PENGUIN-faithful, bit-exact vs upstream `preprocess.py` at 8 s for 15/15 subjects): PPG {cfg['preprocess']['ppg']}, ECG {cfg['preprocess']['ecg']}, all statistics per window\n")
    L.append("## Split")
    sp = cfg["split"]
    L.append(f"- Manifest `{sp['manifest']}` (sha256 `{sp['sha256'][:16]}…`), protocol {sp['protocol']} seed {sp['seed']}")
    L.append(f"- train ({len(sp['train'])}): {', '.join(sp['train'])}; val: {', '.join(sp['val'])}; test: {', '.join(sp['test'])}")
    nw = prov["n_windows"]
    L.append(f"- windows: train {nw['train']}, val {nw['val']}, test {nw['test']}")
    lc = prov["leakage_checks"]
    L.append(f"- Leakage checks at preflight: subject-disjoint {lc['subject_disjoint']['ok']} {lc['subject_disjoint']['overlaps']}; window-hash-disjoint {lc['window_disjoint']['ok']} {lc['window_disjoint']['overlaps']}; window-local normalisation PPG {lc['windowwise_normalization_ppg']['ok']} / ECG {lc['windowwise_normalization_ecg']['ok']}")
    L.append("- Limitation: upstream's own split is glob-order dependent (on this machine it would be val S4 / test S10); the paper's held-out subject is unknown, so this is **not** guaranteed to be the paper's split.\n")
    L.append("## Window-length decision")
    L.append("8 s windows were fixed **before training** from the audit (`docs/PENGUIN_AUDIT.md` §5/§20, `PREREGISTRATION_V0.md` §6): with the shipped 4 s config the upstream HR metric compresses the 8 s evaluation window 2× (true 60 bpm → 119.7), masks high-HR windows and zero-fills failures; upstream's `sample_num=16181` equals the number of 8 s windows exactly; the paper does not state the window length (a figure caption mentions 4 s). The 4 s configuration remains a documented ambiguity and was not trained in this stage.\n")
    L.append("## Training")
    last = log[-1]
    L.append(f"- Epochs run: {summ.get('epochs_run', len(log))} (max {cfg['epochs_max']}); best epoch **{best}**; early stopped: {summ.get('early_stopped')}; total training time {float(summ.get('total_train_time_s', 0))/3600:.2f} h; peak GPU memory {float(summ.get('peak_mem_MiB', 0))/1024:.1f} GiB")
    L.append(f"- Best validation MAE (batch-mean, 50-NFE samples, val subject {sp['val'][0]}): {f(summ.get('best_val_mae_batchmean'), 4)}; first-epoch val MAE {f(log[0]['val_mae_batchmean'], 4)}; last-epoch val MAE {f(last['val_mae_batchmean'], 4)}")
    L.append(f"- Train CFM loss: epoch 1 {f(log[0]['train_loss'], 4)} → final {f(last['train_loss'], 4)}; val CFM loss (fixed noise): {f(log[0]['val_cfm_loss'], 4)} → {f(last['val_cfm_loss'], 4)}; LR constant {log[0]['lr']}")
    L.append(f"- Per-epoch time ≈ {np.mean([float(r['epoch_time_s']) for r in log]):.0f} s; full log `{args.out_dir}/training_log.csv`\n")
    L.append("## Paper-vs-code discrepancies")
    L.append("See `docs/PENGUIN_AUDIT.md` §22/§25. Consequential for this run: (1) window length ambiguity (handled: 8 s); (2) paper claims a 6:1:1 subject split, code does 13/1/1 with an unlogged, filesystem-dependent test subject (handled: deterministic manifest, single test subject S2); (3) upstream HR metric pathology (handled: corrected + our own HR error; as-shipped reported only as diagnostic); (4) PPG conditioning is MLP-based, not a linear projection; dead `cross_attn`; (5) no seeds/variance in the paper.\n")
    L.append("## Main result")
    L.append(f"Reference arm {ref['solver'].capitalize()} {ref['solver_steps']} steps (**{ref['actual_NFE']} NFE**) on test subject {sp['test'][0]} ({nw['test']} × 8 s windows):")
    L.append(f"- **corrected HR error (ours) = {ref_hr:.2f} bpm**, R-peak F1 {f(ref['rpeak_f1'])}, RR MAE {f(ref['rr_mae_ms'],1)} ms, QRS-width error {f(ref['qrs_width_err_ms'],1)} ms, beat morphology corr {f(ref['morph_corr'])}, MAE {f(ref['mae'])}, RMSE {f(ref['rmse'])}, PCC {f(ref['pcc'])}")
    L.append(f"- upstream `HeartRateError` corrected {f(ref['hr_err_penguin_corrected'],2)} bpm; **as-shipped (diagnostic only) {f(ref['hr_err_penguin_as_shipped'],2)} bpm**")
    L.append(f"- windows with no detected predicted beats: {100*float(ref['frac_windows_no_pred_beats']):.1f} %\n")
    L.append("## NFE-quality curve")
    L.append("| Solver | Steps | Actual NFE | HR Error (bpm) | R-F1 | RR MAE (ms) | RMSE | PCC | QRS Error (ms) | Morph corr | Latency (ms / batch 64) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['solver'].capitalize()} | {r['solver_steps']} | {r['actual_NFE']} | {f(r['hr_abs_err_bpm'],2)} | {f(r['rpeak_f1'])} | {f(r['rr_mae_ms'],1)} | {f(r['rmse'])} | {f(r['pcc'])} | {f(r['qrs_width_err_ms'],1)} | {f(r['morph_corr'])} | {f(r['latency_ms_batch64'],0)} |")
    L.append("")
    L.append("Pre-registered non-inferiority margins vs the 50-NFE reference (§5: F1 drop > 0.02, HR error rise > 1.0 bpm, morph-corr drop > 0.05, QRS-width error rise > 10 ms):")
    L.append("| Solver | Steps | NFE | ΔF1 | ΔHR (bpm) | Δmorph | ΔQRS (ms) | metrics failing |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        fails = []
        d = {}
        for k, (kind, m) in MARGINS.items():
            delta = float(r[k]) - float(ref[k])
            d[k] = delta
            if (kind == "drop" and -delta > m) or (kind == "rise" and delta > m):
                fails.append(k)
        L.append(f"| {r['solver'].capitalize()} | {r['solver_steps']} | {r['actual_NFE']} | {d['rpeak_f1']:+.3f} | {d['hr_abs_err_bpm']:+.2f} | {d['morph_corr']:+.3f} | {d['qrs_width_err_ms']:+.1f} | {', '.join(fails) or '—'} |")
    L.append(f"\nFigure: `{args.out_dir}/figures/nfe_curve.png`; per-window metrics in `predictions/*.npz`.\n")
    L.append("## Morphology analysis")
    L.append("| Solver | Steps | NFE | HF-energy ratio pred / target | seed std (mean) | seed pairwise corr | cond. gain (bpm) | HR err vs right target (shuffled PPG) | HR err vs wrong target |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['solver'].capitalize()} | {r['solver_steps']} | {r['actual_NFE']} | {f(r['hf_ratio_pred'])} / {f(r['hf_ratio_target'])} | {f(r['seed_std_mean'])} | {f(r['seed_pairwise_corr'])} | {f(r['cond_gain_bpm'],2)} | {f(r['hr_err_shuffled_right_target'],2)} | {f(r['hr_err_shuffled_wrong_target'],2)} |")
    L.append("")
    L.append(narrative("Morphology analysis"))
    L.append(f"\nExample figures (deterministic selection: fixed positions {met.get('examples',{}).get('fixed_positions')} and 10/50/90 % HR-error quantiles of the 50-NFE arm {met.get('examples',{}).get('ref_arm_hr_err_quantiles_10_50_90')}): `{args.out_dir}/figures/example_*.png`\n")
    L.append("## Efficiency")
    L.append("| Solver | Steps | Actual NFE | Latency (ms / batch 64, median) | samples / s | peak GPU mem (MiB) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['solver'].capitalize()} | {r['solver_steps']} | {r['actual_NFE']} | {f(r['latency_ms_batch64'],1)} | {f(r['samples_per_s'],1)} | {f(r['peak_mem_MiB'],0)} |")
    L.append("\nMeasured on the same GPU, fp32, no compile, 3 warm-ups, fixed batch of 64 test windows.\n")
    L.append("## Upstream HR metric pathology")
    L.append("| Solver | Steps | NFE | ours corrected HR err | upstream corrected | upstream as-shipped (4 s path, diagnostic) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['solver'].capitalize()} | {r['solver_steps']} | {r['actual_NFE']} | {f(r['hr_abs_err_bpm'],2)} | {f(r['hr_err_penguin_corrected'],2)} | {f(r['hr_err_penguin_as_shipped'],2)} |")
    L.append("\nThe as-shipped column reproduces upstream's 4 s-config code path on the 8 s windows (2× time compression, high-HR masking, 0.0 fallback). It is reported only to show whether that pathology lands near the paper's number; it is **not** a reproduction metric.\n")
    L.append("## Paper number comparison")
    L.append(f"- Paper (arXiv:2602.03858, Table 1, PPG-DaLiA): HR Error **{PAPER_HR} bpm** (RDDM 16.43, CycleGAN 23.61, RespDiff 22.75, PaPaGei-S 40.89; w/o PPG conditioning 24.40)")
    L.append(f"- Ours, 50 NFE, corrected HR error: **{ref_hr:.2f} bpm** → **{verdict(ref_hr)}** (PASS ≤ 17.2, BORDERLINE ≤ 20, FAIL > 20)")
    L.append(f"- Upstream corrected: {f(ref['hr_err_penguin_corrected'],2)} bpm; upstream as-shipped: {f(ref['hr_err_penguin_as_shipped'],2)} bpm (|Δ paper| = {abs(float(ref['hr_err_penguin_as_shipped'])-PAPER_HR):.2f})")
    L.append("- Caveat: single test subject, single seed, unknown paper split/window length → no binary 'paper reproduced' claim; the verdict is a feasibility gate.\n")
    L.append("## Limitations")
    L.append(narrative("Limitations", "- single seed (42); single test subject (S2) and single val subject (S11) — subject-level variance unknown\n- window length and split not guaranteed to match the paper\n- metrics computed on per-window min-max-normalised ECG (no absolute amplitude)\n- QRS-width proxy (QS-trough) and neurokit R-peak detector are ours, not the paper's\n- upstream HR metric pathology means the paper's number itself may be in doubled units if it was produced at 4 s"))
    L.append("")
    L.append("## GO / NO-GO")
    L.append(narrative("GO / NO-GO"))
    L.append("")
    L.append("## Recommended next experiment")
    L.append(narrative("Recommended next experiment"))
    L.append("")
    (ROOT / args.report).write_text("\n".join(L))
    print("wrote", ROOT / args.report)


if __name__ == "__main__":
    main()
