"""R2 pre-training preparation (prereg f954e07 sections 2, 3, 6, 7, 9): checkpoint manifests, the trainable
set, the SHUFFLE manifest for the train / eval / viz populations, the ORACLE training-field cache, and the
loader-order provenance. No optimizer step. Validation data is touched only for metadata (site,
window_index) and for the eval-population shuffle descriptive (GT R-peaks of the frozen 2,048 windows).
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r2_rhythm_transfer"
MANIFEST, PROCESSED = "data/manifests/split_a4_wildppg_seed42.json", "data/processed/wildppg_8s"
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024
V1_MANIFEST = ROOT / "artifacts/v1_stepwise_visualization/cohort_manifest.csv"
SITES = ("sternum", "head", "wrist", "ankle")


def wcsv(p, rows):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    split = read_manifest(ROOT / MANIFEST)[0]
    ER.assert_no_test_subjects(list(split["train"]) + list(VAL))
    git = git_sha(ROOT)
    t_all = time.perf_counter()

    # ---- checkpoint manifests + trainable set (CPU) ----
    net, _ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, torch.device("cpu"))
    _tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, torch.device("cpu"))
    RT.assert_only_adapter_trainable(net)
    (ART / "generator_checkpoint_manifest.json").write_text(json.dumps(gmeta, indent=2, default=str))
    (ART / "rhythm_checkpoint_manifest.json").write_text(json.dumps(tmeta, indent=2, default=str))
    (ART / "trainable_parameters.json").write_text(json.dumps({
        "trainable": RT.trainable_names(net), "n_trainable": RT.n_trainable(net), "h_dim": gmeta["h_dim"],
        "generator_params_frozen": gmeta["n_params_total"], "rhythm_tcn_params_frozen": tmeta["params"],
        "adapter": "Conv1d(1->h_dim, k=1, bias=False), zero-initialised"}, indent=2))
    print(f"[M] generator {gmeta['state_dict_sha256'][:16]} round {gmeta['round']} | tcn {tmeta['state_dict_sha256'][:16]} | "
          f"trainable {RT.trainable_names(net)} = {RT.n_trainable(net)}", flush=True)
    del net

    # ---- SHUFFLE manifest: train / eval / viz ----
    rows = []
    offsets, off = {}, 0
    tr_sub, tr_site, tr_wi, tr_y = [], [], [], []
    for s in split["train"]:
        d = np.load(ROOT / PROCESSED / f"{s}.npz")
        S, W, Y = np.asarray(d["site"]).astype(str), d["window_index"].astype(np.int64), d["y"]
        offsets[s] = off; off += len(S)
        tr_sub += [s] * len(S); tr_site.append(S); tr_wi.append(W); tr_y.append(Y)
    tr_sub, tr_site, tr_wi = np.array(tr_sub), np.concatenate(tr_site), np.concatenate(tr_wi)
    partner = RT.shuffle_partner(tr_sub, tr_site, tr_wi)
    RT.assert_derangement(partner)
    apos = np.concatenate([np.arange(np.sum(tr_sub == s)) for s in split["train"]])
    for i in range(len(tr_sub)):
        j = int(partner[i])
        rows.append({"population": "train", "subject": tr_sub[i], "site": tr_site[i], "window_index": int(tr_wi[i]),
                     "array_pos": int(apos[i]), "partner_window_index": int(tr_wi[j]), "partner_array_pos": int(apos[j]),
                     "train_row": i, "partner_train_row": j, "pop_row": "", "partner_pop_row": ""})
    print(f"[S] train shuffle: {len(tr_sub)} rows, {len(set(zip(tr_sub, tr_site)))} strata, bijective, no fixed point", flush=True)

    ev_sub, ev_site, ev_wi, ev_pos, ev_y = [], [], [], [], []
    for s in VAL:
        d = np.load(ROOT / PROCESSED / f"{s}.npz")
        S, W, Y = np.asarray(d["site"]).astype(str), d["window_index"].astype(np.int64), d["y"]
        idx = ER.select_subset(SALT, s, len(S), TAKE)
        ev_sub += [s] * len(idx); ev_site.append(S[idx]); ev_wi.append(W[idx]); ev_pos.append(idx); ev_y.append(Y[idx])
    ev_sub, ev_site, ev_wi, ev_pos = np.array(ev_sub), np.concatenate(ev_site), np.concatenate(ev_wi), np.concatenate(ev_pos)
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s in VAL:
        if ev_pos[ev_sub == s].tolist() != list(frozen[s]):
            raise RuntimeError(f"frozen subset mismatch for {s}")
    p_ev = RT.shuffle_partner(ev_sub, ev_site, ev_wi)
    RT.assert_derangement(p_ev)
    for i in range(len(ev_sub)):
        j = int(p_ev[i])
        rows.append({"population": "eval", "subject": ev_sub[i], "site": ev_site[i], "window_index": int(ev_wi[i]),
                     "array_pos": int(ev_pos[i]), "partner_window_index": int(ev_wi[j]), "partner_array_pos": int(ev_pos[j]),
                     "train_row": "", "partner_train_row": "", "pop_row": i, "partner_pop_row": j})
    # free descriptive: beat-count / mean-RR differences between partners (GT peaks of the eval windows)
    Yev = np.concatenate(ev_y).astype(np.float64)
    pk = [R.detect_rpeaks(Yev[i], 128) for i in range(len(Yev))]
    nb = np.array([len(p) for p in pk], float)
    rr = np.array([np.mean(np.diff(p)) / 128 * 1000 if len(p) >= 2 else np.nan for p in pk])
    dnb, drr = np.abs(nb[p_ev] - nb), np.abs(rr[p_ev] - rr)
    desc = {"n": int(len(ev_sub)), "abs_beat_count_diff_mean": float(dnb.mean()), "abs_beat_count_diff_median": float(np.median(dnb)),
            "frac_same_beat_count": float(np.mean(dnb == 0)), "abs_mean_rr_diff_ms_median": float(np.nanmedian(drr)),
            "abs_mean_rr_diff_ms_p90": float(np.nanpercentile(drr, 90))}
    print(f"[S] eval shuffle: {desc}", flush=True)

    viz = [r for r in csv.DictReader(open(V1_MANIFEST)) if r["cohort"] == "viz" and r["split"] == "val"]
    vz_sub, vz_site, vz_wi, vz_pos = (np.array([r["subject"] for r in viz]), np.array([r["site"] for r in viz]),
                                      np.array([int(r["window_index"]) for r in viz]), np.array([int(r["array_pos"]) for r in viz]))
    if len(viz) != 64:
        raise RuntimeError(f"V1 val VIZ cohort must have 64 windows, got {len(viz)}")
    p_vz = RT.shuffle_partner(vz_sub, vz_site, vz_wi)
    RT.assert_derangement(p_vz)
    for i in range(len(viz)):
        j = int(p_vz[i])
        rows.append({"population": "viz", "subject": vz_sub[i], "site": vz_site[i], "window_index": int(vz_wi[i]),
                     "array_pos": int(vz_pos[i]), "partner_window_index": int(vz_wi[j]), "partner_array_pos": int(vz_pos[j]),
                     "train_row": "", "partner_train_row": "", "pop_row": i, "partner_pop_row": j})
    wcsv(ART / "shuffle_manifest.csv", rows)
    (ART / "shuffle_eval_descriptive.json").write_text(json.dumps(desc, indent=2))
    print(f"[S] viz shuffle: 64 rows, 8 strata of 8", flush=True)

    # ---- ORACLE training-field cache (all 293,271 rows; GT-R leakage by design; CPU) ----
    t0 = time.perf_counter()
    Ytr = np.concatenate(tr_y)
    field = RT.oracle_fields(Ytr, workers=12)
    del Ytr
    sha = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
    np.savez(ART / "_cache_oracle_train.npz", field=field, n_rows=np.int64(len(field)))
    build_s = time.perf_counter() - t0
    (ART / "cache_build.json").write_text(json.dumps({"oracle_cache_rows": int(len(field)), "oracle_cache_sha256": sha,
                                                       "oracle_cache_build_s": build_s, "field_mean": float(field.mean()),
                                                       "rows_without_beats": int(np.sum(field.max(axis=1) == 0)),
                                                       "detector": "rpeaks.detect_rpeaks(neurokit) + rhythm_tcn.soft_event_field sigma 12.8"}, indent=2))
    print(f"[O] oracle cache {field.shape} sha256 {sha[:16]} in {build_s:.0f} s (CPU, outside the GPU-hour rule)", flush=True)
    del field

    # ---- loader-order provenance from the permutation alone (same generator rule as the driver) ----
    gen = torch.Generator(); gen.manual_seed(RT.SEED)
    loader = DataLoader(TensorDataset(torch.arange(len(tr_sub))), batch_size=RT.BATCH, shuffle=True, generator=gen)
    it = iter(loader); vis = np.zeros(len(tr_sub), dtype=bool); order = []
    for _ in range(RT.STEPS):
        (b,) = next(it); b = b.numpy(); order.append(b); vis[b] = True
    order = np.concatenate(order)
    lp = {"steps": RT.STEPS, "windows_visited": int(vis.sum()), "no_window_twice": bool(len(order) == len(np.unique(order))),
          "visits_per_subject_site": {f"{s}/{st}": int(np.sum(vis & (tr_sub == s) & (tr_site == st))) for s in split["train"] for st in SITES},
          "noisy_ecg_subject_share": float(np.sum(vis & np.isin(tr_sub, ("fex", "p5d"))) / vis.sum()),
          "first_batch_sha256": hashlib.sha256(order[:RT.BATCH].tobytes()).hexdigest()}
    (ART / "loader_order_provenance.json").write_text(json.dumps(lp, indent=2))
    (ART / "build_provenance.json").write_text(json.dumps({"git": git, "prereg": "f954e07", "utc": datetime.now(timezone.utc).isoformat(),
                                                            "wall_s": time.perf_counter() - t_all, "test_subjects_loaded": []}, indent=2))
    print(f"[done] caches built in {time.perf_counter()-t_all:.0f} s; visited {lp['windows_visited']} windows, noisy share {lp['noisy_ecg_subject_share']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
