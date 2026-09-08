"""Build the D3 corpora with PENGUIN's preprocessing (docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md §3-§4).

  bidmc_resp_4s   PPG -> respiration, 4 s (60 % 4 == 0, the only lengths the RespRate metric accepts)
  wesad_resp_4s   PPG -> respiration, 4 s
  uci_bp_8s       PPG -> ABP,         8 s, label left in RAW mmHg (no filter, no z-score, no min-max)

Run: .venv/bin/python scripts/build_processed_d3.py --corpus {bidmc_resp,wesad_resp,uci_bp}
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ppg2ecg.data import penguin_tasks as T
from ppg2ecg.data.preprocess import PPG_KW, preprocess_windows

ROOT = Path(__file__).resolve().parents[1]
# label kwargs verbatim from external/PENGUIN/config/preprocess.yaml
RESP_KW = dict(bandpass=True, freq_range=(-1, 1), zscore=True, normalize=True)     # 1 Hz LOW-pass
ABP_KW = dict(bandpass=False, freq_range=(-1, -1), zscore=False, normalize=False)  # raw mmHg
SPEC = {
    "bidmc_resp": dict(seg=4, label_kw=RESP_KW, label="Resp"),
    "wesad_resp": dict(seg=4, label_kw=RESP_KW, label="Resp"),
    "uci_bp": dict(seg=8, label_kw=ABP_KW, label="ABP"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def subjects(corpus: str):
    if corpus == "bidmc_resp":
        return [(p.stem.replace("_Signals", ""), p) for p in T.bidmc_files(ROOT / "data/raw/BIDMC/1.0.0/bidmc_csv") if p.exists()]
    if corpus == "wesad_resp":
        return [(p.parent.name, p) for p in T.wesad_files(ROOT / "data/raw/WESAD")]
    return [(f"uci{i:02d}", i) for i in T.uci_subject_ids()]


def load(corpus: str, key, seg: int):
    if corpus == "bidmc_resp":
        return T.load_bidmc_resp(key, seg)
    if corpus == "wesad_resp":
        return T.load_wesad_resp(key, seg)
    return T.load_uci_abp(ROOT / "data/raw/UCI-BP", key, seg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(SPEC))
    ap.add_argument("--resample-rate", type=int, default=128)
    args = ap.parse_args()
    spec = SPEC[args.corpus]
    seg = spec["seg"]
    out = ROOT / "data/processed" / f"{args.corpus}_{seg}s"
    out.mkdir(parents=True, exist_ok=True)

    files, total, dropped = {}, 0, 0
    for name, key in subjects(args.corpus):
        w = load(args.corpus, key, seg)
        if len(w.ppg) == 0:
            print(f"  {name}: no windows, skipped", flush=True)
            continue
        # the frozen drop rule (scripts/build_processed_wildppg.py): non-finite OR zero-std before, non-finite after
        finite = np.isfinite(w.ppg).all(1) & np.isfinite(w.label).all(1)
        std_ok = (w.ppg.std(1) > 0) & (w.label.std(1) > 0)
        keep = finite & std_ok
        x = preprocess_windows(w.ppg[keep], args.resample_rate, seg, **PPG_KW).astype(np.float32)
        y = preprocess_windows(w.label[keep], args.resample_rate, seg, **spec["label_kw"]).astype(np.float32)
        ok2 = np.isfinite(x).all(1) & np.isfinite(y).all(1)
        x, y, widx = x[ok2], y[ok2], w.window_index[keep][ok2]
        p = out / f"{name}.npz"
        np.savez(p, x=x, y=y, window_index=widx.astype(np.int32),
                 window_start_s=(widx * seg).astype(np.int32), subject=name)
        n_drop = int((~keep).sum() + (~ok2).sum())
        files[name] = {"path": str(p.relative_to(ROOT)), "n_windows": int(len(x)),
                       "n_dropped_nonfinite_or_constant": n_drop, "fs_ppg": w.fs_ppg, "fs_label": w.fs_label,
                       "sha256": sha256(p), "notes": w.notes}
        total += len(x)
        dropped += n_drop
        print(f"  {name}: {x.shape} dropped {n_drop}", flush=True)

    (out / "MANIFEST.json").write_text(json.dumps({
        "built": datetime.now().isoformat(timespec="seconds"), "dataset": args.corpus, "label": spec["label"],
        "segment_len_s": seg, "resample_rate": args.resample_rate, "samples_per_window": args.resample_rate * seg,
        "ppg_preprocess": PPG_KW, "label_preprocess": spec["label_kw"], "dtype": "float32",
        "total_windows": total, "total_dropped": dropped,
        "segment_len_rationale": "prereg §3: PENGUIN's sample_num bookkeeping is 8 s for every dataset, but "
                                 "train.py:43 asserts window_size %% segment_len == 0 and RespRateError has "
                                 "window_size 60, so the respiratory task must use 4 s; ABP uses 8 s.",
        "files": files}, indent=1, default=str))
    print(f"total {total} windows (dropped {dropped}) -> {out}/MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
