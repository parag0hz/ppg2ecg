"""Scan MIMIC-III Waveform Matched Subset for records carrying BOTH PLETH and ECG lead II.

Index scan only: fetches the master header and its layout header per record (~2 KB each) and writes an
index CSV. No waveform data is downloaded here.
"""
from __future__ import annotations
import csv, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/MIMIC-AFib"; OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://physionet.org/files/mimic3wdb-matched/1.0"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
S = requests.Session()
A = requests.adapters.HTTPAdapter(pool_connections=WORKERS * 2, pool_maxsize=WORKERS * 2, max_retries=2)
S.mount("https://", A)


def sigs_of(rec: str):
    sub, name = rec.rsplit("/", 1)
    try:
        m = S.get(f"{BASE}/{sub}/{name}.hea", timeout=45)
        if m.status_code != 200:
            return rec, None, None
        lines = [l for l in m.text.splitlines() if l.strip()]
        dur = int(lines[0].split()[3]) if len(lines[0].split()) > 3 else 0
        fs = float(lines[0].split()[2]) if len(lines[0].split()) > 2 else 0.0
        lay = next((l.split()[0] for l in lines[1:] if l.split()[0].endswith("_layout")), None)
        if lay is None:
            return rec, None, None
        h = S.get(f"{BASE}/{sub}/{lay}.hea", timeout=45)
        if h.status_code != 200:
            return rec, None, None
        names = [l.split()[-1] for l in h.text.splitlines()[1:] if l.strip() and len(l.split()) >= 5]
        return rec, names, (fs, dur)
    except Exception:
        return rec, None, None


def main() -> int:
    recs = [l.strip() for l in
            S.get(f"{BASE}/RECORDS-waveforms", timeout=120).text.splitlines() if l.strip()]
    print(f"[mimic] scanning {len(recs)} waveform records with {WORKERS} workers", flush=True)
    rows, both, t0 = [], 0, time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(sigs_of, r) for r in recs]
        for i, f in enumerate(as_completed(futs), 1):
            rec, names, meta = f.result()
            if names:
                b = ("PLETH" in names) and ("II" in names)
                both += b
                rows.append({"record": rec, "has_pleth_and_II": int(b), "fs": meta[0],
                             "n_samples": meta[1], "hours": round(meta[1] / max(meta[0], 1) / 3600, 3),
                             "signals": "|".join(names)})
            if i % 500 == 0 or i == len(recs):
                el = time.perf_counter() - t0
                print(f"[mimic] {i}/{len(recs)} both={both} {el/60:.1f} min "
                      f"eta {(el/max(i,1))*(len(recs)-i)/60:.0f} min", flush=True)
    with open(OUT / "mimic3wdb_matched_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["record", "has_pleth_and_II", "fs", "n_samples", "hours", "signals"])
        w.writeheader(); w.writerows(rows)
    hrs = sum(r["hours"] for r in rows if r["has_pleth_and_II"])
    print(f"[mimic] DONE indexed={len(rows)} both={both} hours(both)={hrs:.0f} "
          f"-> raw16bit 2ch estimate {hrs*3600*125*2*2/1e9:.0f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
