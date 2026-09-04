"""VitalDB bulk download: SNUADC/PLETH + SNUADC/ECG_II per case, raw 500 Hz float32, one npz per case.

Data acquisition only. No preprocessing decision is baked in: signals are stored as delivered by the API.
"""
from __future__ import annotations
import io, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/VitalDB/cases"; OUT.mkdir(parents=True, exist_ok=True)
TRACKS = ("SNUADC/PLETH", "SNUADC/ECG_II")
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
S = requests.Session()

def fetch(tid: str) -> np.ndarray | None:
    for attempt in range(4):
        try:
            r = S.get(f"https://api.vitaldb.net/{tid}", timeout=180)
            if r.status_code != 200 or not r.content:
                time.sleep(2 * (attempt + 1)); continue
            df = pd.read_csv(io.BytesIO(r.content))
            return df.iloc[:, 1].to_numpy(dtype=np.float32)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None

def one_case(caseid: int, tids: dict) -> tuple[int, str, int]:
    p = OUT / f"case_{caseid:05d}.npz"
    if p.exists() and p.stat().st_size > 1024:
        return caseid, "skip", p.stat().st_size
    arrs = {}
    for name, tid in tids.items():
        a = fetch(tid)
        if a is None or a.size == 0:
            return caseid, "fail", 0
        arrs[name.split("/")[-1]] = a
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, srate=np.float32(500.0), caseid=np.int32(caseid), **arrs)
    tmp.rename(p)
    return caseid, "ok", p.stat().st_size

def main() -> int:
    trks = pd.read_csv(ROOT / "data/raw/VitalDB/trks.csv")
    trks = trks[trks.tname.isin(TRACKS)]
    per = {}
    for cid, g in trks.groupby("caseid"):
        d = dict(zip(g.tname, g.tid))
        if all(t in d for t in TRACKS):
            per[int(cid)] = d
    todo = sorted(per)
    print(f"[vitaldb] {len(todo)} cases with both tracks | workers={WORKERS}", flush=True)
    ok = skip = fail = 0; nbytes = 0; t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one_case, c, per[c]): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            cid, st, sz = f.result()
            ok += st == "ok"; skip += st == "skip"; fail += st == "fail"; nbytes += sz
            if i % 25 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"[vitaldb] {i}/{len(todo)} ok={ok} skip={skip} fail={fail} "
                      f"{nbytes/1e9:.1f} GB {el/60:.1f} min "
                      f"eta {(el/max(i,1))*(len(todo)-i)/60:.0f} min", flush=True)
    print(f"[vitaldb] DONE ok={ok} skip={skip} fail={fail} total={nbytes/1e9:.1f} GB", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
