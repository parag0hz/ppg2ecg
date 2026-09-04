"""CapnoBase (Borealis Dataverse doi:10.5683/SP2/NLB8IT) — per-file download with retry.

The bulk-zip endpoint truncates, so each file is fetched individually and size-verified.
"""
from __future__ import annotations
import sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/CapnoBase/files"; OUT.mkdir(parents=True, exist_ok=True)
DOI = "doi:10.5683/SP2/NLB8IT"
API = "https://borealisdata.ca/api"
S = requests.Session()


def get_one(f):
    df = f["dataFile"]
    name = df.get("originalFileName") or df["filename"]
    want = int(df["filesize"])
    p = OUT / name
    if p.exists() and p.stat().st_size == want:
        return name, "skip", p.stat().st_size
    for a in range(4):
        try:
            r = S.get(f"{API}/access/datafile/{df['id']}", params={"format": "original"}, timeout=300)
            if r.status_code == 200 and len(r.content) > 0:
                p.write_bytes(r.content)
                return name, ("ok" if p.stat().st_size == want else f"size {p.stat().st_size}!={want}"), p.stat().st_size
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    return name, "fail", 0


def main() -> int:
    d = S.get(f"{API}/datasets/:persistentId/", params={"persistentId": DOI}, timeout=120).json()["data"]
    files = d["latestVersion"]["files"]
    print(f"[capno] {len(files)} files, {sum(f['dataFile']['filesize'] for f in files)/1e6:.1f} MB", flush=True)
    ok = bad = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(get_one, f) for f in files]):
            n, st, sz = fut.result()
            ok += st in ("ok", "skip"); bad += st not in ("ok", "skip")
            if bad and st not in ("ok", "skip"):
                print(f"  [capno] {n}: {st}", flush=True)
    tot = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"[capno] DONE ok={ok} bad={bad} total={tot/1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
