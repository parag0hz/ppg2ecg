"""Build the 112 ChatGPT-review contact sheets by LOSSLESS TILING of the frozen V1 PNGs.

Each source PNG is pasted at 1:1 — no crop, no resample, no aspect change, no re-plot. The frozen
matplotlib output remains the review source of truth. Cohort, window selection and NFE traces are untouched.
"""
from __future__ import annotations

import csv, hashlib, re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ppg2ecg.evaluation import v1_timing as V

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/v1_stepwise_visualization"
OUT = A / "chatgpt_review"; OUT.mkdir(parents=True, exist_ok=True)
FIGS, ZOOMS = A / "figures", A / "beat_zooms"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLS, GUT, PAD, HDR, CAP = 2, 26, 26, 150, 46


def f(sz):
    return ImageFont.truetype(FONT, sz)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sheet(paths, title, sub, out: Path):
    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = ims[0].size
    assert all(i.size == (w, h) for i in ims), "source PNGs differ in size; tiling would distort"
    rows = -(-len(ims) // COLS)
    W = PAD * 2 + COLS * w + (COLS - 1) * GUT
    H = PAD * 2 + HDR + rows * (h + CAP) + (rows - 1) * GUT
    cv = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(cv)
    d.rectangle([0, 0, W, HDR + PAD // 2], fill="#101820")
    d.text((PAD, 26), title, font=f(64), fill="white")
    d.text((PAD, 100), sub, font=f(30), fill="#9fd0ff")
    for k, (im, p) in enumerate(zip(ims, paths)):
        r, c = divmod(k, COLS)
        x = PAD + c * (w + GUT)
        y = PAD + HDR + r * (h + CAP + GUT)
        d.text((x, y + 8), Path(p).stem, font=f(30), fill="#101820")
        cv.paste(im, (x, y + CAP))                       # 1:1 paste, no resize
        d.rectangle([x - 1, y + CAP - 1, x + w, y + CAP + h], outline="#c8c8c8")
    cv.save(out, "PNG", optimize=True)
    return cv.size, out.stat().st_size


def main() -> int:
    rows = []
    for sub in V.VAL + V.TRAIN:
        split = "VAL" if sub in V.VAL else "TRAIN"
        for site in V.SITES:
            for kind, srcdir, pat, suffix in (("stepwise", FIGS, f"{sub}_{site}_w*.png", "stepwise"),
                                              ("rzoom", ZOOMS, f"{sub}_{site}_w*_zoom.png", "rzoom")):
                ps = sorted(srcdir.glob(pat), key=lambda p: int(re.search(r"_w(\d+)", p.stem).group(1)))
                if len(ps) != 8:
                    print(f"[!] {sub}/{site}/{kind}: found {len(ps)} (expected 8) — skipped"); continue
                wi = [int(re.search(r"_w(\d+)", p.stem).group(1)) for p in ps]
                name = f"{split}_{sub}_{site}" + ("" if kind == "stepwise" else "_RZOOM")
                out = OUT / f"{split}_{sub}_{site}_{suffix}.png"
                size, nbytes = sheet(ps, name,
                                     f"subject: {sub}   split: {split.lower()}   site: {site}   "
                                     f"NFE 1/2/4/8/50, source seed 0   windows: {', '.join(map(str, wi))}", out)
                rows.append({"figma_frame_name": name, "subject": sub, "split": split.lower(), "site": site,
                             "type": kind, "source_png": ";".join(p.name for p in ps),
                             "window_indices": ";".join(map(str, wi)),
                             "sheet_png": out.name, "sheet_px": f"{size[0]}x{size[1]}",
                             "sheet_bytes": nbytes, "sha256": sha256(out),
                             "source_sha256": ";".join(sha256(p)[:16] for p in ps)})
                print(f"[S] {out.name:38s} {size[0]}x{size[1]}  {nbytes/1e6:.2f} MB", flush=True)
    for g, src in (("GROUP_NFE_METRICS", "all_subject_nfe_metrics.png"),
                   ("GROUP_VAL_NFE_SUMMARY", "val_only_nfe_summary.png"),
                   ("GROUP_R_PPG_DELAY", "all_subject_r_ppg_delay.png"),
                   ("GROUP_SITE_EFFECT", "site_delay_boxplot.png"),
                   ("GROUP_DELAY_HEATMAP", "subject_site_delay_heatmap.png"),
                   ("GROUP_TIMING_PRIOR", "timing_prior_coverage.png")):
        p = FIGS / src
        if not p.exists():
            continue
        dst = OUT / f"{g}.png"
        dst.write_bytes(p.read_bytes())                   # byte-identical copy
        rows.append({"figma_frame_name": g, "subject": "-", "split": "-", "site": "-", "type": "group",
                     "source_png": src, "window_indices": "-", "sheet_png": dst.name,
                     "sheet_px": "x".join(map(str, Image.open(dst).size)),
                     "sheet_bytes": dst.stat().st_size, "sha256": sha256(dst),
                     "source_sha256": sha256(p)[:16]})
    with open(A / "figma_review_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_step = sum(r["type"] == "stepwise" for r in rows)
    n_zoom = sum(r["type"] == "rzoom" for r in rows)
    n_grp = sum(r["type"] == "group" for r in rows)
    mx = max(r["sheet_bytes"] for r in rows) / 1e6
    print(f"\n[done] stepwise {n_step}  rzoom {n_zoom}  group {n_grp}  total {len(rows)}  max {mx:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
