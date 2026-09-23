"""PART B — PPGFlowECG checkpoint provenance audit (no inference). Reproduces every check of
docs/PPGFLOWECG_PROVENANCE_AUDIT.md from primary sources and writes artifacts/pfe_provenance/provenance.json.

Safety: each checkpoint's pickle is first parsed STATICALLY (pickletools opcodes, nothing executed); only if every global
it references is a torch / collections type is it loaded with torch.load(weights_only=True) (restricted unpickler).

Run: .venv/bin/python scripts/pfe_provenance.py
"""
from __future__ import annotations

import hashlib
import io
import json
import pickletools
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "data/pretrained/ppgflowecg"
OUT = ROOT / "artifacts/pfe_provenance"
HF = "XiaochengFang/PPGFlowECG"
GH = "https://github.com/PKUDigitalHealth/PPGFlowECG.git"
GH_COMMIT = "56b2cd2cfa738388c60daccd788d511aa8698085"
REPO = ROOT / "outputs/pfe_audit_repo"                      # audit clone (gitignored)
SAFE = re.compile(r"^(collections OrderedDict|torch \w+Storage|torch\._utils _rebuild_\w+|torch Size|builtins (set|frozenset)|_codecs encode)$")
CORPUS = re.compile(r"mcmed|mc-med|vitaldb|bidmc|mimic|afib|capno|dalia|wesad|ptb", re.I)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def static_pickle(p):
    if not zipfile.is_zipfile(p):
        return {"format": "legacy (non-zip) torch file", "all_globals_safe": False}
    z = zipfile.ZipFile(p)
    names = z.namelist()
    pk = [n for n in names if n.endswith("data.pkl")][0]
    globs, strs, last = [], [], []
    for op, arg, _ in pickletools.genops(io.BytesIO(z.read(pk))):
        if op.name == "GLOBAL":
            globs.append(arg.replace("\n", " ").strip())
        elif "UNICODE" in op.name or op.name in ("SHORT_BINSTRING", "BINSTRING", "STRING"):
            strs.append(str(arg)); last = (last + [str(arg)])[-2:]
        elif op.name == "STACK_GLOBAL":
            globs.append(" ".join(last))
    meta = sorted({s for s in strs if "." not in s and not s.isdigit()})
    return {"zip_entries": len(names), "non_tensor_entries": [n for n in names if "/data/" not in n],
            "globals": sorted(set(globs)), "all_globals_safe": all(SAFE.match(g) for g in set(globs)),
            "n_string_constants": len(strs), "non_dotted_strings": meta,
            "corpus_like_strings": sorted({s for s in strs if CORPUS.search(s)}),
            "path_like_strings": sorted({s for s in strs if "/" in s or "\\" in s})[:50]}


def describe(obj, depth=0):
    if isinstance(obj, dict):
        return {str(k): describe(v, depth + 1) if depth < 1 else type(v).__name__ for k, v in list(obj.items())[:40]}
    if torch.is_tensor(obj):
        return f"tensor{tuple(obj.shape)}" if obj.numel() > 1 else obj.item()
    return obj if isinstance(obj, (int, float, str, bool, type(None))) else type(obj).__name__


def hf(path):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{HF}{path}", timeout=60) as r:
        return json.loads(r.read())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"hf_repo": HF, "github": GH, "github_commit": GH_COMMIT, "files": {}}
    info = hf("?blobs=true")
    res["hf_sha"], res["hf_created"], res["hf_last_modified"], res["hf_card_data"] = info.get("sha"), info.get("createdAt"), info.get("lastModified"), info.get("cardData")
    lfs = {s["rfilename"]: (s.get("size"), (s.get("lfs") or {}).get("sha256")) for s in info.get("siblings", [])}
    res["hf_commits"] = [{"id": c["id"], "date": c["date"], "title": c["title"], "message": c.get("message", "")} for c in hf("/commits/main")]
    res["hf_trees"] = {c["id"][:12]: [x["path"] for x in hf(f"/tree/{c['id']}")] for c in res["hf_commits"]}
    for fn in ("checkpoint-10.pt", "VAE-iter-40000.pth"):
        p = CK / fn
        r = {"size_local": p.stat().st_size if p.exists() else None, "size_hf": lfs.get(fn, (None,))[0], "sha256_hf": lfs.get(fn, (None, None))[1]}
        if p.exists() and r["size_local"] == r["size_hf"]:
            r["sha256_local"] = sha256(p)
            r["sha256_match"] = r["sha256_local"] == r["sha256_hf"]
            r["static"] = st = static_pickle(p)
            if st["all_globals_safe"]:
                ck = torch.load(p, map_location="cpu", weights_only=True)
                r["top_level"] = describe(ck)
                if "model" in ck:
                    r["n_params_model"] = int(sum(t.numel() for t in ck["model"].values()))
                del ck
        else:
            r["note"] = "not fully downloaded"
        res["files"][fn] = r
    if not REPO.exists():
        subprocess.run(["git", "clone", "-q", GH, str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "-q", GH_COMMIT], check=True)
    res["github_log"] = subprocess.run(["git", "-C", str(REPO), "log", "--format=%h %ad %s", "--date=short"], capture_output=True, text=True).stdout.splitlines()
    grep = subprocess.run(["grep", "-rn", "-i", "-E", r"checkpoint-10|VAE-iter|mcmed|mc-med|vitaldb|bidmc|mimic|afib|results_folder|save_dir|dataset_dir|saved_dir|split|pretrain",
                           "--include=*.py", "--include=*.yaml", "--include=*.md", "."], cwd=REPO, capture_output=True, text=True).stdout.splitlines()
    res["github_grep"] = grep
    hist = []
    for c in subprocess.run(["git", "-C", str(REPO), "log", "--format=%h", "--", "README.md"], capture_output=True, text=True).stdout.split():
        txt = subprocess.run(["git", "-C", str(REPO), "show", f"{c}:README.md"], capture_output=True, text=True).stdout
        hist.append({"commit": c, "mentions_training_corpus_of_released_weights": bool(re.search(r"(trained|pre-?trained)[^.\n]{0,80}(mcmed|vitaldb|bidmc|mimic)", txt, re.I)),
                     "checkpoint_lines": [l for l in txt.splitlines() if re.search(r"checkpoint|weights|huggingface|baidu", l, re.I)]})
    res["readme_history"] = hist
    (OUT / "provenance.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("github_grep", "readme_history")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
