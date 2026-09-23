"""Verbatim reuse of official PPGFlowECG functions (external/PPGFlowECG @ 56b2cd2, never edited).

The official modules cannot be imported as released for our purpose: `evaluation/calculate_metric.py` imports
`utils.data`, which does not exist in the repository, and `data_process_to_npz/step1.py` / `step2.py` import `biobss`
at module level. So each needed function is taken **verbatim from the pinned source file** (AST extraction of its exact
source text), executed in a namespace holding only the imports it uses, and the sha256 of every extracted source text is
recorded. Nothing is re-implemented.

Library versions for these functions are the paper's pins, installed in isolated directories:
`outputs/pfe_env/paper_eval` → biosppy 2.2.3, neurokit2 0.1.7, mne 1.8.0 (put on PYTHONPATH only for the processes that
run preprocessing or the Hamilton evaluator).
"""
from __future__ import annotations

import ast
import hashlib
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UP = ROOT / "external/PPGFlowECG"
UP_COMMIT = "56b2cd2cfa738388c60daccd788d511aa8698085"

EVAL_FUNCS = ("get_Rpeaks_ECG", "heartbeats_ecg", "ecg_bpm_array", "MAE_hr", "get_peaks_PPG", "heartbeats_ppg", "ppg_bpm_array")
PREP_METHODS = ("ppg_clean_elgendi_mne", "ecg_clean_nk_mne", "data_resampler", "data_normalizer")
SG_FUNCS = ("_safe_savgol", "_batch_savgol")


def _extract(path, names, cls=None):
    src = path.read_text()
    tree = ast.parse(src)
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    out = {}
    for n in body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            seg = ast.get_source_segment(src, n, padded=True)
            if cls is not None:                                   # method → plain function: drop decorators / self
                seg = textwrap.dedent(seg)
            out[n.name] = seg
    missing = set(names) - set(out)
    assert not missing, (path, missing)
    return out


def source_hashes():
    h = {}
    for f, names, cls in (("evaluation/calculate_metric.py", EVAL_FUNCS, None), ("data_process_to_npz/step1.py", PREP_METHODS, "MCMEDProcessor"),
                          ("data_process_to_npz/step2.py", SG_FUNCS, None)):
        for k, v in _extract(UP / f, names, cls).items():
            h[f"{f}::{k}"] = hashlib.sha256(v.encode()).hexdigest()
    return h


def load_eval():
    """Official Hamilton-based HR functions (evaluation/calculate_metric.py)."""
    import numpy as np
    import neurokit2 as nk
    from biosppy.signals import ecg as ecg_func
    from biosppy.signals import ppg as ppg_func
    from biosppy.signals import tools
    ns = {"np": np, "nk": nk, "ecg_func": ecg_func, "ppg_func": ppg_func, "tools": tools}
    for src in _extract(UP / "evaluation/calculate_metric.py", EVAL_FUNCS).values():
        exec(compile(src, "calculate_metric.py", "exec"), ns)
    return ns


def load_prep():
    """Official preprocessing transforms: step1 filters / resampler / normaliser, step2 Savitzky-Golay smoothing."""
    from typing import Any, Dict, Optional

    import mne
    import numpy as np
    from scipy.signal import resample_poly, savgol_filter
    ns = {"np": np, "mne": mne, "resample_poly": resample_poly, "savgol_filter": savgol_filter, "Dict": Dict, "Any": Any, "Optional": Optional}
    for name, src in _extract(UP / "data_process_to_npz/step1.py", PREP_METHODS, "MCMEDProcessor").items():
        src = src.replace("@staticmethod\n", "")
        exec(compile(src, "step1.py", "exec"), ns)
    for src in _extract(UP / "data_process_to_npz/step2.py", SG_FUNCS).values():
        exec(compile(src, "step2.py", "exec"), ns)
    return ns
