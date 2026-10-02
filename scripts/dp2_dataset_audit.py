"""DP2 Stage A — local PPG / ECG dataset audit (metadata only).

Discovers the physiological datasets present on this machine and classifies each for a frozen external confirmation of
DualReadout-ECG S1 MIDDLE. Nothing here loads a full waveform: directory walks use file sizes only; WFDB / CSV headers
are read as text (first lines); .npy arrays are opened with mmap_mode="r" for their shape; MATLAB v7.3 files are opened
with h5py for dataset shapes only; project-generated metadata (manifests, inventories) are read as JSON. No model is
run. Excluded from discovery by standing project rules: /home/kwy00/sci, /home/kwy00/taeho. The old VitalDB V1 TEST is
not opened.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/dp2_dataset_audit.py
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
ART = ROOT / "artifacts/dp2_external"
MITDB = Path("/home/kwy00/DESTINATION")
EXCLUDED_ROOTS = ("/home/kwy00/sci", "/home/kwy00/taeho")
SEARCHED_ROOTS = {"/home": "searched (kwy00 only readable; excluding sci, taeho, anaconda3, caches, site-packages)", "/mnt": "empty",
                  "/opt": "no physiological data (innorix-ex only)", "/data": "absent", "/datasets": "absent", "/scratch": "absent",
                  "/workspace": "absent"}
LABELS = ("FRESH-PRIMARY-CANDIDATE", "FRESH-SECONDARY-CANDIDATE", "CONTAMINATED", "INELIGIBLE", "PROVENANCE-UNCERTAIN")
U = "UNKNOWN"


# ----------------------------------------------------------------------------------------------- metadata-only readers
def dir_stats(path: Path) -> dict:
    """Walk a directory: file count, total bytes, extension histogram (no file contents read)."""
    n, size, ext = 0, 0, {}
    for dp, _, fs in os.walk(path):
        for f in fs:
            n += 1
            size += os.path.getsize(os.path.join(dp, f))
            e = f.rsplit(".", 1)[-1].lower() if "." in f else ""
            ext[e] = ext.get(e, 0) + 1
    return {"n_files": n, "bytes": size, "gib": round(size / 2 ** 30, 3), "extensions": dict(sorted(ext.items(), key=lambda kv: -kv[1])[:8])}


def head_lines(path: Path, n: int = 8) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [next(f, "").rstrip("\n") for _ in range(n)]


def wfdb_header(path: Path) -> dict:
    lines = [ln for ln in head_lines(path, 20) if ln]
    rec = lines[0].split()
    sigs = [ln.split()[-1].rstrip(",") for ln in lines[1:1 + int(rec[1])]]
    comments = " ".join(ln for ln in lines if ln.startswith("#"))
    return {"fs": float(rec[2]), "n_samples": int(rec[3]) if len(rec) > 3 else None, "signals": sigs, "comments": comments}


def npy_shape(path: Path):
    a = np.load(path, mmap_mode="r")
    return list(a.shape), str(a.dtype)


# ----------------------------------------------------------------------------------------------- per-dataset schema
def bidmc() -> dict:
    hs = sorted((RAW / "BIDMC").glob("bidmc[0-9][0-9].hea"))
    h = [wfdb_header(p) for p in hs]
    src = sorted({m.group(0) for x in h for m in [re.search(r"mimic2wdb/matched/s\d+", x["comments"])] if m})
    return {"subjects": len(hs), "records": len(hs), "fs_ppg": sorted({x["fs"] for x in h}), "fs_ecg": sorted({x["fs"] for x in h}),
            "signals": sorted({s for x in h for s in x["signals"]}), "duration_s": sorted({x["n_samples"] / x["fs"] for x in h}),
            "ecg_leads_records": {ld: sum(ld in x["signals"] for x in h) for ld in ("II", "V", "AVR", "I", "III", "MCL")},
            "pleth_records": sum("PLETH" in x["signals"] for x in h),
            "ppg_channels": ["PLETH"], "subject_id": "bidmcNN (header source: MIMIC-II matched subject sNNNNN)",
            "source_records_mimic2": len(src), "source_example": src[:2]}


def capnobase() -> dict:
    d = RAW / "CapnoBase/files"
    par = sorted(d.glob("*_param.csv"))
    fs = set()
    for p in par:
        r = list(csv.DictReader(open(p)))
        fs.add((int(float(r[0]["samplingrate_pleth"])), int(float(r[0]["samplingrate_ecg"]))))
    chans = head_lines(next(d.glob("*_signal.csv")), 1)[0]
    return {"subjects": len(par), "records": len(par), "fs_ppg_ecg": sorted(fs), "channels": chans, "duration_s": 480,
            "ecg_leads": U, "ppg_channels": ["pleth"], "subject_id": "case id NNNN_8min (one record per subject)"}


def mimicbp() -> dict:
    d = RAW / "MIMIC-BP"
    lists = {k: json.loads((d / f"{k}_subjects.txt").read_text().replace("'", '"')) for k in ("train", "val", "test")}
    kinds = {k: len(list((d / k).glob("*.npy"))) for k in ("ppg", "ecg", "abp", "resp", "labels")}
    pid = sorted(p.name.split("_")[0] for p in (d / "ppg").glob("*_ppg.npy"))
    shapes = {k: npy_shape(d / k / f"{pid[0]}_{k}.npy") for k in ("ppg", "ecg", "abp")}
    fs = re.search(r"fs\s*=\s*(\d+)", (d / "read_data.py").read_text()).group(1)
    return {"subjects": len(pid), "records": len(pid) * shapes["ppg"][0][0], "official_split": {k: len(v) for k, v in lists.items()},
            "files_per_kind": kinds, "array_shape_example": shapes, "fs_ppg": int(fs), "fs_ecg": int(fs),
            "duration_s": f"{shapes['ppg'][0][0]} segments x {shapes['ppg'][0][1] / int(fs):.0f} s per subject",
            "ecg_leads": U, "ppg_channels": ["ppg"], "subject_id": "pNNNNNN (MIMIC-III Waveform Matched Subset subject id)"}


def dalia() -> dict:
    inv = json.loads((ROOT / "data/manifests/dalia_raw_inventory.json").read_text())
    r = inv["rows"]
    return {"subjects": len(r), "records": len(r), "fs_ecg": sorted({round(x["ecg_len"] / x["ecg_s"]) for x in r}),
            "fs_ppg": sorted({round(x["bvp_len"] / x["bvp_s"]) for x in r}), "duration_s_range": [min(x["ecg_s"] for x in r), max(x["ecg_s"] for x in r)],
            "ecg_leads": "chest RespiBAN ECG (single channel)", "ppg_channels": ["wrist Empatica E4 BVP"], "subject_id": "S1..S15",
            "metadata_source": "data/manifests/dalia_raw_inventory.json (project-generated)"}


def wesad() -> dict:
    subj = sorted(p.name for p in (RAW / "WESAD").iterdir() if p.is_dir() and p.name.startswith("S"))
    return {"subjects": len(subj), "records": len(subj), "fs_ecg": 700, "fs_ppg": 64, "ecg_leads": "chest RespiBAN ECG (single channel)",
            "ppg_channels": ["wrist Empatica E4 BVP"], "subject_id": "S2..S17 (no S1, S12)",
            "metadata_source": "docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md (signal.wrist.BVP @64 Hz, chest @700 Hz); pkl not opened"}


def wildppg() -> dict:
    inv = json.loads((RAW / "WildPPG/INVENTORY.json").read_text())
    f = inv["files"]
    loc = f[0]["locations"]
    return {"subjects": len(f), "records": len(f), "ids": [x["id"] for x in f], "fs": sorted({v["fs"] for site in loc.values() for v in site.values() if isinstance(v, dict) and "fs" in v}),
            "ecg_duration_h": [round(x["ecg_duration_s"] / 3600, 1) for x in f], "sites": list(loc), "ecg_leads": "sternum ECG (single channel)",
            "ppg_channels": ["ppg_g", "ppg_ir", "ppg_r"] + ["at sites " + ",".join(loc)], "subject_id": "3-character participant code",
            "metadata_source": "data/raw/WildPPG/INVENTORY.json (project-generated, 2026-08-26)"}


def ucibp() -> dict:
    import h5py
    parts = sorted((RAW / "UCI-BP").glob("Part_*.mat"))
    n = 0
    for p in parts:
        with h5py.File(p, "r") as f:
            n += int(np.prod(f[p.stem].shape))
    return {"subjects": U, "records": n, "fs": 125, "channels": "per record [n, 3] = PPG, ABP, ECG (A7 audit)", "ecg_leads": U,
            "subject_id": "none (records from MIMIC-II, identifiers suppressed; docs/A7_ABP_DATASET_AUDIT.md)"}


def vitaldb() -> dict:
    s = json.loads((ROOT / "artifacts/v0_vitaldb_audit/summary.json").read_text())
    m = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())
    sp = m["splits"][0]
    v1 = set(sp["train"]) | set(sp["val"]) | set(sp["test"])
    poc = {c: int(p) for c, p in m["extra"]["patient_of_case"].items()}
    cs = {int(r["caseid"]): int(r["subjectid"]) for r in csv.DictReader(open(RAW / "VitalDB/cases.csv", encoding="utf-8-sig"))}
    local = sorted(int(p.stem[5:]) for p in (RAW / "VitalDB/cases").glob("case_*.npz"))
    rest = [c for c in local if f"case_{c:05d}" not in v1]
    rest_subj = {cs[c] for c in rest}
    return {"cases_local": len(local), "eligible_cases_v0": s["n_eligible_cases"], "eligible_subjects_v0": s["n_eligible_subjects"],
            "v1_cases": len(v1), "v1_patients": m["extra"]["n_patients"], "fs_ppg": 500, "fs_ecg": 500, "ecg_leads": "ECG_II",
            "ppg_channels": ["PLETH (SNUADC)"], "subject_id": "VitalDB subjectid", "v0_rule": s["rule"], "v0_fail_reasons": s["fail_reasons"],
            "residual_not_in_v1": {"cases": len(rest), "subjects": len(rest_subj), "subjects_also_in_v1": len(rest_subj & set(poc.values())),
                                   "reason": "V0 rule: finite fraction < 0.95 (nan)"}}


def mitbih() -> dict:
    hs = sorted(MITDB.glob("[0-9][0-9][0-9].hea"))
    h = [wfdb_header(p) for p in hs]
    return {"records": len(hs), "subjects": U, "fs_ecg": sorted({x["fs"] for x in h}), "signals": sorted({s for x in h for s in x["signals"]}),
            "ppg_present": False, "also": f"{len(list((MITDB / 'x_mitdb').glob('*.hea')))} headers in x_mitdb/"}


def mimic_afib() -> dict:
    p = RAW / "MIMIC-AFib/mimic3wdb_matched_index.csv"
    rows = list(csv.DictReader(open(p)))
    return {"index_rows": len(rows), "records_with_pleth_and_II": sum(r["has_pleth_and_II"] == "1" for r in rows),
            "waveforms_present": False, "content": "header index of the MIMIC-III Waveform Matched Subset written by scripts/scan_mimic3wdb.py"}


# ----------------------------------------------------------------------------------------------- frozen classification
PRIOR_USE = {   # verified commit / record evidence (git log, docs/); ordered by first use
    "PPG-DaLiA": (["USED FOR TRAINING", "USED FOR DEVELOPMENT", "USED FOR FINAL PERFORMANCE REPORTING", "USED FOR HYPOTHESIS FORMATION"], [
        ("5e0aa35", "docs/A0_PENGUIN_REPRODUCTION_REPORT.md", "A0 PENGUIN reproduction trained / tested on PPG-DaLiA (train 13, val S11, test S2)"),
        ("ce13126", "A3 results", "one-step recovery tested on DaLiA subject S1"),
        ("150669b", "docs/D1_MULTI_DATASET_BENCHMARK_REPORT.md", "five-corpus benchmark incl. PPG-DaLiA"),
        ("c753d13", "docs/U2_PAIRED_IMEANFLOW_VS_PENGUIN_REPORT.md", "U2 paired iMF vs PENGUIN incl. DaLiA"),
        ("542e6be", "P1 RESULT", "'DaLiA samples disagree on beat positions' (hypothesis)"),
        ("c3d672d", "docs/MC1_MULTI_CORPUS_REPORT.md", "MC1 PPG-DaLiA 15 subjects, 4 test")]),
    "WESAD": (["USED FOR TRAINING", "USED FOR FINAL PERFORMANCE REPORTING"], [
        ("8d53f65", "docs/D3_PENGUIN_SIX_DATASET_REPORT.md", "D3 six-dataset replication incl. WESAD (respiration target)"),
        ("2842c11", "docs/U1_UPSTREAM_PENGUIN_ASSHIPPED_REPORT.md", "U1 upstream PENGUIN on six datasets"),
        ("8bd682b", "U2", "'U2: WESAD evaluated -- non-inferior at k=1, but only just'")]),
    "WildPPG": (["USED FOR TRAINING", "USED FOR DEVELOPMENT", "USED FOR FINAL PERFORMANCE REPORTING", "USED FOR HYPOTHESIS FORMATION"], [
        ("bae0142", "docs/A4_WILDPPG_REPLICATION_REPORT.md", "A4: test participants kjd, ssx; val an0, k2s"),
        ("3f93a4d", "A9", "ECG target-representation robustness on WildPPG"),
        ("ec7f1d4", "docs/WP1_WILDPPG_4FOLD_REPORT.md", "WP1 subject-level 4-fold"),
        ("74858d0", "ED2", "decoding transferred to WildPPG"), ("d22d2ae", "WD1", "WildPPG regression + allocation"),
        ("504395d", "docs/BEAT_FIRST_ARCHITECTURE_DESIGN_KO.md", "cites 'WildPPG에서 같은 구조(R1) 0.62' as design evidence")]),
    "BIDMC": (["USED FOR TRAINING", "USED FOR FINAL PERFORMANCE REPORTING"], [
        ("150669b", "docs/D1_MULTI_DATASET_BENCHMARK_REPORT.md", "D1 BIDMC 36 train / 8 test subjects"),
        ("8d53f65", "D3", "BIDMC respiration"), ("c3d672d", "docs/MC1_MULTI_CORPUS_REPORT.md", "MC1 BIDMC 51 subjects, 15 test"),
        ("a4d66e5", "EXP-D respiration", "BIDMC (6 subj) width/depth")]),
    "CapnoBase": (["USED FOR TRAINING", "USED FOR FINAL PERFORMANCE REPORTING"], [
        ("150669b", "docs/D1_MULTI_DATASET_BENCHMARK_REPORT.md", "D1 five-corpus benchmark incl. CapnoBase"),
        ("c3d672d", "docs/MC1_MULTI_CORPUS_REPORT.md", "MC1 CapnoBase 42 subjects, 13 test")]),
    "MIMIC-BP": (["USED FOR TRAINING", "USED FOR DEVELOPMENT", "USED FOR FINAL PERFORMANCE REPORTING", "USED FOR HYPOTHESIS FORMATION",
                  "USED FOR THRESHOLD / MODEL / LOSS SELECTION"], [
        ("fed5b8c", "docs/A7_ABP_GENERALIZATION_REPORT.md", "A7 PPG->ABP on the official split (train 1,100 / val 195 / test 229 subjects; all 1,524)"),
        ("9ff77b5", "A8", "ABP target-scale sensitivity (target representation decided from A7 outcomes)"),
        ("e09ee6d", "U2", "'U2: MIMIC-BP evaluated -- and it exposes a defect in my own frozen gate'"),
        ("b1eb66a", "docs/EXP_D_ABP_REPORT.md", "EXP-D ABP width / depth allocation"),
        ("—", "docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md", "excluded from ECG work: 'target is ABP, not ECG'; ECG channel never used as a target")]),
    "UCI-BP": (["USED FOR TRAINING", "USED FOR FINAL PERFORMANCE REPORTING"], [
        ("8d53f65", "D3", "D3 incl. UCI-BP"), ("2842c11", "U1", "U1 incl. UCI-BP"), ("—", "data/manifests/split_u2_ucibp_seed42.json", "U2 split")]),
    "VitalDB": (["USED FOR TRAINING", "USED FOR DEVELOPMENT", "USED FOR FINAL PERFORMANCE REPORTING", "USED FOR HYPOTHESIS FORMATION",
                 "USED FOR THRESHOLD / MODEL / LOSS SELECTION"], [
        ("080ec1c", "V0", "VitalDB availability audit"), ("5d3d274", "docs/V1_VITALDB_PAIRED_REPORT.md", "V1 test 1,156 patients"),
        ("724b883", "docs/DP0_DUALREADOUT_REPORT.md", "all DP0 training / development / lock populations"),
        ("17b43a6", "docs/DP1_TEST_FRESHNESS_AUDIT.md", "old V1 TEST NOT FRESH")]),
    "VitalDB residual (V0-excluded)": (["REFERENCED ONLY"], [
        ("080ec1c", "artifacts/v0_vitaldb_audit/cases_audit.csv", "V0 eligibility audit only (finite fraction); excluded, never windowed or modelled"),
        ("—", "data/manifests/split_v1_vitaldb_seed42.json", "9 of the 104+9 residual subjects also own V1 cases (used)")]),
    "MIT-BIH (DESTINATION)": (["NEVER REFERENCED"], [("—", "repository grep", "no 'mitdb' / 'MIT-BIH' in docs, scripts, src or artifacts")]),
    "MIMIC-AFib (index only)": (["REFERENCED ONLY"], [
        ("—", "scripts/scan_mimic3wdb.py", "index scan only; no waveform downloaded"),
        ("—", "docs/PREPROCESSING_CONVENTIONS_SURVEY.md", "'Do not use MIMIC-AFib for any comparative claim until a subject list is fixed'")]),
}

ELIGIBILITY = {   # frozen precedence: INELIGIBLE (structural) > CONTAMINATED (prior outcome use) > PROVENANCE-UNCERTAIN > FRESH
    "PPG-DaLiA": ("CONTAMINATED", "training, final reporting and hypothesis formation in this project; 15 subjects; PPG/ECG only second-level synchronised (A0)"),
    "WESAD": ("CONTAMINATED", "training and final reporting (D3 / U1 / U2); 15 subjects"),
    "WildPPG": ("CONTAMINATED", "training, final reporting and hypothesis formation (A4 / A9 / D1 / U1 / U2 / WP1 / ED2 / WD1; cited by the BF0 design document); 16 participants"),
    "BIDMC": ("CONTAMINATED", "training and final reporting (D1 / D3 / U1 / U2 / MC1 / EXP-D); 53 subjects; MIMIC-II provenance"),
    "CapnoBase": ("CONTAMINATED", "training and final reporting (D1 / MC1); 42 subjects"),
    "MIMIC-BP": ("CONTAMINATED", "prior outcome-based model / loss / selection use in this project for PPG->ABP (A7, A8, U1, U2, EXP-D) on all 1,524 "
                                 "subjects (criterion 7 fails); its ECG channel and PPG->ECG outcomes were never computed; MIMIC-III provenance"),
    "UCI-BP": ("INELIGIBLE", "no subject identifiers (patient-level split infeasible); also used in D3 / U1 / U2; MIMIC-II provenance"),
    "VitalDB": ("CONTAMINATED", "the DP0 development source (training, development, lock); old V1 TEST NOT FRESH (DP1 audit)"),
    "VitalDB residual (V0-excluded)": ("INELIGIBLE", "113 cases / 104 subjects not in V1: excluded by the frozen V0 finite-fraction rule (< 0.95); same source, "
                                                     "devices and institution as all DP0 development data, so not an external dataset shift"),
    "MIT-BIH (DESTINATION)": ("INELIGIBLE", "ECG only (MLII + one precordial lead, 360 Hz); no PPG"),
    "MIMIC-AFib (index only)": ("INELIGIBLE", "only a header index CSV; no waveform present locally"),
}


def select_primary(elig: dict, inv: dict):
    """Frozen DP2 rule over FRESH-PRIMARY candidates: provenance clarity, no prior use, most subjects, sync metadata, least
    dataset-specific preprocessing. Returns None when there is no candidate."""
    c = [k for k, (lab, _) in elig.items() if lab == "FRESH-PRIMARY-CANDIDATE"]
    if not c:
        return None
    return sorted(c, key=lambda k: -int(inv[k].get("subjects") or 0))[0]


def main():
    ART.mkdir(parents=True, exist_ok=True)
    schema = {"PPG-DaLiA": dalia(), "WESAD": wesad(), "WildPPG": wildppg(), "BIDMC": bidmc(), "CapnoBase": capnobase(), "MIMIC-BP": mimicbp(),
              "UCI-BP": ucibp(), "VitalDB": vitaldb(), "MIT-BIH (DESTINATION)": mitbih(), "MIMIC-AFib (index only)": mimic_afib()}
    paths = {"PPG-DaLiA": RAW / "PPG-DaLiA", "WESAD": RAW / "WESAD", "WildPPG": RAW / "WildPPG", "BIDMC": RAW / "BIDMC", "CapnoBase": RAW / "CapnoBase",
             "MIMIC-BP": RAW / "MIMIC-BP", "UCI-BP": RAW / "UCI-BP", "VitalDB": RAW / "VitalDB", "MIT-BIH (DESTINATION)": MITDB,
             "MIMIC-AFib (index only)": RAW / "MIMIC-AFib"}
    stats = {k: dir_stats(p) for k, p in paths.items()}
    v = schema["VitalDB"]
    common = dict(raw_or_processed="raw (project processed copies under data/processed are derived from it)")
    inv = {
        "PPG-DaLiA": dict(source="PPG-DaLiA (Reiss et al. 2019), 15 subjects, daily-life protocol", subjects=15, records=15, subject_id=True, ppg=True, ecg=True, abp=False,
                          simultaneous="YES", fs_ppg=64, fs_ecg=700, duration="2.5 h per subject (ecg_s 9212 s for S1)", ecg_leads="chest single channel",
                          ppg_channels="wrist BVP", timestamps=False, sync="UNCERTAIN (second-level only, A0 report)", windows=True, patient_split=True),
        "WESAD": dict(source="WESAD (Schmidt et al. 2018), 15 subjects", subjects=15, records=15, subject_id=True, ppg=True, ecg=True, abp=False,
                      simultaneous="YES", fs_ppg=64, fs_ecg=700, duration=U, ecg_leads="chest single channel", ppg_channels="wrist BVP", timestamps=False,
                      sync="UNCERTAIN (same device pair as PPG-DaLiA)", windows=True, patient_split=True),
        "WildPPG": dict(source="WildPPG (ETH Zurich, polybox), 16 participants, multi-site wearable", subjects=16, records=16, subject_id=True, ppg=True, ecg=True,
                        abp=False, simultaneous="YES", fs_ppg=128, fs_ecg=128, duration="~12 h per participant", ecg_leads="sternum single channel",
                        ppg_channels="green / IR / red at head, sternum, wrist, ankle", timestamps=True, sync="YES (one recording device per site, 128 Hz)",
                        windows=True, patient_split=True),
        "BIDMC": dict(source="BIDMC PPG and Respiration (PhysioNet), records from MIMIC-II matched subset", subjects=53, records=53, subject_id=True, ppg=True,
                      ecg=True, abp=False, simultaneous="YES", fs_ppg=125, fs_ecg=125, duration="480 s", ecg_leads="II in every record (also V, AVR, ...)", ppg_channels="PLETH",
                      timestamps=True, sync="YES (one WFDB multi-signal record)", windows=True, patient_split=True),
        "CapnoBase": dict(source="CapnoBase IEEE TBME RR benchmark (Karlen et al. 2013)", subjects=42, records=42, subject_id=True, ppg=True, ecg=True, abp=False,
                          simultaneous="YES", fs_ppg=300, fs_ecg=300, duration="480 s", ecg_leads=U, ppg_channels="pleth", timestamps=True,
                          sync="YES (one acquisition, common 300 Hz clock)", windows=True, patient_split=True),
        "MIMIC-BP": dict(source="MIMIC-BP (curated from the MIMIC-III Waveform Matched Subset; Dataverse)", subjects=schema["MIMIC-BP"]["subjects"],
                         records=schema["MIMIC-BP"]["records"], subject_id=True, ppg=True, ecg=True, abp=True, simultaneous="YES", fs_ppg=125, fs_ecg=125,
                         duration="30 segments x 30 s per subject", ecg_leads=U, ppg_channels="ppg (single)", timestamps=False,
                         sync="YES (curated: ECG/PPG fundamental frequency within 0.3 Hz, pulse-arrival-time consistency; A7 audit)", windows=True,
                         patient_split=True),
        "UCI-BP": dict(source="UCI Cuff-Less Blood Pressure Estimation (records from MIMIC-II)", subjects=U, records=schema["UCI-BP"]["records"], subject_id=False,
                       ppg=True, ecg=True, abp=True, simultaneous="YES", fs_ppg=125, fs_ecg=125, duration="variable per record", ecg_leads=U,
                       ppg_channels="PPG (single)", timestamps=False, sync="UNCERTAIN", windows=True, patient_split=False),
        "VitalDB": dict(source="VitalDB (Seoul National University Hospital, intra-operative)", subjects=v["eligible_subjects_v0"], records=v["cases_local"],
                        subject_id=True, ppg=True, ecg=True, abp=U, simultaneous="YES", fs_ppg=500, fs_ecg=500, duration="median case > 10 min (V0 rule)",
                        ecg_leads="ECG_II", ppg_channels="PLETH", timestamps=True, sync="YES (one 500 Hz clock)", windows=True, patient_split=True),
        "VitalDB residual (V0-excluded)": dict(source="VitalDB (same source as DP0 development)", subjects=v["residual_not_in_v1"]["subjects"],
                                               records=v["residual_not_in_v1"]["cases"], subject_id=True, ppg=True, ecg=True, abp=U, simultaneous="YES",
                                               fs_ppg=500, fs_ecg=500, duration=U, ecg_leads="ECG_II", ppg_channels="PLETH", timestamps=True,
                                               sync="YES", windows="UNCERTAIN (failed finite-fraction rule)", patient_split=True),
        "MIT-BIH (DESTINATION)": dict(source="MIT-BIH Arrhythmia Database (PhysioNet)", subjects=U, records=schema["MIT-BIH (DESTINATION)"]["records"],
                                      subject_id=False, ppg=False, ecg=True, abp=False, simultaneous="NO", fs_ppg=None, fs_ecg=360, duration="~30 min",
                                      ecg_leads="MLII + V1/V2/V4/V5", ppg_channels=None, timestamps=True, sync="NO", windows=False, patient_split=False),
        "MIMIC-AFib (index only)": dict(source="index of the MIMIC-III Waveform Matched Subset (no waveforms)", subjects=U, records=0, subject_id=U, ppg=False,
                                        ecg=False, abp=False, simultaneous="NO", fs_ppg=None, fs_ecg=None, duration=None, ecg_leads=None, ppg_channels=None,
                                        timestamps=False, sync="NO", windows=False, patient_split=False),
    }
    path_of = dict(paths) | {"VitalDB residual (V0-excluded)": RAW / "VitalDB/cases"}
    rows = []
    for k, x in inv.items():
        base = k if k in stats else "VitalDB"
        rows.append({"dataset_name": k, "absolute_path": str(path_of[k]), "source_provenance": x["source"],
                     "disk_size_gib": stats[base]["gib"] if k in stats else None, "file_format": list(stats[base]["extensions"]) if k in stats else ["npz"],
                     "number_of_files": stats[base]["n_files"] if k in stats else v["residual_not_in_v1"]["cases"],
                     "estimated_subject_count": x["subjects"], "estimated_record_count": x["records"],
                     "subject_identifier_available": "YES" if x["subject_id"] is True else "NO" if x["subject_id"] is False else U,
                     "PPG_present": "YES" if x["ppg"] else "NO", "ECG_present": "YES" if x["ecg"] else "NO",
                     "ABP_present": "YES" if x["abp"] is True else "NO" if x["abp"] is False else U, "simultaneous_PPG_ECG": x["simultaneous"],
                     "sampling_rate_PPG": x["fs_ppg"] if x["fs_ppg"] is not None else "n/a", "sampling_rate_ECG": x["fs_ecg"] if x["fs_ecg"] is not None else "n/a",
                     "signal_length_duration": x["duration"] or "n/a", "ECG_leads": x["ecg_leads"] or "n/a", "PPG_channels": x["ppg_channels"] or "n/a",
                     "timestamps_available": "YES" if x["timestamps"] else "NO", "known_synchronization": x["sync"],
                     "window_extraction_feasible": "YES" if x["windows"] is True else "NO" if x["windows"] is False else x["windows"],
                     "patient_disjoint_split_feasible": "YES" if x["patient_split"] else "NO", "raw_or_processed": common["raw_or_processed"],
                     "notes": ELIGIBILITY[k][1], "schema_probe": schema.get(k)})
    (ART / "dataset_inventory.json").write_text(json.dumps({"searched_roots": SEARCHED_ROOTS, "excluded_by_project_rule": EXCLUDED_ROOTS,
                                                            "derived_copies_not_separate_datasets": sorted(p.name for p in (ROOT / "data/processed").iterdir()) +
                                                            ["data/raw_u1_wildppg_view"], "model_weights_not_datasets": ["data/pretrained/{imf,ppgflowecg,rddm}"],
                                                            "datasets": rows}, indent=1, default=str))
    prov = {"groups": {"MIMIC (Beth Israel Deaconess ICU)": {"members": ["BIDMC (MIMIC-II matched)", "UCI-BP (MIMIC-II)", "MIMIC-BP (MIMIC-III matched)",
                                                                         "MIMIC-AFib index (MIMIC-III matched)"],
                                                             "overlap": "possible: MIMIC-III waveform records span the MIMIC-II period and no local mapping between "
                                                                        "MIMIC-II sNNNNN and MIMIC-III pNNNNNN identifiers exists; never independent replications"},
                       "Empatica E4 + RespiBAN (Univ. Siegen / Bosch)": {"members": ["PPG-DaLiA", "WESAD"], "overlap": "same device pair and research group; subject overlap UNKNOWN"},
                       "VitalDB (SNUH)": {"members": ["VitalDB", "VitalDB residual (V0-excluded)"], "overlap": "same source as all DP0 development data; 9 residual subjects also have V1 cases"},
                       "independent single sources": ["WildPPG (ETH Zurich)", "CapnoBase (UBC)", "MIT-BIH (ECG only)"]},
            "dp0_development_population": "VitalDB only; no MIMIC / wearable dataset shares institutions with it"}
    (ART / "dataset_provenance.json").write_text(json.dumps(prov, indent=1))
    elig = {k: {"label": lab, "reason": why, "prior_use": PRIOR_USE.get(k if k in PRIOR_USE else "VitalDB")[0],
                "evidence": [{"commit": c, "record": f, "description": d} for c, f, d in PRIOR_USE.get(k if k in PRIOR_USE else "VitalDB")[1]]}
            for k, (lab, why) in ELIGIBILITY.items()}
    assert all(e["label"] in LABELS for e in elig.values()) and set(elig) == set(inv)
    (ART / "dataset_eligibility.json").write_text(json.dumps({"precedence": "INELIGIBLE > CONTAMINATED > PROVENANCE-UNCERTAIN > FRESH (exactly one label)",
                                                              "datasets": elig}, indent=1))
    win = select_primary(ELIGIBILITY, {k: {"subjects": x["subjects"] if isinstance(x["subjects"], int) else 0} for k, x in inv.items()})
    (ART / "primary_dataset_selection.json").write_text(json.dumps({
        "selected_dataset": win, "verdict": "DP2 NO ELIGIBLE PRIMARY DATASET" if win is None else "selected",
        "selection_reason": "no dataset satisfies all FRESH-PRIMARY criteria" if win is None else "frozen rule",
        "all_candidates": {k: e["label"] for k, e in elig.items()},
        "why_other_candidates_were_not_selected": {k: e["reason"] for k, e in elig.items()},
        "closest_to_eligible": {"dataset": "MIMIC-BP", "fails": "criterion 7 (prior outcome-based model / loss / selection use in this project, PPG->ABP)",
                                "would_otherwise_meet": "simultaneous PPG + ECG, 1,524 subject ids, curated synchronisation, 125 Hz, no overlap with VitalDB, "
                                                        "PPG->ECG outcomes never computed; ECG lead UNKNOWN locally"}}, indent=1))
    with open(ART / "table_dataset_inventory.csv", "w", newline="") as f:
        cols = ["dataset_name", "estimated_subject_count", "estimated_record_count", "PPG_present", "ECG_present", "sampling_rate_PPG", "sampling_rate_ECG",
                "known_synchronization", "subject_identifier_available", "absolute_path"]
        w = csv.writer(f)
        w.writerow(cols + ["prior_use", "eligibility"])
        for r in rows:
            e = elig[r["dataset_name"]]
            w.writerow([r[c] for c in cols] + ["; ".join(e["prior_use"]), e["label"]])
    figure(rows, elig)
    print(json.dumps({k: e["label"] for k, e in elig.items()}, indent=1), "\nprimary:", win)


def figure(rows, elig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"CONTAMINATED": "#a23b52", "INELIGIBLE": "0.55", "PROVENANCE-UNCERTAIN": "#d59a54", "FRESH-SECONDARY-CANDIDATE": "#7a6fd0",
           "FRESH-PRIMARY-CANDIDATE": "#2a7f3f"}
    names = [r["dataset_name"] for r in rows]
    subj = [r["estimated_subject_count"] if isinstance(r["estimated_subject_count"], int) else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(names))
    ax.barh(y, [max(s, 1) for s in subj], color=[col[elig[n]["label"]] for n in names])
    for i, (n, s) in enumerate(zip(names, subj)):
        ax.text(max(s, 1) * 1.1, i, f"{s if s else 'unknown'} subjects — {elig[n]['label']}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(1, 2e5)
    ax.set_xlabel("subjects (log scale; unknown shown at 1)")
    ax.set_title("DP2 local dataset audit — no FRESH-PRIMARY-CANDIDATE (DP2 NO ELIGIBLE PRIMARY DATASET)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(ART / "figure_dataset_audit.png", dpi=110)


if __name__ == "__main__":
    main()
