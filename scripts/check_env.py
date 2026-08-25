"""Environment audit for ppg2ecg-one-step. Run: .venv/bin/python scripts/check_env.py
Prints the facts recorded in docs/ENVIRONMENT.md so they can be re-verified at any time."""
import platform
import shutil
import subprocess
import sys

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"

print("python      :", sys.version.split()[0], sys.executable)
print("platform    :", platform.platform())
print("nvidia-smi  :", sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"))
print("git         :", sh("git --version"), "| git-lfs:", sh("git lfs version || echo missing"))
print("gcc         :", sh("gcc --version | head -1"))
print("nvcc        :", shutil.which("nvcc") or "not on PATH")
print("RAM         :", sh("free -h | awk '/Mem:/{print $2\" total, \"$7\" available\"}'"))
print("disk        :", sh("df -h /home/kwy00 | awk 'NR==2{print $4\" free of \"$2}'"))
try:
    import torch
    print("torch       :", torch.__version__, "| cuda build", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
    print("cuda avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu0        :", p.name, f"{p.total_memory/2**30:.1f} GiB", f"sm_{p.major}{p.minor}")
        print("bf16        :", torch.cuda.is_bf16_supported())
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        print("bf16 matmul :", "ok" if torch.isfinite((x @ x).float()).all().item() else "NaN")
        print("tf32 matmul :", torch.backends.cuda.matmul.allow_tf32)
except Exception as e:  # noqa: BLE001
    print("torch       : FAILED", repr(e))
for m in ["numpy", "scipy", "hydra", "omegaconf", "neurokit2", "biosppy", "thop", "einops", "wandb", "pytest"]:
    try:
        mod = __import__(m)
        print(f"{m:12}: {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:12}: MISSING ({e.__class__.__name__})")
