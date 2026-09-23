# Query to the PPGFlowECG authors (prepared 2026-09-23 — NOT sent)

To: the corresponding author of arXiv 2509.19774 (PPGFlowECG), repository https://github.com/PKUDigitalHealth/PPGFlowECG

Subject: Provenance of the released PPGFlowECG checkpoints on Hugging Face

We would like to evaluate the released checkpoints (`XiaochengFang/PPGFlowECG` @ `fdebf3e3`: `checkpoint-10.pt`, sha256
`50f1af67…ecdb`; `VAE-iter-40000.pth`, sha256 `c186633a…d7d6d`) without retraining, and could not determine the following
from the paper, repository or model files:

1. Which dataset / corpus was the public `checkpoint-10.pt` (and `VAE-iter-40000.pth`) trained on?
2. Which split was used (for MCMED, the official split; for any other corpus, the subject lists)?
3. Does this checkpoint correspond to a reported table or result in the paper, and if so which one?
4. Is varying the Euler / ODE inference step count valid for this released checkpoint without retraining (in particular
   below T = 5, which the paper's Fig. 5 does not cover)?

Thank you.
