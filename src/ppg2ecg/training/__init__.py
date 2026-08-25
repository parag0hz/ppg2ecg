"""Training loops.

v0 (this session): the baseline reproduction runs the *upstream* loop unchanged
(`scripts/run_upstream_train.sh` -> external/PENGUIN/src/train.py with configs/upstream).
Our own loop (needed for the objective swap OT-CFM -> iMeanFlow on the same backbone) is added only after
the upstream numbers are reproduced; it must reuse `ppg2ecg.flow.cfm` and pass tests/test_flow_parity.py.
"""
