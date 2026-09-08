#!/usr/bin/env bash
# Download the three PENGUIN datasets we lack, into the layout external/PENGUIN/src/utils/load_data.py expects:
#   BIDMC   -> data/raw/BIDMC/1.0.0/bidmc_csv/bidmc_NN_Signals.csv   (columns " PLETH", " RESP", 125 Hz)
#   WESAD   -> data/raw/WESAD/S*/S*.pkl                              (signal.wrist.BVP 64 Hz, signal.chest.Resp 700 Hz)
#   UCI-BP  -> data/raw/UCI-BP/Part_N.mat                            (HDF5; column 0 PPG, column 1 ABP, 125 Hz)
set -u
ROOT=/home/kwy00/ppg2ecg-one-step
cd "$ROOT" || exit 1
log(){ echo "$(date -Is) $*"; }

case "${1:-all}" in
bidmc)
  log "BIDMC csv: 53 records"
  mkdir -p data/raw/BIDMC/1.0.0/bidmc_csv
  for i in $(seq -w 1 53); do
    f="data/raw/BIDMC/1.0.0/bidmc_csv/bidmc_${i}_Signals.csv"
    [ -s "$f" ] && continue
    curl -sfL --retry 3 -o "$f" "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_${i}_Signals.csv" || echo "  FAIL $i"
  done
  log "BIDMC done: $(ls data/raw/BIDMC/1.0.0/bidmc_csv/*.csv 2>/dev/null | wc -l)/53 files, $(du -sh data/raw/BIDMC/1.0.0/bidmc_csv | cut -f1)"
  ;;
uci)
  log "UCI-BP: Cuff-Less Blood Pressure Estimation (UCI #340)"
  mkdir -p data/raw/UCI-BP
  curl -fL --retry 3 -o data/raw/UCI-BP/uci_bp.zip \
    "https://archive.ics.uci.edu/static/public/340/cuff+less+blood+pressure+estimation.zip" \
    && (cd data/raw/UCI-BP && unzip -o -q uci_bp.zip && ls)
  # the release nests a second zip in some mirrors; unpack anything that is still zipped
  (cd data/raw/UCI-BP && for z in *.zip; do [ "$z" = uci_bp.zip ] && continue; unzip -o -q "$z" 2>/dev/null; done)
  log "UCI-BP done: $(ls data/raw/UCI-BP/ | tr '\n' ' ')"
  ;;
wesad)
  log "WESAD: 2.25 GB"
  mkdir -p data/raw/WESAD
  curl -fL --retry 3 -C - -o data/raw/WESAD/WESAD.zip "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download" \
    && (cd data/raw/WESAD && unzip -o -q WESAD.zip && rm -f WESAD.zip)
  # the archive unpacks to WESAD/S2 ... S17; flatten one level if needed
  [ -d data/raw/WESAD/WESAD ] && mv data/raw/WESAD/WESAD/* data/raw/WESAD/ && rmdir data/raw/WESAD/WESAD
  log "WESAD done: $(ls -d data/raw/WESAD/S* 2>/dev/null | wc -l) subject dirs, $(du -sh data/raw/WESAD | cut -f1)"
  ;;
esac
