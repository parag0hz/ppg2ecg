#!/usr/bin/env bash
# Download PPG-DaLiA (UCI ML Repository #495, CC BY 4.0) into data/raw and verify.
# Usage: bash scripts/download_dalia.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
URL="https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
ZIP="$RAW/ppg+dalia.zip"
mkdir -p "$RAW"
if [ ! -s "$ZIP" ]; then
  echo "[download] $URL -> $ZIP"
  curl -sS -L --retry 3 --max-time 7200 -o "$ZIP" "$URL"
else
  echo "[download] zip already present: $(du -h "$ZIP" | cut -f1)"
fi
echo "[verify] zip integrity"
unzip -tq "$ZIP"
echo "[extract] -> $RAW/PPG-DaLiA/"
mkdir -p "$RAW/PPG-DaLiA"
unzip -q -o "$ZIP" -d "$RAW/PPG-DaLiA"
# The UCI zip contains data.zip (nested). Extract that too if present.
if [ -f "$RAW/PPG-DaLiA/data.zip" ]; then
  unzip -q -o "$RAW/PPG-DaLiA/data.zip" -d "$RAW/PPG-DaLiA"
fi
echo "[checksum] writing $RAW/CHECKSUMS.sha256"
( cd "$RAW" && sha256sum ppg+dalia.zip > CHECKSUMS.sha256 && find PPG-DaLiA -name 'S*.pkl' -print0 | sort -z | xargs -0 sha256sum >> CHECKSUMS.sha256 )
echo "[done] subjects found:"; find "$RAW/PPG-DaLiA" -name 'S*.pkl' | sort
