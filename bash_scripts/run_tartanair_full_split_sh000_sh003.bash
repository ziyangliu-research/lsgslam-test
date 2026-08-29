#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEQS=(SH000 SH001 SH002 SH003)
DATA_ROOT="${TARTANAIR_DATA_ROOT:-/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge}"
WORK_ROOT="${LSG_WORKDIR:-experiments/tartanair_split}"

export TARTANAIR_DATA_ROOT="$DATA_ROOT"
export LSG_WORKDIR="$WORK_ROOT"
export TARTANAIR_START=0
export TARTANAIR_END=-1
export TARTANAIR_STRIDE=1

echo "============================================================"
echo "Precomputing full TartanAir sequences (existing files skipped)"
echo "Sequences: ${SEQS[*]}"
echo "============================================================"

python tools/tartanair_parser/operate_tartanair_data.py \
  --sequences "${SEQS[@]}"

for seq in "${SEQS[@]}"; do
    export TARTANAIR_SEQUENCE="$seq"
    out="${WORK_ROOT}/${seq}_full_split5"
    summary="$out/benchmark_summary_split.json"

    echo
    echo "============================================================"
    echo "Running ${seq}: full sequence, 8:2 split"
    echo "Test frames: 4,9,14,... (pose only; no mapping/keyframe)"
    echo "============================================================"

    if [ -f "$summary" ]; then
        echo "SKIP: ${seq} already completed: $summary"
        continue
    fi

    # Remove only an incomplete previous attempt. Completed runs are preserved.
    rm -rf "$out"
    mkdir -p "$out"

    set +e
    python -u scripts/tartanair_split_splatam.py \
      configs/tartanair/lsgslam_split_8_2.py \
      2>&1 | tee "$out/run.log"
    status=${PIPESTATUS[0]}
    set -e

    if [ "$status" -ne 0 ]; then
        echo "FAILED: ${seq} (exit=${status})"
        echo "Re-run the same command after fixing the issue; completed sequences will be skipped."
        exit "$status"
    fi

    echo "FINISHED: ${seq}"
    cat "$summary"
done

echo
echo "============================================================"
echo "Final summary"
echo "============================================================"

python tools/summarize_tartanair_split_benchmarks.py \
  SH000 SH001 SH002 SH003 \
  --root "$WORK_ROOT" \
  --csv "$WORK_ROOT/summary_SH000_SH003_split5.csv"
