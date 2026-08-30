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

missing_preprocess=()
for seq in "${SEQS[@]}"; do
    seq_dir="$DATA_ROOT/stereo/$seq"
    image_count=$(find "$seq_dir/image_left" -maxdepth 1 -type f -name '*_left.png' | wc -l)
    depth_count=$(find "$seq_dir/depth_sceneflow" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l)
    feature_count=$(find "$seq_dir/global_features" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l)

    echo "[Preprocess check] $seq images=$image_count depth=$depth_count features=$feature_count"
    if [ "$image_count" -eq 0 ] || [ "$depth_count" -ne "$image_count" ] || [ "$feature_count" -ne "$image_count" ]; then
        missing_preprocess+=("$seq")
    fi
done

if [ "${#missing_preprocess[@]}" -gt 0 ]; then
    echo "============================================================"
    echo "Precomputing missing full TartanAir sequences"
    echo "Sequences: ${missing_preprocess[*]}"
    echo "============================================================"

    python tools/tartanair_parser/operate_tartanair_data.py \
      --sequences "${missing_preprocess[@]}"
else
    echo "All four sequences already have complete depth/global-feature caches; skipping preprocessing."
fi

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
