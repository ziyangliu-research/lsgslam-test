#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -gt 0 ]; then
    SEQS=("$@")
else
    SEQS=(SE000 SE001 SE002 SE003 SH000 SH001 SH002 SH003)
fi

DATA_ROOT="${TARTANAIR_DATA_ROOT:-/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge}"

# The underlying final runner uses `set -euo pipefail` and counts preprocessing
# cache files with `find`.  If a cache directory does not exist yet, `find`
# returns non-zero and the runner exits before it reaches its preprocessing
# fallback.  Create only the empty cache directories here; the original runner
# will then see count=0 and correctly generate the missing cache files.
for seq in "${SEQS[@]}"; do
    image_dir="$DATA_ROOT/stereo/$seq/image_left"
    if [ ! -d "$image_dir" ]; then
        echo "Missing image directory: $image_dir"
        exit 1
    fi
    mkdir -p \
        "$DATA_ROOT/stereo/$seq/depth_sceneflow" \
        "$DATA_ROOT/stereo/$seq/global_features"
done

echo "Safe final benchmark launcher"
echo "Sequences: ${SEQS[*]}"
echo "Data root: $DATA_ROOT"
echo

exec bash "$ROOT_DIR/bash_scripts/run_tartanair_final_benchmark_sequences.bash" "${SEQS[@]}"
