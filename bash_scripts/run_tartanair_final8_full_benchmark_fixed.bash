#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FULL_ROOT="${LSG_FULL_ROOT:-experiments/tartanair_official_full_final8}"
SEQS=(SE000 SE001 SE002 SE003 SH000 SH001 SH002 SH003)

# Clean only incomplete Stage-3 artifacts.  Stage-1 submaps and the completed
# Stage-2 loop folder/timing are preserved, so a failed backend resumes without
# repeating the expensive frontend or loop-constraint generation.
for seq in "${SEQS[@]}"; do
    seq_root="$FULL_ROOT/$seq"
    backend_marker="$seq_root/.full_backend_complete"
    if [ -d "$seq_root" ] && [ ! -f "$backend_marker" ]; then
        echo "[resume-clean] $seq: removing incomplete Stage-3 artifacts only"
        rm -rf \
            "$seq_root/PoseGraphResult" \
            "$seq_root/RenderingResult"
        rm -f \
            "$seq_root/benchmark_summary_full_split.json" \
            "$seq_root/backend_optimization_timing.json"
    fi
done

# Reuse the final8 runner verbatim, changing only the Stage-3 launcher to the
# timing-order-fixed version.  Put the temporary script in bash_scripts so its
# ROOT_DIR calculation remains identical to the original runner.
tmp_runner="$(mktemp "$ROOT_DIR/bash_scripts/.final8_fixed.XXXXXX.bash")"
trap 'rm -f "$tmp_runner"' EXIT
sed \
    's#python -u tools/loop_closure/tartanair_pose_graph_part_optim.py#python -u tools/loop_closure/tartanair_pose_graph_part_optim_final.py#' \
    "$ROOT_DIR/bash_scripts/run_tartanair_final8_full_benchmark.bash" \
    > "$tmp_runner"
chmod +x "$tmp_runner"

exec bash "$tmp_runner"
