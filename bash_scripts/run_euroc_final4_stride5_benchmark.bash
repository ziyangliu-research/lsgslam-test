#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -gt 0 ]; then
    SEQS=("$@")
else
    SEQS=(MH02 V101 V201 MH05)
fi

DATA_ROOT="${EUROC_DATA_ROOT:-/home/shiyo/Desktop/Datasets/EuRoC}"
FULL_ROOT="${LSG_EUROC_FULL_ROOT:-experiments/euroc_official_full_stride5_final4}"
STRIDE=5
CHUNK_RAW="${LSG_EUROC_CHUNK_RAW:-200}"
CONFIG="configs/euroc/lsgslam_full_stride5_8_2.py"

export MPLBACKEND=Agg
export EUROC_STRIDE="$STRIDE"

write_wall_timing_json() {
    local start_ns="$1"
    local end_ns="$2"
    local out_json="$3"
    local scope="$4"
    python - "$start_ns" "$end_ns" "$out_json" "$scope" <<'PY'
import json, sys
start_ns=int(sys.argv[1]); end_ns=int(sys.argv[2]); out=sys.argv[3]; scope=sys.argv[4]
seconds=(end_ns-start_ns)/1e9
with open(out,'w',encoding='utf-8') as f:
    json.dump({'wall_seconds':seconds,'scope':scope},f,indent=2)
print(f'Timing saved: {out} = {seconds:.3f} s')
PY
}

resolve_sequence() {
    case "$1" in
        MH02)
            SCENE="MH_02_easy"
            SEQ_DIR="$DATA_ROOT/extracted/machine_hall/machine_hall/MH_02_easy"
            ;;
        V101)
            SCENE="V1_01_easy"
            SEQ_DIR="$DATA_ROOT/extracted/vicon_room1/vicon_room1/V1_01_easy"
            ;;
        V201)
            SCENE="V2_01_easy"
            SEQ_DIR="$DATA_ROOT/extracted/vicon_room2/vicon_room2/V2_01_easy"
            ;;
        MH05)
            SCENE="MH_05_difficult"
            SEQ_DIR="$DATA_ROOT/extracted/machine_hall/machine_hall/MH_05_difficult"
            ;;
        *)
            echo "Unknown EuRoC alias: $1 (expected MH02 V101 V201 MH05)" >&2
            exit 2
            ;;
    esac
}

count_synced_frames() {
    local cam0="$1"
    python - "$cam0" <<'PY'
import glob, os, sys
from pathlib import Path
cam0=sys.argv[1]
traj=os.path.join(cam0,'traj.txt')
with open(traj,'r',encoding='utf-8') as f:
    ts={line.split()[0] for line in f if line.strip()}
paths=sorted(glob.glob(os.path.join(cam0,'data_rect','*.png')), key=lambda p:int(Path(p).stem))
synced=[p for p in paths if Path(p).stem in ts]
print(len(synced))
PY
}

for alias in "${SEQS[@]}"; do
    resolve_sequence "$alias"
    CAM0_DIR="$SEQ_DIR/mav0/cam0"

    echo
    echo "================================================================================"
    echo "$alias / $SCENE : EuRoC FINAL benchmark"
    echo "Dataset: $SEQ_DIR"
    echo "Protocol: full synchronized stream -> stride=5 -> strict 8:2 holdout"
    echo "Test retained ids: 4,9,14,... (raw/synced dataset indices 20,45,70,...)"
    echo "LSG submaps: released EuRoC convention, raw span=$CHUNK_RAW with stride=5"
    echo "Metrics: full-image PSNR + single-scale SSIM + LPIPS"
    echo "================================================================================"

    # ------------------------------------------------------------------
    # Preprocessing: released EuRoC rectification + IGEV depth + GT traj +
    # TransVPR features. Cached outputs are skipped; time is excluded.
    # ------------------------------------------------------------------
    python -u tools/euroc_parser/prepare_euroc_final4.py "$alias" --data-root "$DATA_ROOT"

    if [ ! -d "$CAM0_DIR/data_rect" ] || [ ! -f "$CAM0_DIR/traj.txt" ]; then
        echo "Preprocessing incomplete for $alias: $CAM0_DIR" >&2
        exit 1
    fi

    num_sync="$(count_synced_frames "$CAM0_DIR")"
    if [ "$num_sync" -le 0 ]; then
        echo "No synchronized EuRoC frames for $alias" >&2
        exit 1
    fi
    last_idx=$((num_sync - 1))
    retained=$((last_idx / STRIDE + 1))
    test_count=$((retained / 5))
    # For N retained samples, test ids are 4,9,...; integer division N/5 is exact count.
    train_count=$((retained - test_count))

    echo "Synchronized frames: $num_sync (indices 0..$last_idx)"
    echo "After stride=5:      $retained"
    echo "Expected 8:2 split:  train=$train_count test=$test_count"

    seq_root="$FULL_ROOT/$alias"
    mkdir -p "$seq_root"
    export LSG_WORKDIR="$seq_root"
    export EUROC_SEQUENCE="$SCENE"
    export EUROC_CAM0_DIR="$CAM0_DIR"

    # ------------------------------------------------------------------
    # Stage 1: official-style EuRoC submaps.  The released LSG-SLAM EuRoC
    # script uses raw span=200 and config stride=5; preserve that here.
    # ------------------------------------------------------------------
    start=0
    while [ "$start" -lt "$last_idx" ]; do
        end=$((start + CHUNK_RAW))
        if [ "$end" -gt "$last_idx" ]; then end="$last_idx"; fi

        export EUROC_START="$start"
        export EUROC_END="$end"
        run_name="${SCENE}_${start}_${end}_${STRIDE}"
        out="$seq_root/$run_name"

        if [ -f "$out/params.npz" ] && [ -f "$out/benchmark_summary_split.json" ]; then
            echo "[Stage 1] SKIP completed submap: $run_name"
        else
            echo "[Stage 1] Run submap: $run_name"
            rm -rf "$out"
            mkdir -p "$out"
            set +e
            python -u scripts/euroc_split_splatam.py "$CONFIG" 2>&1 | tee "$out/run.log"
            status=${PIPESTATUS[0]}
            set -e
            if [ "$status" -ne 0 ]; then
                echo "FAILED Stage 1: $run_name (exit=$status)" >&2
                exit "$status"
            fi
        fi
        start="$end"
    done

    # ------------------------------------------------------------------
    # Stage 2: full downsampled sequence loop detection/constraint generation.
    # Temporary loop-pair metric eval is disabled; wall time is saved.
    # ------------------------------------------------------------------
    export EUROC_START=0
    export EUROC_END="$last_idx"
    loop_marker="$seq_root/.loop_stage_complete"
    loop_timing="$seq_root/loop_stage_timing.json"
    loop_folder="$seq_root/${SCENE}_0_${last_idx}_${STRIDE}_loops"

    if [ -f "$loop_marker" ] && [ -f "$loop_timing" ]; then
        echo "[Stage 2] SKIP completed loop stage with timing"
    else
        echo "[Stage 2] Full-sequence loop detection + constraints"
        rm -f "$loop_marker" "$loop_timing"
        rm -rf "$loop_folder"
        stage2_start_ns="$(date +%s%N)"
        set +e
        python -u scripts/euroc_loop_closure.py "$CONFIG" 2>&1 | tee "$seq_root/loop_closure.log"
        status=${PIPESTATUS[0]}
        set -e
        stage2_end_ns="$(date +%s%N)"
        if [ "$status" -ne 0 ]; then
            echo "FAILED Stage 2: $alias (exit=$status)" >&2
            exit "$status"
        fi
        write_wall_timing_json \
            "$stage2_start_ns" "$stage2_end_ns" "$loop_timing" \
            "released EuRoC loop detection + loop constraint generation wall time; temporary metric eval excluded"
        touch "$loop_marker"
    fi

    # ------------------------------------------------------------------
    # Stage 3: released PGO + Gaussian deformation + train-only 5000-iter SR.
    # Backend timing excludes before/after rendering metric loops.
    # ------------------------------------------------------------------
    export LSG_FULL_BASE_FOLDER="$seq_root"
    backend_marker="$seq_root/.full_backend_complete"
    backend_summary="$seq_root/benchmark_summary_full_split.json"
    backend_timing="$seq_root/backend_optimization_timing.json"

    if [ -f "$backend_marker" ] && [ -f "$backend_summary" ] && [ -f "$backend_timing" ]; then
        echo "[Stage 3] SKIP completed backend with timing"
    else
        echo "[Stage 3] PGO + Gaussian deformation + train-only SR"
        rm -rf "$seq_root/PoseGraphResult" "$seq_root/RenderingResult"
        rm -f "$backend_marker" "$backend_summary" "$backend_timing"
        set +e
        python -u tools/loop_closure/euroc_pose_graph_part_optim_final.py 2>&1 | tee "$seq_root/full_backend.log"
        status=${PIPESTATUS[0]}
        set -e
        if [ "$status" -ne 0 ]; then
            echo "FAILED Stage 3: $alias (exit=$status)" >&2
            exit "$status"
        fi
        if [ ! -f "$backend_timing" ]; then
            echo "FAILED Stage 3 timing: $backend_timing not produced" >&2
            exit 1
        fi
        touch "$backend_marker"
    fi

    # Evaluate each completed sequence immediately so partial batch results are
    # always inspectable. Metric time is outside all recorded algorithm timers.
    echo "[Metrics] Unified full-image PSNR / SSIM / LPIPS"
    PYTHONPATH="$ROOT_DIR" python tools/evaluate_euroc_full_unified.py \
        "$alias" --root "$FULL_ROOT" --data-root "$DATA_ROOT"

    echo "[$alias] COMPLETE"
done

echo
echo "=============================== FINAL TABLE ==============================="
PYTHONPATH="$ROOT_DIR" python tools/summarize_euroc_final4_benchmarks.py \
    --root "$FULL_ROOT"

echo
echo "Done. Results root: $FULL_ROOT"
