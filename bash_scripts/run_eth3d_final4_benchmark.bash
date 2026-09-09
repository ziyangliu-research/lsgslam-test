#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -gt 0 ]; then
    SEQS=("$@")
else
    SEQS=(mannequin_face_1 einstein_1 sofa_3 plant_scene_3)
fi

DATA_ROOT="${ETH3D_RECTIFIED_ROOT:-/home/shiyo/Desktop/Datasets/ETH3D_rectified}"
FULL_ROOT="${LSG_ETH3D_FULL_ROOT:-experiments/eth3d_full_split_final4}"
STRIDE=1
CHUNK="${LSG_ETH3D_CHUNK_SIZE:-200}"
CONFIG="configs/eth3d/lsgslam_full_split_8_2.py"

export MPLBACKEND=Agg
export ETH3D_STRIDE="$STRIDE"

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

for scene in "${SEQS[@]}"; do
    seq_dir="$DATA_ROOT/$scene"
    if [ ! -d "$seq_dir" ]; then
        echo "Missing rectified ETH3D sequence: $seq_dir" >&2
        exit 1
    fi

    echo
    echo "================================================================================"
    echo "$scene : ETH3D FINAL LSG-SLAM benchmark"
    echo "Dataset: $seq_dir"
    echo "Protocol: full rectified stereo stream, stride=1, strict 8:2 holdout"
    echo "Test frames: 4,9,14,... = pose-only; excluded from map insertion/optimization/SR"
    echo "Metrics: FULL IMAGE PSNR + single-scale SSIM + LPIPS; NO mask"
    echo "Held-out render triplets: saved after unified evaluation"
    echo "================================================================================"

    # Cached IGEV depth + TransVPR features + aligned traj/YAML.
    python -u tools/eth3d_parser/prepare_eth3d_lsg.py "$scene" --root "$DATA_ROOT"

    image_count="$(find "$seq_dir/image_left" -maxdepth 1 -type f -name '*.png' | wc -l)"
    if [ "$image_count" -le 0 ]; then
        echo "No rectified left images for $scene" >&2
        exit 1
    fi
    last_idx=$((image_count - 1))
    test_count=$((image_count / 5))
    train_count=$((image_count - test_count))

    echo "Frames:             $image_count (indices 0..$last_idx)"
    echo "Expected 8:2 split: train=$train_count test=$test_count"
    echo "Chunk size:         $CHUNK (one-frame boundary overlap)"

    seq_root="$FULL_ROOT/$scene"
    mkdir -p "$seq_root"
    export LSG_WORKDIR="$seq_root"
    export ETH3D_SEQUENCE="$scene"
    export ETH3D_SEQ_DIR="$seq_dir"

    # ------------------------------------------------------------------
    # Stage 1: official-style 200-frame submaps, strict test pose-only split.
    # ------------------------------------------------------------------
    start=0
    while [ "$start" -lt "$last_idx" ]; do
        end=$((start + CHUNK))
        if [ "$end" -gt "$last_idx" ]; then end="$last_idx"; fi

        export ETH3D_START="$start"
        export ETH3D_END="$end"
        run_name="${scene}_${start}_${end}_${STRIDE}"
        out="$seq_root/$run_name"

        if [ -f "$out/params.npz" ] && [ -f "$out/benchmark_summary_split.json" ]; then
            echo "[Stage 1] SKIP completed submap: $run_name"
        else
            echo "[Stage 1] Run submap: $run_name"
            rm -rf "$out"
            mkdir -p "$out"
            set +e
            python -u scripts/eth3d_split_splatam.py "$CONFIG" 2>&1 | tee "$out/run.log"
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
    # Stage 2: released loop detection/constraint generation over full stream.
    # ------------------------------------------------------------------
    export ETH3D_START=0
    export ETH3D_END="$last_idx"
    loop_marker="$seq_root/.loop_stage_complete"
    loop_timing="$seq_root/loop_stage_timing.json"
    loop_folder="$seq_root/${scene}_0_${last_idx}_${STRIDE}_loops"

    if [ -f "$loop_marker" ] && [ -f "$loop_timing" ]; then
        echo "[Stage 2] SKIP completed loop stage with timing"
    else
        echo "[Stage 2] Full-sequence loop detection + constraints"
        rm -f "$loop_marker" "$loop_timing"
        rm -rf "$loop_folder"
        stage2_start_ns="$(date +%s%N)"
        set +e
        python -u scripts/eth3d_loop_closure.py "$CONFIG" 2>&1 | tee "$seq_root/loop_closure.log"
        status=${PIPESTATUS[0]}
        set -e
        stage2_end_ns="$(date +%s%N)"
        if [ "$status" -ne 0 ]; then
            echo "FAILED Stage 2: $scene (exit=$status)" >&2
            exit "$status"
        fi
        write_wall_timing_json \
            "$stage2_start_ns" "$stage2_end_ns" "$loop_timing" \
            "released loop detection + loop constraint generation; temporary metric eval excluded"
        touch "$loop_marker"
    fi

    # ------------------------------------------------------------------
    # Stage 3: released PGO + deformation + train-only 5000-iter SR.
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
        python -u tools/loop_closure/eth3d_pose_graph_part_optim_final.py \
            2>&1 | tee "$seq_root/full_backend.log"
        status=${PIPESTATUS[0]}
        set -e
        if [ "$status" -ne 0 ]; then
            echo "FAILED Stage 3: $scene (exit=$status)" >&2
            exit "$status"
        fi
        if [ ! -f "$backend_timing" ]; then
            echo "FAILED Stage 3 timing: $backend_timing not produced" >&2
            exit 1
        fi
        touch "$backend_marker"
    fi

    # Unified benchmark metrics are recomputed from raw saved RGB renders with
    # no silhouette/depth mask. This also exports held-out render triplets.
    echo "[Metrics] Full-image NO-MASK PSNR / SSIM / LPIPS + test renders"
    PYTHONPATH="$ROOT_DIR" python tools/evaluate_eth3d_full_unified.py \
        "$scene" --root "$FULL_ROOT" --data-root "$DATA_ROOT"

    echo "[$scene] COMPLETE"
done

echo
echo "=============================== FINAL TABLE ==============================="
PYTHONPATH="$ROOT_DIR" python tools/summarize_eth3d_final4_benchmarks.py \
    --root "$FULL_ROOT"

echo
echo "Done. Results root: $FULL_ROOT"
