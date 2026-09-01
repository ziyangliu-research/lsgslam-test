#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEQS=(SH000 SH001 SH002 SH003)
DATA_ROOT="${TARTANAIR_DATA_ROOT:-/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge}"
FULL_ROOT="${LSG_FULL_ROOT:-experiments/tartanair_official_full_split}"
CHUNK_SIZE="${LSG_FULL_CHUNK_SIZE:-200}"

export TARTANAIR_DATA_ROOT="$DATA_ROOT"
export TARTANAIR_STRIDE=1
export MPLBACKEND=Agg

for seq in "${SEQS[@]}"; do
    export TARTANAIR_SEQUENCE="$seq"

    image_dir="$DATA_ROOT/stereo/$seq/image_left"
    depth_dir="$DATA_ROOT/stereo/$seq/depth_sceneflow"
    feat_dir="$DATA_ROOT/stereo/$seq/global_features"

    if [ ! -d "$image_dir" ]; then
        echo "Missing image directory: $image_dir"
        exit 1
    fi

    last_file="$(find "$image_dir" -maxdepth 1 -type f -name '*_left.png' | sort | tail -n 1)"
    if [ -z "$last_file" ]; then
        echo "No TartanAir images found for $seq"
        exit 1
    fi
    last_name="$(basename "$last_file")"
    last_str="${last_name%_left.png}"
    last_idx=$((10#$last_str))
    num_frames=$((last_idx + 1))

    image_count="$(find "$image_dir" -maxdepth 1 -type f -name '*_left.png' | wc -l)"
    depth_count="$(find "$depth_dir" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l)"
    feat_count="$(find "$feat_dir" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l)"

    echo
    echo "================================================================"
    echo "$seq official-style FULL LSG-SLAM + strict 8:2 holdout"
    echo "Frames: $num_frames (0..$last_idx)"
    echo "Test frames: 4,9,14,... = pose-only; no map/SR loss"
    echo "Caches: images=$image_count depth=$depth_count features=$feat_count"
    echo "Chunk size: $CHUNK_SIZE raw frames"
    echo "================================================================"

    if [ "$depth_count" -ne "$image_count" ] || [ "$feat_count" -ne "$image_count" ]; then
        echo "Incomplete preprocessing cache for $seq; filling missing files first..."
        python tools/tartanair_parser/operate_tartanair_data.py --sequences "$seq"
    fi

    seq_root="$FULL_ROOT/$seq"
    mkdir -p "$seq_root"
    export LSG_WORKDIR="$seq_root"

    # Stage 1: released 200-frame Gaussian submaps with the common 8:2 holdout.
    start=0
    while [ "$start" -lt "$last_idx" ]; do
        end=$((start + CHUNK_SIZE))
        if [ "$end" -gt "$last_idx" ]; then
            end="$last_idx"
        fi

        export TARTANAIR_START="$start"
        export TARTANAIR_END="$end"
        export TARTANAIR_STRIDE=1

        run_name="${seq}_${start}_${end}_1"
        out="$seq_root/$run_name"

        if [ -f "$out/params.npz" ]; then
            echo "[Stage 1] SKIP completed split submap: $run_name"
        else
            echo "[Stage 1] Split submap $run_name"
            rm -rf "$out"
            mkdir -p "$out"
            set +e
            python -u scripts/tartanair_split_splatam.py \
                configs/tartanair/lsgslam_full_split_8_2.py \
                2>&1 | tee "$out/run.log"
            status=${PIPESTATUS[0]}
            set -e
            if [ "$status" -ne 0 ]; then
                echo "FAILED Stage 1: $run_name (exit=$status)"
                exit "$status"
            fi
        fi

        start="$end"
    done

    # Stage 2: released loop-closure logic on the full sequence.
    export TARTANAIR_START=0
    export TARTANAIR_END="$last_idx"
    export TARTANAIR_STRIDE=1

    loop_marker="$seq_root/.loop_stage_complete"
    if [ -f "$loop_marker" ]; then
        echo "[Stage 2] SKIP completed loop-closure stage"
    else
        echo "[Stage 2] Full-sequence loop detection + pose constraints"
        set +e
        python -u scripts/tartanair_loop_closure.py \
            configs/tartanair/lsgslam_full_split_8_2.py \
            2>&1 | tee "$seq_root/loop_closure.log"
        status=${PIPESTATUS[0]}
        set -e
        if [ "$status" -ne 0 ]; then
            echo "FAILED Stage 2: $seq (exit=$status)"
            exit "$status"
        fi
        touch "$loop_marker"
    fi

    # Stage 3: released pose graph + Gaussian deformation + train-only 5000-iter SR.
    export LSG_FULL_BASE_FOLDER="$seq_root"
    backend_marker="$seq_root/.full_backend_complete"
    if [ -f "$backend_marker" ] && [ -f "$seq_root/benchmark_summary_full_split.json" ]; then
        echo "[Stage 3] SKIP completed split-aware full backend"
    else
        echo "[Stage 3] Pose graph optimization + train-only structure refinement"
        set +e
        python -u tools/loop_closure/tartanair_pose_graph_part_optim.py \
            2>&1 | tee "$seq_root/full_backend.log"
        status=${PIPESTATUS[0]}
        set -e
        if [ "$status" -ne 0 ]; then
            echo "FAILED Stage 3: $seq (exit=$status)"
            exit "$status"
        fi
        touch "$backend_marker"
    fi

    echo
    echo "==================== $seq COMPLETE ===================="
    summary="$seq_root/benchmark_summary_full_split.json"
    if [ -f "$summary" ]; then
        cat "$summary"
    else
        echo "Backend completed but split summary was not found. Check:"
        echo "  $seq_root/full_backend.log"
        exit 1
    fi
    echo "========================================================"
done

echo
echo "All SH000-SH003 full LSG-SLAM 8:2 runs completed."
echo "Results root: $FULL_ROOT"

echo
echo "[Final Evaluation] Unified raw-RGB PSNR/SSIM + Stage-1 FPS aggregation"
PYTHONPATH="$ROOT_DIR" python tools/evaluate_tartanair_full_unified.py \
    SH000 SH001 SH002 SH003 \
    --root "$FULL_ROOT" \
    --data-root "$DATA_ROOT"

echo
echo "[Final Table]"
PYTHONPATH="$ROOT_DIR" python tools/summarize_tartanair_full_split_benchmarks.py \
    SH000 SH001 SH002 SH003 \
    --root "$FULL_ROOT"
