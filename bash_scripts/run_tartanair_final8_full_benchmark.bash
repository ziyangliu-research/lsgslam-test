#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEQS=(SE000 SE001 SE002 SE003 SH000 SH001 SH002 SH003)
DATA_ROOT="${TARTANAIR_DATA_ROOT:-/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge}"
FULL_ROOT="${LSG_FULL_ROOT:-experiments/tartanair_official_full_final8}"
CHUNK_SIZE="${LSG_FULL_CHUNK_SIZE:-200}"

export TARTANAIR_DATA_ROOT="$DATA_ROOT"
export TARTANAIR_STRIDE=1
export MPLBACKEND=Agg

write_wall_timing_json() {
    local start_ns="$1"
    local end_ns="$2"
    local out_json="$3"
    local scope="$4"
    python - "$start_ns" "$end_ns" "$out_json" "$scope" <<'PY'
import json
import sys
start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])
out = sys.argv[3]
scope = sys.argv[4]
seconds = (end_ns - start_ns) / 1e9
with open(out, "w", encoding="utf-8") as f:
    json.dump({"wall_seconds": seconds, "scope": scope}, f, indent=2)
print(f"Timing saved: {out} = {seconds:.3f} s")
PY
}

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
    echo "================================================================================"
    echo "$seq FINAL LSG-SLAM benchmark"
    echo "Frames: $num_frames (0..$last_idx)"
    echo "Split: test = global 4,9,14,... (pose-only; no map/SR loss)"
    echo "Metrics: full RGB PSNR + single-scale SSIM + LPIPS"
    echo "Timing: online Stage-1 + offline loop/PGO/deformation/SR; final metric render excluded"
    echo "Caches: images=$image_count depth=$depth_count features=$feat_count"
    echo "Chunk size: $CHUNK_SIZE"
    echo "Results: $FULL_ROOT/$seq"
    echo "================================================================================"

    if [ "$depth_count" -ne "$image_count" ] || [ "$feat_count" -ne "$image_count" ]; then
        echo "Incomplete preprocessing cache for $seq; filling missing files first..."
        python tools/tartanair_parser/operate_tartanair_data.py --sequences "$seq"
    fi

    seq_root="$FULL_ROOT/$seq"
    mkdir -p "$seq_root"
    export LSG_WORKDIR="$seq_root"

    # ----------------------------------------------------------------------
    # Stage 1: official 200-frame submaps. Per-submap benchmark_summary_split
    # already contains online_seconds measured from first-map initialization to
    # sequence completion, excluding final evaluation. Test frames remain pose-only.
    # ----------------------------------------------------------------------
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

        if [ -f "$out/params.npz" ] && [ -f "$out/benchmark_summary_split.json" ]; then
            echo "[Stage 1] SKIP completed submap: $run_name"
        else
            echo "[Stage 1] Run submap: $run_name"
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

    # ----------------------------------------------------------------------
    # Stage 2: released full-sequence loop detection + loop constraint generation.
    # This is an offline backend stage; wall time is recorded separately.
    # ----------------------------------------------------------------------
    export TARTANAIR_START=0
    export TARTANAIR_END="$last_idx"
    export TARTANAIR_STRIDE=1

    loop_marker="$seq_root/.loop_stage_complete"
    loop_timing="$seq_root/loop_stage_timing.json"
    if [ -f "$loop_marker" ] && [ -f "$loop_timing" ]; then
        echo "[Stage 2] SKIP completed loop-closure stage with timing"
    else
        echo "[Stage 2] Full-sequence loop detection + pose constraints"
        rm -f "$loop_marker" "$loop_timing"
        stage2_start_ns="$(date +%s%N)"
        set +e
        python -u scripts/tartanair_loop_closure.py \
            configs/tartanair/lsgslam_full_split_8_2.py \
            2>&1 | tee "$seq_root/loop_closure.log"
        status=${PIPESTATUS[0]}
        set -e
        stage2_end_ns="$(date +%s%N)"
        if [ "$status" -ne 0 ]; then
            echo "FAILED Stage 2: $seq (exit=$status)"
            exit "$status"
        fi
        write_wall_timing_json \
            "$stage2_start_ns" "$stage2_end_ns" "$loop_timing" \
            "released loop detection + loop constraint generation wall time"
        touch "$loop_marker"
    fi

    # ----------------------------------------------------------------------
    # Stage 3: released PGO + Gaussian deformation + train-only 5000-iter SR.
    # The TartanAir wrapper records optimization-only time and excludes the
    # before/after metric-render loops from backend_optimization_timing.json.
    # ----------------------------------------------------------------------
    export LSG_FULL_BASE_FOLDER="$seq_root"
    backend_marker="$seq_root/.full_backend_complete"
    backend_summary="$seq_root/benchmark_summary_full_split.json"
    backend_timing="$seq_root/backend_optimization_timing.json"
    if [ -f "$backend_marker" ] && [ -f "$backend_summary" ] && [ -f "$backend_timing" ]; then
        echo "[Stage 3] SKIP completed backend with optimization timing"
    else
        echo "[Stage 3] PGO + Gaussian deformation + train-only SR"
        rm -f "$backend_marker" "$backend_timing"
        set +e
        python -u tools/loop_closure/tartanair_pose_graph_part_optim.py \
            2>&1 | tee "$seq_root/full_backend.log"
        status=${PIPESTATUS[0]}
        set -e
        if [ "$status" -ne 0 ]; then
            echo "FAILED Stage 3: $seq (exit=$status)"
            exit "$status"
        fi
        if [ ! -f "$backend_timing" ]; then
            echo "FAILED Stage 3 timing: $backend_timing was not produced"
            exit 1
        fi
        touch "$backend_marker"
    fi

    echo "[$seq] algorithm stages complete. Unified metrics are evaluated after all sequences."
done

echo
echo "================================================================================"
echo "All 8 sequences completed. Running unified full-RGB PSNR/SSIM/LPIPS evaluation..."
echo "Metric evaluation time is NOT included in online/offline algorithm time."
echo "================================================================================"

PYTHONPATH="$ROOT_DIR" python tools/evaluate_tartanair_full_unified.py \
    "${SEQS[@]}" \
    --root "$FULL_ROOT" \
    --data-root "$DATA_ROOT"

echo
echo "=============================== FINAL TABLE ==============================="
PYTHONPATH="$ROOT_DIR" python tools/summarize_tartanair_full_split_benchmarks.py \
    "${SEQS[@]}" \
    --root "$FULL_ROOT"

echo
echo "Done."
echo "Results root: $FULL_ROOT"
echo "Per sequence: benchmark_summary_unified.json + benchmark_metrics_unified_per_frame.csv"
