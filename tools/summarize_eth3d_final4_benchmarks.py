#!/usr/bin/env python3
import argparse
import json
import os

DEFAULT_SEQS = ["mannequin_face_1", "einstein_1", "sofa_3", "plant_scene_3"]


def fmt_gk(n):
    return f"{int(n) / 1000.0:.1f}k"


def status_for(root, seq):
    seq_root = os.path.join(root, seq)
    unified = os.path.join(seq_root, "benchmark_summary_unified.json")
    backend_marker = os.path.join(seq_root, ".full_backend_complete")
    backend_summary = os.path.join(seq_root, "benchmark_summary_full_split.json")
    backend_timing = os.path.join(seq_root, "backend_optimization_timing.json")
    loop_marker = os.path.join(seq_root, ".loop_stage_complete")
    if os.path.isfile(unified):
        return "DONE"
    if os.path.isfile(backend_marker) and os.path.isfile(backend_summary) and os.path.isfile(backend_timing):
        return "READY (eval pending)"
    if os.path.isdir(seq_root) and (os.path.isfile(loop_marker) or os.listdir(seq_root)):
        return "IN PROGRESS"
    return "MISSING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--root", default="experiments/eth3d_full_split_final4")
    args = ap.parse_args()

    print(
        f"{'Sequence':<19} {'Method / Status':<28} {'MaxMap':>9} {'ATE↓':>10} "
        f"{'Train P/S/L':>24} {'Test P/S/L':>24} {'FPS':>8} {'Time(s)':>10} {'G':>11}"
    )
    print("-" * 150)

    counts = {"DONE": 0, "READY (eval pending)": 0, "IN PROGRESS": 0, "MISSING": 0}
    for seq in args.sequences:
        status = status_for(args.root, seq)
        counts[status] += 1
        p = os.path.join(args.root, seq, "benchmark_summary_unified.json")
        if status != "DONE":
            print(f"{seq:<19} {status:<28} {'—':>9} {'—':>10} {'—':>24} {'—':>24} {'—':>8} {'—':>10} {'—':>11}")
            continue

        with open(p, "r", encoding="utf-8") as f:
            x = json.load(f)
        for i, key in enumerate(("without_pgo_sr", "with_pgo_sr")):
            r = x[key]
            seq_col = seq if i == 0 else ""
            fps = "—" if r.get("fps") is None else f"{r['fps']:.4f}"
            print(
                f"{seq_col:<19} {r['label']:<28} "
                f"{100*r['maxmap_ratio']:>8.2f}% {r['ate_rmse_se3_m']:>8.4f}m "
                f"{r['train_psnr']:>6.2f}/{r['train_ssim']:.4f}/{r['train_lpips']:.4f} "
                f"{r['test_psnr']:>6.2f}/{r['test_ssim']:.4f}/{r['test_lpips']:.4f} "
                f"{fps:>8} {r['time_seconds']:>10.1f} {fmt_gk(r['gaussians']):>11}"
            )

    print()
    for k in ("DONE", "READY (eval pending)", "IN PROGRESS", "MISSING"):
        print(f"{k:<22}: {counts[k]}/{len(args.sequences)}")
    print("Metric protocol: full image, NO silhouette/depth mask; PSNR + single-scale SSIM + LPIPS(AlexNet).")


if __name__ == "__main__":
    main()
