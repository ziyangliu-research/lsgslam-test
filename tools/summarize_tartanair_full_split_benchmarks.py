#!/usr/bin/env python3
import argparse
import json
import os


DEFAULT_SEQS = ["SE000", "SE001", "SE002", "SE003", "SH000", "SH001", "SH002", "SH003"]


def fmt_gk(n):
    return f"{int(n) / 1000.0:.1f}k"


def status_for(seq_root):
    unified = os.path.join(seq_root, "benchmark_summary_unified.json")
    if os.path.isfile(unified):
        return "DONE"

    backend_marker = os.path.join(seq_root, ".full_backend_complete")
    backend_summary = os.path.join(seq_root, "benchmark_summary_full_split.json")
    backend_timing = os.path.join(seq_root, "backend_optimization_timing.json")
    render_before = os.path.join(seq_root, "RenderingResult", "before_opt_render_rgb")
    render_after = os.path.join(seq_root, "RenderingResult", "after_opt_render_rgb")

    if (
        os.path.isfile(backend_marker)
        and os.path.isfile(backend_summary)
        and os.path.isfile(backend_timing)
        and os.path.isdir(render_before)
        and os.path.isdir(render_after)
    ):
        return "READY (eval pending)"

    if os.path.isdir(seq_root):
        return "IN PROGRESS"
    return "MISSING"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", nargs="*", default=DEFAULT_SEQS)
    parser.add_argument("--root", default="experiments/tartanair_official_full_final8")
    args = parser.parse_args()

    header = (
        f"{'Seq':<7} {'Method / Status':<27} {'MaxMap':>8} {'ATE↓':>10} "
        f"{'Train P/S/L':>24} {'Test P/S/L':>24} "
        f"{'FPS':>8} {'Time(s)':>10} {'G':>11}"
    )
    print(header)
    print("-" * len(header))

    done = []
    ready = []
    running = []
    missing = []

    for seq in args.sequences:
        seq_root = os.path.join(args.root, seq)
        p = os.path.join(seq_root, "benchmark_summary_unified.json")
        status = status_for(seq_root)

        if status != "DONE":
            if status == "READY (eval pending)":
                ready.append(seq)
            elif status == "IN PROGRESS":
                running.append(seq)
            else:
                missing.append(seq)
            print(
                f"{seq:<7} {status:<27} {'—':>8} {'—':>10} "
                f"{'—':>24} {'—':>24} {'—':>8} {'—':>10} {'—':>11}"
            )
            continue

        done.append(seq)
        with open(p, "r", encoding="utf-8") as f:
            x = json.load(f)
        a = x["without_pgo_sr"]
        b = x["with_pgo_sr"]

        print(
            f"{seq:<7} {'LSG-SLAM (w/o PGO/SR)':<27} "
            f"{100*a['maxmap_ratio']:>7.2f}% {a['ate_rmse_se3_m']:>8.4f}m "
            f"{a['train_psnr']:>6.2f}/{a['train_ssim']:.4f}/{a['train_lpips']:.4f} "
            f"{a['test_psnr']:>6.2f}/{a['test_ssim']:.4f}/{a['test_lpips']:.4f} "
            f"{a['fps']:>8.4f} {a['time_seconds']:>10.1f} {fmt_gk(a['gaussians']):>11}"
        )
        print(
            f"{'':<7} {'LSG-SLAM (+ PGO/SR)':<27} "
            f"{100*b['maxmap_ratio']:>7.2f}% {b['ate_rmse_se3_m']:>8.4f}m "
            f"{b['train_psnr']:>6.2f}/{b['train_ssim']:.4f}/{b['train_lpips']:.4f} "
            f"{b['test_psnr']:>6.2f}/{b['test_ssim']:.4f}/{b['test_lpips']:.4f} "
            f"{'—':>8} {b['time_seconds']:>10.1f} {fmt_gk(b['gaussians']):>11}"
        )

    print()
    print(f"DONE:                {len(done)}/{len(args.sequences)}" + (f"  ({', '.join(done)})" if done else ""))
    print(f"READY (eval pending):{len(ready):>2}/{len(args.sequences)}" + (f"  ({', '.join(ready)})" if ready else ""))
    print(f"IN PROGRESS:          {len(running)}/{len(args.sequences)}" + (f"  ({', '.join(running)})" if running else ""))
    print(f"MISSING:              {len(missing)}/{len(args.sequences)}" + (f"  ({', '.join(missing)})" if missing else ""))
    print()
    print("DONE means unified PSNR/SSIM/LPIPS has been computed.")
    print("READY means SLAM + PGO/SR finished and raw renders/timing exist, but unified metrics have not been computed yet.")
    print("Metric protocol: full RGB; PSNR + single-scale SSIM + LPIPS(AlexNet); no silhouette/depth mask.")
    print("Time(s): w/o PGO/SR = online time; +PGO/SR = offline time. Gaussians are shown in k.")


if __name__ == "__main__":
    main()
