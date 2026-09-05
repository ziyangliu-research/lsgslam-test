#!/usr/bin/env python3
import argparse
import json
import os


DEFAULT_SEQS = ["SE000", "SE001", "SE002", "SE003", "SH000", "SH001", "SH002", "SH003"]


def fmt_gk(n):
    return f"{int(n) / 1000.0:.1f}k"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", nargs="*", default=DEFAULT_SEQS)
    parser.add_argument("--root", default="experiments/tartanair_official_full_final8")
    args = parser.parse_args()

    completed = []
    missing = []

    for seq in args.sequences:
        p = os.path.join(args.root, seq, "benchmark_summary_unified.json")
        if not os.path.isfile(p):
            missing.append(seq)
            print()
            print(seq)
            print("MISSING")
            continue

        completed.append(seq)
        with open(p, "r", encoding="utf-8") as f:
            x = json.load(f)

        a = x["without_pgo_sr"]
        b = x["with_pgo_sr"]

        print()
        print(seq)
        print(
            f"{'Method':<26} {'MaxMap':>9} {'ATE↓':>11} "
            f"{'Train P/S/L':>24} {'Test P/S/L':>24} "
            f"{'FPS':>8} {'Time(s)':>11} {'Gaussians':>12}"
        )
        print("-" * 134)
        print(
            f"{a['label']:<26} "
            f"{100*a['maxmap_ratio']:>8.2f}% "
            f"{a['ate_rmse_se3_m']:>8.4f}m "
            f"{a['train_psnr']:>6.2f}/{a['train_ssim']:.4f}/{a['train_lpips']:.4f} "
            f"{a['test_psnr']:>6.2f}/{a['test_ssim']:.4f}/{a['test_lpips']:.4f} "
            f"{a['fps']:>8.4f} {a['time_seconds']:>11.1f} {fmt_gk(a['gaussians']):>12}"
        )
        print(
            f"{b['label']:<26} "
            f"{100*b['maxmap_ratio']:>8.2f}% "
            f"{b['ate_rmse_se3_m']:>8.4f}m "
            f"{b['train_psnr']:>6.2f}/{b['train_ssim']:.4f}/{b['train_lpips']:.4f} "
            f"{b['test_psnr']:>6.2f}/{b['test_ssim']:.4f}/{b['test_lpips']:.4f} "
            f"{'—':>8} {b['time_seconds']:>11.1f} {fmt_gk(b['gaussians']):>12}"
        )

        t = x["timing"]
        print(
            f"  Time scope: online={t['online_seconds']:.1f}s; "
            f"offline={t['offline_seconds']:.1f}s "
            f"[loop={t['loop_closure_seconds']:.1f}, PGO={t['pgo_seconds']:.1f}, "
            f"deform={t['gaussian_deformation_seconds']:.1f}, SR={t['structure_refinement_seconds']:.1f}]; "
            f"end-to-end={t['end_to_end_algorithm_seconds']:.1f}s"
        )

    print()
    print("============================== SUMMARY ==============================")
    print(f"Completed: {len(completed)}/{len(args.sequences)}" + (f"  ({', '.join(completed)})" if completed else ""))
    print(f"Missing:   {len(missing)}/{len(args.sequences)}" + (f"  ({', '.join(missing)})" if missing else ""))
    print("=====================================================================")
    print("Metric protocol: full RGB; PSNR + single-scale SSIM + LPIPS(AlexNet); no silhouette/depth mask.")
    print("Time(s): w/o PGO/SR row = online time; +PGO/SR row = offline backend time.")
    print("FPS: unique sequence frames / online time. +PGO/SR FPS is intentionally not reported.")
    print("End-to-end algorithm time (online + offline) is saved in each benchmark_summary_unified.json.")


if __name__ == "__main__":
    main()
