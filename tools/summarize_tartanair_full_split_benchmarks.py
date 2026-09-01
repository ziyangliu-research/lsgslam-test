#!/usr/bin/env python3
import argparse
import json
import os


def fmt_g(n):
    n = int(n)
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequences",
        nargs="*",
        default=["SH000", "SH001", "SH002", "SH003"],
    )
    parser.add_argument("--root", default="experiments/tartanair_official_full_split")
    args = parser.parse_args()

    for seq in args.sequences:
        p = os.path.join(args.root, seq, "benchmark_summary_unified.json")
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Unified summary not found: {p}\n"
                "Run: PYTHONPATH=. python tools/evaluate_tartanair_full_unified.py"
            )
        with open(p, "r", encoding="utf-8") as f:
            x = json.load(f)

        a = x["without_pgo_sr"]
        b = x["with_pgo_sr"]

        print()
        print(seq)
        print(
            f"{'Method':<26} {'MaxMap':>9} {'ATE RMSE↓':>12} "
            f"{'Train PSNR/SSIM':>21} {'Test PSNR/SSIM':>21} "
            f"{'FPS':>9} {'Gaussians':>12}"
        )
        print("-" * 116)
        print(
            f"{a['label']:<26} "
            f"{100*a['maxmap_ratio']:>8.2f}% "
            f"{a['ate_rmse_se3_m']:>10.4f} m "
            f"{a['train_psnr']:>8.2f}/{a['train_ssim']:<10.4f} "
            f"{a['test_psnr']:>8.2f}/{a['test_ssim']:<10.4f} "
            f"{a['fps']:>9.4f} {fmt_g(a['gaussians']):>12}"
        )
        print(
            f"{b['label']:<26} "
            f"{100*b['maxmap_ratio']:>8.2f}% "
            f"{b['ate_rmse_se3_m']:>10.4f} m "
            f"{b['train_psnr']:>8.2f}/{b['train_ssim']:<10.4f} "
            f"{b['test_psnr']:>8.2f}/{b['test_ssim']:<10.4f} "
            f"{'—':>9} {fmt_g(b['gaussians']):>12}"
        )

    print()
    print("Metric protocol: full RGB, no silhouette/depth mask, single-scale SSIM.")
    print("FPS protocol: unique sequence frames / summed Stage-1 online time; boundary-overlap work is charged in time.")
    print("PGO/SR is staged final processing, so FPS is intentionally not reported for the +PGO/SR row.")


if __name__ == "__main__":
    main()
