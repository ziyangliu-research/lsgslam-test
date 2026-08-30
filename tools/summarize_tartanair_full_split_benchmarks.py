#!/usr/bin/env python3
import argparse
import json
import os


def fmt_g(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", nargs="+", default=["SH000", "SH001", "SH002", "SH003"])
    parser.add_argument("--root", default="experiments/tartanair_official_full_split")
    args = parser.parse_args()

    print(
        f"{'Sequence':<10} {'MaxMap':>9} {'ATE RMSE↓':>12} "
        f"{'Train PSNR/SSIM':>21} {'Test PSNR/SSIM':>21} {'Gaussians':>12}"
    )
    print("-" * 93)

    for seq in args.sequences:
        p = os.path.join(args.root, seq, "benchmark_summary_full_split.json")
        with open(p, "r", encoding="utf-8") as f:
            x = json.load(f)
        print(
            f"{seq:<10} "
            f"{100*x['maxmap_ratio']:>8.2f}% "
            f"{x['ate_rmse_se3_m']:>10.4f} m "
            f"{x['after_sr_train_psnr']:>8.2f}/{x['after_sr_train_ssim']:<10.4f} "
            f"{x['after_sr_test_psnr']:>8.2f}/{x['after_sr_test_ssim']:<10.4f} "
            f"{fmt_g(x['gaussians']):>12}"
        )


if __name__ == "__main__":
    main()
