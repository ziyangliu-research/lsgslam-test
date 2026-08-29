#!/usr/bin/env python3
import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", nargs="+")
    parser.add_argument(
        "--root",
        default="experiments/tartanair_split",
        help="Root containing <SEQ>_full_split5 directories",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Output CSV path; default: <root>/summary_split5.csv",
    )
    args = parser.parse_args()

    rows = []
    for seq in args.sequences:
        path = os.path.join(
            args.root, f"{seq}_full_split5", "benchmark_summary_split.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)
        rows.append(
            {
                "Sequence": seq,
                "MaxMap": x["maxmap_ratio"],
                "Train PSNR": x["train_psnr"],
                "Train SSIM": x["train_ssim"],
                "Test PSNR": x["test_psnr"],
                "Test SSIM": x["test_ssim"],
                "ATE(m)": x["ate_rmse_se3_m"],
                "FPS": x["online_fps"],
                "Gaussians": x["gaussians"],
            }
        )

    headers = [
        "Sequence",
        "MaxMap",
        "Train PSNR",
        "Train SSIM",
        "Test PSNR",
        "Test SSIM",
        "ATE(m)",
        "FPS",
        "Gaussians",
    ]

    print(
        f"{'Sequence':<10} {'MaxMap':>9} {'Train PSNR/SSIM':>21} "
        f"{'Test PSNR/SSIM':>21} {'ATE(m)':>11} {'FPS':>10} {'Gaussians':>12}"
    )
    print("-" * 101)
    for r in rows:
        print(
            f"{r['Sequence']:<10} "
            f"{100.0*r['MaxMap']:>8.2f}% "
            f"{r['Train PSNR']:>8.4f}/{r['Train SSIM']:<10.6f} "
            f"{r['Test PSNR']:>8.4f}/{r['Test SSIM']:<10.6f} "
            f"{r['ATE(m)']:>11.6f} "
            f"{r['FPS']:>10.4f} "
            f"{r['Gaussians']:>12d}"
        )

    csv_path = args.csv or os.path.join(args.root, "summary_split5.csv")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
