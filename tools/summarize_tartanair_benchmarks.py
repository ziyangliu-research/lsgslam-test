#!/usr/bin/env python3
"""Print a compact table from TartanAir benchmark_summary.json files."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequences",
        nargs="+",
        help="Sequence names, e.g. SH000 SH001 SH002 SH003",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=39)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workdir", default="experiments/tartanair")
    args = parser.parse_args()

    header = (
        f"{'Sequence':<9} {'PSNR':>9} {'MS-SSIM':>9} {'Gaussians':>12} "
        f"{'ATE-SE3(m)':>12} {'FPS':>9}"
    )
    print(header)
    print("-" * len(header))

    for seq in args.sequences:
        run_name = f"{seq}_{args.start}_{args.end}_{args.stride}"
        path = os.path.join(args.workdir, run_name, "benchmark_summary.json")
        if not os.path.exists(path):
            print(f"{seq:<9} {'MISSING':>9}  {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        print(
            f"{seq:<9} "
            f"{s['psnr']:>9.4f} "
            f"{s['ms_ssim']:>9.6f} "
            f"{s['gaussians']:>12d} "
            f"{s['ate_rmse_se3_m']:>12.6f} "
            f"{s['online_fps']:>9.4f}"
        )


if __name__ == "__main__":
    main()
