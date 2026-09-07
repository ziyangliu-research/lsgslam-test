#!/usr/bin/env python3
import argparse
import json
import os

DEFAULT = ["MH02", "V101", "V201", "MH05"]
SCENE = {
    "MH02": "MH_02_easy",
    "V101": "V1_01_easy",
    "V201": "V2_01_easy",
    "MH05": "MH_05_difficult",
}


def gk(n):
    return f"{int(n)/1000.0:.1f}k"


def status(root, alias):
    d = os.path.join(root, alias)
    unified = os.path.join(d, "benchmark_summary_unified.json")
    if os.path.isfile(unified):
        return "DONE", unified
    if os.path.isfile(os.path.join(d, ".full_backend_complete")):
        return "READY (eval pending)", None
    if os.path.isdir(d):
        # Any submap/loop/backend artifact means work has started.
        try:
            if os.listdir(d):
                return "IN PROGRESS", None
        except OSError:
            pass
    return "MISSING", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=DEFAULT)
    ap.add_argument("--root", default="experiments/euroc_official_full_stride5_final4")
    args = ap.parse_args()

    print(
        f"{'Seq':<6} {'Method / Status':<28} {'MaxMap':>8} {'ATE↓':>10} "
        f"{'Train P/S/L':>24} {'Test P/S/L':>24} {'FPS':>8} {'Time(s)':>10} {'G':>11}"
    )
    print("-" * 140)

    counts = {"DONE": 0, "READY (eval pending)": 0, "IN PROGRESS": 0, "MISSING": 0}
    for alias in args.sequences:
        st, p = status(args.root, alias)
        counts[st] += 1
        if st != "DONE":
            print(
                f"{alias:<6} {st:<28} {'—':>8} {'—':>10} {'—':>24} {'—':>24} "
                f"{'—':>8} {'—':>10} {'—':>11}"
            )
            continue

        with open(p, "r", encoding="utf-8") as f:
            x = json.load(f)
        for i, key in enumerate(("without_pgo_sr", "with_pgo_sr")):
            r = x[key]
            fps = "—" if r.get("fps") is None else f"{r['fps']:.4f}"
            seq_col = alias if i == 0 else ""
            print(
                f"{seq_col:<6} {r['label']:<28} {100*r['maxmap_ratio']:>7.2f}% "
                f"{r['ate_rmse_se3_m']:>8.4f}m "
                f"{r['train_psnr']:>6.2f}/{r['train_ssim']:.4f}/{r['train_lpips']:.4f} "
                f"{r['test_psnr']:>6.2f}/{r['test_ssim']:.4f}/{r['test_lpips']:.4f} "
                f"{fps:>8} {r['time_seconds']:>10.1f} {gk(r['gaussians']):>11}"
            )

    total = len(args.sequences)
    print()
    print(
        f"DONE {counts['DONE']}/{total} | READY {counts['READY (eval pending)']}/{total} | "
        f"IN PROGRESS {counts['IN PROGRESS']}/{total} | MISSING {counts['MISSING']}/{total}"
    )
    print("Protocol: full synchronized EuRoC sequence -> stride=5 -> strict 8:2 holdout.")
    print("Metrics: full image PSNR / single-scale SSIM / LPIPS(AlexNet), no silhouette/depth mask.")
    print("Time(s): online row=online time; +PGO/SR row=offline time; metric/preprocess time excluded.")


if __name__ == "__main__":
    main()
