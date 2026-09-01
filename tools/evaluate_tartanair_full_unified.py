#!/usr/bin/env python3
"""Unified evaluator for the official-style TartanAir LSG-SLAM pipeline.

Protocol used for the paper table:
  * global test frames are 4, 9, 14, ... (pose-only during mapping/SR);
  * PSNR and SSIM are computed on the full RGB image, with NO silhouette mask
    and NO depth mask;
  * SSIM is ordinary single-scale SSIM (utils.slam_external.calc_ssim);
  * every global frame is evaluated exactly once.  At the one-frame boundary
    overlap between official 200-frame submaps, the earlier submap owns the
    boundary frame, matching the released backend's stitched trajectory;
  * pre-PGO ATE is a true camera-center RMSE after rigid SE(3) alignment,
    without scale alignment;
  * post-PGO ATE is read from the already-completed full backend summary;
  * FPS for LSG-SLAM (w/o PGO/SR) is unique_sequence_frames divided by the sum
    of Stage-1 submap online_seconds.  Thus repeated boundary-frame work is
    charged in the denominator while the numerator remains the real input
    sequence length.  Preprocessing, model/dataset setup, and final evaluation
    are excluded by the per-submap timer.  PGO/SR FPS is intentionally N/A.

This script does not rerun SLAM, PGO, Gaussian deformation, or SR.  It uses the
raw (unmasked) RGB renders already saved by the full backend.
"""

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.slam_external import calc_ssim
from diagnose_tartanair_full_pose import (
    _camera_centers_from_w2c,
    _load_stitched_odometry,
    _se3_align,
)


_RENDER_RE = re.compile(r"^(\d+)_(\d+)_rgb\.png$")


def _parse_submap_dir(path, seq):
    name = Path(path).name
    if not name.startswith(seq + "_") or name.endswith("_loops"):
        return None
    parts = name[len(seq) + 1 :].split("_")
    if len(parts) != 3:
        return None
    try:
        return tuple(map(int, parts))
    except ValueError:
        return None


def _submaps(seq_root, seq):
    out = []
    for p in glob.glob(os.path.join(seq_root, f"{seq}_*_*_*")):
        info = _parse_submap_dir(p, seq)
        if info is None:
            continue
        if not os.path.isfile(os.path.join(p, "params.npz")):
            continue
        out.append((info, p))
    out.sort(key=lambda x: x[0][0])
    if not out:
        raise FileNotFoundError(f"No completed Stage-1 submaps found under {seq_root}")
    return out


def _collect_unique_renders(render_dir):
    """Map global frame -> render path, keeping the earlier submap on overlap."""
    candidates = []
    for p in glob.glob(os.path.join(render_dir, "*_rgb.png")):
        m = _RENDER_RE.match(os.path.basename(p))
        if not m:
            continue
        start = int(m.group(1))
        local = int(m.group(2))
        global_idx = start + local
        candidates.append((start, local, global_idx, p))
    candidates.sort(key=lambda x: (x[0], x[1]))

    unique = {}
    duplicates = []
    for start, local, global_idx, p in candidates:
        if global_idx in unique:
            duplicates.append((global_idx, unique[global_idx], p))
            continue
        unique[global_idx] = p
    return unique, duplicates


def _read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float32) / 255.0


def _metric_pair(render_path, gt_path):
    pred = _read_rgb(render_path)
    gt = _read_rgb(gt_path)
    if pred.shape != gt.shape:
        raise RuntimeError(
            f"Image shape mismatch: render={pred.shape} ({render_path}), "
            f"GT={gt.shape} ({gt_path})"
        )

    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)

    mse = torch.mean((pred_t - gt_t) ** 2).item()
    psnr = float("inf") if mse <= 0.0 else float(-10.0 * math.log10(mse))
    ssim = float(calc_ssim(pred_t, gt_t, size_average=True).item())
    return psnr, ssim


def _evaluate_render_set(render_dir, gt_dir, expected_frames):
    renders, duplicates = _collect_unique_renders(render_dir)
    expected = set(range(expected_frames))
    available = set(renders.keys())
    missing = sorted(expected - available)
    extra = sorted(available - expected)
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} global render frames in {render_dir}; "
            f"first missing={missing[:10]}"
        )
    if extra:
        raise RuntimeError(
            f"Unexpected global render frames in {render_dir}: {extra[:10]}"
        )

    train_psnr, train_ssim = [], []
    test_psnr, test_ssim = [], []
    per_frame = []

    for idx in range(expected_frames):
        gt_path = os.path.join(gt_dir, f"{idx:06d}_left.png")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(gt_path)
        psnr, ssim = _metric_pair(renders[idx], gt_path)
        is_test = (idx % 5) == 4
        if is_test:
            test_psnr.append(psnr)
            test_ssim.append(ssim)
        else:
            train_psnr.append(psnr)
            train_ssim.append(ssim)
        per_frame.append((idx, psnr, ssim, "test" if is_test else "train"))

    return {
        "train_psnr": float(np.mean(train_psnr)),
        "train_ssim": float(np.mean(train_ssim)),
        "test_psnr": float(np.mean(test_psnr)),
        "test_ssim": float(np.mean(test_ssim)),
        "train_frames": len(train_psnr),
        "test_frames": len(test_psnr),
        "duplicate_boundary_renders_skipped": len(duplicates),
        "per_frame": per_frame,
    }


def _aggregate_stage1(seq_root, seq):
    submaps = _submaps(seq_root, seq)
    total_seconds = 0.0
    processed_frames_with_overlap = 0
    gaussian_total = 0
    starts_ends = []

    for (start, end, stride), folder in submaps:
        summary_path = os.path.join(folder, "benchmark_summary_split.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(
                f"Missing per-submap timing summary: {summary_path}. "
                "Stage-1 must have been run with scripts/tartanair_split_splatam.py."
            )
        with open(summary_path, "r", encoding="utf-8") as f:
            x = json.load(f)
        sec = float(x["online_seconds"])
        if not np.isfinite(sec) or sec <= 0:
            raise RuntimeError(f"Invalid online_seconds in {summary_path}: {sec}")
        total_seconds += sec
        processed_frames_with_overlap += int(x["num_frames"])

        params = np.load(os.path.join(folder, "params.npz"), allow_pickle=True)
        gaussian_total += int(np.asarray(params["means3D"]).shape[0])
        starts_ends.append([start, end, stride])

    unique_frames = max(x[1] for x in starts_ends) - min(x[0] for x in starts_ends) + 1
    fps = unique_frames / total_seconds
    return {
        "online_seconds_sum": total_seconds,
        "unique_sequence_frames": unique_frames,
        "processed_stage1_frames_including_overlap": processed_frames_with_overlap,
        "fps": fps,
        "gaussians": gaussian_total,
        "submaps": starts_ends,
    }


def _pre_pgo_ate(seq_root, seq):
    est_w2c, gt_w2c, _ = _load_stitched_odometry(seq_root, seq)
    est_c = _camera_centers_from_w2c(est_w2c)
    gt_c = _camera_centers_from_w2c(gt_w2c)
    valid = np.isfinite(est_c).all(axis=1) & np.isfinite(gt_c).all(axis=1)
    ate = _se3_align(gt_c, est_c)
    return float(ate), int(valid.sum()), int(len(valid))


def _save_per_frame_csv(path, before_rows, after_rows):
    before = {i: (p, s, role) for i, p, s, role in before_rows}
    after = {i: (p, s, role) for i, p, s, role in after_rows}
    with open(path, "w", encoding="utf-8") as f:
        f.write("frame,split,wopgo_psnr,wopgo_ssim,full_psnr,full_ssim\n")
        for idx in sorted(before):
            bp, bs, role = before[idx]
            ap, ass, _ = after[idx]
            f.write(f"{idx},{role},{bp:.10f},{bs:.10f},{ap:.10f},{ass:.10f}\n")


def evaluate_sequence(root, data_root, seq):
    seq_root = os.path.join(root, seq)
    render_root = os.path.join(seq_root, "RenderingResult")
    before_dir = os.path.join(render_root, "before_opt_render_rgb")
    after_dir = os.path.join(render_root, "after_opt_render_rgb")
    gt_dir = os.path.join(data_root, "stereo", seq, "image_left")

    stage1 = _aggregate_stage1(seq_root, seq)
    num_frames = int(stage1["unique_sequence_frames"])
    before = _evaluate_render_set(before_dir, gt_dir, num_frames)
    after = _evaluate_render_set(after_dir, gt_dir, num_frames)

    pre_ate, pre_tracked, pre_total = _pre_pgo_ate(seq_root, seq)

    full_summary_path = os.path.join(seq_root, "benchmark_summary_full_split.json")
    if not os.path.isfile(full_summary_path):
        raise FileNotFoundError(full_summary_path)
    with open(full_summary_path, "r", encoding="utf-8") as f:
        full = json.load(f)

    summary = {
        "sequence": seq,
        "protocol": {
            "test_rule": "global frame index % 5 == 4; test is pose-only during mapping and excluded from SR loss",
            "rgb_metric": "full RGB image; no silhouette mask; no depth mask; saved raw 8-bit renders",
            "psnr": "-10*log10(mean RGB squared error), data range 1",
            "ssim": "single-scale SSIM, 11x11 Gaussian window, utils.slam_external.calc_ssim",
            "duplicate_submap_boundary_rule": "each global frame counted once; earlier submap owns duplicated boundary frame",
            "ate": "camera-center true RMSE after rigid SE(3) alignment; no scale alignment",
            "fps": "unique sequence frames / sum(Stage-1 submap online_seconds); overlap work remains in denominator; preprocessing/setup/final eval excluded",
        },
        "without_pgo_sr": {
            "label": "LSG-SLAM (w/o PGO/SR)",
            "maxmap_ratio": float(pre_tracked / pre_total),
            "tracked_frames": pre_tracked,
            "num_frames": pre_total,
            "ate_rmse_se3_m": pre_ate,
            "train_psnr": before["train_psnr"],
            "train_ssim": before["train_ssim"],
            "test_psnr": before["test_psnr"],
            "test_ssim": before["test_ssim"],
            "fps": float(stage1["fps"]),
            "online_seconds": float(stage1["online_seconds_sum"]),
            "gaussians": int(stage1["gaussians"]),
        },
        "with_pgo_sr": {
            "label": "LSG-SLAM (+ PGO/SR)",
            "maxmap_ratio": float(full["maxmap_ratio"]),
            "tracked_frames": int(full["tracked_frames"]),
            "num_frames": int(full["num_frames"]),
            "ate_rmse_se3_m": float(full["ate_rmse_se3_m"]),
            "train_psnr": after["train_psnr"],
            "train_ssim": after["train_ssim"],
            "test_psnr": after["test_psnr"],
            "test_ssim": after["test_ssim"],
            "fps": None,
            "gaussians": int(full["gaussians"]),
        },
        "stage1_timing": stage1,
        "render_accounting": {
            "before_duplicate_boundaries_skipped": before["duplicate_boundary_renders_skipped"],
            "after_duplicate_boundaries_skipped": after["duplicate_boundary_renders_skipped"],
            "train_frames": before["train_frames"],
            "test_frames": before["test_frames"],
        },
    }

    out_json = os.path.join(seq_root, "benchmark_summary_unified.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _save_per_frame_csv(
        os.path.join(seq_root, "benchmark_metrics_unified_per_frame.csv"),
        before["per_frame"],
        after["per_frame"],
    )
    return summary


def _fmt_g(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _print_summary(x):
    seq = x["sequence"]
    a = x["without_pgo_sr"]
    b = x["with_pgo_sr"]
    print()
    print(seq)
    print(
        f"{'Method':<26} {'MaxMap':>9} {'ATE RMSE↓':>12} "
        f"{'Train PSNR/SSIM':>21} {'Test PSNR/SSIM':>21} {'FPS':>9} {'Gaussians':>12}"
    )
    print("-" * 116)
    print(
        f"{a['label']:<26} {100*a['maxmap_ratio']:>8.2f}% {a['ate_rmse_se3_m']:>10.4f} m "
        f"{a['train_psnr']:>8.2f}/{a['train_ssim']:<10.4f} "
        f"{a['test_psnr']:>8.2f}/{a['test_ssim']:<10.4f} "
        f"{a['fps']:>9.4f} {_fmt_g(a['gaussians']):>12}"
    )
    print(
        f"{b['label']:<26} {100*b['maxmap_ratio']:>8.2f}% {b['ate_rmse_se3_m']:>10.4f} m "
        f"{b['train_psnr']:>8.2f}/{b['train_ssim']:<10.4f} "
        f"{b['test_psnr']:>8.2f}/{b['test_ssim']:<10.4f} "
        f"{'—':>9} {_fmt_g(b['gaussians']):>12}"
    )
    t = x["stage1_timing"]
    print(
        f"FPS scope: {t['unique_sequence_frames']} unique frames / "
        f"{t['online_seconds_sum']:.3f}s Stage-1 online time "
        f"({t['processed_stage1_frames_including_overlap']} processed incl. overlaps)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=["SH000", "SH001", "SH002", "SH003"])
    ap.add_argument("--root", default="experiments/tartanair_official_full_split")
    ap.add_argument(
        "--data-root",
        default=os.environ.get(
            "TARTANAIR_DATA_ROOT",
            "/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge",
        ),
    )
    args = ap.parse_args()

    for seq in args.sequences:
        x = evaluate_sequence(args.root, args.data_root, seq)
        _print_summary(x)

    print()
    print("Unified metrics written to <seq>/benchmark_summary_unified.json")
    print("Per-frame metrics written to <seq>/benchmark_metrics_unified_per_frame.csv")


if __name__ == "__main__":
    main()
