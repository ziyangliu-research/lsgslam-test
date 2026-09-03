#!/usr/bin/env python3
"""Unified evaluator for the official-style TartanAir LSG-SLAM pipeline.

Paper protocol:
  * test frames are global indices 4,9,14,...;
  * full RGB metrics: no silhouette mask and no depth mask;
  * PSNR, single-scale SSIM, and LPIPS(AlexNet) are reported;
  * each global frame is counted once across one-frame submap overlaps;
  * ATE is camera-center true RMSE after rigid SE(3) alignment, no scale;
  * online FPS = unique sequence frames / summed Stage-1 online time;
  * online Time(s) is Stage-1 online time;
  * offline Time(s) is loop-closure/constraint-generation wall time plus
    backend PGO + Gaussian-deformation + SR optimization time, excluding final
    benchmark metric rendering;
  * end-to-end algorithm time = online + offline and is saved in JSON.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.slam_external import calc_ssim
from diagnose_tartanair_full_pose import (
    _camera_centers_from_w2c,
    _load_stitched_odometry,
    _se3_align,
)


_RENDER_RE = re.compile(r"^(\d+)_(\d+)_rgb\.png$")
DEFAULT_SEQS = ["SE000", "SE001", "SE002", "SE003", "SH000", "SH001", "SH002", "SH003"]


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


def _metric_pair(render_path, gt_path, lpips_model, device):
    pred = _read_rgb(render_path)
    gt = _read_rgb(gt_path)
    if pred.shape != gt.shape:
        raise RuntimeError(
            f"Image shape mismatch: render={pred.shape} ({render_path}), "
            f"GT={gt.shape} ({gt_path})"
        )

    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        mse = torch.mean((pred_t - gt_t) ** 2).item()
        psnr = float("inf") if mse <= 0.0 else float(-10.0 * math.log10(mse))
        ssim = float(calc_ssim(pred_t, gt_t, size_average=True).item())
        lpips = float(lpips_model(pred_t, gt_t).item())
    return psnr, ssim, lpips


def _evaluate_render_set(render_dir, gt_dir, expected_frames, lpips_model, device):
    renders, duplicates = _collect_unique_renders(render_dir)
    expected = set(range(expected_frames))
    available = set(renders.keys())
    missing = sorted(expected - available)
    extra = sorted(available - expected)
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} global render frames in {render_dir}; first missing={missing[:10]}"
        )
    if extra:
        raise RuntimeError(f"Unexpected global render frames in {render_dir}: {extra[:10]}")

    train_psnr, train_ssim, train_lpips = [], [], []
    test_psnr, test_ssim, test_lpips = [], [], []
    per_frame = []

    for idx in range(expected_frames):
        gt_path = os.path.join(gt_dir, f"{idx:06d}_left.png")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(gt_path)
        psnr, ssim, lpips = _metric_pair(renders[idx], gt_path, lpips_model, device)
        is_test = (idx % 5) == 4
        if is_test:
            test_psnr.append(psnr)
            test_ssim.append(ssim)
            test_lpips.append(lpips)
        else:
            train_psnr.append(psnr)
            train_ssim.append(ssim)
            train_lpips.append(lpips)
        per_frame.append((idx, psnr, ssim, lpips, "test" if is_test else "train"))

    return {
        "train_psnr": float(np.mean(train_psnr)),
        "train_ssim": float(np.mean(train_ssim)),
        "train_lpips": float(np.mean(train_lpips)),
        "test_psnr": float(np.mean(test_psnr)),
        "test_ssim": float(np.mean(test_ssim)),
        "test_lpips": float(np.mean(test_lpips)),
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
                "Stage-1 must use scripts/tartanair_split_splatam.py."
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


def _load_offline_timing(seq_root):
    loop_path = os.path.join(seq_root, "loop_stage_timing.json")
    backend_path = os.path.join(seq_root, "backend_optimization_timing.json")
    if not os.path.isfile(loop_path):
        raise FileNotFoundError(
            f"Missing offline loop timing: {loop_path}. Run the final 8-sequence runner."
        )
    if not os.path.isfile(backend_path):
        raise FileNotFoundError(
            f"Missing backend optimization timing: {backend_path}. Run the final 8-sequence runner."
        )
    with open(loop_path, "r", encoding="utf-8") as f:
        loop = json.load(f)
    with open(backend_path, "r", encoding="utf-8") as f:
        backend = json.load(f)

    loop_seconds = float(loop["wall_seconds"])
    backend_seconds = float(backend["backend_optimization_seconds"])
    offline_seconds = loop_seconds + backend_seconds
    return {
        "loop_closure_seconds": loop_seconds,
        "pgo_seconds": float(backend["pgo_seconds"]),
        "gaussian_deformation_seconds": float(backend["gaussian_deformation_seconds"]),
        "structure_refinement_seconds": float(backend["structure_refinement_seconds"]),
        "backend_optimization_seconds": backend_seconds,
        "offline_seconds": offline_seconds,
    }


def _save_per_frame_csv(path, before_rows, after_rows):
    before = {i: (p, s, l, role) for i, p, s, l, role in before_rows}
    after = {i: (p, s, l, role) for i, p, s, l, role in after_rows}
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "split",
            "wopgo_psnr", "wopgo_ssim", "wopgo_lpips",
            "full_psnr", "full_ssim", "full_lpips",
        ])
        for idx in sorted(before):
            bp, bs, bl, role = before[idx]
            ap, ass, al, _ = after[idx]
            w.writerow([idx, role, f"{bp:.10f}", f"{bs:.10f}", f"{bl:.10f}",
                        f"{ap:.10f}", f"{ass:.10f}", f"{al:.10f}"])


def evaluate_sequence(root, data_root, seq, lpips_model, device):
    seq_root = os.path.join(root, seq)
    render_root = os.path.join(seq_root, "RenderingResult")
    before_dir = os.path.join(render_root, "before_opt_render_rgb")
    after_dir = os.path.join(render_root, "after_opt_render_rgb")
    gt_dir = os.path.join(data_root, "stereo", seq, "image_left")

    stage1 = _aggregate_stage1(seq_root, seq)
    num_frames = int(stage1["unique_sequence_frames"])
    before = _evaluate_render_set(before_dir, gt_dir, num_frames, lpips_model, device)
    after = _evaluate_render_set(after_dir, gt_dir, num_frames, lpips_model, device)
    pre_ate, pre_tracked, pre_total = _pre_pgo_ate(seq_root, seq)
    offline = _load_offline_timing(seq_root)

    full_summary_path = os.path.join(seq_root, "benchmark_summary_full_split.json")
    if not os.path.isfile(full_summary_path):
        raise FileNotFoundError(full_summary_path)
    with open(full_summary_path, "r", encoding="utf-8") as f:
        full = json.load(f)

    online_seconds = float(stage1["online_seconds_sum"])
    offline_seconds = float(offline["offline_seconds"])
    total_seconds = online_seconds + offline_seconds

    summary = {
        "sequence": seq,
        "protocol": {
            "test_rule": "global frame index % 5 == 4; test is pose-only during mapping and excluded from SR loss",
            "rgb_metric": "full RGB image; no silhouette mask; no depth mask; saved raw 8-bit renders",
            "psnr": "-10*log10(mean RGB squared error), data range 1",
            "ssim": "single-scale SSIM, 11x11 Gaussian window, utils.slam_external.calc_ssim",
            "lpips": "LPIPS AlexNet via torchmetrics, normalize=True, full RGB image",
            "duplicate_submap_boundary_rule": "each global frame counted once; earlier submap owns duplicated boundary frame",
            "ate": "camera-center true RMSE after rigid SE(3) alignment; no scale alignment",
            "fps": "unique sequence frames / sum(Stage-1 submap online_seconds); overlap work remains in denominator",
            "time": "w/o row reports online seconds; +PGO/SR row reports offline seconds; JSON also stores online+offline end-to-end seconds",
        },
        "without_pgo_sr": {
            "label": "LSG-SLAM (w/o PGO/SR)",
            "maxmap_ratio": float(pre_tracked / pre_total),
            "tracked_frames": pre_tracked,
            "num_frames": pre_total,
            "ate_rmse_se3_m": pre_ate,
            "train_psnr": before["train_psnr"],
            "train_ssim": before["train_ssim"],
            "train_lpips": before["train_lpips"],
            "test_psnr": before["test_psnr"],
            "test_ssim": before["test_ssim"],
            "test_lpips": before["test_lpips"],
            "fps": float(stage1["fps"]),
            "time_seconds": online_seconds,
            "online_seconds": online_seconds,
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
            "train_lpips": after["train_lpips"],
            "test_psnr": after["test_psnr"],
            "test_ssim": after["test_ssim"],
            "test_lpips": after["test_lpips"],
            "fps": None,
            "time_seconds": offline_seconds,
            "offline_seconds": offline_seconds,
            "total_pipeline_seconds": total_seconds,
            "gaussians": int(full["gaussians"]),
        },
        "timing": {
            "online_seconds": online_seconds,
            **offline,
            "end_to_end_algorithm_seconds": total_seconds,
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
        before["per_frame"], after["per_frame"],
    )
    return summary


def _fmt_gk(n):
    return f"{int(n) / 1000.0:.1f}k"


def _print_summary(x):
    seq = x["sequence"]
    a = x["without_pgo_sr"]
    b = x["with_pgo_sr"]
    print()
    print(seq)
    print(
        f"{'Method':<26} {'MaxMap':>9} {'ATE↓':>11} "
        f"{'Train P/S/L':>24} {'Test P/S/L':>24} {'FPS':>8} {'Time(s)':>11} {'G':>11}"
    )
    print("-" * 132)
    print(
        f"{a['label']:<26} {100*a['maxmap_ratio']:>8.2f}% {a['ate_rmse_se3_m']:>8.4f}m "
        f"{a['train_psnr']:>6.2f}/{a['train_ssim']:.4f}/{a['train_lpips']:.4f} "
        f"{a['test_psnr']:>6.2f}/{a['test_ssim']:.4f}/{a['test_lpips']:.4f} "
        f"{a['fps']:>8.4f} {a['time_seconds']:>11.1f} {_fmt_gk(a['gaussians']):>11}"
    )
    print(
        f"{b['label']:<26} {100*b['maxmap_ratio']:>8.2f}% {b['ate_rmse_se3_m']:>8.4f}m "
        f"{b['train_psnr']:>6.2f}/{b['train_ssim']:.4f}/{b['train_lpips']:.4f} "
        f"{b['test_psnr']:>6.2f}/{b['test_ssim']:.4f}/{b['test_lpips']:.4f} "
        f"{'—':>8} {b['time_seconds']:>11.1f} {_fmt_gk(b['gaussians']):>11}"
    )
    t = x["timing"]
    print(
        f"Timing: online={t['online_seconds']:.1f}s, offline={t['offline_seconds']:.1f}s "
        f"(loop={t['loop_closure_seconds']:.1f}s, PGO={t['pgo_seconds']:.1f}s, "
        f"deform={t['gaussian_deformation_seconds']:.1f}s, SR={t['structure_refinement_seconds']:.1f}s), "
        f"end-to-end={t['end_to_end_algorithm_seconds']:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--root", default="experiments/tartanair_official_full_final8")
    ap.add_argument(
        "--data-root",
        default=os.environ.get(
            "TARTANAIR_DATA_ROOT",
            "/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge",
        ),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Metric device: {device}")
    print("Initializing LPIPS(AlexNet)...")
    lpips_model = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device).eval()

    for seq in args.sequences:
        x = evaluate_sequence(args.root, args.data_root, seq, lpips_model, device)
        _print_summary(x)

    print()
    print("Unified metrics written to <seq>/benchmark_summary_unified.json")
    print("Per-frame PSNR/SSIM/LPIPS written to <seq>/benchmark_metrics_unified_per_frame.csv")


if __name__ == "__main__":
    main()
