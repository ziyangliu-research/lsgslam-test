#!/usr/bin/env python3
"""Unified evaluator for the ETH3D LSG-SLAM benchmark.

Protocol:
  * full rectified ETH3D stereo sequence, stride=1;
  * strict 8:2 split: frame ids 4,9,14,... are pose-only/test;
  * full-image PSNR, single-scale SSIM, LPIPS(AlexNet), no silhouette/depth mask;
  * ATE is camera-center true RMSE after rigid SE(3) alignment, no scale;
  * online FPS = unique frames / summed Stage-1 online time;
  * +PGO/SR Time(s) = loop + PGO + deformation + SR;
  * preprocessing and metric evaluation are excluded from algorithm time;
  * held-out GT / pre-PGO / post-SR renders are copied to test_render_results/.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
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

DEFAULT_SEQS = ["mannequin_face_1", "einstein_1", "sofa_3", "plant_scene_3"]
_RENDER_RE = re.compile(r"^(\d+)_(\d+)_rgb\.png$")


def _parse_submap_dir(path, scene):
    name = Path(path).name
    prefix = scene + "_"
    if not name.startswith(prefix) or name.endswith("_loops"):
        return None
    parts = name[len(prefix):].split("_")
    if len(parts) != 3:
        return None
    try:
        return tuple(map(int, parts))
    except ValueError:
        return None


def _submaps(seq_root, scene):
    out = []
    for p in glob.glob(os.path.join(seq_root, f"{scene}_*_*_*")):
        info = _parse_submap_dir(p, scene)
        if info is None or not os.path.isfile(os.path.join(p, "params.npz")):
            continue
        out.append((info, p))
    out.sort(key=lambda x: x[0][0])
    if not out:
        raise FileNotFoundError(f"No completed Stage-1 submaps found under {seq_root}")
    return out


def _selected_indices(submaps):
    seen = set()
    ordered = []
    duplicates = 0
    for (start, end, stride), _folder in submaps:
        for idx in range(start, end + 1, stride):
            if idx in seen:
                duplicates += 1
                continue
            seen.add(idx)
            ordered.append(idx)
    return ordered, duplicates


def _collect_unique_renders(render_dir):
    candidates = []
    for p in glob.glob(os.path.join(render_dir, "*_rgb.png")):
        m = _RENDER_RE.match(os.path.basename(p))
        if not m:
            continue
        start = int(m.group(1))
        saved_offset = int(m.group(2))
        dataset_idx = start + saved_offset
        candidates.append((start, saved_offset, dataset_idx, p))
    candidates.sort(key=lambda x: (x[0], x[1]))

    unique = {}
    duplicates = []
    for _start, _offset, idx, p in candidates:
        if idx in unique:
            duplicates.append((idx, unique[idx], p))
            continue
        unique[idx] = p
    return unique, duplicates


def _ordered_gt_paths(seq_dir):
    image_dir = Path(seq_dir) / "image_left"
    stamps_file = Path(seq_dir) / "timestamps.txt"
    if stamps_file.is_file():
        stamps = [x.strip() for x in stamps_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        paths = []
        for stamp in stamps:
            p = image_dir / f"{stamp}.png"
            if not p.is_file():
                raise FileNotFoundError(p)
            paths.append(str(p))
        return paths
    paths = list(image_dir.glob("*.png"))
    try:
        paths.sort(key=lambda p: float(p.stem))
    except ValueError:
        paths.sort(key=lambda p: p.name)
    return [str(p) for p in paths]


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
            f"Image shape mismatch: render={pred.shape} ({render_path}), gt={gt.shape} ({gt_path})"
        )
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        mse = torch.mean((pred_t - gt_t) ** 2).item()
        psnr = float("inf") if mse <= 0 else float(-10.0 * math.log10(mse))
        ssim = float(calc_ssim(pred_t, gt_t, size_average=True).item())
        lpips = float(lpips_model(pred_t, gt_t).item())
    return psnr, ssim, lpips


def _evaluate_render_set(render_dir, gt_paths, selected, lpips_model, device):
    renders, duplicate_renders = _collect_unique_renders(render_dir)
    expected = set(selected)
    available = set(renders)
    missing = sorted(expected - available)
    extra = sorted(available - expected)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} renders in {render_dir}; first={missing[:10]}")
    if extra:
        raise RuntimeError(f"Unexpected render indices in {render_dir}; first={extra[:10]}")

    train = {"psnr": [], "ssim": [], "lpips": []}
    test = {"psnr": [], "ssim": [], "lpips": []}
    rows = []
    for idx in selected:
        if idx >= len(gt_paths):
            raise RuntimeError(f"Frame index {idx} exceeds ETH3D GT image list ({len(gt_paths)})")
        p, s, l = _metric_pair(renders[idx], gt_paths[idx], lpips_model, device)
        is_test = (idx % 5) == 4
        dst = test if is_test else train
        dst["psnr"].append(p)
        dst["ssim"].append(s)
        dst["lpips"].append(l)
        rows.append((idx, Path(gt_paths[idx]).stem, "test" if is_test else "train", p, s, l, renders[idx]))

    def mean(x):
        return float(np.mean(x)) if x else float("nan")

    return {
        "train_psnr": mean(train["psnr"]),
        "train_ssim": mean(train["ssim"]),
        "train_lpips": mean(train["lpips"]),
        "test_psnr": mean(test["psnr"]),
        "test_ssim": mean(test["ssim"]),
        "test_lpips": mean(test["lpips"]),
        "train_frames": len(train["psnr"]),
        "test_frames": len(test["psnr"]),
        "duplicate_boundary_renders_skipped": len(duplicate_renders),
        "rows": rows,
        "renders": renders,
    }


def _aggregate_stage1(seq_root, scene):
    submaps = _submaps(seq_root, scene)
    selected, duplicate_samples = _selected_indices(submaps)
    total_seconds = 0.0
    processed_with_overlap = 0
    gaussians = 0
    infos = []
    for info, folder in submaps:
        summary_path = os.path.join(folder, "benchmark_summary_split.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            x = json.load(f)
        sec = float(x["online_seconds"])
        if not np.isfinite(sec) or sec <= 0:
            raise RuntimeError(f"Invalid online_seconds in {summary_path}: {sec}")
        total_seconds += sec
        processed_with_overlap += int(x["num_frames"])
        params = np.load(os.path.join(folder, "params.npz"), allow_pickle=True)
        gaussians += int(np.asarray(params["means3D"]).shape[0])
        infos.append(list(info))
    return {
        "online_seconds_sum": total_seconds,
        "unique_frames": len(selected),
        "processed_stage1_frames_including_overlap": processed_with_overlap,
        "duplicate_boundary_samples": duplicate_samples,
        "fps": len(selected) / total_seconds,
        "gaussians": gaussians,
        "submaps": infos,
        "selected_indices": selected,
    }


def _pre_pgo_ate(seq_root, scene):
    est_w2c, gt_w2c, _ = _load_stitched_odometry(seq_root, scene)
    est_c = _camera_centers_from_w2c(est_w2c)
    gt_c = _camera_centers_from_w2c(gt_w2c)
    valid = np.isfinite(est_c).all(axis=1) & np.isfinite(gt_c).all(axis=1)
    ate = _se3_align(gt_c, est_c)
    return float(ate), int(valid.sum()), int(len(valid))


def _load_offline_timing(seq_root):
    loop_path = os.path.join(seq_root, "loop_stage_timing.json")
    backend_path = os.path.join(seq_root, "backend_optimization_timing.json")
    for p in (loop_path, backend_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    with open(loop_path, "r", encoding="utf-8") as f:
        loop = json.load(f)
    with open(backend_path, "r", encoding="utf-8") as f:
        backend = json.load(f)
    loop_s = float(loop["wall_seconds"])
    backend_s = float(backend["backend_optimization_seconds"])
    return {
        "loop_closure_seconds": loop_s,
        "pgo_seconds": float(backend["pgo_seconds"]),
        "gaussian_deformation_seconds": float(backend["gaussian_deformation_seconds"]),
        "structure_refinement_seconds": float(backend["structure_refinement_seconds"]),
        "backend_optimization_seconds": backend_s,
        "offline_seconds": loop_s + backend_s,
    }


def _save_per_frame_csv(path, before_rows, after_rows):
    b = {r[0]: r for r in before_rows}
    a = {r[0]: r for r in after_rows}
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "timestamp", "split",
            "wopgo_psnr", "wopgo_ssim", "wopgo_lpips",
            "full_psnr", "full_ssim", "full_lpips",
        ])
        for idx in sorted(b):
            bi = b[idx]
            ai = a[idx]
            w.writerow([
                idx, bi[1], bi[2],
                f"{bi[3]:.10f}", f"{bi[4]:.10f}", f"{bi[5]:.10f}",
                f"{ai[3]:.10f}", f"{ai[4]:.10f}", f"{ai[5]:.10f}",
            ])


def _save_test_renders(seq_root, gt_paths, selected, before_renders, after_renders):
    out_root = Path(seq_root) / "test_render_results"
    if out_root.exists():
        shutil.rmtree(out_root)
    dirs = {
        "gt": out_root / "gt",
        "without_pgo_sr": out_root / "without_pgo_sr",
        "with_pgo_sr": out_root / "with_pgo_sr",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in selected:
        if idx % 5 != 4:
            continue
        stamp = Path(gt_paths[idx]).stem
        name = f"{idx:06d}_{stamp}.png"
        shutil.copy2(gt_paths[idx], dirs["gt"] / name)
        shutil.copy2(before_renders[idx], dirs["without_pgo_sr"] / name)
        shutil.copy2(after_renders[idx], dirs["with_pgo_sr"] / name)
        count += 1
    (out_root / "README.txt").write_text(
        "Held-out ETH3D test views (frame % 5 == 4).\n"
        "gt: rectified left-camera ground truth image\n"
        "without_pgo_sr: raw render before PGO/deformation/SR\n"
        "with_pgo_sr: raw render after PGO/deformation/train-only SR\n",
        encoding="utf-8",
    )
    return count


def evaluate_sequence(root, data_root, scene, lpips_model, device):
    seq_root = os.path.join(root, scene)
    seq_dir = os.path.join(data_root, scene)
    gt_paths = _ordered_gt_paths(seq_dir)

    stage1 = _aggregate_stage1(seq_root, scene)
    selected = stage1["selected_indices"]
    render_root = os.path.join(seq_root, "RenderingResult")
    before = _evaluate_render_set(
        os.path.join(render_root, "before_opt_render_rgb"), gt_paths, selected, lpips_model, device
    )
    after = _evaluate_render_set(
        os.path.join(render_root, "after_opt_render_rgb"), gt_paths, selected, lpips_model, device
    )

    pre_ate, pre_tracked, pre_total = _pre_pgo_ate(seq_root, scene)
    offline = _load_offline_timing(seq_root)
    full_path = os.path.join(seq_root, "benchmark_summary_full_split.json")
    if not os.path.isfile(full_path):
        raise FileNotFoundError(full_path)
    with open(full_path, "r", encoding="utf-8") as f:
        full = json.load(f)

    online = float(stage1["online_seconds_sum"])
    offline_s = float(offline["offline_seconds"])
    end_to_end = online + offline_s

    saved_test = _save_test_renders(
        seq_root, gt_paths, selected, before["renders"], after["renders"]
    )

    gt_stats_path = os.path.join(seq_dir, "gt_association_stats.json")
    gt_stats = None
    if os.path.isfile(gt_stats_path):
        with open(gt_stats_path, "r", encoding="utf-8") as f:
            gt_stats = json.load(f)

    summary = {
        "sequence": scene,
        "protocol": {
            "input": "rectified ETH3D left/right stereo; full sequence; stride=1",
            "test_rule": "frame index % 5 == 4; test is pose-only during mapping and excluded from SR loss",
            "rgb_metric": "full RGB image; no silhouette mask; no depth mask; raw saved renders",
            "psnr": "-10*log10(mean RGB squared error), data range 1",
            "ssim": "single-scale SSIM, utils.slam_external.calc_ssim",
            "lpips": "LPIPS AlexNet via torchmetrics, normalize=True",
            "ate": "camera-center true RMSE after rigid SE(3) alignment; no scale",
            "fps": "unique sequence frames / summed Stage-1 online time; submap-overlap work remains in denominator",
            "time": "w/o row=online time; +PGO/SR row=offline time; preprocessing/final metric eval excluded",
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
            "time_seconds": online,
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
            "time_seconds": offline_s,
            "gaussians": int(full["gaussians"]),
        },
        "timing": {
            "online_seconds": online,
            **offline,
            "end_to_end_algorithm_seconds": end_to_end,
        },
        "stage1_timing": {k: v for k, v in stage1.items() if k != "selected_indices"},
        "render_accounting": {
            "train_frames": before["train_frames"],
            "test_frames": before["test_frames"],
            "saved_test_render_triplets": saved_test,
            "test_render_dir": os.path.join(seq_root, "test_render_results"),
            "before_duplicate_boundaries_skipped": before["duplicate_boundary_renders_skipped"],
            "after_duplicate_boundaries_skipped": after["duplicate_boundary_renders_skipped"],
        },
        "gt_association_stats": gt_stats,
    }

    out = os.path.join(seq_root, "benchmark_summary_unified.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _save_per_frame_csv(
        os.path.join(seq_root, "benchmark_metrics_unified_per_frame.csv"),
        before["rows"], after["rows"],
    )
    return summary


def _fmt_g(n):
    return f"{int(n) / 1000.0:.1f}k"


def _print_summary(x):
    print(f"\n{x['sequence']}")
    print(f"{'Method':<26} {'MaxMap':>9} {'ATE↓':>10} {'Train P/S/L':>24} {'Test P/S/L':>24} {'FPS':>8} {'Time(s)':>10} {'G':>11}")
    print("-" * 130)
    for key in ("without_pgo_sr", "with_pgo_sr"):
        r = x[key]
        fps = "—" if r["fps"] is None else f"{r['fps']:.4f}"
        print(
            f"{r['label']:<26} {100*r['maxmap_ratio']:>8.2f}% {r['ate_rmse_se3_m']:>8.4f}m "
            f"{r['train_psnr']:>6.2f}/{r['train_ssim']:.4f}/{r['train_lpips']:.4f} "
            f"{r['test_psnr']:>6.2f}/{r['test_ssim']:.4f}/{r['test_lpips']:.4f} "
            f"{fps:>8} {r['time_seconds']:>10.1f} {_fmt_g(r['gaussians']):>11}"
        )
    print(f"Saved held-out render triplets: {x['render_accounting']['saved_test_render_triplets']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--root", default="experiments/eth3d_full_split_final4")
    ap.add_argument("--data-root", default="/home/shiyo/Desktop/Datasets/ETH3D_rectified")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Metric device: {device}")
    print("Metric protocol: FULL IMAGE, NO silhouette mask, NO depth mask")
    print("Initializing LPIPS(AlexNet)...")
    lpips_model = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device).eval()

    for scene in args.sequences:
        x = evaluate_sequence(args.root, args.data_root, scene, lpips_model, device)
        _print_summary(x)


if __name__ == "__main__":
    main()
