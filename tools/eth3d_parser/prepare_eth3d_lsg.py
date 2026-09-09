#!/usr/bin/env python3
"""Prepare rectified ETH3D stereo sequences for the LSG-SLAM benchmark.

Input layout (already produced by rectify_eth3d_stereo.py):
  <root>/<sequence>/image_left/*.png
  <root>/<sequence>/image_right/*.png
  <root>/<sequence>/calibration.json
  <root>/<sequence>/groundtruth_left.txt
  <root>/<sequence>/timestamps.txt

This script is non-destructive. It adds cached LSG-SLAM inputs next to the
rectified data:
  data_rect -> image_left (relative symlink, for native EurocDataset reuse)
  depth_sceneflow/*.npy      IGEV-Stereo depth
  global_features/*.npy      TransVPR descriptors
  traj.txt                    image-timestamp-aligned rectified-left c2w poses
  eth3d_lsg.yaml              per-sequence rectified intrinsics
  gt_association_stats.json   timestamp-association audit

Preprocessing time is not part of benchmark runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

_BASE_DIR = Path(__file__).resolve().parents[2]
_IGEV_DIR = _BASE_DIR / "third_party" / "IGEV-Stereo"
sys.path.insert(0, str(_BASE_DIR))
sys.path.insert(0, str(_IGEV_DIR))

from core.igev_stereo import IGEVStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402
from third_party.TransVPR.feature_extractor import Extractor_base  # noqa: E402
from third_party.TransVPR.blocks import POOL  # noqa: E402

DEFAULT_SEQS = ["mannequin_face_1", "einstein_1", "sofa_3", "plant_scene_3"]


def _numeric_pngs(folder: Path) -> list[Path]:
    paths = list(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found under {folder}")
    try:
        return sorted(paths, key=lambda p: float(p.stem))
    except ValueError:
        return sorted(paths, key=lambda p: p.name)


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("Invalid quaternion in ETH3D GT")
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_gt(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = np.fromstring(line, sep=" ", dtype=np.float64)
            if vals.size == 8:
                rows.append(vals)
    if not rows:
        raise RuntimeError(f"No timestamp + tx ty tz qx qy qz qw rows found in {path}")
    gt = np.asarray(rows, dtype=np.float64)
    order = np.argsort(gt[:, 0])
    return gt[order]


def _write_aligned_traj(seq_dir: Path, image_paths: list[Path]) -> dict:
    gt_path = seq_dir / "groundtruth_left.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    gt = _load_gt(gt_path)
    gt_ts = gt[:, 0]

    diffs = []
    associated = []
    for p in image_paths:
        t = float(p.stem)
        k = int(np.searchsorted(gt_ts, t))
        candidates = []
        if k < len(gt_ts):
            candidates.append(k)
        if k > 0:
            candidates.append(k - 1)
        if not candidates:
            raise RuntimeError(f"Could not associate image timestamp {t} to GT")
        j = min(candidates, key=lambda idx: abs(float(gt_ts[idx] - t)))
        dt = abs(float(gt_ts[j] - t))
        diffs.append(dt)
        associated.append((p.stem, gt[j]))

    traj_path = seq_dir / "traj.txt"
    with traj_path.open("w", encoding="utf-8") as f:
        for image_stamp, row in associated:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _quat_xyzw_to_rot(row[4:8])
            T[:3, 3] = row[1:4]
            vals = T[:3, :].reshape(-1)
            f.write(image_stamp + " " + " ".join(f"{x:.12g}" for x in vals) + "\n")

    arr = np.asarray(diffs, dtype=np.float64)
    stats = {
        "num_images": len(image_paths),
        "num_gt_rows": int(len(gt)),
        "nearest_gt_dt_seconds": {
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "over_0p05s": int((arr > 0.05).sum()),
            "over_0p10s": int((arr > 0.10).sum()),
        },
        "note": (
            "traj.txt uses the nearest rectified-left GT pose for each image timestamp. "
            "Inspect large timestamp gaps before using ATE in a paper."
        ),
    }
    (seq_dir / "gt_association_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(
        "GT association: "
        f"max_dt={stats['nearest_gt_dt_seconds']['max']:.6f}s, "
        f">0.05s={stats['nearest_gt_dt_seconds']['over_0p05s']}, "
        f">0.10s={stats['nearest_gt_dt_seconds']['over_0p10s']}"
    )
    if stats["nearest_gt_dt_seconds"]["over_0p10s"]:
        print("WARNING: some ETH3D image timestamps are >0.10 s from nearest GT; inspect gt_association_stats.json.")
    return stats


def _write_camera_yaml(seq_dir: Path, calib: dict) -> None:
    K = np.asarray(calib["K_rectified_left"], dtype=np.float64)
    w, h = [int(v) for v in calib["rectified_size"]]
    text = f"""dataset_name: 'euroc'\ncamera_params:\n  image_height: {h}\n  image_width: {w}\n  fx: {K[0,0]:.12g}\n  fy: {K[1,1]:.12g}\n  cx: {K[0,2]:.12g}\n  cy: {K[1,2]:.12g}\n  png_depth_scale: 1.0\n  crop_edge: 0\n  depth_filter_near: 0.1\n  depth_filter_far: 30.0\n"""
    (seq_dir / "eth3d_lsg.yaml").write_text(text, encoding="utf-8")


def _ensure_data_rect_link(seq_dir: Path) -> None:
    link = seq_dir / "data_rect"
    if link.is_symlink():
        if link.resolve() != (seq_dir / "image_left").resolve():
            raise RuntimeError(f"Existing {link} points to unexpected target {link.resolve()}")
        return
    if link.exists():
        # Allow a real directory only when it contains the same filenames.
        a = {p.name for p in link.glob("*.png")}
        b = {p.name for p in (seq_dir / "image_left").glob("*.png")}
        if a != b:
            raise RuntimeError(f"Existing {link} does not match image_left")
        return
    link.symlink_to("image_left", target_is_directory=True)


def _igev_args() -> SimpleNamespace:
    return SimpleNamespace(
        mixed_precision=False,
        valid_iters=32,
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        slow_fast_gru=False,
        n_gru_layers=3,
        max_disp=192,
    )


def _load_igev(device: torch.device):
    ckpt = _IGEV_DIR / "pretrained_models" / "sceneflow.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model = torch.nn.DataParallel(IGEVStereo(_igev_args()), device_ids=[device.index or 0])
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model = model.module.to(device).eval()
    return model


def _load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = np.asarray(Image.open(path)).astype(np.uint8)
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def _prepare_depth(seq_dir: Path, left: list[Path], right_by_name: dict[str, Path], calib: dict) -> None:
    depth_dir = seq_dir / "depth_sceneflow"
    depth_dir.mkdir(exist_ok=True)
    missing = [p for p in left if not (depth_dir / f"{p.stem}.npy").is_file()]
    if not missing:
        print(f"IGEV depth: complete ({len(left)}) -> SKIP")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("IGEV preprocessing requires CUDA in this benchmark environment")
    K = np.asarray(calib["K_rectified_left"], dtype=np.float64)
    baseline = float(calib["baseline_rectified_m"])
    fB = float(K[0, 0]) * baseline
    model = _load_igev(device)

    print(f"IGEV depth: generating {len(missing)} missing frames; fx*B={fB:.6f}")
    with torch.no_grad():
        for lp in tqdm(missing, desc="ETH3D IGEV"):
            rp = right_by_name[lp.name]
            l = _load_image_tensor(lp, device)
            r = _load_image_tensor(rp, device)
            padder = InputPadder(l.shape, divis_by=32)
            l, r = padder.pad(l, r)
            disp = model(l, r, iters=32, test_mode=True)
            disp = padder.unpad(disp).detach().cpu().numpy().squeeze().astype(np.float32)
            depth = np.zeros_like(disp, dtype=np.float32)
            valid = np.isfinite(disp) & (disp > 1e-6)
            depth[valid] = fB / disp[valid]
            depth[(depth < 0.1) | (~np.isfinite(depth))] = 0.0
            np.save(depth_dir / f"{lp.stem}.npy", depth)

    del model
    torch.cuda.empty_cache()


def _prepare_features(seq_dir: Path, left: list[Path]) -> None:
    feat_dir = seq_dir / "global_features"
    feat_dir.mkdir(exist_ok=True)
    missing = [p for p in left if not (feat_dir / f"{p.stem}.npy").is_file()]
    if not missing:
        print(f"TransVPR features: complete ({len(left)}) -> SKIP")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = _BASE_DIR / "third_party" / "TransVPR" / "TransVPR_MSLS.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    model = Extractor_base()
    model.add_module("pool", POOL(model.embedding_dim))
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model = model.to(device).eval()
    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([480, 640]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(f"TransVPR: generating {len(missing)} missing descriptors")
    with torch.no_grad():
        for p in tqdm(missing, desc="ETH3D TransVPR"):
            img = Image.open(p).convert("RGB")
            x = input_transform(img)[None].to(device)
            patch_feat = model(x)
            global_feat, _attention = model.pool(patch_feat)
            np.save(feat_dir / f"{p.stem}.npy", global_feat.detach().cpu().numpy()[0])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_one(root: Path, sequence: str) -> None:
    seq_dir = root / sequence
    left_dir = seq_dir / "image_left"
    right_dir = seq_dir / "image_right"
    calib_path = seq_dir / "calibration.json"
    for p in (seq_dir, left_dir, right_dir):
        if not p.exists():
            raise FileNotFoundError(p)
    if not calib_path.is_file():
        raise FileNotFoundError(calib_path)

    left = _numeric_pngs(left_dir)
    right = _numeric_pngs(right_dir)
    right_by_name = {p.name: p for p in right}
    left_names = {p.name for p in left}
    right_names = set(right_by_name)
    if left_names != right_names:
        raise RuntimeError(
            f"Left/right filename mismatch for {sequence}: "
            f"left_only={sorted(left_names-right_names)[:5]}, "
            f"right_only={sorted(right_names-left_names)[:5]}"
        )

    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    w, h = [int(v) for v in calib["rectified_size"]]
    K = np.asarray(calib["K_rectified_left"], dtype=np.float64)
    baseline = float(calib["baseline_rectified_m"])

    print("\n" + "=" * 80)
    print(f"ETH3D prepare: {sequence}")
    print(f"Path:       {seq_dir}")
    print(f"Frames:     {len(left)}")
    print(f"Resolution: {w} x {h}")
    print(f"fx/fy:      {K[0,0]:.6f} / {K[1,1]:.6f}")
    print(f"baseline:   {baseline:.6f} m")
    print("=" * 80)

    _ensure_data_rect_link(seq_dir)
    _write_camera_yaml(seq_dir, calib)
    _write_aligned_traj(seq_dir, left)
    _prepare_depth(seq_dir, left, right_by_name, calib)
    _prepare_features(seq_dir, left)

    depth_count = len(list((seq_dir / "depth_sceneflow").glob("*.npy")))
    feat_count = len(list((seq_dir / "global_features").glob("*.npy")))
    if depth_count != len(left) or feat_count != len(left):
        raise RuntimeError(
            f"Incomplete ETH3D preprocessing for {sequence}: "
            f"frames={len(left)}, depth={depth_count}, features={feat_count}"
        )
    print(f"[{sequence}] preprocessing COMPLETE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--root", default="/home/shiyo/Desktop/Datasets/ETH3D_rectified")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    for seq in args.sequences:
        prepare_one(root, seq)


if __name__ == "__main__":
    main()
