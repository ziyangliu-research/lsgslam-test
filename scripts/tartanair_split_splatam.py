"""TartanAir 8:2 split runner for LSG-SLAM.

Every fifth frame (global indices 4, 9, 14, ...) is a test frame:
- participates in feature matching, PnP/ICP, and camera-pose tracking;
- does NOT add Gaussians;
- does NOT run mapping optimization;
- does NOT enter the mapping keyframe pool.

The upstream scripts/splatam.py is left untouched.  This entry point loads that
source at runtime and applies three guarded source-level edits so KITTI/EuRoC
and the original baseline remain unchanged.
"""

import argparse
import json
import os
import shutil
import sys
import time
import types
from importlib.machinery import SourceFileLoader

import numpy as np
import torch

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from datasets.gradslam_datasets.tartanair import TartanAirDataset  # noqa: E402
import utils.eval_helpers as eval_helpers_module  # noqa: E402
from utils.slam_external import calc_ssim  # noqa: E402


def _replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not apply split patch '{label}': expected 1 match, found {count}. "
            "The upstream splatam.py may have changed."
        )
    return source.replace(old, new, 1)


def _load_split_splatam():
    source_path = os.path.join(_BASE_DIR, "scripts", "splatam.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    source = _replace_once(
        source,
        "        iter_time_idx = time_idx\n",
        """        iter_time_idx = time_idx

        # TartanAir benchmark split: every fifth GLOBAL frame is held out from
        # mapping, but remains in the sequential pose-tracking stream.
        split_every = int(config.get("eval_split_every", 0))
        default_offset = split_every - 1 if split_every > 0 else -1
        split_offset = int(config.get("eval_split_offset", default_offset))
        if hasattr(dataset, "retained_inds"):
            absolute_frame_idx = int(dataset.retained_inds[time_idx].item())
        else:
            absolute_frame_idx = int(dataset_config.get("start", 0)) + time_idx * int(dataset_config.get("stride", 1))
        is_test_frame = split_every > 0 and (absolute_frame_idx % split_every == split_offset)
        print(f"[Split] frame={absolute_frame_idx} role={'TEST-pose-only' if is_test_frame else 'TRAIN-map'}")
""",
        "frame-role flag",
    )

    source = _replace_once(
        source,
        "        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:\n",
        "        if (not is_test_frame) and (time_idx == 0 or (time_idx+1) % config['map_every'] == 0):\n",
        "skip mapping on test frames",
    )

    source = _replace_once(
        source,
        """        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \\
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
""",
        """        if (not is_test_frame) and (((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \\
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any())):
""",
        "exclude test frames from keyframes",
    )

    module = types.ModuleType("lsg_splatam_tartanair_split")
    module.__file__ = source_path
    module.__package__ = None
    exec(compile(source, source_path, "exec"), module.__dict__)
    return module


lsg_splatam = _load_split_splatam()

_original_get_dataset = lsg_splatam.get_dataset
_original_initialize_first_timestep = lsg_splatam.initialize_first_timestep
_original_eval = lsg_splatam.eval
_original_match_feature = lsg_splatam.match_feature
_timing = {}


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["tartanair", "tartan"]:
        return TartanAirDataset(config_dict, basedir, sequence, **kwargs)
    return _original_get_dataset(config_dict, basedir, sequence, **kwargs)


def _timed_initialize_first_timestep(*args, **kwargs):
    if "online_start" not in _timing:
        _sync_cuda()
        _timing["online_start"] = time.perf_counter()
    return _original_initialize_first_timestep(*args, **kwargs)


def _timed_eval(*args, **kwargs):
    # The upstream final eval starts after the online sequence is complete.
    if "online_end" not in _timing:
        _sync_cuda()
        _timing["online_end"] = time.perf_counter()
    return _original_eval(*args, **kwargs)


def _match_feature_without_plot(*args, **kwargs):
    # Diagnostic match figures are not part of SLAM and would contaminate FPS.
    kwargs["save_path"] = None
    return _original_match_feature(*args, **kwargs)


def _single_scale_ssim(img1, img2, data_range=1.0, size_average=True, **_kwargs):
    # eval_helpers calls a symbol named ms_ssim. Replace only the metric function
    # for this benchmark so all reported "SSIM" values are ordinary SSIM.
    return calc_ssim(img1, img2, size_average=size_average)


def _quat_wxyz_to_rot(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if not np.isfinite(q).all():
        return None
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return None
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _trajectory_from_params(params):
    gt_w2c = np.asarray(params["gt_w2c_all_frames"], dtype=np.float64)
    num_frames = gt_w2c.shape[0]
    quats = np.asarray(params["cam_unnorm_rots"])
    trans = np.asarray(params["cam_trans"])
    first_w2c = np.asarray(params["w2c"], dtype=np.float64)

    est_w2c = []
    valid = np.zeros(num_frames, dtype=bool)
    for idx in range(num_frames):
        if idx == 0:
            T = first_w2c.copy()
        else:
            q = np.asarray(quats[..., idx]).squeeze()
            t = np.asarray(trans[..., idx]).squeeze()
            R = _quat_wxyz_to_rot(q)
            if R is None or not np.isfinite(t).all():
                est_w2c.append(np.full((4, 4), np.nan, dtype=np.float64))
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t.reshape(3)
        if np.isfinite(T).all() and abs(np.linalg.det(T[:3, :3])) > 1e-12:
            valid[idx] = True
        est_w2c.append(T)

    est_w2c = np.stack(est_w2c, axis=0)

    gt_centers = np.full((num_frames, 3), np.nan, dtype=np.float64)
    est_centers = np.full((num_frames, 3), np.nan, dtype=np.float64)
    for idx in range(num_frames):
        try:
            gt_centers[idx] = np.linalg.inv(gt_w2c[idx])[:3, 3]
        except np.linalg.LinAlgError:
            valid[idx] = False
        if valid[idx]:
            try:
                est_centers[idx] = np.linalg.inv(est_w2c[idx])[:3, 3]
            except np.linalg.LinAlgError:
                valid[idx] = False
    valid &= np.isfinite(gt_centers).all(axis=1) & np.isfinite(est_centers).all(axis=1)
    return gt_centers, est_centers, valid


def _ate_rmse(gt_centers, est_centers, valid, align=True):
    gt = gt_centers[valid]
    est = est_centers[valid]
    if gt.shape[0] == 0:
        return float("nan")

    if align and gt.shape[0] >= 3:
        gt_mean = gt.mean(axis=0)
        est_mean = est.mean(axis=0)
        X = est - est_mean
        Y = gt - gt_mean
        U, _, Vt = np.linalg.svd(X.T @ Y)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = gt_mean - R @ est_mean
        est = (R @ est.T).T + t

    err = np.linalg.norm(est - gt, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def _load_metric(path, expected):
    values = np.atleast_1d(np.loadtxt(path, dtype=np.float64))
    if values.size != expected:
        raise RuntimeError(
            f"Expected {expected} values in {path}, found {values.size}. "
            "Keep eval_every=1 for split benchmarking."
        )
    return values


def _write_split_summary(results_dir, config):
    params_path = os.path.join(results_dir, "params.npz")
    if not os.path.exists(params_path):
        raise FileNotFoundError(params_path)

    params = dict(np.load(params_path, allow_pickle=True))
    num_frames = int(np.asarray(params["gt_w2c_all_frames"]).shape[0])
    eval_dir = os.path.join(results_dir, "eval")

    psnr = _load_metric(os.path.join(eval_dir, "psnr.txt"), num_frames)
    ssim = _load_metric(os.path.join(eval_dir, "ssim.txt"), num_frames)

    start = int(config["data"]["start"])
    stride = int(config["data"]["stride"])
    absolute_indices = start + np.arange(num_frames, dtype=np.int64) * stride
    split_every = int(config["eval_split_every"])
    split_offset = int(config["eval_split_offset"])
    test_mask = (absolute_indices % split_every) == split_offset
    train_mask = ~test_mask

    gt_centers, est_centers, valid_pose = _trajectory_from_params(params)
    ate = _ate_rmse(gt_centers, est_centers, valid_pose, align=True)

    online_seconds = float("nan")
    online_fps = float("nan")
    if "online_start" in _timing and "online_end" in _timing:
        online_seconds = _timing["online_end"] - _timing["online_start"]
        if online_seconds > 0:
            online_fps = num_frames / online_seconds

    tracked_frames = int(valid_pose.sum())
    maxmap_ratio = float(tracked_frames / num_frames) if num_frames else float("nan")
    num_gaussians = int(np.asarray(params["means3D"]).shape[0])

    summary = {
        "sequence": str(config["data"]["sequence"]),
        "num_frames": num_frames,
        "train_frames": int(train_mask.sum()),
        "test_frames": int(test_mask.sum()),
        "split_every": split_every,
        "split_offset": split_offset,
        "test_rule": "global frame indices 4,9,14,...; pose tracking only; no Gaussian insertion/mapping/keyframe",
        "maxmap_ratio": maxmap_ratio,
        "tracked_frames": tracked_frames,
        "train_psnr": float(psnr[train_mask].mean()),
        "train_ssim": float(ssim[train_mask].mean()),
        "test_psnr": float(psnr[test_mask].mean()),
        "test_ssim": float(ssim[test_mask].mean()),
        "ate_rmse_se3_m": ate,
        "online_seconds": online_seconds,
        "online_fps": online_fps,
        "gaussians": num_gaussians,
        "metric_pose_scope": "rendering uses the online estimated camera pose for each frame",
        "fps_scope": (
            "first Gaussian initialization through final online frame; includes feature matching, "
            "PnP/ICP, tracking and train-frame mapping; excludes IGEV/TransVPR preprocessing, "
            "model/dataset setup, match visualization, and final PSNR/SSIM evaluation"
        ),
        "ssim_type": "single-scale SSIM",
    }

    summary_path = os.path.join(results_dir, "benchmark_summary_split.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez(
        os.path.join(results_dir, "split_indices.npz"),
        train_indices=absolute_indices[train_mask],
        test_indices=absolute_indices[test_mask],
    )

    print("\n================ TartanAir 8:2 Benchmark ====================")
    print(f"Sequence:             {summary['sequence']}")
    print(f"Frames:               {num_frames} (train={summary['train_frames']}, test={summary['test_frames']})")
    print(f"MaxMap:               {100.0 * maxmap_ratio:.2f}% ({tracked_frames}/{num_frames})")
    print(f"Train PSNR / SSIM:    {summary['train_psnr']:.4f} / {summary['train_ssim']:.6f}")
    print(f"Test PSNR / SSIM:     {summary['test_psnr']:.4f} / {summary['test_ssim']:.6f}")
    print(f"ATE RMSE (SE3):       {ate:.6f} m")
    print(f"Online FPS:           {online_fps:.4f}")
    print(f"Gaussians:            {num_gaussians}")
    print(f"Saved summary:        {summary_path}")
    print("=============================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    args = parser.parse_args()

    # Route TartanAir without modifying upstream dataset dispatch.
    lsg_splatam.get_dataset = _get_dataset

    # Benchmark instrumentation.
    lsg_splatam.initialize_first_timestep = _timed_initialize_first_timestep
    lsg_splatam.eval = _timed_eval
    lsg_splatam.match_feature = _match_feature_without_plot

    # Standardize the reported SSIM to single-scale SSIM.
    eval_helpers_module.ms_ssim = _single_scale_ssim

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    if int(experiment.config.get("eval_split_every", 0)) != 5 or int(
        experiment.config.get("eval_split_offset", -1)
    ) != 4:
        raise ValueError("This runner expects eval_split_every=5 and eval_split_offset=4.")

    lsg_splatam.seed_everything(seed=experiment.config["seed"])

    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config["load_checkpoint"]:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    lsg_splatam.rgbd_slam(experiment.config)
    _write_split_summary(results_dir, experiment.config)


if __name__ == "__main__":
    main()
