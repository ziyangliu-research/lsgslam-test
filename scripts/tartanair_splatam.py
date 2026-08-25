"""TartanAir entry point for the LSG-SLAM frontend with benchmark reporting.

The original LSG-SLAM algorithm remains unchanged.  This wrapper adds:
- TartanAir dataset routing.
- Disabled match-figure rendering during benchmark timing.
- Online end-to-end timing from first Gaussian initialization to final frame,
  excluding model/dataset setup, offline stereo/global-feature preprocessing,
  and final rendering evaluation.
- A machine-readable benchmark_summary.json with PSNR, MS-SSIM, Gaussian
  count, proper camera-center ATE RMSE, and FPS.
"""

import argparse
import json
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

import numpy as np
import torch


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import splatam as lsg_splatam  # noqa: E402
from datasets.gradslam_datasets.tartanair import TartanAirDataset  # noqa: E402


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
    # rgbd_slam calls final evaluation only after the complete online scan.
    if "online_end" not in _timing:
        _sync_cuda()
        _timing["online_end"] = time.perf_counter()
    return _original_eval(*args, **kwargs)


def _match_feature_without_plot(*args, **kwargs):
    # Match visualization uses matplotlib + disk I/O and is not part of the
    # SLAM algorithm; suppress it so online FPS is not contaminated by logging.
    kwargs["save_path"] = None
    return _original_match_feature(*args, **kwargs)


def _quat_wxyz_to_rot(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-norm camera quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_centers_from_params(params):
    gt_w2c = np.asarray(params["gt_w2c_all_frames"], dtype=np.float64)
    num_frames = gt_w2c.shape[0]

    quats = np.asarray(params["cam_unnorm_rots"])
    trans = np.asarray(params["cam_trans"])
    first_w2c = np.asarray(params["w2c"], dtype=np.float64)

    est_w2c = [first_w2c]
    for idx in range(1, num_frames):
        q = np.asarray(quats[..., idx]).squeeze()
        t = np.asarray(trans[..., idx]).squeeze()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_wxyz_to_rot(q)
        T[:3, 3] = t.reshape(3)
        est_w2c.append(T)
    est_w2c = np.stack(est_w2c, axis=0)

    gt_centers = np.stack([np.linalg.inv(T)[:3, 3] for T in gt_w2c], axis=0)
    est_centers = np.stack([np.linalg.inv(T)[:3, 3] for T in est_w2c], axis=0)
    return gt_centers, est_centers


def _ate_rmse(gt_centers, est_centers, align=True):
    gt = np.asarray(gt_centers, dtype=np.float64)
    est = np.asarray(est_centers, dtype=np.float64)
    valid = np.isfinite(gt).all(axis=1) & np.isfinite(est).all(axis=1)
    gt = gt[valid]
    est = est[valid]
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


def _safe_mean_txt(path):
    if not os.path.exists(path):
        return float("nan")
    values = np.atleast_1d(np.loadtxt(path, dtype=np.float64))
    return float(np.mean(values))


def _write_benchmark_summary(results_dir, config):
    params_path = os.path.join(results_dir, "params.npz")
    if not os.path.exists(params_path):
        print(f"[Benchmark] params.npz not found: {params_path}")
        return

    params = dict(np.load(params_path, allow_pickle=True))
    eval_dir = os.path.join(results_dir, "eval")

    num_gaussians = int(np.asarray(params["means3D"]).shape[0])
    num_frames = int(np.asarray(params["gt_w2c_all_frames"]).shape[0])
    gt_centers, est_centers = _camera_centers_from_params(params)
    ate_se3 = _ate_rmse(gt_centers, est_centers, align=True)
    ate_no_align = _ate_rmse(gt_centers, est_centers, align=False)

    online_seconds = float("nan")
    online_fps = float("nan")
    if "online_start" in _timing and "online_end" in _timing:
        online_seconds = _timing["online_end"] - _timing["online_start"]
        if online_seconds > 0:
            online_fps = num_frames / online_seconds

    summary = {
        "sequence": config["data"]["sequence"],
        "start": int(config["data"]["start"]),
        "end": int(config["data"]["end"]),
        "stride": int(config["data"]["stride"]),
        "num_frames": num_frames,
        "psnr": _safe_mean_txt(os.path.join(eval_dir, "psnr.txt")),
        "ms_ssim": _safe_mean_txt(os.path.join(eval_dir, "ssim.txt")),
        "gaussians": num_gaussians,
        "ate_rmse_se3_m": ate_se3,
        "ate_rmse_no_align_m": ate_no_align,
        "online_seconds": online_seconds,
        "online_fps": online_fps,
        "fps_scope": (
            "first Gaussian initialization through final online frame; excludes "
            "IGEV/TransVPR preprocessing, model/dataset setup, match visualization, "
            "and final PSNR/MS-SSIM evaluation"
        ),
    }

    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================ TartanAir Benchmark Summary ================")
    print(f"Sequence:              {summary['sequence']}")
    print(f"Frames:                {num_frames}")
    print(f"PSNR:                  {summary['psnr']:.4f} dB")
    print(f"MS-SSIM:               {summary['ms_ssim']:.6f}")
    print(f"Gaussians:             {num_gaussians}")
    print(f"ATE RMSE (SE3):        {ate_se3:.6f} m")
    print(f"ATE RMSE (no align):   {ate_no_align:.6f} m")
    print(f"Online time:           {online_seconds:.3f} s")
    print(f"Online FPS:            {online_fps:.4f}")
    print(f"Saved summary:         {summary_path}")
    print("=============================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to TartanAir experiment file")
    args = parser.parse_args()

    lsg_splatam.get_dataset = _get_dataset
    lsg_splatam.initialize_first_timestep = _timed_initialize_first_timestep
    lsg_splatam.eval = _timed_eval
    lsg_splatam.match_feature = _match_feature_without_plot

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    lsg_splatam.seed_everything(seed=experiment.config["seed"])

    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config["load_checkpoint"]:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    lsg_splatam.rgbd_slam(experiment.config)
    _write_benchmark_summary(results_dir, experiment.config)


if __name__ == "__main__":
    main()
