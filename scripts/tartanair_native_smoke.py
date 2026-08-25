"""Staged native smoke test for the modernized LSG-SLAM TartanAir port.

This intentionally exercises the first-frame code path in small, named stages so
segmentation faults from native libraries can be localized precisely.
"""

import faulthandler
import os
import sys
from importlib.machinery import SourceFileLoader

faulthandler.enable(all_threads=True)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import open3d as o3d
import torch

import splatam as lsg
from datasets.gradslam_datasets import load_dataset_config
from datasets.gradslam_datasets.tartanair import TartanAirDataset
from feature_matching import extract_feature
from sp_lg.superpoint import SuperPoint


def stage(name):
    print(f"\n[NATIVE-SMOKE] >>> {name}", flush=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    exp_path = os.environ.get("LSG_TARTANAIR_CONFIG", "configs/tartanair/lsgslam.py")
    experiment = SourceFileLoader("tartanair_smoke_config", exp_path).load_module()
    config = experiment.config
    device = torch.device(config["primary_device"])

    stage("versions")
    print("python:", sys.version.replace("\n", " "), flush=True)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)
    print("numpy:", np.__version__, flush=True)
    print("open3d:", o3d.__version__, flush=True)
    print("gpu:", torch.cuda.get_device_name(device), flush=True)
    print("capability:", torch.cuda.get_device_capability(device), flush=True)

    stage("dataset construction")
    data_cfg = config["data"]
    grad_cfg = load_dataset_config(data_cfg["gradslam_data_cfg"])
    dataset = TartanAirDataset(
        grad_cfg,
        data_cfg["basedir"],
        os.path.basename(data_cfg["sequence"]),
        start=data_cfg["start"],
        end=data_cfg["end"],
        stride=data_cfg["stride"],
        desired_height=data_cfg["desired_image_height"],
        desired_width=data_cfg["desired_image_width"],
        device=device,
        relative_pose=True,
    )
    print("frames:", len(dataset), flush=True)

    stage("dataset[0]")
    color_hwc, depth_hwc, intrinsics4, pose, depth_orig_hwc, global_feature = dataset[0]
    sync()
    print("color:", tuple(color_hwc.shape), "depth:", tuple(depth_hwc.shape), flush=True)
    print("depth min/max:", float(depth_hwc.min()), float(depth_hwc.max()), flush=True)
    print("pose finite:", bool(torch.isfinite(pose).all()), flush=True)

    color = color_hwc.permute(2, 0, 1) / 255.0
    depth = depth_hwc.permute(2, 0, 1)
    depth_orig = depth_orig_hwc.permute(2, 0, 1)
    intrinsics = intrinsics4[:3, :3]

    stage("SuperPoint construction")
    sp = SuperPoint(max_num_keypoints=1024).eval().to(device)
    sync()

    stage("SuperPoint dense feature extraction")
    color_feature = color.clone()
    mask = (depth_orig < 0.1) | (depth_orig > min(dataset.depth_filter_far, 15.0))
    color_feature[:, mask[0]] = 0
    feats, dense_desc = extract_feature(color_feature, depth_orig, sp, device)
    sync()
    print("keypoints:", tuple(feats["keypoints"].shape), flush=True)
    print("dense_desc:", tuple(dense_desc.shape), flush=True)

    stage("Open3D point-cloud construction")
    depth_tmp = depth_orig.squeeze(0)
    h, w = depth_tmp.shape
    x, y = np.meshgrid(range(w), range(h))
    pts = np.vstack((x.reshape(-1), y.reshape(-1), np.ones(h * w)))
    pts = torch.tensor(pts, dtype=torch.float32, device=device)
    d = depth_tmp.reshape(-1)
    valid = torch.where((d > max(0.1, dataset.depth_filter_near)) & (d < max(30.0, dataset.depth_filter_far)))[0]
    X = torch.multiply(d[valid], torch.linalg.inv(intrinsics) @ pts[:, valid])
    rgb = color.permute(1, 2, 0).reshape(-1, 3)[valid]
    X_np = np.ascontiguousarray(X.t().detach().cpu().numpy(), dtype=np.float64)
    rgb_np = np.ascontiguousarray(rgb.detach().cpu().numpy(), dtype=np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(X_np)
    pc.colors = o3d.utility.Vector3dVector(rgb_np)
    print("Open3D points:", len(pc.points), flush=True)

    stage("Gaussian parameter initialization")
    config["pixel_gs_depth_threshold"] = config["pixel_gs_depth_gamma"] * float(depth_orig_hwc.max())
    params, variables, intr, first_w2c, cam = lsg.initialize_first_timestep(
        dataset,
        len(dataset),
        config["scene_radius_depth_ratio"],
        config["mean_sq_dist_method"],
        gaussian_distribution=config["gaussian_distribution"],
        config=config,
    )
    sync()
    print("gaussians:", int(params["means3D"].shape[0]), flush=True)

    stage("Gaussian rasterizer forward (RGB + depth/silhouette)")
    curr_data = {
        "cam": cam,
        "im": color,
        "depth": depth,
        "depth_original": depth_orig,
        "id": 0,
        "intrinsics": intr,
        "w2c": first_w2c,
        "iter_gt_w2c_list": [torch.linalg.inv(pose)],
    }
    loss, variables, losses = lsg.get_loss(
        params,
        curr_data,
        variables,
        0,
        config["mapping"]["loss_weights"],
        config["mapping"]["use_sil_for_loss"],
        config["mapping"]["sil_thres"],
        config["mapping"]["use_l1"],
        config["mapping"]["ignore_outlier_depth_loss"],
        mapping=True,
    )
    sync()
    print("loss:", float(loss.detach()), flush=True)

    stage("Gaussian rasterizer backward")
    loss.backward()
    sync()
    print("backward: OK", flush=True)

    stage("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
