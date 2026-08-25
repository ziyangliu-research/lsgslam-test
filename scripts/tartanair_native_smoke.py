"""Staged native smoke test for the modernized LSG-SLAM TartanAir port.

This intentionally exercises both the first-frame mapping path and the first
0->1 tracking-prior path in small, named stages so native crashes can be localized.
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
from feature_matching import extract_feature, match_feature, estimate_pnp
from sp_lg.lightglue import LightGlue
from sp_lg.superpoint import SuperPoint


def stage(name):
    print(f"\n[NATIVE-SMOKE] >>> {name}", flush=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_open3d_pc(color, depth, intrinsics, dataset, device):
    depth_tmp = depth.squeeze(0)
    h, w = depth_tmp.shape
    x, y = np.meshgrid(range(w), range(h))
    pts = np.vstack((x.reshape(-1), y.reshape(-1), np.ones(h * w)))
    pts = torch.tensor(pts, dtype=torch.float32, device=device)
    d = depth_tmp.reshape(-1)
    valid = torch.where(
        (d > max(0.1, dataset.depth_filter_near))
        & (d < max(30.0, dataset.depth_filter_far))
    )[0]
    X = torch.multiply(d[valid], torch.linalg.inv(intrinsics) @ pts[:, valid])
    rgb = color.permute(1, 2, 0).reshape(-1, 3)[valid]
    X_np = np.ascontiguousarray(X.t().detach().cpu().numpy(), dtype=np.float64)
    rgb_np = np.ascontiguousarray(rgb.detach().cpu().numpy(), dtype=np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(X_np)
    pc.colors = o3d.utility.Vector3dVector(rgb_np)
    return pc


def preprocess_frame(dataset, idx):
    color_hwc, depth_hwc, intrinsics4, pose, depth_orig_hwc, global_feature = dataset[idx]
    color = color_hwc.permute(2, 0, 1) / 255.0
    depth = depth_hwc.permute(2, 0, 1)
    depth_orig = depth_orig_hwc.permute(2, 0, 1)
    intrinsics = intrinsics4[:3, :3]
    return color, depth, intrinsics, pose, depth_orig, global_feature


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
    color, depth, intrinsics, pose0, depth_orig, _ = preprocess_frame(dataset, 0)
    sync()
    print("color:", tuple(color.shape), "depth:", tuple(depth.shape), flush=True)
    print("depth min/max:", float(depth.min()), float(depth.max()), flush=True)
    print("pose finite:", bool(torch.isfinite(pose0).all()), flush=True)

    stage("SuperPoint + LightGlue construction")
    sp = SuperPoint(max_num_keypoints=1024).eval().to(device)
    lg = LightGlue(
        pretrained="superpoint",
        width_confidence=0.99,
        depth_confidence=0.95,
    ).eval().to(device)
    sync()

    stage("SuperPoint dense feature extraction frame 0")
    color_feature0 = color.clone()
    mask0 = (depth_orig < 0.1) | (depth_orig > min(dataset.depth_filter_far, 15.0))
    color_feature0[:, mask0[0]] = 0
    feats0, dense_desc0 = extract_feature(color_feature0, depth_orig, sp, device)
    sync()
    print("keypoints0:", tuple(feats0["keypoints"].shape), flush=True)
    print("dense_desc0:", tuple(dense_desc0.shape), flush=True)

    stage("Open3D point-cloud construction frame 0")
    pc0 = build_open3d_pc(color, depth_orig, intrinsics, dataset, device)
    print("Open3D points0:", len(pc0.points), flush=True)

    stage("Gaussian parameter initialization")
    config["pixel_gs_depth_threshold"] = config["pixel_gs_depth_gamma"] * float(depth_orig.max())
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
    curr_data0 = {
        "cam": cam,
        "im": color,
        "depth": depth,
        "depth_original": depth_orig,
        "id": 0,
        "intrinsics": intr,
        "w2c": first_w2c,
        "iter_gt_w2c_list": [torch.linalg.inv(pose0)],
    }
    loss, variables, losses = lsg.get_loss(
        params,
        curr_data0,
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

    # ------------------------------------------------------------------
    # Two-frame tracking-prior path, where the full run previously crashed.
    # ------------------------------------------------------------------
    if len(dataset) < 2:
        stage("ALL FIRST-FRAME STAGES PASSED (dataset has <2 frames)")
        return

    stage("dataset[1]")
    color1, depth1, intrinsics1, pose1, depth_orig1, _ = preprocess_frame(dataset, 1)
    sync()
    print("depth1 min/max:", float(depth1.min()), float(depth1.max()), flush=True)

    stage("SuperPoint dense feature extraction frame 1")
    color_feature1 = color1.clone()
    mask1 = (depth_orig1 < 0.1) | (depth_orig1 > min(dataset.depth_filter_far, 15.0))
    color_feature1[:, mask1[0]] = 0
    feats1, dense_desc1 = extract_feature(color_feature1, depth_orig1, sp, device)
    sync()
    print("keypoints1:", tuple(feats1["keypoints"].shape), flush=True)

    stage("LightGlue + EssentialMat matching 0 -> 1")
    mkpts1, mkpts0, mscores = match_feature(
        config,
        color1,
        feats1,
        intrinsics,
        color,
        feats0,
        lg,
        device,
        topk=1024,
        save_path=None,
    )
    sync()
    if mkpts1 is None:
        print("match_feature returned no valid matches", flush=True)
        init_pose = np.eye(4, dtype=np.float64)
        pnp_inliers = 0
    else:
        print("essential inlier matches:", int(mkpts1.shape[0]), flush=True)

        stage("PnP prior 0 -> 1")
        gt_w2c0 = torch.linalg.inv(pose0)
        gt_w2c1 = torch.linalg.inv(pose1)
        last_data = {
            "depth_original": depth_orig,
            "depth": depth,
            "intrinsics": intrinsics,
            "est_w2c": gt_w2c0,
            "iter_gt_w2c_list": [gt_w2c0],
            "id": 0,
        }
        curr_data = {
            "iter_gt_w2c_list": [gt_w2c0, gt_w2c1],
            "id": 1,
        }
        pnp_result = estimate_pnp(mkpts1, mkpts0, curr_data, last_data, dataset)
        sync()
        if pnp_result is None:
            print("PnP returned None; falling back to identity for ICP smoke", flush=True)
            init_pose = np.eye(4, dtype=np.float64)
            pnp_inliers = 0
        else:
            _, init_pose, pnp_inliers = pnp_result
            print("PnP inliers:", int(pnp_inliers), flush=True)
            print("PnP relative pose finite:", bool(np.isfinite(init_pose).all()), flush=True)

    stage("Open3D point-cloud construction frame 1")
    pc1 = build_open3d_pc(color1, depth_orig1, intrinsics1, dataset, device)
    print("Open3D points1:", len(pc1.points), flush=True)

    stage("Open3D ICP 0 -> 1")
    corr_threshold = config["tracking"]["icp_corr_threshold"]
    if pnp_inliers < 50:
        corr_threshold = max(3.0, corr_threshold)
    T_icp, fitness, inlier_rmse = lsg.icp(pc1, pc0, init_pose, corr_threshold)
    print("ICP fitness:", float(fitness), "rmse:", float(inlier_rmse), flush=True)
    print("ICP pose finite:", bool(np.isfinite(T_icp).all()), flush=True)

    stage("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
