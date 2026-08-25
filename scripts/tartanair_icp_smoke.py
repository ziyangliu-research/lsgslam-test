"""Isolate Open3D ICP behavior on TartanAir frame 0 -> 1.

Runs exactly the geometric preprocessing used by LSG-SLAM's ICP path, but lets
point-to-point and point-to-plane registration be tested independently so a
native segfault can be localized without taking down the full SLAM process.
"""

import argparse
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

from datasets.gradslam_datasets import load_dataset_config
from datasets.gradslam_datasets.tartanair import TartanAirDataset


def build_pc(dataset, idx, device):
    color_hwc, _, intr4, _, depth_hwc, _ = dataset[idx]
    color = color_hwc.permute(2, 0, 1) / 255.0
    depth = depth_hwc.permute(2, 0, 1)
    K = intr4[:3, :3]

    d = depth.squeeze(0)
    h, w = d.shape
    x, y = np.meshgrid(range(w), range(h))
    pix = np.vstack((x.reshape(-1), y.reshape(-1), np.ones(h * w)))
    pix = torch.tensor(pix, dtype=torch.float32, device=device)
    dflat = d.reshape(-1)
    valid = torch.where(
        (dflat > max(0.1, dataset.depth_filter_near))
        & (dflat < max(30.0, dataset.depth_filter_far))
    )[0]
    X = torch.multiply(dflat[valid], torch.linalg.inv(K) @ pix[:, valid])
    rgb = color.permute(1, 2, 0).reshape(-1, 3)[valid]

    xyz_np = np.ascontiguousarray(X.t().detach().cpu().numpy(), dtype=np.float64)
    rgb_np = np.ascontiguousarray(rgb.detach().cpu().numpy(), dtype=np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz_np)
    pc.colors = o3d.utility.Vector3dVector(rgb_np)
    return pc


def prepare(pc, name):
    print(f"{name}: raw={len(pc.points)}", flush=True)
    pc = pc.voxel_down_sample(0.1)
    print(f"{name}: voxel={len(pc.points)}", flush=True)
    pc, _ = pc.remove_radius_outlier(nb_points=20, radius=0.4)
    print(f"{name}: radius_filtered={len(pc.points)}", flush=True)
    if len(pc.points) == 0:
        raise RuntimeError(f"{name} became empty after filtering")
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
    print(f"{name}: normals={pc.has_normals()} count={len(pc.normals)}", flush=True)
    return pc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", choices=["point_to_point", "point_to_plane"], required=True)
    args = parser.parse_args()

    experiment = SourceFileLoader("tartanair_icp_config", "configs/tartanair/lsgslam.py").load_module()
    cfg = experiment.config
    data_cfg = cfg["data"]
    device = torch.device(cfg["primary_device"])
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

    print("Open3D:", o3d.__version__, flush=True)
    print("NumPy:", np.__version__, flush=True)

    # Match LSG-SLAM icp(now_pc, pre_pc, ...): source=previous, target=current.
    target = prepare(build_pc(dataset, 1, device), "target(frame1)")
    source = prepare(build_pc(dataset, 0, device), "source(frame0)")

    if args.estimator == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    print(f"running estimator={args.estimator}", flush=True)
    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        0.5,
        np.eye(4, dtype=np.float64),
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=0.1,
            max_iteration=100,
        ),
    )
    print("ICP OK", flush=True)
    print("fitness:", reg.fitness, flush=True)
    print("rmse:", reg.inlier_rmse, flush=True)
    print("correspondences:", np.asarray(reg.correspondence_set).shape[0], flush=True)
    print(reg.transformation, flush=True)


if __name__ == "__main__":
    main()
