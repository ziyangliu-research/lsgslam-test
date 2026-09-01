#!/usr/bin/env python3
"""Causal PGO diagnostic for saved TartanAir LSG-SLAM runs.

Replays ONLY the released pose-graph optimization from already-saved odometry
submaps and loop-pair measurements.  No tracking, mapping, loop search,
Gaussian deformation, or structure refinement is rerun.

By default for SH000 it compares:
  1) released PGO with all backend-accepted loop factors;
  2) the same PGO with only loop 487->90 excluded.

This is a diagnostic/causal test only.  The exclusion result must not be used as
an official benchmark result because the offending loop was identified using GT.
"""

import argparse
import json
import os

import gtsam
import numpy as np

from diagnose_tartanair_full_pose import (
    _camera_centers_from_w2c,
    _find_loop_folder,
    _load_stitched_odometry,
    _load_w2cs,
    _se3_align,
)


# These values and factor definitions mirror released
# tools/loop_closure/pose_graph_part_optim.py.
PRIOR_SIGMAS = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10])
ODOM_SIGMAS = np.array([2e-1, 2e-1, 2e-1, 1e1, 1e1, 1e1])
LOOP_SIGMAS = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])


def _graph_pose_matrix(values, idx):
    pose = values.atPose3(gtsam.symbol("x", idx))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = np.array([pose.x(), pose.y(), pose.z()], dtype=np.float64)
    return T


class ReleasedPoseGraph:
    """Minimal replay of the released PoseGraphManager."""

    def __init__(self):
        self.prior_cov = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_SIGMAS)
        self.odom_cov = gtsam.noiseModel.Diagonal.Sigmas(ODOM_SIGMAS)
        self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(LOOP_SIGMAS)
        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_initials = gtsam.Values()
        self.curr_se3 = None
        self.curr_node_idx = None
        self.prev_node_idx = None
        self.graph_optimized = None

    def add_prior(self):
        self.curr_node_idx = 0
        self.prev_node_idx = 0
        self.curr_se3 = np.eye(4, dtype=np.float64)
        self.graph_initials.insert(gtsam.symbol("x", 0), gtsam.Pose3(self.curr_se3))
        self.graph_factors.add(
            gtsam.PriorFactorPose3(
                gtsam.symbol("x", 0), gtsam.Pose3(self.curr_se3), self.prior_cov
            )
        )

    def add_odometry(self, odom_transform):
        self.graph_initials.insert(
            gtsam.symbol("x", self.curr_node_idx), gtsam.Pose3(self.curr_se3)
        )
        self.graph_factors.add(
            gtsam.BetweenFactorPose3(
                gtsam.symbol("x", self.prev_node_idx),
                gtsam.symbol("x", self.curr_node_idx),
                gtsam.Pose3(odom_transform),
                self.odom_cov,
            )
        )

    def add_loop(self, loop_transform, loop_idx):
        self.graph_factors.add(
            gtsam.BetweenFactorPose3(
                gtsam.symbol("x", loop_idx),
                gtsam.symbol("x", self.curr_node_idx),
                gtsam.Pose3(loop_transform),
                self.loop_cov,
            )
        )

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        opt = gtsam.LevenbergMarquardtOptimizer(
            self.graph_factors, self.graph_initials, params
        )
        self.graph_optimized = opt.optimize()
        self.curr_se3 = _graph_pose_matrix(self.graph_optimized, self.curr_node_idx)


def _load_backend_loop_infos(loop_folder):
    """Reproduce the released backend's loop_infos dict exactly.

    Important: the release keys the dict only by query index, so when multiple
    accepted pairs share a query, the LAST accepted pair overwrites earlier
    ones.  This function preserves that behavior.
    """
    found_path = os.path.join(loop_folder, "found_loops.npy")
    found = np.asarray(np.load(found_path)).reshape(-1, 2)
    loop_infos = {}
    accepted_rows = []

    for q_raw, r_raw in found:
        q, r = int(q_raw), int(r_raw)
        inlier_path = os.path.join(
            loop_folder, f"eval_{q}_{r}", "match_res", "1_inliers.txt"
        )
        if not os.path.isfile(inlier_path):
            continue
        try:
            inliers = float(open(inlier_path, "r", encoding="utf-8").readline().strip())
        except Exception:
            continue
        if inliers < 100:
            continue

        params_path = os.path.join(loop_folder, f"eval_{q}_{r}", "params.npz")
        if not os.path.isfile(params_path):
            continue
        loop_est_w2cs, _ = _load_w2cs(params_path)
        if len(loop_est_w2cs) < 2:
            continue

        # Exact released convention:
        # loop_est_r2q = load_params(loop_scene)[0][1]
        # loop_transform = inv(loop_est_r2q)
        loop_transform = np.linalg.inv(loop_est_w2cs[1])
        loop_infos[q] = (r, loop_transform, inliers)
        accepted_rows.append((q, r, inliers))

    return loop_infos, accepted_rows


def _replay_pgo(odo_est_w2c, loop_infos, excluded_pairs):
    """Replay release PGO including its pose-list update semantics."""
    n = len(odo_est_w2c)
    pg = ReleasedPoseGraph()
    pg.add_prior()

    # Mirrors PoseGraphResultSaver(init_pose=I).
    pose_list = [np.eye(4, dtype=np.float64)]
    applied = []
    excluded = []

    for odo_idx in range(n):
        pg.curr_node_idx = odo_idx
        if odo_idx == 0:
            pg.prev_node_idx = 0
            continue

        odo_transform = odo_est_w2c[odo_idx - 1] @ np.linalg.inv(odo_est_w2c[odo_idx])
        pg.curr_se3 = pg.curr_se3 @ odo_transform
        pg.add_odometry(odo_transform)
        pg.prev_node_idx = odo_idx

        if odo_idx in loop_infos:
            ref_idx, loop_transform, inliers = loop_infos[odo_idx]
            pair = (odo_idx, ref_idx)
            if pair in excluded_pairs:
                excluded.append((odo_idx, ref_idx, inliers))
            else:
                pg.add_loop(loop_transform, ref_idx)
                pg.optimize()
                applied.append((odo_idx, ref_idx, inliers))

                # Mirrors saveOptimizedPoseGraphResult(): its saved list contains
                # optimized nodes 1..curr-1 (node 0 is later re-prepended by the
                # backend trajectory evaluator), followed by current below.
                pose_list = [
                    _graph_pose_matrix(pg.graph_optimized, j)
                    for j in range(1, odo_idx)
                ]

        # Mirrors saveUnoptimizedPoseGraphResult after optional optimize().
        pose_list.append(pg.curr_se3.copy())

    pose_arr = np.asarray(pose_list, dtype=np.float64)

    # Released backend constructs loop_traj_pts as origin + pose_list translations
    # for the remaining N-1 frames.
    centers = [np.zeros(3, dtype=np.float64)]
    for i in range(n - 1):
        centers.append(pose_arr[i, :3, 3].copy())
    return np.asarray(centers), applied, excluded


def _parse_pair(text):
    try:
        q, r = text.split(":", 1)
        return int(q), int(r)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid pair '{text}'. Use QUERY:REF, e.g. 487:90"
        ) from exc


def run(root, seq, excludes):
    seq_root = os.path.join(root, seq)
    odo_est_w2c, odo_gt_w2c, submaps = _load_stitched_odometry(seq_root, seq)
    gt_centers = _camera_centers_from_w2c(odo_gt_w2c)
    odo_centers = _camera_centers_from_w2c(odo_est_w2c)
    pre_pgo_ate = _se3_align(gt_centers, odo_centers)

    loop_folder = _find_loop_folder(seq_root, seq)
    if loop_folder is None:
        raise FileNotFoundError(f"No loop folder found under {seq_root}")
    loop_infos, accepted_rows = _load_backend_loop_infos(loop_folder)

    all_centers, all_applied, _ = _replay_pgo(odo_est_w2c, loop_infos, set())
    all_ate = _se3_align(gt_centers, all_centers)

    excl_set = set(excludes)
    excl_centers, excl_applied, actually_excluded = _replay_pgo(
        odo_est_w2c, loop_infos, excl_set
    )
    excl_ate = _se3_align(gt_centers, excl_centers)

    summary_path = os.path.join(seq_root, "benchmark_summary_full_split.json")
    saved_ate = np.nan
    if os.path.isfile(summary_path):
        try:
            saved_ate = float(json.load(open(summary_path, "r", encoding="utf-8"))["ate_rmse_se3_m"])
        except Exception:
            pass

    print(f"\n{seq} PGO causal diagnostic")
    print("=" * 74)
    print("Submaps: " + ", ".join(name for name, _ in submaps))
    print(f"Frames: {len(gt_centers)}")
    print(f"Accepted loop rows before query-overwrite: {len(accepted_rows)}")
    print(f"Actual loop factors in backend dict:       {len(loop_infos)}")
    print(f"Odometry / pre-PGO SE3 ATE:               {pre_pgo_ate:.6f} m")
    print(f"Replayed PGO, all loops:                   {all_ate:.6f} m")
    if np.isfinite(saved_ate):
        print(f"Saved full-run PGO ATE:                    {saved_ate:.6f} m")
        print(f"Replay-vs-saved difference:                {all_ate - saved_ate:+.6f} m")
    print()
    print("Diagnostic exclusion(s):")
    if actually_excluded:
        for q, r, ins in actually_excluded:
            print(f"  EXCLUDED {q} -> {r}   inliers={ins:.0f}")
    else:
        print("  WARNING: none of the requested pairs were actual backend factors")
        print("  Requested: " + ", ".join(f"{q}->{r}" for q, r in excludes))
    print(f"Replayed PGO after exclusion:              {excl_ate:.6f} m")
    print(f"ATE change vs all-loop PGO:                {excl_ate - all_ate:+.6f} m")
    print(f"ATE change vs pre-PGO odometry:            {excl_ate - pre_pgo_ate:+.6f} m")
    print()
    print("NOTE: exclusion result is diagnostic only; do not use it as benchmark data.")

    if np.isfinite(saved_ate) and abs(all_ate - saved_ate) > 1e-3:
        print(
            "WARNING: replay does not reproduce the saved PGO ATE within 1 mm. "
            "Treat the exclusion result cautiously and send this output for inspection."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence", nargs="?", default="SH000")
    ap.add_argument(
        "--exclude",
        action="append",
        type=_parse_pair,
        default=None,
        help="Loop pair QUERY:REF to exclude; repeat for multiple pairs. Default: 487:90 for SH000.",
    )
    ap.add_argument("--root", default="experiments/tartanair_official_full_split")
    args = ap.parse_args()

    excludes = args.exclude
    if excludes is None:
        if args.sequence == "SH000":
            excludes = [(487, 90)]
        else:
            ap.error("For sequences other than SH000, provide --exclude QUERY:REF")

    run(args.root, args.sequence, excludes)


if __name__ == "__main__":
    main()
