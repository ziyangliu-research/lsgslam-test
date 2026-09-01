#!/usr/bin/env python3
"""Diagnose TartanAir full-pipeline trajectory changes without rerunning SLAM/SR.

Reports, per sequence:
  * stitched odometry ATE before pose-graph optimization (SE3 and Sim3),
  * saved pose-graph trajectory ATE (SE3 and Sim3),
  * fitted Sim3 scale,
  * loop pairs and the backend inlier count used to accept them.

The submap stitching follows released pose_graph_part_optim.py exactly.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np


def _quat_wxyz_to_rot(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.full((3, 3), np.nan)
    w, x, y, z = q / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _load_w2cs(params_path):
    p = dict(np.load(params_path, allow_pickle=True))
    rots = np.asarray(p["cam_unnorm_rots"])
    trans = np.asarray(p["cam_trans"])
    gt = np.asarray(p["gt_w2c_all_frames"], dtype=np.float64)

    rots = np.squeeze(rots)
    trans = np.squeeze(trans)
    if rots.shape[0] != 4 and rots.shape[-1] == 4:
        rots = rots.T
    if trans.shape[0] != 3 and trans.shape[-1] == 3:
        trans = trans.T

    n = rots.shape[-1]
    est = []
    for i in range(n):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_wxyz_to_rot(rots[:, i])
        T[:3, 3] = trans[:, i]
        est.append(T)
    return np.asarray(est), gt


def _parse_submap_name(path, seq):
    name = Path(path).name
    prefix = seq + "_"
    if not name.startswith(prefix) or "_loops" in name:
        return None
    parts = name[len(prefix):].split("_")
    if len(parts) != 3:
        return None
    try:
        start, end, stride = map(int, parts)
    except ValueError:
        return None
    return start, end, stride


def _load_stitched_odometry(seq_root, seq):
    entries = []
    for p in glob.glob(os.path.join(seq_root, f"{seq}_*_*_*")):
        info = _parse_submap_name(p, seq)
        if info is None or not os.path.isfile(os.path.join(p, "params.npz")):
            continue
        entries.append((info, p))
    entries.sort(key=lambda x: x[0][0])
    if not entries:
        raise FileNotFoundError(f"No completed submaps found under {seq_root}")

    all_est, all_gt = [], []
    names = []
    for info, folder in entries:
        est, gt = _load_w2cs(os.path.join(folder, "params.npz"))
        names.append((Path(folder).name, len(est)))
        if not all_est:
            all_est = list(est)
            all_gt = list(gt)
        else:
            # Exact released backend chaining rule.
            last_est = all_est[-1]
            last_gt = all_gt[-1]
            est = [pose @ last_est for pose in est[1:]]
            gt = [pose @ last_gt for pose in gt[1:]]
            all_est.extend(est)
            all_gt.extend(gt)
    return np.asarray(all_est), np.asarray(all_gt), names


def _camera_centers_from_w2c(w2cs):
    out = []
    for T in w2cs:
        try:
            out.append(np.linalg.inv(T)[:3, 3])
        except np.linalg.LinAlgError:
            out.append([np.nan, np.nan, np.nan])
    return np.asarray(out, dtype=np.float64)


def _se3_align(gt, est):
    valid = np.isfinite(gt).all(1) & np.isfinite(est).all(1)
    gt, est = gt[valid], est[valid]
    if len(gt) < 3:
        return np.nan
    gm, em = gt.mean(0), est.mean(0)
    X, Y = est - em, gt - gm
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = gm - R @ em
    aligned = (R @ est.T).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1))))


def _sim3_align(gt, est):
    valid = np.isfinite(gt).all(1) & np.isfinite(est).all(1)
    gt, est = gt[valid], est[valid]
    if len(gt) < 3:
        return np.nan, np.nan
    gm, em = gt.mean(0), est.mean(0)
    X, Y = est - em, gt - gm
    cov = (Y.T @ X) / len(X)
    U, svals, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    var_x = np.mean(np.sum(X * X, axis=1))
    scale = float(np.sum(svals * np.diag(D)) / var_x) if var_x > 1e-15 else np.nan
    t = gm - scale * (R @ em)
    aligned = scale * (R @ est.T).T + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1))))
    return rmse, scale


def _load_pgo_centers(seq_root, seq):
    paths = glob.glob(os.path.join(seq_root, "PoseGraphResult", "csvs", f"pose{seq}unoptimized_*.csv"))
    if not paths:
        return None, None
    p = max(paths, key=os.path.getmtime)
    arr = np.loadtxt(p, delimiter=",")
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 16:
        raise RuntimeError(f"Unexpected pose graph CSV shape {arr.shape}: {p}")
    mats = arr.reshape(-1, 4, 4)
    # PoseGraphResultSaver stores C2W-like graph poses; translation is camera center.
    return mats[:, :3, 3].astype(np.float64), p


def _find_loop_folder(seq_root, seq):
    cands = [p for p in glob.glob(os.path.join(seq_root, f"{seq}_*_loops")) if os.path.isdir(p)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _loop_rows(loop_folder):
    if loop_folder is None:
        return []
    found = os.path.join(loop_folder, "found_loops.npy")
    if not os.path.isfile(found):
        return []
    loops = np.asarray(np.load(found)).reshape(-1, 2)
    rows = []
    for q, r in loops:
        q, r = int(q), int(r)
        ip = os.path.join(loop_folder, f"eval_{q}_{r}", "match_res", "1_inliers.txt")
        inliers = np.nan
        if os.path.isfile(ip):
            try:
                inliers = float(Path(ip).read_text().strip().splitlines()[0])
            except Exception:
                pass
        rows.append((q, r, inliers, bool(np.isfinite(inliers) and inliers >= 100)))
    return rows


def diagnose(root, seq):
    seq_root = os.path.join(root, seq)
    est_w2c, gt_w2c, submaps = _load_stitched_odometry(seq_root, seq)
    pre_est = _camera_centers_from_w2c(est_w2c)
    gt = _camera_centers_from_w2c(gt_w2c)

    pre_se3 = _se3_align(gt, pre_est)
    pre_sim3, pre_scale = _sim3_align(gt, pre_est)

    pgo, pgo_path = _load_pgo_centers(seq_root, seq)
    if pgo is not None:
        n = min(len(gt), len(pgo))
        pgo_se3 = _se3_align(gt[:n], pgo[:n])
        pgo_sim3, pgo_scale = _sim3_align(gt[:n], pgo[:n])
    else:
        n = 0
        pgo_se3 = pgo_sim3 = pgo_scale = np.nan

    loops = _loop_rows(_find_loop_folder(seq_root, seq))

    print(f"\n{seq}")
    print("=" * 76)
    print("Submaps:", ", ".join(f"{name}[{nf}]" for name, nf in submaps))
    print(f"Stitched trajectory frames: {len(gt)}")
    print(f"Before PGO  SE3 ATE RMSE : {pre_se3:.6f} m")
    print(f"Before PGO  Sim3 ATE RMSE: {pre_sim3:.6f} m   scale={pre_scale:.6f}")
    if pgo is None:
        print("After PGO: no PoseGraphResult CSV found")
    else:
        print(f"After PGO   SE3 ATE RMSE : {pgo_se3:.6f} m   ({n} frames)")
        print(f"After PGO   Sim3 ATE RMSE: {pgo_sim3:.6f} m   scale={pgo_scale:.6f}")
        print(f"PGO CSV: {pgo_path}")
        if np.isfinite(pre_se3) and np.isfinite(pgo_se3):
            print(f"PGO SE3 change            : {pgo_se3 - pre_se3:+.6f} m")

    accepted = sum(row[3] for row in loops)
    print(f"Loops: found={len(loops)}, backend-accepted(inliers>=100)={accepted}")
    for q, r, inliers, ok in loops:
        ins = "nan" if not np.isfinite(inliers) else f"{inliers:.0f}"
        print(f"  {q:4d} -> {r:4d}   inliers={ins:>5}   {'USED' if ok else 'REJECT'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=["SH000", "SH001", "SH002", "SH003"])
    ap.add_argument("--root", default="experiments/tartanair_official_full_split")
    args = ap.parse_args()
    for seq in args.sequences:
        diagnose(args.root, seq)


if __name__ == "__main__":
    main()
