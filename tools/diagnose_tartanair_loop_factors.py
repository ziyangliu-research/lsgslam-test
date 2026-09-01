#!/usr/bin/env python3
"""Check the exact loop factors fed to the released LSG-SLAM PGO.

No SLAM, loop search, PGO, or SR is rerun.  The script reads the saved
submaps and loop-pair params, then compares every backend-accepted loop
measurement against the stitched GT relative pose in exactly the convention
used by tools/loop_closure/pose_graph_part_optim.py.
"""

import argparse
import os

import numpy as np

from diagnose_tartanair_full_pose import (
    _find_loop_folder,
    _load_stitched_odometry,
    _load_w2cs,
    _loop_rows,
)


def _rotation_error_deg(T_err):
    R = T_err[:3, :3]
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _pose_error(est_T, gt_T):
    d = np.linalg.inv(est_T) @ gt_T
    return _rotation_error_deg(d), float(np.linalg.norm(d[:3, 3]))


def diagnose(root, seq):
    seq_root = os.path.join(root, seq)
    _, stitched_gt_w2c, _ = _load_stitched_odometry(seq_root, seq)
    loop_folder = _find_loop_folder(seq_root, seq)
    rows = _loop_rows(loop_folder)

    print(f"\n{seq}")
    print("=" * 94)
    print(f"Stitched GT frames: {len(stitched_gt_w2c)}")
    if loop_folder is None:
        print("No loop folder found")
        return

    print(f"Loop folder: {loop_folder}")
    print(
        f"{'query':>6} {'ref':>6} {'inliers':>8} {'used':>6} "
        f"{'PGO rot err':>13} {'PGO trans err':>15} "
        f"{'opposite rot':>13} {'opposite trans':>15}"
    )
    print("-" * 94)

    used_errors = []
    for q, r, inliers, used in rows:
        if q >= len(stitched_gt_w2c) or r >= len(stitched_gt_w2c):
            print(f"{q:6d} {r:6d} {inliers:8.0f} {str(used):>6} INDEX-OUT-OF-RANGE")
            continue

        p = os.path.join(loop_folder, f"eval_{q}_{r}", "params.npz")
        if not os.path.isfile(p):
            print(f"{q:6d} {r:6d} {inliers:8.0f} {str(used):>6} missing params.npz")
            continue

        loop_est_w2c, _ = _load_w2cs(p)
        if len(loop_est_w2c) < 2:
            print(f"{q:6d} {r:6d} {inliers:8.0f} {str(used):>6} bad loop params")
            continue

        # Exact released backend convention:
        #   loop_est_r2q = load_params(loop_scene)[0][1]
        #   loop_transform = inv(loop_est_r2q)
        #   gt_loop_transform = gt_w2c[ref] @ inv(gt_w2c[query])
        loop_est_r2q = loop_est_w2c[1]
        pgo_measurement = np.linalg.inv(loop_est_r2q)
        gt_measurement = stitched_gt_w2c[r] @ np.linalg.inv(stitched_gt_w2c[q])

        rerr, terr = _pose_error(pgo_measurement, gt_measurement)
        # Also show the non-inverted estimate.  If this is dramatically better,
        # it is a strong sign of a convention mismatch at the adapter boundary.
        rerr_opp, terr_opp = _pose_error(loop_est_r2q, gt_measurement)

        ins = "nan" if not np.isfinite(inliers) else f"{inliers:.0f}"
        print(
            f"{q:6d} {r:6d} {ins:>8} {('YES' if used else 'no'):>6} "
            f"{rerr:10.3f} deg {terr:11.4f} m "
            f"{rerr_opp:10.3f} deg {terr_opp:11.4f} m"
        )
        if used:
            used_errors.append((rerr, terr, rerr_opp, terr_opp))

    if used_errors:
        x = np.asarray(used_errors, dtype=np.float64)
        print("\nAccepted-loop summary")
        print(f"  count: {len(x)}")
        print(f"  PGO convention median: rot={np.median(x[:,0]):.3f} deg, trans={np.median(x[:,1]):.4f} m")
        print(f"  PGO convention max:    rot={np.max(x[:,0]):.3f} deg, trans={np.max(x[:,1]):.4f} m")
        print(f"  Opposite median:       rot={np.median(x[:,2]):.3f} deg, trans={np.median(x[:,3]):.4f} m")
    else:
        print("No backend-accepted loop factors found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=["SH000", "SH001", "SH002", "SH003"])
    ap.add_argument("--root", default="experiments/tartanair_official_full_split")
    args = ap.parse_args()
    for seq in args.sequences:
        diagnose(args.root, seq)


if __name__ == "__main__":
    main()
