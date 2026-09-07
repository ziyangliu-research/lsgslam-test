#!/usr/bin/env python3
"""Prepare the four raw downloaded EuRoC sequences for released LSG-SLAM.

This is a non-destructive launcher around the repository's released
operate_euroc_data.py preprocessing logic (official rectification calibration,
IGEV-Stereo depth, GT camera trajectory conversion, TransVPR global features).
It only adapts dataset/model paths and scene selection at runtime.

Expected raw layout under --data-root:
  extracted/machine_hall/machine_hall/MH_02_easy/mav0/...
  extracted/machine_hall/machine_hall/MH_05_difficult/mav0/...
  extracted/vicon_room1/vicon_room1/V1_01_easy/mav0/...
  extracted/vicon_room2/vicon_room2/V2_01_easy/mav0/...
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEMPLATE = os.path.join(_BASE_DIR, "tools", "euroc_parser", "operate_euroc_data.py")
DEFAULT_ALIASES = ["MH02", "V101", "V201", "MH05"]
SEQ_INFO = {
    "MH02": ("MH_02_easy", "extracted/machine_hall/machine_hall/MH_02_easy"),
    "V101": ("V1_01_easy", "extracted/vicon_room1/vicon_room1/V1_01_easy"),
    "V201": ("V2_01_easy", "extracted/vicon_room2/vicon_room2/V2_01_easy"),
    "MH05": ("MH_05_difficult", "extracted/machine_hall/machine_hall/MH_05_difficult"),
}


def _png_names(folder):
    return {os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.png"))}


def _cache_status(seq_dir):
    cam0 = os.path.join(seq_dir, "mav0", "cam0")
    cam1 = os.path.join(seq_dir, "mav0", "cam1")
    raw0 = _png_names(os.path.join(cam0, "data"))
    raw1 = _png_names(os.path.join(cam1, "data"))
    if not raw0 or not raw1:
        raise FileNotFoundError(f"Missing raw stereo images under {seq_dir}/mav0/cam[01]/data")
    if raw0 != raw1:
        only0 = sorted(raw0 - raw1)[:5]
        only1 = sorted(raw1 - raw0)[:5]
        raise RuntimeError(
            "cam0/cam1 raw filenames differ. Refusing to run the released parser because "
            f"it deletes unmatched raw images. only_cam0={only0}, only_cam1={only1}"
        )

    gt = os.path.join(seq_dir, "mav0", "state_groundtruth_estimate0", "data.csv")
    y0 = os.path.join(cam0, "sensor.yaml")
    y1 = os.path.join(cam1, "sensor.yaml")
    for p in (gt, y0, y1):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    expected = len(raw0)
    rect0 = len(glob.glob(os.path.join(cam0, "data_rect", "*.png")))
    rect1 = len(glob.glob(os.path.join(cam1, "data_rect", "*.png")))
    depth = len(glob.glob(os.path.join(cam0, "depth_sceneflow", "*.npy")))
    feat = len(glob.glob(os.path.join(cam0, "global_features", "*.npy")))
    traj = os.path.join(cam0, "traj.txt")
    complete = (
        rect0 == expected and rect1 == expected and depth == expected and feat == expected
        and os.path.isfile(traj) and os.path.getsize(traj) > 0
    )
    return {
        "expected": expected,
        "rect0": rect0,
        "rect1": rect1,
        "depth": depth,
        "features": feat,
        "traj": os.path.isfile(traj),
        "complete": complete,
    }


def _replace_once(source, old, new, label):
    n = source.count(old)
    if n != 1:
        raise RuntimeError(f"Could not adapt EuRoC parser '{label}': expected 1 match, found {n}")
    return source.replace(old, new, 1)


def _run_released_parser(seq_dir, scene):
    with open(_TEMPLATE, "r", encoding="utf-8") as f:
        source = f.read()

    parent = str(Path(seq_dir).parent)
    igev = os.path.join(
        _BASE_DIR, "third_party", "IGEV-Stereo", "pretrained_models", "sceneflow.pth"
    )
    transvpr = os.path.join(
        _BASE_DIR, "third_party", "TransVPR", "TransVPR_MSLS.pth"
    )
    for p in (igev, transvpr):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Required released preprocessing weight not found: {p}"
            )

    source = _replace_once(
        source,
        "base_path = '' # path to euroc dataset",
        f"base_path = r'{parent}' # runtime EuRoC dataset parent",
        "base path",
    )
    source = _replace_once(
        source,
        "igev_sceneflow_model_path = '../../third_party/IGEV-Stereo/pretrained_models/sceneflow.pth'",
        f"igev_sceneflow_model_path = r'{igev}'",
        "IGEV path",
    )
    source = _replace_once(
        source,
        "vpr_model_path = '../../third_party/TransVPR/TransVPR_MSLS.pth'",
        f"vpr_model_path = r'{transvpr}'",
        "TransVPR path",
    )
    source = _replace_once(
        source,
        '''scene_names = [\n    "V2_01_easy",\n]\n''',
        f'''scene_names = [\n    "{scene}",\n]\n''',
        "scene selection",
    )

    parser_dir = os.path.dirname(_TEMPLATE)
    fd, tmp = tempfile.mkstemp(prefix=".prepare_final4_", suffix=".py", dir=parser_dir)
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(source)
        print(f"[prepare] Running released preprocessing for {scene}")
        subprocess.run([sys.executable, "-u", tmp], cwd=_BASE_DIR, check=True)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="*", default=DEFAULT_ALIASES)
    ap.add_argument("--data-root", default="/home/shiyo/Desktop/Datasets/EuRoC")
    ap.add_argument("--force", action="store_true", help="rerun preprocessing even when cache looks complete")
    args = ap.parse_args()

    for alias in args.sequences:
        if alias not in SEQ_INFO:
            raise ValueError(f"Unknown sequence alias {alias}; choose from {DEFAULT_ALIASES}")
        scene, rel = SEQ_INFO[alias]
        seq_dir = os.path.join(args.data_root, rel)
        print(f"\n[{alias}] {scene}")
        print(f"  path: {seq_dir}")
        status = _cache_status(seq_dir)
        print(
            "  cache: raw={expected}, rect0={rect0}, rect1={rect1}, "
            "depth={depth}, features={features}, traj={traj}".format(**status)
        )
        if status["complete"] and not args.force:
            print("  -> preprocessing already complete; SKIP")
            continue
        _run_released_parser(seq_dir, scene)
        status2 = _cache_status(seq_dir)
        if not status2["complete"]:
            raise RuntimeError(f"Preprocessing did not produce a complete cache for {alias}: {status2}")
        print("  -> preprocessing COMPLETE")


if __name__ == "__main__":
    main()
