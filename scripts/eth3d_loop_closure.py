#!/usr/bin/env python3
"""ETH3D loop-closure entry point with benchmark-clean timing.

The rectified ETH3D cache is exposed through native EurocDataset-compatible
files. This wrapper reuses the released LSG-SLAM loop-closure logic and skips
only the read-only final metric eval of each temporary loop-pair run.
"""

import argparse
import os
import shutil
import sys
from importlib.machinery import SourceFileLoader

import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import loop_closure as lsg_loop  # noqa: E402


def _skip_loop_pair_final_eval(*_args, **_kwargs):
    print("[Benchmark timing] Skip temporary loop-pair final metric eval.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to ETH3D experiment config")
    args = parser.parse_args()

    lsg_loop.eval = _skip_loop_pair_final_eval

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    lsg_loop.seed_everything(seed=experiment.config["seed"])

    experiment.config["run_name"] = experiment.config["run_name"] + "_loops"
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config["load_checkpoint"]:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    lsg_loop.find_loops(experiment.config)
    found_loops = np.load(os.path.join(results_dir, "found_loops.npy"))
    print(found_loops)

    for loop in found_loops:
        experiment = SourceFileLoader(
            os.path.basename(args.experiment), args.experiment
        ).load_module()

        experiment.config["run_name"] = experiment.config["run_name"] + "_loops"
        experiment.config["tracking"]["num_iters"] = 300
        experiment.config["mapping"]["num_iters"] = 100
        experiment.config["use_warp_loss"] = True

        start_idx = experiment.config["data"]["start"]
        stride = experiment.config["data"]["stride"]
        ref_kf_idx = int(loop[1])
        query_kf_idx = int(loop[0])
        start = start_idx + ref_kf_idx * stride
        end = start_idx + query_kf_idx * stride

        experiment.config["data"]["start"] = start
        experiment.config["data"]["end"] = end
        experiment.config["data"]["stride"] = end - start
        print(experiment.config["data"])

        lsg_loop.rgbd_slam(experiment.config, loop)


if __name__ == "__main__":
    main()
