"""TartanAir entry point for the unmodified LSG-SLAM submap frontend."""

import argparse
import os
import shutil
import sys
from importlib.machinery import SourceFileLoader


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import splatam as lsg_splatam  # noqa: E402
from datasets.gradslam_datasets.tartanair import TartanAirDataset  # noqa: E402


_original_get_dataset = lsg_splatam.get_dataset


def _get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["tartanair", "tartan"]:
        return TartanAirDataset(config_dict, basedir, sequence, **kwargs)
    return _original_get_dataset(config_dict, basedir, sequence, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to TartanAir experiment file")
    args = parser.parse_args()

    lsg_splatam.get_dataset = _get_dataset

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


if __name__ == "__main__":
    main()
