#!/usr/bin/env python3
"""Final ETH3D backend launcher for the strict 8:2 benchmark.

The released pose_graph_part_optim.py remains untouched. The rectified ETH3D
cache is exposed through native EurocDataset-compatible files. This launcher
adapts the audited benchmark wrapper to the ETH3D sequence path, keeps the
strict frame-id modulo-5 holdout for train-only SR, and applies the backend
timing hotfix before executing released PGO/deformation/SR code.
"""

import os
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.join(_THIS_DIR, "tartanair_pose_graph_part_optim.py")


def _replace_once(source, old, new, label):
    n = source.count(old)
    if n != 1:
        raise RuntimeError(
            f"Could not adapt ETH3D backend '{label}': expected 1 match, found {n}."
        )
    return source.replace(old, new, 1)


def main():
    with open(_TEMPLATE, "r", encoding="utf-8") as f:
        source = f.read()

    # Route the generated backend through native EurocDataset using the
    # ETH3D-compatible cache produced by prepare_eth3d_lsg.py.
    source = source.replace('os.environ["TARTANAIR_SEQUENCE"]', 'os.environ["ETH3D_SEQUENCE"]')
    source = source.replace("dataset_type = 'tartanair'", "dataset_type = 'euroc'")
    source = source.replace(
        "from configs.tartanair.lsgslam_full_split_8_2 import config",
        "from configs.eth3d.lsgslam_full_split_8_2 import config",
    )

    tartan_basedir = '''    dataset_config['basedir'] = os.path.join(\n        os.environ.get(\n            "TARTANAIR_DATA_ROOT",\n            "/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge",\n        ),\n        "stereo",\n    )\n'''
    eth3d_basedir = '''    dataset_config['basedir'] = os.environ["ETH3D_SEQ_DIR"]\n'''
    source = _replace_once(source, tartan_basedir, eth3d_basedir, "dataset basedir")

    source = source.replace(
        "'split_rule': 'global frames 4,9,14,... are pose-only; excluded from mapping and SR loss',",
        "'split_rule': 'ETH3D frames 4,9,14,... are pose-only and excluded from mapping/SR loss',",
    )
    source = source.replace("TartanAir adapter", "ETH3D adapter")

    # Timing hotfix: initialize accumulators before the PGO loop and do not
    # reset them in the later metric-accumulator section.
    source = _replace_once(
        source,
        """    total_gaussians = 0
    pgo_opt_seconds = 0.0
    gaussian_deformation_seconds = 0.0
    sr_opt_seconds = 0.0
""",
        """    total_gaussians = 0
""",
        "remove late timing initialization",
    )

    anchor = """    source = _replace_once(
        source,
        '''            PGM.optimizePoseGraph()\n''',
"""
    early_patch = """    source = _replace_once(
        source,
        '''    # Pose Graph Manager (for back-end optimization) initialization\n    PGM = PoseGraphManager()\n''',
        '''    pgo_opt_seconds = 0.0\n    gaussian_deformation_seconds = 0.0\n    sr_opt_seconds = 0.0\n\n    # Pose Graph Manager (for back-end optimization) initialization\n    PGM = PoseGraphManager()\n''',
        \"early backend timing accumulator initialization\",
    )

"""
    source = _replace_once(
        source,
        anchor,
        early_patch + anchor,
        "insert early timing initialization patch",
    )

    module = types.ModuleType("lsg_eth3d_pose_graph_part_optim_final")
    module.__file__ = _TEMPLATE
    module.__name__ = "__main__"
    exec(compile(source, _TEMPLATE, "exec"), module.__dict__)


if __name__ == "__main__":
    main()
