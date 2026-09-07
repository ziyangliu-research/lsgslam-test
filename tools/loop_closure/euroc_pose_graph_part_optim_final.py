#!/usr/bin/env python3
"""Final EuRoC backend launcher for the stride=5 strict 8:2 benchmark.

The released pose_graph_part_optim.py remains untouched.  This launcher adapts
our audited TartanAir benchmark wrapper in memory to native EuRoC routing,
changes the holdout rule to raw-index modulo 25 == 20 (equivalent to retained
stride-5 sample ids 4,9,14,...), and applies the timing-accumulator hotfix before
executing the released PGO/deformation/train-only-SR backend.
"""

import os
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.join(_THIS_DIR, "tartanair_pose_graph_part_optim.py")


def _replace_once(source, old, new, label):
    n = source.count(old)
    if n != 1:
        raise RuntimeError(
            f"Could not adapt EuRoC backend '{label}': expected 1 match, found {n}."
        )
    return source.replace(old, new, 1)


def main():
    with open(_TEMPLATE, "r", encoding="utf-8") as f:
        source = f.read()

    # ------------------------------------------------------------------
    # Convert the benchmark wrapper's generated-backend routing to EuRoC.
    # Upstream pose_graph_part_optim.py already supports EurocDataset.
    # ------------------------------------------------------------------
    source = source.replace('os.environ["TARTANAIR_SEQUENCE"]', 'os.environ["EUROC_SEQUENCE"]')
    source = source.replace("dataset_type = 'tartanair'", "dataset_type = 'euroc'")
    source = source.replace(
        "from configs.tartanair.lsgslam_full_split_8_2 import config",
        "from configs.euroc.lsgslam_full_stride5_8_2 import config",
    )

    tartan_basedir = '''    dataset_config['basedir'] = os.path.join(\n        os.environ.get(\n            "TARTANAIR_DATA_ROOT",\n            "/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge",\n        ),\n        "stereo",\n    )\n'''
    euroc_basedir = '''    dataset_config['basedir'] = os.environ["EUROC_CAM0_DIR"]\n'''
    source = _replace_once(source, tartan_basedir, euroc_basedir, "dataset basedir")

    # Strict 8:2 split AFTER stride=5: retained ids 4,9,14,... map to raw
    # indices 20,45,70,... .
    source = source.replace("if global_frame_idx % 5 == 4:", "if global_frame_idx % 25 == 20:")
    source = source.replace(
        "if ((start_idx + idx * stride) % 5) != 4",
        "if ((start_idx + idx * stride) % 25) != 20",
    )
    source = source.replace(
        "'split_rule': 'global frames 4,9,14,... are pose-only; excluded from mapping and SR loss',",
        "'split_rule': 'after stride=5, retained ids 4,9,14,... (raw 20,45,70,...) are pose-only and excluded from mapping/SR loss',",
    )
    source = source.replace("TartanAir adapter", "EuRoC adapter")

    # ------------------------------------------------------------------
    # Timing hotfix: initialize PGO/deformation/SR accumulators before PGO,
    # not later in the rendering accumulator block.
    # ------------------------------------------------------------------
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

    module = types.ModuleType("lsg_euroc_pose_graph_part_optim_final")
    module.__file__ = _TEMPLATE
    module.__name__ = "__main__"
    exec(compile(source, _TEMPLATE, "exec"), module.__dict__)


if __name__ == "__main__":
    main()
