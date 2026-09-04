"""Fixed launcher for the TartanAir LSG-SLAM backend benchmark.

This launcher fixes the timing-accumulator initialization order in
`tartanair_pose_graph_part_optim.py` without changing the released
`pose_graph_part_optim.py` algorithm.  In the first timing version,
`pgo_opt_seconds` was injected after the PGO loop but referenced inside it.
Here the three backend timing accumulators are initialized before the released
pose-graph loop and are not reset later.
"""

import os
import types


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WRAPPER = os.path.join(_THIS_DIR, "tartanair_pose_graph_part_optim.py")


def _replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not apply final backend timing hotfix '{label}': "
            f"expected 1 match, found {count}."
        )
    return source.replace(old, new, 1)


def main():
    with open(_WRAPPER, "r", encoding="utf-8") as f:
        source = f.read()

    # The first timing wrapper initialized these inside the later rendering
    # accumulator block.  Remove that late initialization so PGO time is not
    # reset after it has already been measured.
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

    # Inject an additional source patch into the wrapper.  This modifies only
    # the generated TartanAir backend source in memory; the released backend
    # file on disk remains untouched.
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

    module = types.ModuleType("lsg_tartanair_pose_graph_part_optim_final_launcher")
    module.__file__ = _WRAPPER
    module.__name__ = "__main__"
    exec(compile(source, _WRAPPER, "exec"), module.__dict__)


if __name__ == "__main__":
    main()
