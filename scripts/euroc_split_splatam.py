#!/usr/bin/env python3
"""EuRoC stride=5 + strict 8:2 split entry point.

This reuses the already-audited TartanAir split instrumentation without
modifying upstream scripts/splatam.py.  EuRoC is natively supported by the
upstream dataset dispatcher, so the only semantic change required is the split
period: after stride=5, held-out sample ids 4,9,14,... correspond to raw dataset
indices 20,45,70,..., i.e. raw-index modulo 25 == 20.
"""

import os
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.join(_THIS_DIR, "tartanair_split_splatam.py")


def _replace(source, old, new, label, count=1):
    n = source.count(old)
    if n < count:
        raise RuntimeError(
            f"Could not adapt split runner '{label}': expected at least {count} match(es), found {n}."
        )
    return source.replace(old, new, count)


def main():
    with open(_TEMPLATE, "r", encoding="utf-8") as f:
        source = f.read()

    # Raw-index representation of an 8:2 split applied AFTER stride=5.
    source = _replace(
        source,
        '''    if int(experiment.config.get("eval_split_every", 0)) != 5 or int(\n        experiment.config.get("eval_split_offset", -1)\n    ) != 4:\n        raise ValueError("This runner expects eval_split_every=5 and eval_split_offset=4.")\n''',
        '''    if int(experiment.config.get("eval_split_every", 0)) != 25 or int(\n        experiment.config.get("eval_split_offset", -1)\n    ) != 20:\n        raise ValueError(\n            "EuRoC stride-5 benchmark expects raw-index split: "\n            "eval_split_every=25 and eval_split_offset=20 "\n            "(retained sample ids 4,9,14,... are test)."\n        )\n''',
        "split validation",
    )

    source = source.replace(
        "TartanAir benchmark split: every fifth GLOBAL frame is held out from",
        "EuRoC benchmark split: every fifth RETAINED stride-5 frame is held out from",
    )
    source = source.replace(
        '"test_rule": "global frame indices 4,9,14,...; pose tracking only; no Gaussian insertion/mapping/keyframe",',
        '"test_rule": "after stride=5, retained sample ids 4,9,14,... (raw indices 20,45,70,...) are pose-only; no Gaussian insertion/mapping/keyframe",',
    )
    source = source.replace("TartanAir 8:2 Benchmark", "EuRoC stride5 8:2 Benchmark")
    source = source.replace(
        "Route TartanAir without modifying upstream dataset dispatch.",
        "Use upstream EuRoC dataset dispatch; TartanAir routing remains dormant.",
    )

    module = types.ModuleType("lsg_euroc_split_splatam")
    # Keep the template path so its relative _BASE_DIR calculation is unchanged.
    module.__file__ = _TEMPLATE
    module.__name__ = "__main__"
    exec(compile(source, _TEMPLATE, "exec"), module.__dict__)


if __name__ == "__main__":
    main()
