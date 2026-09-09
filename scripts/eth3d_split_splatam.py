#!/usr/bin/env python3
"""ETH3D stride=1 + strict 8:2 split entry point.

The rectified ETH3D cache is exposed through the native EurocDataset interface,
so the already-audited split instrumentation can be reused without modifying
upstream scripts/splatam.py. Every fifth frame (4,9,14,...) is pose-only:
tracking continues, but Gaussian insertion, mapping optimization, and mapping
keyframe insertion are skipped.
"""

import os
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.join(_THIS_DIR, "tartanair_split_splatam.py")


def main():
    with open(_TEMPLATE, "r", encoding="utf-8") as f:
        source = f.read()

    source = source.replace(
        "TartanAir benchmark split: every fifth GLOBAL frame is held out from",
        "ETH3D benchmark split: every fifth frame is held out from",
    )
    source = source.replace(
        '"test_rule": "global frame indices 4,9,14,...; pose tracking only; no Gaussian insertion/mapping/keyframe",',
        '"test_rule": "ETH3D frames 4,9,14,... are pose-only; no Gaussian insertion/mapping/keyframe",',
    )
    source = source.replace("TartanAir 8:2 Benchmark", "ETH3D 8:2 Benchmark")
    source = source.replace(
        "Route TartanAir without modifying upstream dataset dispatch.",
        "Use native EurocDataset dispatch for the ETH3D-compatible cache.",
    )

    module = types.ModuleType("lsg_eth3d_split_splatam")
    module.__file__ = _TEMPLATE
    module.__name__ = "__main__"
    exec(compile(source, _TEMPLATE, "exec"), module.__dict__)


if __name__ == "__main__":
    main()
