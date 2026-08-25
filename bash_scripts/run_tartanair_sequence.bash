#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash bash_scripts/run_tartanair_sequence.bash SE000 0 39 1
#
# The end index is inclusive, matching configs/tartanair/lsgslam.py.

SEQUENCE="${1:-SE000}"
START="${2:-0}"
END="${3:-39}"
STRIDE="${4:-1}"

export TARTANAIR_SEQUENCE="$SEQUENCE"
export TARTANAIR_START="$START"
export TARTANAIR_END="$END"
export TARTANAIR_STRIDE="$STRIDE"
export PYTHONFAULTHANDLER=1

python -u scripts/tartanair_splatam.py configs/tartanair/lsgslam.py
