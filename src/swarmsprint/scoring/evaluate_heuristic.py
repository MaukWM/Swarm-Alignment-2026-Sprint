"""Runs inside the throwaway scorer container: pack the hidden instances with the submitted heuristic, print packings.

The hub checks the syntax beforehand; anything that goes wrong here is reported as just the exception's type name.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from swarmsprint.scoring import funsearch_bin_packing as fs  # noqa: E402

job = json.load(sys.stdin)
out = sys.stdout
sys.stdout = sys.stderr = open(os.devnull, "w")  # the program's own output goes nowhere
try:
    ns: dict[str, Any] = {
        "np": np
    }  # own namespace, so a helper named like one of DeepMind's functions doesn't replace it
    exec(job["program"], ns)
    fs.priority = ns["priority"]
    packings = []
    for inst in job["instances"]:
        packing, _ = fs.online_binpack(tuple(inst["items"]), np.array([inst["capacity"] for _ in inst["items"]]))
        packings.append([[int(x) for x in b] for b in packing if b])
    print(json.dumps({"packings": packings}), file=out)
except BaseException as e:  # noqa: BLE001 - every failure of the program is a result, including SystemExit
    print(json.dumps({"error": type(e).__name__}), file=out)
