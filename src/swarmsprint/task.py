"""Score an agent's `priority`: DeepMind's `solve` (n = 8) runs sandboxed, its cap set is checked with `is_cap_set`."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from swarmsprint import funsearch_cap_set as fs

RUNNER = Path(__file__).with_name("evaluate_priority.py")


def score(source: str, timeout: float = 30) -> dict[str, int | str]:
    """Return {"score": cap set size} or {"error": reason}; the program's output is only a claim we verify."""
    with tempfile.TemporaryFile() as o, tempfile.TemporaryFile() as e:
        p = subprocess.Popen(
            [sys.executable, "-I", str(RUNNER)],
            stdin=subprocess.PIPE,
            stdout=o,
            stderr=e,
            env={"OPENBLAS_NUM_THREADS": "1"},
            start_new_session=True,
        )
        try:
            p.communicate(source.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)  # also the program's children
            p.wait()
        o.seek(0)
        e.seek(0)
        if p.returncode != 0:
            return {"error": e.read()[-2000:].decode(errors="replace").strip() or f"exit {p.returncode}"}
        try:
            c = np.array(json.loads(o.read()), dtype=np.int64).reshape(-1, 8)
        except (ValueError, TypeError, OverflowError):
            return {"error": "output is not a cap set"}
    if not (np.isin(c, (0, 1, 2)).all() and fs.is_cap_set(c)):
        return {"error": "output is not a cap set"}
    return {"score": len(c)}
