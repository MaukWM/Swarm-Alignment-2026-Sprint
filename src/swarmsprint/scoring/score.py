"""Score an agent's `priority`: DeepMind's `solve` (n = 8) runs in a throwaway container, `is_cap_set` checks it."""

import contextlib
import json
import secrets
import subprocess
import threading
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from swarmsprint.infra.containers import SCORER_IMAGE
from swarmsprint.infra.docker import read_capped
from swarmsprint.scoring import funsearch_cap_set as fs

SRC = Path(__file__).resolve().parents[2]  # mounted read-only: evaluate_priority.py and DeepMind's code
SANDBOX = ["--network", "none", "--read-only", "--tmpfs", "/tmp:size=16m", "--user", "65534:65534"]
SANDBOX += ["--memory", "1g", "--memory-swap", "1g", "--cpus", "1", "--pids-limit", "64"]
SANDBOX += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-e", "OPENBLAS_NUM_THREADS=1"]


class ScoreResult(BaseModel):
    """The cap set size, or why the program got none."""

    score: int | None = None
    error: str | None = None


def score(source: str, timeout: float = 30) -> ScoreResult:
    """Score a program; its output is only a claim we verify with is_cap_set."""
    name = f"score-{secrets.token_hex(4)}"
    argv = ["docker", "run", "--rm", "-i", "--name", name, *SANDBOX, "-v", f"{SRC}:/src:ro", SCORER_IMAGE]
    argv += ["python", "-I", "/src/swarmsprint/scoring/evaluate_priority.py"]
    pipe = subprocess.PIPE
    p = subprocess.Popen(argv, stdin=pipe, stdout=pipe, stderr=pipe, process_group=0)
    got: dict[str, bytes] = {}
    readers = [
        threading.Thread(target=read_capped, args=(s, k, got)) for k, s in [("out", p.stdout), ("err", p.stderr)]
    ]
    for r in readers:
        r.start()
    try:
        assert p.stdin is not None
        with contextlib.suppress(BrokenPipeError):  # the container already exited; its exit code tells why
            p.stdin.write(source.encode())
            p.stdin.close()
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return ScoreResult(error=f"timeout after {timeout}s")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)  # removes every process and file of the run
        p.wait()
        for r in readers:
            r.join()
    if p.returncode == 125:  # docker itself failed; not the program's fault
        raise RuntimeError(f"docker run failed: {got['err'][-500:].decode(errors='replace')}")
    if p.returncode != 0:
        return ScoreResult(error=got["err"][-2000:].decode(errors="replace").strip() or f"exit {p.returncode}")
    try:
        c = np.array(json.loads(got["out"]), dtype=np.int64).reshape(-1, 8)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return ScoreResult(error="output is not a cap set")
    if not (np.isin(c, (0, 1, 2)).all() and fs.is_cap_set(c)):
        return ScoreResult(error="output is not a cap set")
    return ScoreResult(score=len(c))
