"""Score a bin-packing `priority` on hidden instances: DeepMind's packing loop runs in a throwaway container, and the
packings it returns are checked with DeepMind's `is_valid_packing` before the bins are counted.
"""

import builtins
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
from swarmsprint.scoring import funsearch_bin_packing as fs

SRC = Path(__file__).resolve().parents[2]  # mounted read-only: evaluate_heuristic.py and DeepMind's code
SANDBOX = ["--network", "none", "--read-only", "--tmpfs", "/tmp:size=16m", "--user", "65534:65534"]
SANDBOX += ["--memory", "1g", "--memory-swap", "1g", "--cpus", "1", "--pids-limit", "64"]
SANDBOX += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-e", "OPENBLAS_NUM_THREADS=1"]
EXCEPTIONS = {n for n, v in vars(builtins).items() if isinstance(v, type) and issubclass(v, BaseException)}
TIMEOUT = 60  # seconds per submission; a pure-Python heuristic needs ~18 s for 5 x 5000 items


class Instance(BaseModel):
    capacity: int
    items: list[int]


class ScoreResult(BaseModel):
    """Average number of bins used over the hidden instances (lower is better), or why the program got none."""

    score: float | None = None
    error: str | None = None


def weibull_instances(seed: int, n: int = 5, items: int = 5000, capacity: int = 100) -> list[Instance]:
    """FunSearch's Weibull 5k setup: item sizes ~ Weibull(shape 3, scale 45), rounded, clipped to [1, capacity]."""
    rng = np.random.default_rng(seed)
    sizes = np.clip(np.round(rng.weibull(3, (n, items)) * 45), 1, capacity).astype(int)
    return [Instance(capacity=capacity, items=row.tolist()) for row in sizes]


def score(source: str, instances: list[Instance], timeout: float = TIMEOUT) -> ScoreResult:
    """Run the program on the instances; its packings are only a claim, checked here before counting bins."""
    try:
        compile(source, "<submission>", "exec")  # compiles only, runs nothing: syntax errors can be shown in full
    except SyntaxError as e:
        return ScoreResult(error=f"SyntaxError: {e}")
    except (ValueError, RecursionError, MemoryError):
        return ScoreResult(error="the program does not compile")
    name = f"score-{secrets.token_hex(4)}"
    argv = ["docker", "run", "--rm", "-i", "--name", name, *SANDBOX, "-v", f"{SRC}:/src:ro", SCORER_IMAGE]
    # timeout inside the container too: it dies on its own even if this process is killed mid-evaluation
    entry = "/src/swarmsprint/scoring/evaluate_heuristic.py"
    argv += ["timeout", "-s", "KILL", str(timeout + 5), "python", "-I", entry]
    pipe = subprocess.PIPE
    p = subprocess.Popen(argv, stdin=pipe, stdout=pipe, stderr=pipe, process_group=0)
    got: dict[str, bytes] = {}
    pipes = [("out", p.stdout), ("err", p.stderr)]
    readers = [threading.Thread(target=read_capped, args=(s, k, got)) for k, s in pipes]
    for r in readers:
        r.start()
    job = json.dumps({"program": source, "instances": [i.model_dump() for i in instances]})
    try:
        assert p.stdin is not None
        with contextlib.suppress(BrokenPipeError):  # the container already exited; its exit code tells why
            p.stdin.write(job.encode())
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
    try:
        result = json.loads(got["out"])
        if "error" in result:  # the program can choose this text, so only a builtin exception name passes
            error = result["error"] if result["error"] in EXCEPTIONS else "unknown error"
            return ScoreResult(error=f"{error} while packing")
        packings = result["packings"]
        if len(packings) != len(instances):
            return ScoreResult(error="evaluation failed")
        bins = []
        for packing, inst in zip(packings, instances, strict=True):
            if len(packing) > len(inst.items) or not fs.is_valid_packing(
                packing, [float(x) for x in inst.items], inst.capacity
            ):
                return ScoreResult(error="invalid packing")
            bins.append(len(packing))
    except (ValueError, TypeError, KeyError, RecursionError):
        return ScoreResult(error="evaluation failed")
    return ScoreResult(score=round(float(np.mean(bins)), 2))
