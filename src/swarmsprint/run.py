"""Run one experiment: start the containers, drive every agent in parallel, archive /team drives, tear down.

usage: python -m swarmsprint.run experiments/<config>.toml   (writes runs/<name>-<timestamp>/)
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from swarmsprint import containers
from swarmsprint.agent import BASH, Log, run_agent
from swarmsprint.config import RunConfig, load


def git(*args: str) -> str:
    """Return the output of a git command in the current repo."""
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def main(path: str) -> int:
    """Run the config; return 0 only if every agent finished and every /team drive was archived."""
    cfg = load(path)
    out = Path("runs") / f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    lock, f, seq, t0 = threading.Lock(), open(out / "events.jsonl", "w"), [0], time.monotonic()

    def log(ev: str, **fields: Any) -> None:
        with lock:
            seq[0] += 1
            meta = {"seq": seq[0], "t": time.time(), "mono": round(time.monotonic() - t0, 3), "run": cfg.name}
            f.write(json.dumps({**meta, "ev": ev, **fields}, default=str) + "\n")
            f.flush()

    stop, failed, workers = threading.Event(), list[str](), list[tuple[str, str, int]]()
    try:
        workers = containers.up(cfg)
        source = {
            "commit": git("rev-parse", "HEAD").strip(),
            "status": git("status", "--porcelain"),
            "diff": git("diff", "HEAD"),
        }
        image = containers.docker("image", "inspect", containers.IMAGE, "--format", "{{.Id}}")
        endpoint = os.environ["LLM_BASE_URL"]
        log(
            "start",
            config=cfg.model_dump(),
            source=source,
            image=image,
            endpoint=endpoint,
            tools=[BASH],
            workers=workers,
        )
        prompts = {g.name: g.prompt for g in cfg.groups}
        threads = [
            threading.Thread(target=drive, args=(cfg, *w, prompts[w[1]], log, stop, failed), daemon=True)
            for w in workers
        ]
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            stop.set()
            failed.append("interrupted")
            for t in threads:
                t.join()
    finally:
        archived = all(archive(cfg, workers, out, log))
        if archived:
            containers.down(cfg)
        log("end", failed=failed, archived=archived)
        f.close()
    print(out if archived else f"{out} (archive failed: containers and drives kept)")
    return 0 if archived and not failed else 1


def archive(cfg: RunConfig, workers: list[tuple[str, str, int]], out: Path, log: Log) -> list[bool]:
    """Copy each group's /team drive into the run directory; log success or failure per group."""
    results = []
    for g in cfg.groups if workers else []:
        worker = next(w for w, group, _ in workers if group == g.name)
        try:
            containers.docker("cp", f"{worker}:/team", str(out / f"team-{g.name}"))
            log("archive", group=g.name, ok=True)
            results.append(True)
        except Exception as e:
            log("archive", group=g.name, ok=False, error=repr(e))
            results.append(False)
    return results


def drive(
    cfg: RunConfig, worker: str, group: str, index: int, prompt: str, log: Log, stop: threading.Event, failed: list[str]
) -> None:
    """Run one agent with every log line tagged by its worker, group and index; record a crash instead of raising."""

    def tagged(ev: str, **fields: Any) -> None:
        log(ev, worker=worker, group=group, agent=index, **fields)

    try:
        run_agent(cfg, worker, prompt, tagged, stop)
    except Exception as e:
        tagged("error", error=repr(e))
        failed.append(worker)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
