"""Run one experiment: start the containers, drive every agent in parallel, archive /team drives, tear down.

usage: python -m swarmsprint.runner.run experiments/<config>.toml   (writes runs/<name>-<timestamp>/)
"""

import contextlib
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarmsprint.agents.loop import BASH, Log, run_agent
from swarmsprint.agents.prompts import render
from swarmsprint.config import Group, RunConfig, load
from swarmsprint.hub.function import HIGH, LOW, Function
from swarmsprint.infra import containers
from swarmsprint.infra.docker import docker
from swarmsprint.runner import hub
from swarmsprint.viewer import server as viewer

ARCHIVE_CAP = 300 * 2**20  # bytes per /team tar; the drive holds 256 MB but sparse files can expand


def ignore_signals() -> None:
    """Ignore further Ctrl-C, SIGTERM and SIGHUP."""
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)


def interrupt(signum: int, frame: Any) -> None:
    """Turn SIGTERM and SIGHUP into the same clean stop as Ctrl-C."""
    raise KeyboardInterrupt


def git(*args: str) -> str:
    """Return the output of a git command in the current repo."""
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def write_data(cfg: RunConfig, seed: int, out: Path) -> Path:
    """Each group's starting knowledge in data/<group>/initial.csv: random points and their values, new per group."""
    function = Function(cfg.function, cfg.dimension, seed)
    header = ",".join([f"x{j + 1}" for j in range(cfg.dimension)] + ["f"])
    for i, group in enumerate(cfg.groups):
        points = np.random.default_rng([seed, i]).uniform(LOW, HIGH, (cfg.initial_points, cfg.dimension))
        rows = [[*p, function(p.tolist())] for p in points]
        (out / "data" / group.name).mkdir(parents=True)
        np.savetxt(out / "data" / group.name / "initial.csv", rows, "%.10g", ",", header=header, comments="")
    return out / "data"


def main(path: str) -> int:
    """Run the config; return 0 only if every agent finished and every /team drive was archived."""
    cfg = load(path)
    if left := containers.existing(cfg):
        sys.exit(f"run {cfg.name!r} still has docker resources {left}; inspect them, then `containers down`")
    if not hub.port_free(cfg.hub_port):
        sys.exit(f"port {cfg.hub_port} is in use; the hub needs it (config: hub_port)")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    out = Path("runs") / f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    print(f"Viewer: {viewer.serve(out)}", flush=True)
    lock, f, seq, t0 = threading.Lock(), open(out / "events.jsonl", "w"), [0], time.monotonic()

    def log(ev: str, **fields: Any) -> None:
        with lock:
            seq[0] += 1
            meta = {"seq": seq[0], "t": time.time(), "mono": round(time.monotonic() - t0, 3), "run": cfg.name}
            f.write(json.dumps({**meta, "ev": ev, **fields}, default=str) + "\n")
            f.flush()

    stop, failed, workers = threading.Event(), list[str](), list[containers.Worker]()
    server: subprocess.Popen[bytes] | None = None
    try:
        seed = cfg.seed if cfg.seed is not None else secrets.randbits(31) + 1
        data = write_data(cfg, seed, out)
        workers = containers.up(cfg)
        for w in workers:
            docker(
                "exec",
                "-i",
                w.name,
                "sh",
                "-c",
                "cat > /data/initial.csv",
                stdin=(data / w.group / "initial.csv").read_text(),
            )
        server, tokens = hub.serve(cfg, workers, out, seed)
        hub.hand_out(cfg, workers, server, tokens)
        source = {
            "commit": git("rev-parse", "HEAD").strip(),
            "status": git("status", "--porcelain"),
            "diff": git("diff", "HEAD"),
        }
        image = docker("image", "inspect", containers.IMAGE, "--format", "{{.Id}}")
        endpoint = os.environ["LLM_BASE_URL"]
        log(
            "start",
            config=cfg.model_dump(),
            source=source,
            image=image,
            endpoint=endpoint,
            tools=[BASH],
            workers=[w._asdict() for w in workers],
            seed=seed,  # of the hidden function; agents never see it
        )
        scores, groups = hub.Scores(out / "hub.jsonl"), {g.name: g for g in cfg.groups}
        threads = []
        for w in workers:
            args = (cfg, w, groups[w.group], scores, log, stop, failed)
            threads.append(threading.Thread(target=drive, args=args, daemon=True))
        log("agents_start")  # hub events before this moment are the runner's own setup and checks
        if cfg.duration_minutes:

            def time_up() -> None:
                log("time_limit")
                stop.set()

            timer = threading.Timer(cfg.duration_minutes * 60, time_up)
            timer.daemon = True
            timer.start()
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            ignore_signals()  # uv forwards a second Ctrl-C; let the agents finish their turn and log their stop
            stop.set()
            failed.append("interrupted")
            for t in threads:
                t.join()
    except BaseException as e:
        failed.append(f"aborted: {e!r}")
        raise
    finally:
        ignore_signals()  # a second signal must not break the cleanup
        if server and (died := hub.stop(server)) is not None:
            failed.append(f"hub exited during the run with code {died}")
        archived = all(archive(cfg, workers, out, log))
        if archived:
            containers.down(cfg)
        log("end", failed=failed, archived=archived)
        f.close()
    print(out if archived else f"{out} (archive failed: containers and drives kept)")
    return 0 if archived and not failed else 1


def archive(cfg: RunConfig, workers: list[containers.Worker], out: Path, log: Log) -> list[bool]:
    """Freeze all workers, then archive each group's /team; one result per group."""
    for w in workers:
        with contextlib.suppress(subprocess.CalledProcessError):  # already exited or paused: nothing to freeze
            docker("pause", w.name)
    return [archive_group(cfg, g.name, out, log) for g in cfg.groups] if workers else []


def archive_group(cfg: RunConfig, group: str, out: Path, log: Log) -> bool:
    """Stream a group's /team from its keeper container into team-<group>.tar (cap + 1 MB at most); never raises."""
    tar, digest, size = out / f"team-{group}.tar", hashlib.sha256(), 0
    try:
        labels = [f"label=swarmsprint.run={cfg.name}", f"label=swarmsprint.group={group}", "label=swarmsprint.keeper=1"]
        keeper = docker("ps", "-q", *(a for lbl in labels for a in ("--filter", lbl)))
        p = subprocess.Popen(["docker", "cp", f"{keeper}:/team", "-"], stdout=subprocess.PIPE, process_group=0)
        assert p.stdout is not None
        with tar.open("wb") as f:
            while size <= ARCHIVE_CAP and (chunk := p.stdout.read(2**20)):
                f.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        p.kill()
        ok = p.wait() == 0 and size <= ARCHIVE_CAP
        skipped = unpack(tar, out / f"team-{group}") if ok else []
        log(
            "archive",
            group=group,
            ok=ok,
            bytes=size,
            sha256=digest.hexdigest(),
            file=tar.name,
            exit=p.returncode,
            skipped=skipped,
        )
        return ok
    except Exception as e:
        log("archive", group=group, ok=False, bytes=size, error=repr(e))
        return False


def unpack(tar: Path, dest: Path) -> list[str]:
    """Extract a readable copy with tarfile's data filter; return the entries it refused (links out, devices, FIFOs)."""
    skipped = []
    dest.mkdir()
    with tarfile.open(tar) as tf:
        for member in tf:
            if "/" not in member.name:
                continue  # the top-level team/ folder itself
            member = member.replace(name=member.name.split("/", 1)[1], deep=False)
            try:
                tf.extract(member, dest, filter="data")
            except tarfile.FilterError as e:
                skipped.append(f"{member.name}: {e}")
    return skipped


def drive(
    cfg: RunConfig,
    w: containers.Worker,
    group: Group,
    scores: hub.Scores,
    log: Log,
    stop: threading.Event,
    failed: list[str],
) -> None:
    """Run one agent with every log line tagged by its worker, group and index; record a crash instead of raising."""

    def tagged(ev: str, **fields: Any) -> None:
        log(ev, worker=w.name, group=w.group, agent=w.agent, **fields)

    try:
        system = render("system", worker=w.name, lab=group.lab, timeout=cfg.cmd_timeout)
        task = render(
            "task",
            lab=group.lab,
            hub=cfg.hub_host,
            n=group.agents,
            dimension=cfg.dimension,
            points=cfg.initial_points,
            start=f"{cfg.credits_start:g}",
            per_minute=f"{cfg.credits_per_minute:g}",
            cost=f"{cfg.credits_per_submission:g}",
        )
        tagged("prompt", system=system, task=task)
        run_agent(cfg, w.name, system, task, lambda: scores.of(w.name), tagged, stop)
    except Exception as e:
        tagged("error", error=repr(e))
        failed.append(w.name)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
