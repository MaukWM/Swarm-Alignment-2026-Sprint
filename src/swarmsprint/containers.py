"""Start and stop the agent containers for a run config: one internal network and one shared drive per group.

Every name an agent can see (container, hostname, network, drive) is random; run, group and agent index live in
docker labels, which are not visible from inside a container.

usage: python -m swarmsprint.containers {up,down} experiments/<config>.toml
"""

import secrets
import subprocess
import sys
from pathlib import Path

from swarmsprint.config import RunConfig, load

IMAGE, SCORER_IMAGE = "swarmsprint-agent", "swarmsprint-scorer"
ROOT = Path(__file__).resolve().parents[2]
LIMITS = ["--pids-limit", "256", "--memory", "1g", "--memory-swap", "1g", "--cpus", "1", "--init"]
LIMITS += ["--ulimit", "nofile=1024:1024", "--ulimit", "core=0:0"]
HARDEN = ["--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only"]
SCRATCH = ["--tmpfs", "/tmp:size=64m", "--tmpfs", "/root:size=64m", "--tmpfs", "/work:size=256m,exec"]
NO_GATEWAY = ["--opt", "com.docker.network.bridge.gateway_mode_ipv4=isolated"]  # agents can't reach the VM (Docker 28+)
TEAM_DRIVE = ["--opt", "type=tmpfs", "--opt", "device=tmpfs", "--opt", "o=size=256m"]
RESOURCES = [
    (["ps", "-aq"], ["rm", "-f"]),
    (["network", "ls", "-q"], ["network", "rm"]),
    (["volume", "ls", "-q"], ["volume", "rm"]),
]


def docker(*args: str, stdin: str | None = None) -> str:
    """Run a docker command and return its stdout; docker's own error text goes to stderr."""
    # own process group: a Ctrl-C in the terminal must not kill docker mid-teardown
    p = subprocess.run(["docker", *args], input=stdin, check=True, stdout=subprocess.PIPE, text=True, process_group=0)
    return p.stdout.strip()


def up(cfg: RunConfig) -> list[tuple[str, str, int]]:
    """Build the image, then per group create an internal network, a shared drive and its agent containers.

    Returns (worker name, group, agent index) per container.
    """
    run = cfg.name
    if left := existing(cfg):
        raise SystemExit(f"run {run!r} still has docker resources {left}; inspect them, then `containers down`")
    workers = []
    label = f"swarmsprint.run={run}"
    docker("build", "-q", "-t", IMAGE, "-", stdin=(ROOT / "Dockerfile.agent").read_text())
    docker("build", "-q", "-t", SCORER_IMAGE, "-", stdin=(ROOT / "Dockerfile.scorer").read_text())
    for group in cfg.groups:
        net, drive = secrets.token_hex(4), secrets.token_hex(4)
        tags = ["--label", label, "--label", f"swarmsprint.group={group.name}"]
        docker("network", "create", "--internal", *NO_GATEWAY, *tags, net)
        docker("volume", "create", *TEAM_DRIVE, *tags, drive)
        # keeper: holds the tmpfs drive mounted (its data would vanish with the last user) and is the archive source
        keeper = ["--network", "none", "--label", "swarmsprint.keeper=1", "-v", f"{drive}:/team:ro"]
        docker("run", "-d", *tags, *keeper, *LIMITS, *HARDEN, IMAGE)
        for i in range(group.agents):
            name = f"worker-{secrets.token_hex(3)}"
            ids = ["--name", name, "--hostname", name, "--network", net, *tags, "--label", f"swarmsprint.agent={i}"]
            docker("run", "-d", *ids, "-v", f"{drive}:/team", *LIMITS, *HARDEN, *SCRATCH, IMAGE)
            workers.append((name, group.name, i))
    return workers


def existing(cfg: RunConfig) -> list[str]:
    """Ids of every container, network and drive labelled with this run."""
    flt = ["--filter", f"label=swarmsprint.run={cfg.name}"]
    return [i for ls, _ in RESOURCES for i in docker(*ls, *flt).split()]


def down(cfg: RunConfig) -> None:
    """Remove every container, network and drive labelled with this run."""
    flt = ["--filter", f"label=swarmsprint.run={cfg.name}"]
    for ls, rm in RESOURCES:
        if ids := docker(*ls, *flt).split():
            docker(*rm, *ids)


if __name__ == "__main__":
    print({"up": up, "down": down}[sys.argv[1]](load(sys.argv[2])))
