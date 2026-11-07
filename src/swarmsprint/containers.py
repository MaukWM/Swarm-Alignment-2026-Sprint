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

IMAGE = "swarmsprint-agent"
DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile.agent"
LIMITS = ["--pids-limit", "256", "--memory", "1g", "--cpus", "1", "--init"]
HARDEN = ["--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
RESOURCES = [
    (["ps", "-aq"], ["rm", "-f"]),
    (["network", "ls", "-q"], ["network", "rm"]),
    (["volume", "ls", "-q"], ["volume", "rm"]),
]


def docker(*args: str, stdin: str | None = None) -> str:
    """Run a docker command and return its stdout; docker's own error text goes to stderr."""
    return subprocess.run(["docker", *args], input=stdin, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()


def up(cfg: RunConfig) -> None:
    """Build the image, then per group create an internal network, a shared drive and its agent containers."""
    run = cfg.name
    down(cfg)
    label = f"swarmsprint.run={run}"
    docker("build", "-q", "-t", IMAGE, "-", stdin=DOCKERFILE.read_text())
    for group in cfg.groups:
        net, drive = secrets.token_hex(4), secrets.token_hex(4)
        tags = ["--label", label, "--label", f"swarmsprint.group={group.name}"]
        docker("network", "create", "--internal", *tags, net)
        docker("volume", "create", *tags, drive)
        for i in range(group.agents):
            name = f"worker-{secrets.token_hex(3)}"
            ids = ["--name", name, "--hostname", name, "--network", net, *tags, "--label", f"swarmsprint.agent={i}"]
            docker("run", "-d", *ids, "-v", f"{drive}:/team", *LIMITS, *HARDEN, IMAGE)
            print(name, group.name, i)


def down(cfg: RunConfig) -> None:
    """Remove every container, network and drive labelled with this run."""
    flt = ["--filter", f"label=swarmsprint.run={cfg.name}"]
    for ls, rm in RESOURCES:
        if ids := docker(*ls, *flt).split():
            docker(*rm, *ids)


if __name__ == "__main__":
    {"up": up, "down": down}[sys.argv[1]](load(sys.argv[2]))
