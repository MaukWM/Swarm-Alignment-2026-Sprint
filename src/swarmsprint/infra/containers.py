"""Start and stop the agent containers for a run config: one internal network and one shared drive per group.

Every name an agent can see (container, hostname, network, drive) is random; run, group and agent index live in
docker labels, which are not visible from inside a container.

usage: python -m swarmsprint.infra.containers {up,down} experiments/<config>.toml
"""

import secrets
import sys
from pathlib import Path
from typing import NamedTuple

from swarmsprint.config import RunConfig, load
from swarmsprint.infra.docker import docker

IMAGE, SCORER_IMAGE = "swarmsprint-agent", "swarmsprint-scorer"
PROXY = "caddy:2.11.6-alpine@sha256:d44355d3c2149dc580ce2cac735955d1c08d3d00882c30489c241aa51a5c10d9"
ROOT = Path(__file__).resolve().parents[3]
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


class Worker(NamedTuple):
    """One agent container: visible name, group, index in the group and IP on the group network."""

    name: str
    group: str
    agent: int
    ip: str


def up(cfg: RunConfig) -> list[Worker]:
    """Build the images, then per group create an internal network, a shared drive, a forwarder and the workers."""
    run = cfg.name
    if left := existing(cfg):
        raise SystemExit(f"run {run!r} still has docker resources {left}; inspect them, then `containers down`")
    workers, forwarders = [], []
    label = f"swarmsprint.run={run}"
    docker("build", "-q", "-t", IMAGE, "-", stdin=(ROOT / "Dockerfile.agent").read_text())
    docker("build", "-q", "-t", SCORER_IMAGE, "-", stdin=(ROOT / "Dockerfile.scorer").read_text())
    uplink = secrets.token_hex(4)
    docker("network", "create", "--label", label, uplink)
    for group in cfg.groups:
        net, drive = secrets.token_hex(4), secrets.token_hex(4)
        tags = ["--label", label, "--label", f"swarmsprint.group={group.name}"]
        docker("network", "create", "--internal", *NO_GATEWAY, *tags, net)
        docker("volume", "create", *TEAM_DRIVE, *tags, drive)
        forwarders.append(start_forwarder(cfg, uplink, net, tags))
        # keeper: holds the tmpfs drive mounted (its data would vanish with the last user) and is the archive source
        keeper = ["--network", "none", "--label", "swarmsprint.keeper=1", "-v", f"{drive}:/team:ro"]
        docker("run", "-d", *tags, *keeper, *LIMITS, *HARDEN, IMAGE)
        for i in range(group.agents):
            name = f"worker-{secrets.token_hex(3)}"
            ids = ["--name", name, "--hostname", name, "--network", net, *tags, "--label", f"swarmsprint.agent={i}"]
            docker("run", "-d", *ids, "-v", f"{drive}:/team", *LIMITS, *HARDEN, *SCRATCH, IMAGE)
            ip = docker("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name)
            workers.append(Worker(name, group.name, i, ip))
    if dead := [f for f in forwarders if docker("inspect", "-f", "{{.State.Running}}", f) != "true"]:
        raise SystemExit(f"forwarder(s) {dead} did not start; see `docker logs`")
    return workers


def start_forwarder(cfg: RunConfig, uplink: str, net: str, tags: list[str]) -> str:
    """Start one group's reverse proxy to the leaderboard; it reports each caller's IP in X-Forwarded-For."""
    on_networks = ["--network", uplink, "--add-host", "host.docker.internal:host-gateway", *tags]
    own = ["--label", "swarmsprint.forwarder=1", "--restart", "unless-stopped", "--sysctl", "net.ipv4.ip_forward=0"]
    caddy_needs = ["--cap-add", "NET_BIND_SERVICE", "--tmpfs", "/data:size=16m", "--tmpfs", "/config:size=16m"]
    many_conns = ["--ulimit", "nofile=65536:65536"]  # after LIMITS, so it wins: a full table would block the group
    proxy = ["caddy", "reverse-proxy", "--from", ":80", "--to", f"host.docker.internal:{cfg.leaderboard_port}"]
    fid = docker("run", "-d", *on_networks, *own, *LIMITS, *HARDEN, *caddy_needs, *many_conns, PROXY, *proxy)
    docker("network", "connect", "--alias", cfg.leaderboard_host, net, fid)
    return fid


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
