"""Start a run's hub on the host, give every worker its token, and check that every group reaches it."""

import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from swarmsprint.config import RunConfig
from swarmsprint.infra.containers import Worker
from swarmsprint.infra.docker import docker

PROBE_AGENT = "swarmsprint-preflight"  # User-Agent of the runner's own checks, so analysis can drop them


def port_free(port: int) -> bool:
    """Whether nothing listens on 127.0.0.1:port (else agent traffic would reach that other service)."""
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def serve(cfg: RunConfig, workers: list[Worker], out: Path) -> tuple[subprocess.Popen[bytes], dict[str, str]]:
    """Create one account and token per worker and start the hub as a subprocess; returns it and the tokens."""
    tokens = {w.name: f"hub_{secrets.token_urlsafe(24)}" for w in workers}
    (out / "accounts.json").write_text(json.dumps(tokens, indent=1))
    env = {**os.environ, "HUB_ACCOUNTS": str(out / "accounts.json"), "HUB_EVENTS": str(out / "hub.jsonl")}
    app = ["uvicorn", "swarmsprint.hub.main:app", "--host", "127.0.0.1", "--port", str(cfg.hub_port)]
    with (out / "hub.log").open("w") as log:
        # own process group: a Ctrl-C in the terminal reaches only the runner, which stops the hub itself
        server = subprocess.Popen([sys.executable, "-m", *app], env=env, stdout=log, stderr=log, process_group=0)
    return server, tokens


def hand_out(cfg: RunConfig, workers: list[Worker], server: subprocess.Popen[bytes], tokens: dict[str, str]) -> None:
    """Wait for the hub, write each token to its worker's /root/.token, and check every group reaches it."""
    wait_ready(cfg, server)
    for w in workers:
        docker("exec", "-i", w.name, "sh", "-c", "umask 077 && cat > /root/.token", stdin=tokens[w.name])
    # end to end: a worker of each group, through its forwarder, must get its own account back from this hub
    for group in cfg.groups:
        w = next(w for w in workers if w.group == group.name)
        auth = '-H "Authorization: Bearer $(cat /root/.token)"'
        me = f"curl -fsS -m 10 -A {PROBE_AGENT} {auth} http://{cfg.hub_host}/me"
        if json.loads(docker("exec", w.name, "sh", "-c", me))["account"] != w.name:
            raise SystemExit(f"group {group.name}: the hub answered for another account")


def stop(server: subprocess.Popen[bytes]) -> int | None:
    """Stop the hub (kill after 10 s); return its exit code if it had already died during the run."""
    died = server.poll()
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:  # uvicorn waits for open requests, e.g. a submission still being scored
        server.kill()
        server.wait()
    return died


def wait_ready(cfg: RunConfig, server: subprocess.Popen[bytes]) -> None:
    """Wait up to 30 s until the hub answers at all (a 401 counts: no token is needed for this check)."""
    request = urllib.request.Request(f"http://127.0.0.1:{cfg.hub_port}/me", headers={"User-Agent": PROBE_AGENT})
    for _ in range(60):
        if server.poll() is not None:
            raise SystemExit("the hub exited at start; see hub.log in the run directory")
        try:
            urllib.request.urlopen(request, timeout=2).close()
            return
        except urllib.error.HTTPError:
            return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("the hub did not answer within 30 s")


class Scores:
    """Each account's best scored submission so far, read from the hub's event log (so it adds no hub traffic)."""

    def __init__(self, events: Path) -> None:
        self.events, self.offset, self.lock = events, 0, threading.Lock()
        self.best: dict[str, tuple[int, int]] = {}

    def of(self, account: str) -> tuple[int, int] | None:
        """(score, submission id) of the account's best submission, or None before its first score."""
        with self.lock, self.events.open("rb") as f:
            f.seek(self.offset)
            complete, newline, _ = f.read().rpartition(b"\n")  # a line still being written waits for the next call
            self.offset += len(complete) + len(newline)
            for line in complete.split(b"\n") if newline else []:
                e = json.loads(line)
                if e["ev"] == "submit" and e["response"]["score"] is not None:
                    score, sid = e["response"]["score"], e["response"]["id"]
                    if e["account"] not in self.best or score > self.best[e["account"]][0]:
                        self.best[e["account"]] = (score, sid)
            return self.best.get(account)
