"""Docker CLI helpers shared by the worker containers and the agent loop."""

import subprocess
from typing import Any

CAP = 1_000_000  # bytes kept per output stream of a command


def docker(*args: str, stdin: str | None = None) -> str:
    """Run a docker command and return its stdout; docker's own error text goes to stderr."""
    # own process group: a Ctrl-C in the terminal must not kill docker mid-teardown
    p = subprocess.run(["docker", *args], input=stdin, check=True, stdout=subprocess.PIPE, text=True, process_group=0)
    return p.stdout.strip()


def read_capped(stream: Any, key: str, into: dict[str, bytes]) -> None:
    """Keep the first CAP bytes of a pipe and discard the rest, so docker still reports the exit code."""
    into[key] = stream.read(CAP)
    while stream.read(2**20):
        pass
