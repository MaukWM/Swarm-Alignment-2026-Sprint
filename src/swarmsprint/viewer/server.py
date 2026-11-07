"""Read-only live view of a run: serves index.html, the run's two logs, and each group's /team files.

Run alongside an experiment (the runner starts it) or on a finished run: python -m swarmsprint.viewer.server runs/<run>
"""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = Path(__file__).with_name("index.html")
LOGS = {"/events": "events.jsonl", "/hub": "hub.jsonl"}
PORT = 18081
FILE_CAP = 200_000


def docker(*args: str) -> str:
    """Output of a docker command, or "" if it fails (the run may have ended in between)."""
    return subprocess.run(["docker", *args], capture_output=True, text=True).stdout


def keeper(run: str, group: str) -> str:
    """Id of the container holding the group's /team drive while the run is live."""
    labels = [f"swarmsprint.run={run}", f"swarmsprint.group={group}", "swarmsprint.keeper=1"]
    return docker("ps", "-q", *[a for lbl in labels for a in ("--filter", f"label={lbl}")]).strip()


def team_files(run_dir: Path, run: str, group: str) -> list[dict[str, object]]:
    """Every file on the group's drive: the archived copy after the run, the live drive during it."""
    local = run_dir / f"team-{group}"
    if local.is_dir():
        files = [p for p in local.rglob("*") if p.is_file()]
        return [
            {"path": str(p.relative_to(local)), "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in files
        ]
    if not (cid := keeper(run, group)):
        return []
    out = docker("exec", cid, "find", "/team", "-type", "f", "-printf", r"%P\t%s\t%T@\n")
    rows = [line.split("\t") for line in out.splitlines()]
    return [{"path": p, "size": int(s), "mtime": float(m)} for p, s, m in rows]


def team_file(run_dir: Path, run: str, group: str, path: str) -> bytes:
    """The first FILE_CAP bytes of one file on the group's drive."""
    if ".." in Path(path).parts or path.startswith("/"):
        return b""
    local = run_dir / f"team-{group}"
    if local.is_dir():
        return (local / path).read_bytes()[:FILE_CAP] if (local / path).is_file() else b""
    if not (cid := keeper(run, group)):
        return b""
    cmd = ["docker", "exec", cid, "head", "-c", str(FILE_CAP), f"/team/{path}"]
    return subprocess.run(cmd, capture_output=True).stdout


def serve(run_dir: Path, port: int = PORT) -> str:
    """Start the viewer in a daemon thread on localhost only; returns its URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path in LOGS:
                path = run_dir / LOGS[url.path]
                data = b""
                if path.exists():
                    with path.open("rb") as log:
                        log.seek(int(q.get("from", "0")))
                        data = log.read()
                self.reply(data[: data.rfind(b"\n") + 1], "application/x-ndjson")  # complete lines only
            elif url.path == "/team":
                self.reply(json.dumps(team_files(run_dir, run_name(), q["group"])).encode(), "application/json")
            elif url.path == "/file":
                body = team_file(run_dir, run_name(), q["group"], q["path"])
                self.reply(body, "text/plain; charset=utf-8")
            else:
                self.send_error(404)

        def reply(self, body: bytes, kind: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:  # quiet: the runner owns the terminal
            pass

    def run_name() -> str:
        with (run_dir / "events.jsonl").open() as f:
            return str(json.loads(f.readline())["run"])

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:  # another run's viewer has the port: take any free one
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


if __name__ == "__main__":
    print(f"Viewer: {serve(Path(sys.argv[1]))}  (Ctrl-C to stop)")
    threading.Event().wait()
