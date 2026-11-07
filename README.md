# Swarm Alignment 2026 Sprint

## Setup

```bash
uv sync
uv run pre-commit install
```

## Run

Needs Docker Engine 28 or newer (tested with Colima on macOS arm64; uses the active Docker context) and an
OpenAI-compatible chat completions endpoint with tool calling. Copy `.env.example` to `.env` and set `LLM_BASE_URL` to
the full `.../v1/chat/completions` URL.

```bash
uv run --frozen --env-file .env python -m swarmsprint.runner.run experiments/smoke.toml
```

Each run writes `runs/<name>-<timestamp>/`: `events.jsonl` (every model call and command, with the config, source commit
and diff, image id and worker mapping in the first line), `leaderboard.jsonl` (every leaderboard request with the
caller's container IP), `accounts.json` (each worker's token) and one `team-<group>.tar` of each shared drive (its
sha256 is in the log). The leaderboard runs on the host (`leaderboard_port`, default 18080) and agents reach it as
`http://capset-leaderboard.org`. Agent containers get no internet, a read-only root filesystem with small scratch space,
one internal network and one 256 MB shared drive (`/team`) per group. A run refuses to start while Docker resources of
an earlier run with the same name exist; inspect them, then remove them with `uv run python -m
swarmsprint.infra.containers down <config>`.
