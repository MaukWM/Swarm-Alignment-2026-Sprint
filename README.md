# Swarm Alignment 2026 Sprint

## Setup

```bash
uv sync
uv run pre-commit install
```

## Run

Needs Docker (tested with Colima on macOS arm64; uses the active Docker context) and an OpenAI-compatible chat
completions endpoint with tool calling. Copy `.env.example` to `.env` and set `LLM_BASE_URL` to the full
`.../v1/chat/completions` URL.

```bash
uv run --frozen --env-file .env python -m swarmsprint.run experiments/smoke.toml
```

Each run writes `runs/<name>-<timestamp>/`: `events.jsonl` (every model call and command, with the config, source
commit and diff, image id and worker mapping in the first line) and one `team-<group>/` copy of each shared drive.
Agent containers get no internet, one internal network and one shared drive (`/team`) per group.
