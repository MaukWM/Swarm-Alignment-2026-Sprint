# Swarm Alignment 2026 Sprint

## Setup

```bash
uv sync
uv run pre-commit install
```

## Run

Needs Docker (tested with Colima on macOS arm64; uses the active Docker context). Agent containers get no internet, one
internal network and one shared drive (`/team`) per group.

```bash
uv run python -m swarmsprint.containers up experiments/smoke.toml
uv run python -m swarmsprint.containers down experiments/smoke.toml
```
