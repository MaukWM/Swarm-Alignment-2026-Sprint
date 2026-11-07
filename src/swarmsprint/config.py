"""Run config: one TOML file per experiment, validated on load."""

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Name = Annotated[str, Field(pattern=r"^[a-z0-9]+$")]


class Group(BaseModel):
    """One swarm: its name, how many agent containers it gets, and the system prompt its agents start with."""

    model_config = ConfigDict(extra="forbid")
    name: Name
    agents: int = Field(gt=0)
    prompt: str


class RunConfig(BaseModel):
    """One experiment run."""

    model_config = ConfigDict(extra="forbid")
    name: Name
    groups: list[Group]
    model: str
    turns: int = Field(gt=0)
    cmd_timeout: int = Field(default=60, gt=0, description="seconds per shell command")
    max_output: int = Field(default=4000, gt=0, description="characters of command output the model sees")
    leaderboard_host: str = Field(default="capset-leaderboard.org", description="hostname agents use, port 80")
    leaderboard_port: int = Field(default=18080, gt=0, lt=65536, description="port of the leaderboard on the host")


def load(path: str | Path) -> RunConfig:
    """Read and validate a run config TOML."""
    with open(path, "rb") as f:
        return RunConfig.model_validate(tomllib.load(f))
