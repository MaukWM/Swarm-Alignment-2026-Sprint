"""Run config: one TOML file per experiment, validated on load."""

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Name = Annotated[str, Field(pattern=r"^[a-z0-9]+$")]


class Group(BaseModel):
    """One swarm: its name and how many agent containers it gets."""

    model_config = ConfigDict(extra="forbid")
    name: Name
    agents: int = Field(gt=0)


class RunConfig(BaseModel):
    """One experiment run."""

    model_config = ConfigDict(extra="forbid")
    name: Name
    groups: list[Group]


def load(path: str | Path) -> RunConfig:
    """Read and validate a run config TOML."""
    with open(path, "rb") as f:
        return RunConfig.model_validate(tomllib.load(f))
