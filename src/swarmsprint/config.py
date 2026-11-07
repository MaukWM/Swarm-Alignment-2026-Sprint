"""Run config: one TOML file per experiment, validated on load."""

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(pattern=r"^[a-z0-9]+$")]


class Group(BaseModel):
    """One swarm: its internal name, how many agent containers it gets, and the lab name its agents work for."""

    model_config = ConfigDict(extra="forbid")
    name: Name
    agents: int = Field(gt=0)
    lab: str = Field(min_length=1, description="the lab name the agents see")


class RunConfig(BaseModel):
    """One experiment run."""

    model_config = ConfigDict(extra="forbid")
    name: Name
    groups: list[Group]
    model: str
    turns: int | None = Field(default=None, gt=0, description="cap on each agent's turns")
    duration_minutes: float | None = Field(default=None, gt=0, description="wall-clock length of the run")
    cmd_timeout: int = Field(default=60, gt=0, description="seconds per shell command")
    max_output: int = Field(default=4000, gt=0, description="characters of command output the model sees")
    hub_host: str = Field(default="capset-hub.org", description="hostname agents use, port 80")
    hub_port: int = Field(default=18080, gt=0, lt=65536, description="port of the hub on the host")

    @model_validator(mode="after")
    def has_an_end(self) -> "RunConfig":
        if self.turns is None and self.duration_minutes is None:
            raise ValueError("set turns, duration_minutes or both, or the run never ends")
        return self


def load(path: str | Path) -> RunConfig:
    """Read and validate a run config TOML."""
    with open(path, "rb") as f:
        return RunConfig.model_validate(tomllib.load(f))
