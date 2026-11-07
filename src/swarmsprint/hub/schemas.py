"""Request and response schemas of the hub API."""

from typing import Annotated

from pydantic import BaseModel, Field

from swarmsprint.hub.function import HIGH, LOW

Coordinate = Annotated[float, Field(ge=LOW, le=HIGH)]  # also rejects NaN and infinity


class SubmitRequest(BaseModel):
    x: list[Coordinate] = Field(
        max_length=1000, description="the point to evaluate: one number in [-5, 5] per variable"
    )
    remark: str = Field(default="", max_length=250, description="shown next to your score on the leaderboard")


class Submission(BaseModel):
    id: int
    account: str
    x: list[float]
    remark: str
    score: float
    time: str


class SubmitResponse(BaseModel):
    id: int
    score: float
    rank: int
    credits: float


class LeaderboardRow(BaseModel):
    rank: int
    account: str
    score: float
    remark: str
    time: str


class MeResponse(BaseModel):
    account: str
    best: float | None
    credits: float
    submissions: list[Submission]
