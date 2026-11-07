"""Request and response schemas of the hub API."""

from pydantic import BaseModel, Field


class SubmitRequest(BaseModel):
    program: str = Field(max_length=100_000, description="Python source defining priority(item, bins) -> np.ndarray")
    remark: str = Field(default="", max_length=250, description="shown next to your score on the leaderboard")


class Submission(BaseModel):
    id: int
    account: str
    program: str
    remark: str
    score: float | None = None
    error: str | None = None
    time: str


class SubmitResponse(BaseModel):
    id: int
    score: float | None = None
    error: str | None = None
    rank: int | None = None
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
