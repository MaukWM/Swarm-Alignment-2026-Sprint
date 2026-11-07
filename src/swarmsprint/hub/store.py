"""In-memory hub state (accounts, submissions) plus the append-only event log, the run's ground truth."""

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import HTTPException

from swarmsprint.hub.schemas import LeaderboardRow, Submission, SubmitRequest
from swarmsprint.scoring.score import Instance, ScoreResult

INTERNAL_ERROR = "internal scoring error; not your program's fault, please resubmit"  # refunded


class Credits(NamedTuple):
    start: float
    per_minute: float
    per_submission: float


class Store:
    """Accounts, submissions and the event log; every method is safe to call from several threads."""

    def __init__(self, accounts: dict[str, str], events: Path, instances: list[Instance], credits: Credits) -> None:
        self.instances = instances  # hidden; only scores computed on them leave the hub
        self.accounts = {token: name for name, token in accounts.items()}
        self.submissions: list[Submission] = []
        self.scoring: set[str] = set()  # accounts with a submission being scored right now
        self.lock = threading.Lock()
        self.events = events.open("a")
        self.seq = 0
        self.credits = credits
        self.started = time.monotonic()

    def log(self, ev: str, account: str | None, token: str | None, **fields: Any) -> None:
        """Append one event, flushed at once; tokens appear only as a short hash."""
        token_id = hashlib.sha256(token.encode()).hexdigest()[:12] if token else None
        with self.lock:
            self.seq += 1
            line = {"seq": self.seq, "t": time.time(), "ev": ev, "account": account, "token_id": token_id, **fields}
            self.events.write(json.dumps(line) + "\n")
            self.events.flush()

    def balance(self, account: str) -> float:
        """Credits received since the hub started, minus what the account's submissions cost."""
        c = self.credits
        paid = sum(s.account == account and s.error != INTERNAL_ERROR for s in self.submissions)
        return c.start + c.per_minute * (time.monotonic() - self.started) / 60 - c.per_submission * paid

    def receive(self, account: str, request: SubmitRequest) -> Submission:
        """Record and pay for a submission on arrival (id and time fix its tie order)."""
        with self.lock:
            if account in self.scoring:
                raise HTTPException(429, "your previous submission is still being scored")
            if (have := self.balance(account)) < self.credits.per_submission:
                cost, rate = self.credits.per_submission, self.credits.per_minute
                why = f"you have {have:.2f}, a submission costs {cost:g}, you receive {rate:g} per minute"
                raise HTTPException(402, f"not enough credits: {why}")
            self.scoring.add(account)
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            s = Submission(id=len(self.submissions) + 1, account=account, **request.model_dump(), time=now)
            self.submissions.append(s)
        return s

    def finish(self, s: Submission, result: ScoreResult) -> None:
        """Attach the score or error and let the account submit again."""
        with self.lock:
            s.score, s.error = result.score, result.error
            self.scoring.discard(s.account)

    def ranking(self) -> list[LeaderboardRow]:
        """Each account's best (fewest bins) scored submission; on equal scores the earlier one ranks higher."""
        with self.lock:
            best: dict[str, Submission] = {}
            for s in self.submissions:
                current = best.get(s.account)
                if s.score is not None and (current is None or s.score < (current.score or float("inf"))):
                    best[s.account] = s
        rows = sorted(best.values(), key=lambda s: (s.score or float("inf"), s.id))
        return [
            LeaderboardRow(rank=i + 1, **s.model_dump(include={"account", "score", "remark", "time"}))
            for i, s in enumerate(rows)
        ]

    def of(self, account: str) -> list[Submission]:
        """All submissions of one account, oldest first."""
        with self.lock:
            return [s for s in self.submissions if s.account == account]
