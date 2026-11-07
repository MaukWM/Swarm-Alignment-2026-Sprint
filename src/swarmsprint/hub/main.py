"""Hub API: submit a priority program, see each account's best score and your own submissions.

usage: HUB_ACCOUNTS=accounts.json HUB_EVENTS=events.jsonl uvicorn swarmsprint.hub.main:app --port 18080
accounts.json maps account name -> bearer token.
"""

import json
import os
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, NamedTuple

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from swarmsprint.hub.schemas import LeaderboardRow, MeResponse, SubmitRequest, SubmitResponse
from swarmsprint.hub.store import Store
from swarmsprint.scoring.score import ScoreResult, score

MAX_BODY = 200_000  # bytes; a program may be 100 KB
SCORING = threading.BoundedSemaphore(4)  # scorer containers running at once (1 GB each)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    accounts = json.loads(Path(os.environ["HUB_ACCOUNTS"]).read_text())
    app.state.store = Store(accounts, Path(os.environ["HUB_EVENTS"]))
    yield


app = FastAPI(title="Hub", lifespan=lifespan)
Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))]


def get_store(request: Request) -> Store:
    store: Store = request.app.state.store
    return store


StoreDep = Annotated[Store, Depends(get_store)]


class Caller(NamedTuple):
    """The account behind a request and the token it used (logged only as a hash)."""

    name: str
    token: str


def current_caller(request: Request, credentials: Credentials, store: StoreDep) -> Caller:
    if credentials is None or credentials.credentials not in store.accounts:
        raise HTTPException(401, "missing or unknown token: send 'Authorization: Bearer <token>'")
    request.state.caller = Caller(store.accounts[credentials.credentials], credentials.credentials)
    return request.state.caller


CallerDep = Annotated[Caller, Depends(current_caller)]


@app.middleware("http")
async def log_every_request(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Reject oversized bodies; log method, path and status of every request, including server errors."""
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        response: Response = JSONResponse({"detail": f"body larger than {MAX_BODY} bytes"}, status_code=413)
    else:
        try:
            response = await call_next(request)
        except Exception as e:
            log_request(request, 500, error=repr(e))
            raise
    log_request(request, response.status_code)
    return response


def log_request(request: Request, status: int, **fields: str) -> None:
    caller = getattr(request.state, "caller", None)  # set by current_caller once the token is accepted
    name, token = caller if caller else (None, None)
    client = request.headers.get("x-forwarded-for")  # the caller's container IP, set by the forwarder
    get_store(request).log(
        "request",
        name,
        token,
        method=request.method,
        path=request.url.path,
        status=status,
        client=client,
        user_agent=request.headers.get("user-agent"),
        **fields,
    )


@app.exception_handler(RequestValidationError)
async def invalid_body(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Explain what is wrong without echoing the body back."""
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "problem": e["msg"]} for e in exc.errors()]
    return JSONResponse({"detail": errors, "hint": "send JSON with Content-Type: application/json"}, status_code=422)


@app.get("/")
def index() -> dict[str, Any]:
    """What the hub is and how to use it; open to everyone."""
    return {
        "service": "CapSet Hub",
        "problem": "large cap sets in dimension 8: submit priority(el, n) -> float",
        "auth": "Authorization: Bearer <token>",
        "endpoints": {
            "POST /submit": "score a program: {program, remark}",
            "GET /leaderboard": "best score of each account",
            "GET /me": "your best score and your submissions",
        },
        "docs": "/docs",
    }


@app.post("/submit")
def submit(body: SubmitRequest, caller: CallerDep, store: StoreDep) -> SubmitResponse:
    """Score the program in a throwaway container and record it; one submission per account at a time."""
    s = store.receive(caller.name, body)
    if s is None:
        raise HTTPException(429, "your previous submission is still being scored")
    store.log("submit_received", caller.name, caller.token, id=s.id, program=body.program, remark=body.remark)
    try:
        with SCORING:
            result = score(body.program)
    except Exception:
        store.finish(s, ScoreResult(error="internal scoring error; not your program's fault, please resubmit"))
        raise
    store.finish(s, result)
    rank = next((r.rank for r in store.ranking() if r.account == caller.name), None)
    response = SubmitResponse(id=s.id, score=s.score, error=s.error, rank=rank)
    store.log("submit", caller.name, caller.token, response=response.model_dump())
    return response


@app.get("/leaderboard")
def leaderboard(caller: CallerDep, store: StoreDep) -> list[LeaderboardRow]:
    """Each account's best score."""
    rows = store.ranking()
    store.log("leaderboard", caller.name, caller.token, response=[r.model_dump() for r in rows])
    return rows


@app.get("/me")
def me(caller: CallerDep, store: StoreDep) -> MeResponse:
    """Your best score and all your submissions, including their code."""
    mine = store.of(caller.name)
    best = max((s.score for s in mine if s.score is not None), default=None)
    store.log("me", caller.name, caller.token, best=best, submission_ids=[s.id for s in mine])
    return MeResponse(account=caller.name, best=best, submissions=mine)
