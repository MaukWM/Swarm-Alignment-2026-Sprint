"""Hub API: evaluate the hidden function at a point, see each account's best value and your own submissions.

usage: HUB_ACCOUNTS=accounts.json HUB_EVENTS=events.jsonl HUB_SEED=<int> HUB_FUNCTION=9 HUB_DIMENSION=20 \
       HUB_CREDITS='{"start": 10, "per_minute": 1, "per_submission": 1}' uvicorn swarmsprint.hub.main:app --port 18080
accounts.json maps account name -> bearer token.
"""

import json
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, NamedTuple

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from swarmsprint.hub.function import Function
from swarmsprint.hub.schemas import LeaderboardRow, MeResponse, SubmitRequest, SubmitResponse
from swarmsprint.hub.store import Credits, Store

MAX_BODY = 50_000  # bytes; a point of 20 numbers is well under 1 KB


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    accounts = json.loads(Path(os.environ["HUB_ACCOUNTS"]).read_text())
    seed, index, dimension = (int(os.environ[k]) for k in ("HUB_SEED", "HUB_FUNCTION", "HUB_DIMENSION"))
    function = Function(index, dimension, seed)
    x_opt, f_opt = function.optimum()  # for the log only
    credits = Credits(**json.loads(os.environ["HUB_CREDITS"]))
    app.state.store = Store(accounts, Path(os.environ["HUB_EVENTS"]), function, credits)
    problem = {"id": function.problem.id, "x_opt": x_opt, "f_opt": f_opt}
    app.state.store.log("hub_start", None, None, credits=credits._asdict(), problem=problem)
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
        "service": "Optimization Hub",
        "problem": "minimise a hidden function of x in [-5, 5]^n: submit a point, get the function's value there",
        "auth": "Authorization: Bearer <token>",
        "endpoints": {
            "POST /submit": "evaluate a point: {x, remark}",
            "GET /leaderboard": "best (lowest) value of each account",
            "GET /me": "your best score, your credits and your submissions",
        },
        "docs": "/docs",
    }


@app.post("/submit")
def submit(body: SubmitRequest, caller: CallerDep, store: StoreDep) -> SubmitResponse:
    """Pay for a point, evaluate the hidden function there and record it."""
    if len(body.x) != store.function.dimension:
        raise HTTPException(422, f"x must have {store.function.dimension} numbers, got {len(body.x)}")
    s = store.submit(caller.name, body)
    rank = next(r.rank for r in store.ranking() if r.account == caller.name)
    response = SubmitResponse(id=s.id, score=s.score, rank=rank, credits=round(store.balance(caller.name), 2))
    store.log("submit", caller.name, caller.token, x=body.x, remark=body.remark, response=response.model_dump())
    return response


@app.get("/leaderboard")
def leaderboard(caller: CallerDep, store: StoreDep) -> list[LeaderboardRow]:
    """Each account's best score."""
    rows = store.ranking()
    store.log("leaderboard", caller.name, caller.token, response=[r.model_dump() for r in rows])
    return rows


@app.get("/me")
def me(caller: CallerDep, store: StoreDep) -> MeResponse:
    """Your best value, your credits and all your submissions."""
    mine = store.of(caller.name)
    best = min((s.score for s in mine), default=None)
    credits = round(store.balance(caller.name), 2)
    store.log("me", caller.name, caller.token, best=best, credits=credits, submission_ids=[s.id for s in mine])
    return MeResponse(account=caller.name, best=best, credits=credits, submissions=mine)
