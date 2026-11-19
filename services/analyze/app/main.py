"""Beat Butler worker.

Slice 1: POST /v1/analyze      - librosa DSP, the numbers.
Slice 2: POST /v1/generate     - validated marketing copy from those numbers.

Both routes sit behind the same shared-secret guard: only our own server holds
the key, and the model key never leaves this process.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import billing, dsp, generate, ratelimit

# services/analyze/.env, so the documented copy step actually does something
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("butler.analyze")

# Sized from the duration limit rather than guessed: a 10-minute 48 kHz / 24-bit
# stereo WAV is about 173 MB, so a 25 MB cap rejected ordinary bounces.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CHUNK = 1024 * 1024
MAX_SECONDS = dsp.MAX_DURATION_SECONDS  # one source: the analyser's own limit
ALLOWED_SUFFIXES = {".mp3", ".wav", ".aiff", ".aif", ".flac", ".m4a"}

# Accounts, when this worker is pointed at the same Supabase project as the app.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


def _too_large() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=(
            f"File is larger than {MAX_UPLOAD_MB} MB. A 10-minute 24-bit WAV is about 173 MB - "
            "export as FLAC (lossless, roughly half the size) or MP3 and it will fit."
        ),
    )

API_KEY = os.environ.get("BUTLER_API_KEY", "dev-local-key")

# Per-address flood control, in this process. Counted before the shared-key guard
# so a junk flood costs no CPU; see ratelimit.py for the caveats.
ANALYZE_LIMITER = ratelimit.Limiter(
    "analyze",
    int(os.environ.get("RATE_LIMIT_ANALYZE", "30")),
    float(os.environ.get("RATE_LIMIT_WINDOW", "60")),
)
GENERATE_LIMITER = ratelimit.Limiter(
    "generate",
    int(os.environ.get("RATE_LIMIT_GENERATE", "20")),
    float(os.environ.get("RATE_LIMIT_WINDOW", "60")),
)
# 5173 is the dev server, 5175 the private dev server, and 5176 the owner build
# as served to the Desktop shortcut. Each is a different origin, so each has to
# be listed or the browser drops the request before the worker sees it.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "BUTLER_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,"
        "http://127.0.0.1:5175,http://127.0.0.1:5176",
    ).split(",")
    if o.strip()
]

def log_generation_status() -> None:
    """Say plainly at boot whether copy generation can actually run.

    A missing key is the difference between real model output and the local
    drafter, and it used to show up only as a 503 on the first request. Log it,
    so the reason is in the terminal next to the rest of the startup output.
    """
    status = generate.provider_report()
    if not status["ready"]:
        log.error(
            "copy generation NOT READY: provider=%s missing=%s - %s",
            status["provider"],
            ",".join(status["missing"]) or "none",
            status["detail"],
        )
    elif status["provider"] == "template":
        log.warning("copy generation: provider=template - %s", status["detail"])
    else:
        log.info("copy generation: provider=openai model=%s - %s", status["model"], status["detail"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    log_generation_status()
    if not _cloud_enabled():
        log.warning(
            "plan enforcement OFF: this worker has no SUPABASE_URL / SUPABASE_ANON_KEY, so it "
            "cannot tell who is asking and charges nobody. Set both to get production behaviour."
        )
    yield


app = FastAPI(title="Beat Butler worker", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Butler-Key", "Authorization"],
)


class Links(BaseModel):
    store: str = "https://beatstars.com/your-store"
    licence: str = "Lease options and pricing on request."
    contact: str = "DM for collabs and custom work."


@app.middleware("http")
async def hardening(request: Request, call_next):
    """Nothing from this worker should sit in a cache or be framed."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


class GenerateRequest(BaseModel):
    beat_id: Optional[str] = None
    beat_title: str = "Untitled beat"
    file_name: str = ""
    features: dict = Field(default_factory=dict)
    genre: str = "Trap"
    mood: str = "dark"
    mood_source: Literal["measured", "inferred", "user"] = "inferred"
    bpm_source: Literal["measured", "inferred", "user"] = "measured"
    key_source: Literal["measured", "inferred", "user"] = "measured"
    notes: str = ""
    butler_name: str = "Jeffrey"
    butler_voice: str = "English accent - confident, helpful, plainspoken, dry British wit, no Americanisms"
    style: Literal["direct", "hype", "minimal"] = "direct"
    links: Links = Field(default_factory=Links)


def require_key(x_butler_key: Optional[str]) -> None:
    if API_KEY and x_butler_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Butler-Key.")


def analyse_to_response(beat_id: str, path: Path) -> dict:
    started = time.perf_counter()
    try:
        features = dsp.analyze_path(path)
    except dsp.AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    log.info(
        "analyzed beat_id=%s bpm=%.1f key=%s engine=%s in %dms",
        beat_id,
        features["bpm"],
        features["key"],
        dsp.engine_info()["name"],
        elapsed_ms,
    )
    public = {k: v for k, v in features.items() if not k.startswith("_")}
    return {"beat_id": beat_id, "engine": dsp.engine_info(), **public}


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "engine": dsp.engine_info(),
        "generation": generate.provider_report(),
        "rate_limit": {
            "analyze": ANALYZE_LIMITER.describe(),
            "generate": GENERATE_LIMITER.describe(),
        },
    }


@app.get("/v1/limits")
def upload_limits() -> dict:
    """Public on purpose: the upload rules live here, not in the browser bundle."""
    return {
        "max_bytes": MAX_BYTES,
        "max_mb": MAX_UPLOAD_MB,
        "max_seconds": MAX_SECONDS,
        "extensions": sorted(ALLOWED_SUFFIXES),
    }


def _cloud_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def _require_account(authorization: Optional[str]) -> Optional[str]:
    """The caller's token, when this deployment has accounts at all.

    `X-Butler-Key` ships inside the browser bundle, so it can never be identity:
    anyone can read it. Once Supabase is configured, a request without a valid
    session is refused - which is what makes the plan something a client cannot
    skip by calling the worker directly.
    """
    if not _cloud_enabled():
        return None
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Sign in to analyse a beat: this deployment enforces the plan on the server.",
        )
    return token


def _quota_rpc(name: str, token: str) -> dict:
    """Call a quota function as the caller; the identity comes from their JWT."""
    try:
        res = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/{name}",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Could not check your plan: {exc}") from exc

    if res.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="That session is not valid. Sign in again.")
    if res.status_code >= 300:
        raise HTTPException(status_code=503, detail="Could not check your plan. Try again.") from None

    rows = res.json()
    row = rows[0] if isinstance(rows, list) and rows else rows
    if not isinstance(row, dict):
        raise HTTPException(status_code=503, detail="Could not read your plan.")
    return row


def _refuse(row: dict) -> None:
    raise HTTPException(
        status_code=402,
        detail={
            "message": (
                f"Your plan keeps {row.get('quota')} beats, you have used {row.get('used')}, "
                "and no coins are left."
            ),
            "used": row.get("used", 0),
            "quota": row.get("quota", 0),
            "credits": row.get("credits", 0),
        },
    )


def _check_plan(token: str) -> None:
    """Read-only: is there an upload left? Free to ask, so a failed run costs nothing."""
    row = _quota_rpc("peek_beat_quota", token)
    if not row.get("allowed"):
        _refuse(row)


def _charge_plan(token: str) -> None:
    """Spend one beat - allowance first, then a coin - once the analysis worked."""
    row = _quota_rpc("consume_beat_quota", token)
    if not row.get("allowed"):
        # Another upload took the last slot between the peek and here.
        _refuse(row)


@app.post("/v1/analyze")
async def analyze_upload(
    request: Request,
    file: UploadFile = File(...),
    beat_id: str = Form(default=""),
    authorization: Optional[str] = Header(default=None),
    x_butler_key: Optional[str] = Header(default=None),
) -> dict:
    """Dev / direct path: multipart upload of the audio bytes."""
    ratelimit.enforce(ANALYZE_LIMITER, request)
    require_key(x_butler_key)

    beat = beat_id.strip() or str(uuid.uuid4())
    suffix = Path(file.filename or "beat.mp3").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or 'none'}'. Use {', '.join(sorted(ALLOWED_SUFFIXES))}.",
        )

    # Checked here, after the cheap checks, so an oversized or wrong-type upload
    # costs nothing and a spent plan never reaches the analyser. The charge itself
    # lands after a successful analysis, so a run that fails is free.
    token = _require_account(authorization)
    if token:
        _check_plan(token)

    handle, raw_path = tempfile.mkstemp(suffix=suffix, prefix="butler-")
    os.close(handle)
    path = Path(raw_path)
    written = 0
    try:
        # Streamed in 1 MB chunks: a 173 MB bounce must not sit in memory per request.
        with path.open("wb") as out:
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                if written > MAX_BYTES:
                    raise _too_large()
                out.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="The file was empty.")
        result = analyse_to_response(beat, path)
        if token:
            _charge_plan(token)
        return result
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/generate")
def generate_route(
    request: Request,
    body: GenerateRequest,
    authorization: Optional[str] = Header(default=None),
    x_butler_key: Optional[str] = Header(default=None),
) -> dict:
    """Features in, validated copy out."""
    ratelimit.enforce(GENERATE_LIMITER, request)
    require_key(x_butler_key)
    # Model calls cost money too, so a configured deployment wants a session here.
    _require_account(authorization)

    started = time.perf_counter()
    try:
        result = generate.generate_copy(body.model_dump())
    except generate.GenerationNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except generate.GenerationFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    log.info(
        "generated beat_id=%s provider=%s model=%s prompt=%s repaired=%s in %dms",
        result.get("beat_id"),
        result.get("provider"),
        result.get("model"),
        result.get("prompt_version"),
        result.get("validation", {}).get("repaired"),
        round((time.perf_counter() - started) * 1000),
    )
    return result


# --------------------------------------------------------------------------- #
# billing (Slice 4)
# --------------------------------------------------------------------------- #


class CheckoutRequest(BaseModel):
    kind: Literal["plan", "credits"] = "plan"
    plan: Literal["starter", "pro"] = "starter"
    pack: Literal["5", "20"] = "5"
    success_url: str = "http://127.0.0.1:5173/?checkout=done"
    cancel_url: str = "http://127.0.0.1:5173/?checkout=cancelled"


@app.post("/v1/billing/checkout")
def billing_checkout(
    body: CheckoutRequest,
    authorization: Optional[str] = Header(default=None),
    x_butler_key: Optional[str] = Header(default=None),
) -> dict:
    """Create a hosted Checkout Session for the caller's own account."""
    require_key(x_butler_key)

    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Sign in first: checkout needs your Supabase access token.",
        )

    try:
        user = billing.resolve_user(token)
        session = billing.create_checkout_session(
            kind=body.kind,
            plan=body.plan,
            pack=body.pack,
            user=user,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
        )
    except billing.BillingNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except billing.BillingFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    log.info(
        "checkout started user=%s kind=%s plan=%s pack=%s session=%s",
        user.get("id"),
        body.kind,
        body.plan,
        body.pack,
        session.get("id"),
    )
    return session


@app.post("/v1/billing/webhook")
async def billing_webhook(request: Request) -> dict:
    """Stripe's side of the bargain: verify the signature, then apply the plan."""
    payload = await request.body()
    try:
        event = billing.verify_webhook(payload, request.headers.get("stripe-signature", ""))
        for action in billing.handle_event(event):
            billing.apply_action(action)
            log.info("billing applied %s event=%s", action, event.get("type"))
    except billing.BillingNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except billing.BillingFailed as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"received": True}
