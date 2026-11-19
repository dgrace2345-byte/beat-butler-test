"""Copy generation for POST /v1/generate.

Two providers behind one function:

  openai     - any OpenAI-compatible /chat/completions endpoint. Used when
               LLM_API_KEY is set (or LLM_PROVIDER=openai).
  template   - a deterministic local draft, no network, no key. It exercises the
               exact same validation and UI, and is labelled "draft" everywhere
               so it is never mistaken for model output.

The provider seam is deliberate: swapping in a different vendor is an env change,
not a rewrite.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from . import prompts, validate

PROMPT_VERSION = prompts.PROMPT_VERSION
TEMPLATE_MODEL = "template-draft"
DEFAULT_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"
REQUEST_TIMEOUT = 60


class GenerationNotConfigured(Exception):
    """No model key available for the requested provider."""


class GenerationFailed(Exception):
    """The model returned something unusable, twice."""


def provider_name() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit in {"openai", "template"}:
        return explicit
    return "openai" if os.environ.get("LLM_API_KEY", "").strip() else "template"


def provider_report() -> dict:
    """Provider state for the startup log and /health, in one place.

    `ready` answers the only question a caller actually has: will /v1/generate
    return copy right now, or a 503? A missing key is otherwise invisible until
    someone hits the route, which reads as a bug rather than as configuration.
    """
    provider = provider_name()
    key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base = os.environ.get("LLM_BASE_URL", DEFAULT_BASE).rstrip("/")

    if provider == "openai":
        ready = bool(key)
        missing = [] if ready else ["LLM_API_KEY"]
        detail = (
            f"Real model output via {base} (model {model})."
            if ready
            else "LLM_PROVIDER=openai is set but LLM_API_KEY is empty, so /v1/generate "
            "answers 503. Add LLM_API_KEY to services/analyze/.env and restart."
        )
    else:
        ready = True
        missing = []
        detail = (
            "Local drafter forced by LLM_PROVIDER=template; LLM_API_KEY is present but unused."
            if key
            else "Local drafter (no LLM_API_KEY). Copy is labelled 'draft'. Add LLM_API_KEY to "
            "services/analyze/.env and restart for real model output."
        )

    return {
        "provider": provider,
        "prompt": PROMPT_VERSION,
        "model": model if provider == "openai" else TEMPLATE_MODEL,
        "ready": ready,
        "missing": missing,
        "detail": detail,
    }


def generate_copy(payload: dict) -> dict:
    """Build validated copy for one beat. Raises the errors above."""
    inp = _flatten(payload)
    allowed = " ".join(
        [inp["notes"], inp["store_link"], inp["licence_text"], inp["contact"], inp["beat_title"]]
    )

    if provider_name() == "template":
        output = validate.normalise(_template_draft(inp))
        violations = validate.validate_output(output, allowed)
        if violations:
            # A bug in the local drafter, not a model problem. Surface it loudly.
            raise GenerationFailed(f"Local draft failed validation: {'; '.join(violations)}")
        return {
            "beat_id": inp["beat_id"],
            "provider": "template",
            "model": TEMPLATE_MODEL,
            "prompt_version": PROMPT_VERSION,
            "style": inp["style"],
            "output": output,
            "validation": {"repaired": False, "violations": []},
            "usage": {"tokens_in": 0, "tokens_out": 0},
            "limits": validate.summary(output),
        }

    user_message = prompts.build_user_message(
        beat_title=inp["beat_title"],
        file_name=inp["file_name"],
        notes=inp["notes"],
        genre=inp["genre"],
        mood=inp["mood"],
        mood_source=inp["mood_source"],
        bpm=inp["bpm"],
        bpm_source=inp["bpm_source"],
        bpm_confidence=inp["bpm_confidence"],
        key=inp["key"],
        key_source=inp["key_source"],
        key_confidence=inp["key_confidence"],
        duration_seconds=inp["duration_seconds"],
        energy=inp["energy"],
        onset_density=inp["onset_density"],
        spectral_centroid_hz=inp["spectral_centroid_hz"],
        butler_name=inp["butler_name"],
        butler_voice=inp["butler_voice"],
        store_link=inp["store_link"],
        licence_text=inp["licence_text"],
        contact=inp["contact"],
        style=inp["style"],
    )

    raw, usage = _call_model(user_message)
    output = validate.normalise(raw)
    violations = validate.validate_output(output, allowed)
    repaired = False

    if violations:
        repair_msg = prompts.build_repair_message(violations)
        raw2, usage2 = _call_model(user_message, repair=repair_msg)
        output2 = validate.normalise(raw2)
        violations2 = validate.validate_output(output2, allowed)
        repaired = True
        if violations2:
            raise GenerationFailed(
                "Generated copy failed validation twice: " + "; ".join(violations2[:4])
            )
        output = output2
        violations = violations2
        usage = {
            "tokens_in": usage.get("tokens_in", 0) + usage2.get("tokens_in", 0),
            "tokens_out": usage.get("tokens_out", 0) + usage2.get("tokens_out", 0),
        }

    return {
        "beat_id": inp["beat_id"],
        "provider": "openai",
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "prompt_version": PROMPT_VERSION,
        "style": inp["style"],
        "output": output,
        "validation": {"repaired": repaired, "violations": violations},
        "usage": usage,
        "limits": validate.summary(output),
    }


# --------------------------------------------------------------------------- #
# model call
# --------------------------------------------------------------------------- #

def _call_model(user_message: str, repair: str | None = None) -> tuple[dict, dict]:
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise GenerationNotConfigured(
            "Generation is not configured. Add LLM_API_KEY to services/analyze/.env "
            "(and optionally LLM_MODEL / LLM_BASE_URL), then restart the worker."
        )

    base = os.environ.get("LLM_BASE_URL", DEFAULT_BASE).rstrip("/")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    if repair:
        messages.append({"role": "user", "content": repair})

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise GenerationFailed(f"Could not reach the model endpoint: {exc}") from exc

    if resp.status_code != 200:
        snippet = resp.text[:300].replace("\n", " ")
        raise GenerationFailed(f"Model endpoint returned HTTP {resp.status_code}: {snippet}")

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise GenerationFailed("Model endpoint returned an unexpected payload.") from exc

    parsed = _parse_json(content)
    raw_usage = data.get("usage") or {}
    usage = {
        "tokens_in": raw_usage.get("prompt_tokens"),
        "tokens_out": raw_usage.get("completion_tokens"),
    }
    return parsed, usage


def _parse_json(content: str) -> dict:
    text = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.+?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationFailed(f"Model did not return valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise GenerationFailed("Model returned JSON that is not an object.")
    return parsed


# --------------------------------------------------------------------------- #
# local drafter
# --------------------------------------------------------------------------- #

def _flatten(payload: dict) -> dict:
    features = payload.get("features") or {}
    links = payload.get("links") or {}
    return {
        "beat_id": payload.get("beat_id") or "local",
        "beat_title": (payload.get("beat_title") or "Untitled beat").strip(),
        "file_name": payload.get("file_name") or "",
        "notes": (payload.get("notes") or "").strip(),
        "genre": (payload.get("genre") or "Trap").strip(),
        "mood": (payload.get("mood") or "dark").strip(),
        "mood_source": payload.get("mood_source") or "inferred",
        "bpm_source": payload.get("bpm_source") or "measured",
        "key_source": payload.get("key_source") or "measured",
        "style": payload.get("style") or "direct",
        "butler_name": (payload.get("butler_name") or "Jeffrey").strip(),
        "butler_voice": payload.get("butler_voice")
        or "English accent - confident, helpful, plainspoken, dry British wit, no Americanisms",
        "store_link": (links.get("store") or "https://beatstars.com/your-store").strip(),
        "licence_text": (links.get("licence") or "Lease options and pricing on request.").strip(),
        "contact": (links.get("contact") or "DM for collabs and custom work.").strip(),
        "bpm": features.get("bpm", 0),
        "bpm_confidence": features.get("bpm_confidence", 0),
        "key": features.get("key") or "",
        "key_confidence": features.get("key_confidence", 0),
        "duration_seconds": features.get("duration_seconds", 0),
        "energy": features.get("energy", 0),
        "onset_density": features.get("onset_density", 0),
        "spectral_centroid_hz": features.get("spectral_centroid_hz", 0),
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _fit(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "."


def _template_draft(inp: dict) -> dict:
    """Deterministic draft. Not AI - every phrase comes from the measured facts."""
    genre = inp["genre"]
    genre_l = genre.lower()
    mood = inp["mood"]
    bpm = inp["bpm"]
    key = inp["key"] or "unknown key"
    bpm_text = f"{bpm:g} BPM" if bpm else ""
    title = _fit(f'{genre} Type Beat - {bpm_text} {key}'.replace("  ", " ").strip(), validate.MAX_TITLE)

    opener = {
        "direct": f"{genre_l.title()} type beat, {mood} and built to sit under a vocal.",
        "hype": f"This is the {genre_l} one. {bpm_text} of pure pressure.",
        "minimal": f"{bpm_text}. {key}. {mood.title()}.",
    }.get(inp["style"], f"{genre_l.title()} type beat, {mood} and built to sit under a vocal.")

    detail = (
        f"Measured at {bpm_text} in {key}, with {mood} energy and a full low end. "
        f"Arrangement is open enough to write to and mixed to translate on phone speakers."
    )
    if inp["notes"]:
        detail = f'{inp["notes"]} {detail}'

    description = "\n\n".join(
        [
            opener,
            detail,
            f"Purchase: {inp['store_link']}",
            f"Licence: {inp['licence_text']}",
            f"Contact: {inp['contact']}",
        ]
    )
    description = _fit(description, validate.MAX_DESCRIPTION)

    tags = validate._dedupe(
        [
            f"{genre_l} type beat",
            f"{genre_l} beat",
            "type beat",
            f"{genre_l} instrumental",
            bpm_text,
            f"{key} type beat",
            f"{mood} type beat",
            f"{mood} {genre_l}",
            "instrumental",
            "beat 2026",
            "buy beats online",
            "rap beat",
            "beatstars",
            "producer",
            "beats for sale",
            f"{bpm_text} {genre_l}".strip(),
        ]
    )
    # keep the combined length inside YouTube's 500-character cap
    trimmed: list[str] = []
    for tag in tags:
        if len(", ".join(trimmed + [tag])) > validate.MAX_TAGS_JOINED:
            break
        trimmed.append(tag[: validate.MAX_TAG_ITEM])

    caption = _fit(
        f"{mood.title()} {genre_l} type beat at {bpm_text}. {inp['notes'] or 'Hit the link for the lease.'}".strip(),
        validate.MAX_CAPTION,
    )

    hashtag_source = ["#typebeat", f"#{_slug(genre_l)}", f"#{_slug(mood)}", "#beatstars", "#instrumental"]
    hashtag_source += [f"#{_slug(bpm_text)}", "#producer", "#beats", "#rapper", "#newmusic"]
    hashtags = [h for h in validate._dedupe(hashtag_source) if validate.HASHTAG_RE.match(h)]

    butler_line = _fit(
        f'{inp["butler_name"]}: {mood} one, {bpm_text}. Filed and ready.',
        validate.MAX_BUTLER_LINE,
    )

    return {
        "youtube": {"title": title, "description": description, "tags": trimmed},
        "type_beat_caption": caption,
        "hashtags": hashtags[: validate.MAX_HASHTAGS],
        "butler_line": butler_line,
    }
