"""Prompt definitions for Beat Butler.

yt-v1 is the prompt from BEAT-BUTLER-SPEC.md section 6. Keep it here, versioned,
so a change to the copy contract is a reviewable diff and not a string buried in
a route handler.
"""

from __future__ import annotations

PROMPT_VERSION = "yt-v1"

SYSTEM_PROMPT = """You write marketing copy for music producers who sell beats.
You are given measured audio features for one beat plus the producer's own notes.
Write copy a producer can paste straight into YouTube, BeatStars and socials.

You are the producer's studio assistant and producer partner: confident, helpful and intelligent.
You are never menacing, never the villain, and never aggressive for effect.

Rules:
- Stay a capable studio partner. Confidence is not swagger: no threats, no horror or "evil" framing,
  and no insults aimed at other producers or artists.
- Use only the facts provided. Never invent collaborators, artist placements, release dates,
  chart positions, sales numbers, or a store that was not named.
- Never call the beat "sold", "exclusive" or "free" unless the input says so.
- Facts marked source=user or notes outrank your own inferences. If they conflict, follow the user.
- Never state an inferred value as measured fact. If energy is inferred, describe the feel instead
  of quoting a number.
- Match tone to the beat's energy, not generic hype. No emoji in the title or description.
- Output one JSON object. No markdown, no code fences, no commentary."""

# The schema from the spec, kept next to the prompt so they cannot drift.
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["youtube", "type_beat_caption", "hashtags", "butler_line"],
    "properties": {
        "youtube": {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "description", "tags"],
            "properties": {
                "title": {"type": "string", "minLength": 10, "maxLength": 100},
                "description": {"type": "string", "minLength": 80, "maxLength": 4800},
                "tags": {
                    "type": "array",
                    "minItems": 8,
                    "maxItems": 24,
                    "items": {"type": "string", "maxLength": 40},
                },
            },
        },
        "type_beat_caption": {"type": "string", "maxLength": 200},
        "hashtags": {
            "type": "array",
            "minItems": 5,
            "maxItems": 15,
            "items": {"type": "string", "pattern": "^#[a-z0-9]+$", "maxLength": 30},
        },
        "butler_line": {"type": "string", "maxLength": 90},
    },
}


def build_user_message(
    *,
    beat_title: str,
    file_name: str,
    notes: str,
    genre: str,
    mood: str,
    mood_source: str,
    bpm: float,
    bpm_source: str,
    bpm_confidence: float,
    key: str,
    key_source: str,
    key_confidence: float,
    duration_seconds: float,
    energy: float,
    onset_density: float,
    spectral_centroid_hz: float,
    butler_name: str,
    butler_voice: str,
    store_link: str,
    licence_text: str,
    contact: str,
    style: str,
) -> str:
    """Renders the user message. Values are inserted verbatim; no facts are invented here."""
    return f"""BEAT
title: {beat_title}
file: {file_name}
producer_notes: {notes or "none"}
genre (source=user): {genre}
mood (source={mood_source}): {mood}

AUDIO FEATURES (source=measured came from the DSP; source=user is the producer's own call)
bpm: {bpm} (source={bpm_source}, confidence {bpm_confidence})
key: {key} (source={key_source}, confidence {key_confidence})
duration_seconds: {duration_seconds}
energy_0_1: {energy}
onset_density: {onset_density}
brightness_hz: {spectral_centroid_hz}

BUTLER
name: {butler_name}
voice: {butler_voice}

PRODUCER LINKS
store_link: {store_link}
licence_text: {licence_text}
contact: {contact}

TASK
1. youtube.title  -- under 100 chars. Front-load "{genre} type beat"; include BPM only if it reads
   naturally. Year optional.
2. youtube.description -- one hook line, then 3-5 lines describing the beat in the producer's voice,
   then the store link and licence block verbatim, then contact. No emoji.
3. youtube.tags -- 12-20 tags, lowercase, each under 40 chars, total under 480 chars when joined
   with ", ". Include genre, bpm, mood and instrument words.
4. type_beat_caption -- 1-2 lines for a short video, under 200 chars, no hashtags inside.
5. hashtags -- 10-15 lowercase hashtags, single words, no spaces or punctuation.
6. butler_line -- one short in-character line, under 90 chars, in the voice above.

STYLE: {style}
Return JSON matching the schema exactly."""


def build_repair_message(violations: list[str]) -> str:
    joined = "\n".join(f"- {v}" for v in violations)
    return (
        "Your previous JSON broke these rules:\n"
        f"{joined}\n\n"
        "Rewrite the whole object, fixing every rule above. Keep the same facts. "
        "Return only the corrected JSON object."
    )
