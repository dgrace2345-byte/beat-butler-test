"""Output validation for generated copy.

The schema is not a suggestion: the platform limits in BEAT-BUTLER-SPEC.md
section 6 are hard caps (YouTube rejects a 101-character title, a description
containing "<", or more than 500 characters of tags), so every generated object
passes through here before it reaches a producer.
"""

from __future__ import annotations

import re

MAX_TITLE = 100
MIN_TITLE = 10
MAX_DESCRIPTION = 4800  # YouTube allows 5000; the buffer covers the licence block
MIN_DESCRIPTION = 80
MAX_TAG_ITEM = 40
MAX_TAGS_JOINED = 480  # YouTube's hard cap is 500 characters for all tags combined
MIN_TAGS, MAX_TAGS = 8, 24
MAX_CAPTION = 200
MAX_HASHTAG_LEN = 30
MIN_HASHTAGS, MAX_HASHTAGS = 5, 15
MAX_BUTLER_LINE = 90

HASHTAG_RE = re.compile(r"^#[a-z0-9]+$")
FORBIDDEN_CHARS = ("<", ">")
BANNED_CLAIMS = re.compile(
    r"\b(sold|exclusive|number one|#1|chart(?:ed|ing)?|platinum|grammy|million|streams)\b",
    re.IGNORECASE,
)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def normalise(output: dict) -> dict:
    """Lowercase and de-duplicate the things the platform treats case-insensitively."""
    youtube = output.get("youtube") or {}
    tags = [str(t) for t in (youtube.get("tags") or [])]
    hashtags = ["#" + str(t).strip().lstrip("#") for t in (output.get("hashtags") or [])]
    return {
        "youtube": {
            "title": str(youtube.get("title") or "").strip(),
            "description": str(youtube.get("description") or "").strip(),
            "tags": _dedupe(tags),
        },
        "type_beat_caption": str(output.get("type_beat_caption") or "").strip(),
        "hashtags": _dedupe(hashtags)[:MAX_HASHTAGS],
        "butler_line": str(output.get("butler_line") or "").strip(),
    }


def validate_output(output: dict, allowed_text: str = "") -> list[str]:
    """Returns human-readable violations. Empty list means the copy is shippable."""
    problems: list[str] = []
    if not isinstance(output, dict):
        return ["output is not a JSON object"]

    youtube = output.get("youtube")
    if not isinstance(youtube, dict):
        return ["youtube block is missing"]

    title = str(youtube.get("title") or "")
    description = str(youtube.get("description") or "")
    tags = youtube.get("tags")
    caption = str(output.get("type_beat_caption") or "")
    hashtags = output.get("hashtags")
    butler_line = str(output.get("butler_line") or "")

    if not MIN_TITLE <= len(title) <= MAX_TITLE:
        problems.append(f"youtube.title is {len(title)} chars (must be {MIN_TITLE}-{MAX_TITLE})")
    if not MIN_DESCRIPTION <= len(description) <= MAX_DESCRIPTION:
        problems.append(
            f"youtube.description is {len(description)} chars (must be {MIN_DESCRIPTION}-{MAX_DESCRIPTION})"
        )
    for char in FORBIDDEN_CHARS:
        if char in title:
            problems.append(f'youtube.title contains "{char}", which YouTube rejects')
        if char in description:
            problems.append(f'youtube.description contains "{char}", which YouTube rejects')

    if not isinstance(tags, list):
        problems.append("youtube.tags is not a list")
    else:
        if not MIN_TAGS <= len(tags) <= MAX_TAGS:
            problems.append(f"youtube.tags has {len(tags)} items (must be {MIN_TAGS}-{MAX_TAGS})")
        long_tags = [t for t in tags if len(str(t)) > MAX_TAG_ITEM]
        if long_tags:
            problems.append(f"{len(long_tags)} tag(s) longer than {MAX_TAG_ITEM} chars")
        joined = ", ".join(str(t) for t in tags)
        if len(joined) > MAX_TAGS_JOINED:
            problems.append(
                f"youtube.tags total {len(joined)} chars joined (must be <= {MAX_TAGS_JOINED}; "
                "YouTube caps all tags at 500 combined)"
            )

    if len(caption) > MAX_CAPTION:
        problems.append(f"type_beat_caption is {len(caption)} chars (max {MAX_CAPTION})")

    if not isinstance(hashtags, list):
        problems.append("hashtags is not a list")
    else:
        if not MIN_HASHTAGS <= len(hashtags) <= MAX_HASHTAGS:
            problems.append(f"hashtags has {len(hashtags)} items (must be {MIN_HASHTAGS}-{MAX_HASHTAGS})")
        bad = [h for h in hashtags if not HASHTAG_RE.match(str(h))]
        if bad:
            problems.append(f"invalid hashtags (lowercase, single word, leading #): {', '.join(map(str, bad[:4]))}")
        long_hashtags = [h for h in hashtags if len(str(h)) > MAX_HASHTAG_LEN]
        if long_hashtags:
            problems.append(f"{len(long_hashtags)} hashtag(s) longer than {MAX_HASHTAG_LEN} chars")

    if len(butler_line) > MAX_BUTLER_LINE:
        problems.append(f"butler_line is {len(butler_line)} chars (max {MAX_BUTLER_LINE})")

    # Claims the producer did not make: scanned only where the model authored text,
    # and exempted when the same word appears in the producer's own inputs
    # (a licence block legitimately says "exclusive").
    allowed = allowed_text.lower()
    for field_name, text in (
        ("youtube.title", title),
        ("youtube.description", description),
        ("type_beat_caption", caption),
        ("butler_line", butler_line),
    ):
        for match in BANNED_CLAIMS.finditer(text):
            term = match.group(1).lower()
            if term not in allowed:
                problems.append(f'{field_name} claims "{term}" which the producer did not state')

    return problems


def summary(output: dict) -> dict:
    """Numbers the UI shows next to each field so a producer can see the headroom."""
    youtube = output.get("youtube") or {}
    tags = youtube.get("tags") or []
    return {
        "title_chars": len(str(youtube.get("title") or "")),
        "title_limit": MAX_TITLE,
        "description_chars": len(str(youtube.get("description") or "")),
        # What the validator enforces, not YouTube's outer 5,000: a counter that
        # says 5,000 next to copy rejected at 4,801 is a lie the producer cannot see.
        "description_limit": MAX_DESCRIPTION,
        "tags_joined_chars": len(", ".join(str(t) for t in tags)),
        "tags_limit": 500,
        "caption_chars": len(str(output.get("type_beat_caption") or "")),
        "caption_limit": MAX_CAPTION,
        "hashtag_count": len(output.get("hashtags") or []),
    }
