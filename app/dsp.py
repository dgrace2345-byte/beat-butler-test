"""DSP engine for Beat Butler (Slice 1).

Measures what DSP can actually measure: tempo, key, energy, onset density,
brightness, duration. Genre and mood are NOT measured here - genre is a
producer choice and mood is inferred, per BEAT-BUTLER-SPEC.md section 3.

Engine: librosa (ISC licence - commercial-safe, unlike Essentia's AGPL +
non-commercial models). Swappable: keep analyze_path()'s return shape stable
and a paid provider can sit behind it.
"""

from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np

SR = 22_050  # downmix target; a beat's character survives this
ANALYSIS_SECONDS = 90.0  # only the first 90s are analysed
HOP = 512
MAX_DURATION_SECONDS = 600
MIN_BPM, MAX_BPM = 60.0, 200.0

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class AnalysisError(Exception):
    """Raised when a file cannot be analysed. The message is user-facing."""


def engine_info() -> dict:
    return {"name": "librosa", "version": librosa.__version__}


def analyze_path(path: Path) -> dict:
    """Analyse one audio file and return the feature fields of the /v1/analyze contract."""
    try:
        y, sr = librosa.load(str(path), sr=SR, mono=True, duration=ANALYSIS_SECONDS)
    except Exception as exc:  # noqa: BLE001
        raise AnalysisError(
            "That file could not be decoded. Try a WAV or MP3 export."
        ) from exc

    if y.size == 0:
        raise AnalysisError("That file contains no audio.")

    peak = float(np.max(np.abs(y)))
    if peak < 1e-4:
        raise AnalysisError("That file is silent.")

    full_duration = _true_duration(path, y, sr)
    if full_duration > MAX_DURATION_SECONDS:
        raise AnalysisError(
            f"That beat is {full_duration / 60:.1f} minutes long. The limit is 10 minutes."
        )

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)

    raw_bpm = _detect_tempo(onset_env)
    bpm = _fold_bpm(raw_bpm) if raw_bpm else 0.0

    key, key_conf = _detect_key(y, sr)

    rms = librosa.feature.rms(y=y, hop_length=HOP).ravel()
    energy = float(np.clip(np.percentile(rms, 95) / 0.35, 0.0, 1.0))

    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, backtrack=True)
    # Density is per second of the audio that was actually analysed. Dividing by the
    # full file length undercounted every beat longer than ANALYSIS_SECONDS, because
    # onsets are only detected inside that window.
    analysed_seconds = len(y) / sr
    onset_density = float(len(onsets) / max(analysed_seconds, 0.001))

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)))

    return {
        "duration_seconds": round(full_duration, 2),
        "bpm": round(bpm, 1),
        "bpm_confidence": _tempo_confidence(onset_env, bpm),
        "bpm_alternatives": _alternatives(bpm),
        "key": key,
        "key_confidence": key_conf,
        "energy": round(energy, 3),
        "onset_density": round(onset_density, 2),
        "spectral_centroid_hz": round(centroid, 1),
        "mood_inferred": _infer_mood(energy, bpm, centroid),
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _true_duration(path: Path, y: np.ndarray, sr: int) -> float:
    """Full file duration, not the 90s analysis window."""
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:  # noqa: BLE001
        return float(len(y) / sr)


def _detect_tempo(onset_env: np.ndarray) -> float:
    """Tempo in BPM from a calibrated estimate, corrected against our range."""
    if onset_env.size < 16 or float(np.max(onset_env)) <= 0:
        return 0.0

    candidates: list[float] = []
    try:
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP)
        candidates.append(float(np.atleast_1d(tempo).ravel()[0]))
    except Exception:  # noqa: BLE001
        pass

    # autocorrelation fallback / cross-check
    ac = librosa.autocorrelate(onset_env - onset_env.mean())
    lag_min = int(math.floor(60.0 * SR / (MAX_BPM * HOP)))
    lag_max = min(int(math.ceil(60.0 * SR / (MIN_BPM * HOP))), len(ac) - 1)
    if lag_max > lag_min:
        window = ac[lag_min:lag_max]
        if window.size and float(np.max(window)) > 0:
            lag = lag_min + int(np.argmax(window))
            candidates.append(60.0 * SR / (lag * HOP))

    candidates = [c for c in candidates if c and math.isfinite(c) and c > 0]
    if not candidates:
        return 0.0

    # prefer the candidate closest to the harmonic centre after folding
    folded = [_fold_bpm(c) for c in candidates]
    folded.sort(key=lambda b: abs(math.log2(b / 120.0)))
    return folded[0]


def _fold_bpm(bpm: float) -> float:
    """Fold an estimate into 70-190 BPM, killing half/double-time reads."""
    if bpm <= 0:
        return 0.0
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 190.0:
        bpm /= 2.0
    return float(bpm)


def _alternatives(bpm: float) -> list[float]:
    if bpm <= 0:
        return []
    alts = {round(bpm, 1)}
    for factor in (0.5, 2.0):
        alt = bpm * factor
        if 45.0 <= alt <= 220.0:
            alts.add(round(alt, 1))
    ordered = sorted(alts)
    # half/double first, then the estimate itself
    return [a for a in ordered if a != round(bpm, 1)] + [round(bpm, 1)]


def _tempo_confidence(onset_env: np.ndarray, bpm: float) -> float:
    """Pulse clarity: how strongly the onset envelope repeats at the beat period."""
    if bpm <= 0 or onset_env.size < 32:
        return 0.0
    centred = onset_env - onset_env.mean()
    ac = librosa.autocorrelate(centred)
    if ac[0] <= 0:
        return 0.0
    lag = 60.0 * SR / (bpm * HOP)
    total, weight = 0.0, 0.0
    for mult, w in ((1.0, 1.0), (2.0, 0.5), (3.0, 0.25)):
        idx = int(round(lag * mult))
        if 0 < idx < len(ac):
            total += w * float(ac[idx] / ac[0])
            weight += w
    if weight == 0:
        return 0.0
    return round(float(np.clip(total / weight * 2.2, 0.0, 1.0)), 2)


def _detect_key(y: np.ndarray, sr: int) -> tuple[str, float]:
    """Krumhansl-Schmuckler key finding on a chroma vector."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    except Exception:  # noqa: BLE001
        return "", 0.0

    profile = np.median(chroma, axis=1)
    if not np.any(profile) or float(np.std(profile)) == 0.0:
        return "", 0.0

    # Pearson correlation against each rotated profile, not cosine: cosine
    # between two profiles is nearly flat across rotations and gives useless
    # margins, which is what made every key look uncertain.
    scores: list[tuple[float, str]] = []
    for root in range(12):
        for mode, template in (("major", _MAJOR), ("minor", _MINOR)):
            rolled = np.roll(template, root)
            r = float(np.corrcoef(profile, rolled)[0, 1])
            if math.isfinite(r):
                scores.append((r, f"{NOTE_NAMES[root]} {mode}"))

    if not scores:
        return "", 0.0
    scores.sort(reverse=True)
    best_score, best_key = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    margin = max(0.0, best_score - second_score)
    confidence = float(np.clip((margin - 0.01) / 0.12, 0.0, 1.0))
    return best_key, round(confidence, 2)


def _infer_mood(energy: float, bpm: float, centroid_hz: float) -> str:
    """Ordered heuristic. Always labelled 'inferred' in the UI - never a fact."""
    if energy >= 0.55:
        return "dark" if centroid_hz < 1700 else "aggressive"
    if energy >= 0.32:
        return "bouncy" if bpm >= 125 else "soulful"
    if energy <= 0.18:
        return "dreamy" if centroid_hz > 1900 else "laid back"
    return "laid back" if bpm < 100 else "soulful"
