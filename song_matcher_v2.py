# song_matcher_v3.py
from __future__ import annotations

import sys, re, unicodedata, math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


# ================================
# Config / Columns
# ================================
REQUIRED_FEATURES = [
    "Danceability","Energy","Key","Loudness","Mode","Speechiness",
    "Acousticness","Instrumentalness","Liveness","Valence","Tempo","Time Signature"
]
ID_COL_TRACKNAME = "Track Name"
OPTIONAL_ID_COLS = [ID_COL_TRACKNAME, "Name","Track","Title","Song","Artist","Artists","Album"]

INTRO_KEYWORDS = ["intro","interlude","skit","spoken","voice note","voicenote","voicemail"]

KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MODE_NAMES = {1: "Major", 0: "Minor"}

ARTIST_CANDIDATE_COLS = ("Artist","Artists","Performer","Primary Artist","Album Artist")
GENRE_CANDIDATE_COLS = ("Genre","Genres","Primary Genre","Style","Styles")

GENRE_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "EDM / Festival": [
        "big room",
        "progressive house",
        "progressive",
        "electro house",
        "festival",
        "mainstage",
        "anthem",
    ],
    "House / Dance": [
        "deep house",
        "tech house",
        "disco house",
        "dance",
        "club",
        "garage house",
    ],
    "Pop / Commercial": [
        "dance pop",
        "electro pop",
        "synth pop",
        "commercial",
        "chart",
        "top 40",
    ],
    "Urban / Hip-Hop": [
        "hip hop",
        "hip-hop",
        "trap",
        "rap",
        "urban",
    ],
    "R&B / Soul": [
        "r&b",
        "neo-soul",
        "neo soul",
        "soul",
    ],
    "Rock / Indie": [
        "indie rock",
        "alt rock",
        "alternative rock",
        "pop rock",
        "indie",
        "rock",
    ],
    "Chill / Lounge": [
        "chillstep",
        "ambient",
        "downtempo",
        "lounge",
        "chill",
        "lofi",
    ],
    "Alternative / Experimental": [
        "experimental",
        "future bass",
        "hyperpop",
        "leftfield",
        "idm",
        "glitch",
        "bass",
    ],
}

GENRE_CATEGORY_TOKEN_MAP: Dict[str, List[str]] = {
    "edm": ["EDM / Festival"],
    "festival": ["EDM / Festival"],
    "big": ["EDM / Festival"],
    "mainstage": ["EDM / Festival"],
    "progressive": ["EDM / Festival"],
    "electro": ["EDM / Festival", "Pop / Commercial"],
    "house": ["House / Dance"],
    "dance": ["House / Dance", "Pop / Commercial"],
    "club": ["House / Dance"],
    "pop": ["Pop / Commercial"],
    "commercial": ["Pop / Commercial"],
    "hip": ["Urban / Hip-Hop"],
    "hop": ["Urban / Hip-Hop"],
    "trap": ["Urban / Hip-Hop"],
    "rap": ["Urban / Hip-Hop"],
    "urban": ["Urban / Hip-Hop"],
    "r&b": ["R&B / Soul"],
    "soul": ["R&B / Soul"],
    "indie": ["Rock / Indie"],
    "rock": ["Rock / Indie"],
    "chill": ["Chill / Lounge"],
    "ambient": ["Chill / Lounge"],
    "downtempo": ["Chill / Lounge"],
    "lounge": ["Chill / Lounge"],
    "future": ["Alternative / Experimental"],
    "bass": ["Alternative / Experimental"],
    "hyperpop": ["Alternative / Experimental"],
    "experimental": ["Alternative / Experimental"],
}

DEFAULT_GENRE_CATEGORY = "Unclassified"
GENRE_CATEGORY_ORDER = list(GENRE_CATEGORY_KEYWORDS.keys())
ALL_GENRES_OPTION = "All Genres"
GENRE_CATEGORY_ADJACENCY: Dict[str, Dict[str, float]] = {
    "EDM / Festival": {
        "House / Dance": 0.8,
        "Pop / Commercial": 0.65,
        "Alternative / Experimental": 0.5,
    },
    "House / Dance": {
        "EDM / Festival": 0.8,
        "Pop / Commercial": 0.75,
        "Chill / Lounge": 0.6,
    },
    "Pop / Commercial": {
        "House / Dance": 0.75,
        "EDM / Festival": 0.65,
        "Urban / Hip-Hop": 0.7,
        "R&B / Soul": 0.6,
    },
    "Urban / Hip-Hop": {
        "Pop / Commercial": 0.7,
        "R&B / Soul": 0.78,
    },
    "R&B / Soul": {
        "Pop / Commercial": 0.6,
        "Urban / Hip-Hop": 0.78,
        "Chill / Lounge": 0.55,
    },
    "Rock / Indie": {
        "Alternative / Experimental": 0.65,
        "Pop / Commercial": 0.5,
    },
    "Chill / Lounge": {
        "R&B / Soul": 0.55,
        "House / Dance": 0.6,
        "Alternative / Experimental": 0.5,
    },
    "Alternative / Experimental": {
        "EDM / Festival": 0.5,
        "Chill / Lounge": 0.5,
        "Rock / Indie": 0.65,
    },
}

ENERGY_INTENTS = ("maintain", "rise", "cool")

DJ_STYLE_PRESETS: Dict[str, Dict[str, float]] = {
    "Warm-up House": {
        "tempo_close": 10.0,
        "genre": 18.0,
        "genre_category": 8.0,
        "energy": 1.2,
        "valence": 1.1,
    },
    "Peak EDM": {
        "tempo_close": 18.0,
        "tempo_mid": 3.0,
        "tempo_far": 1.0,
        "camelot": 11.0,
        "energy": 2.0,
        "danceability": 1.5,
        "genre": 14.0,
    },
    "Commercial Pop": {
        "genre": 20.0,
        "genre_category": 9.0,
        "camelot_rule": 7.0,
        "valence": 1.3,
        "speechiness": 0.4,
    },
}

TIMING_COLUMN_PATTERNS = {
    "intro_end": ["intro end", "introend", "intro_end", "intro outro crossover", "intro cutoff"],
    "fill1": ["fill1", "fill 1", "fill_1"],
    "drop_point": ["drop point", "drop", "drop_point"],
    "fill2": ["fill2", "fill 2", "fill_2"],
    "outro_start": ["outro start", "outro", "outro_start"],
    "fill3": ["fill3", "fill 3", "fill_3"],
}

TIMING_DISPLAY_NAMES = {
    "intro_end": "Intro End",
    "fill1": "Fill 1",
    "drop_point": "Drop Point",
    "fill2": "Fill 2",
    "outro_start": "Outro Start",
    "fill3": "Fill 3",
}


# ================================
# Small helpers
# ================================
def _is_missing(*vals) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals)

def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm_to_orig = {str(c).strip().lower(): c for c in df.columns}
    def resolve(name: str) -> Optional[str]:
        return norm_to_orig.get(name.strip().lower())

    missing, resolved = [], {}
    for f in REQUIRED_FEATURES:
        c = resolve(f)
        if c is None: missing.append(f)
        else: resolved[f] = c
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    id_cols = []
    for c in OPTIONAL_ID_COLS:
        rc = resolve(c)
        if rc and rc not in id_cols: id_cols.append(rc)

    keep = id_cols + [resolved[f] for f in REQUIRED_FEATURES]
    others = [c for c in df.columns if c not in keep]
    return df[keep + others].copy()

def _is_intro_like(name: str) -> bool:
    s = (name or "").strip().lower()
    return any(k in s for k in INTRO_KEYWORDS)

def _value_from_source(source: Any, key: str) -> Any:
    if source is None or key is None:
        return None
    if isinstance(source, dict):
        return source.get(key, None)
    if hasattr(source, "get"):
        try:
            return source.get(key, None)
        except TypeError:
            try:
                return source.get(key)
            except Exception:
                return None
    if hasattr(source, key):
        return getattr(source, key)
    return None

def _normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()

def _resolve_artist(source: Any) -> str:
    for key in ARTIST_CANDIDATE_COLS:
        candidate = _normalize_text_value(_value_from_source(source, key))
        if candidate:
            return candidate
    keys = []
    if isinstance(source, dict):
        keys = source.keys()
    elif hasattr(source, "index"):
        keys = source.index
    for key in keys:
        if isinstance(key, str) and "artist" in key.lower():
            candidate = _normalize_text_value(_value_from_source(source, key))
            if candidate:
                return candidate
    return ""

def _resolve_genre(source: Any) -> str:
    for key in GENRE_CANDIDATE_COLS:
        candidate = _normalize_text_value(_value_from_source(source, key))
        if candidate:
            return candidate
    keys = []
    if isinstance(source, dict):
        keys = source.keys()
    elif hasattr(source, "index"):
        keys = source.index
    for key in keys:
        if isinstance(key, str) and "genre" in key.lower():
            candidate = _normalize_text_value(_value_from_source(source, key))
            if candidate:
                return candidate
    return ""

def _format_metric(value: Any, label: str, *, decimals: int = 2, unit: str = "") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(num):
        return ""
    fmt = f"{{:.{decimals}f}}"
    return f"{label}: {fmt.format(num)}{unit}"

def _parse_time_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\\", ":").replace("-", ":")
    if ":" in text:
        parts = [p for p in text.split(":") if p != ""]
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return None
        if len(parts) == 1:
            return parts[0]
        seconds = 0.0
        multiplier = 1.0
        for number in reversed(parts):
            seconds += number * multiplier
            multiplier *= 60.0
        return seconds
    try:
        num = float(text)
        return num if num >= 0 else None
    except ValueError:
        return None

def _format_time_value(value: Any) -> str:
    secs = _parse_time_to_seconds(value)
    if secs is None:
        return ""
    if secs < 1.0:
        return f"{secs:.2f}s"
    hours = int(secs // 3600)
    minutes = int((secs % 3600) // 60)
    seconds = int(round(secs % 60))
    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60 and hours > 0:
        minutes = 0
        hours += 1
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


_GENRE_SPLIT_RE = re.compile(r"[|/\\;,]+|\s+\&\s+|\s+vs\.?\s+", flags=re.IGNORECASE)

def _clean_text_token(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9+#]+", " ", token).strip().lower()
    return re.sub(r"\s+", " ", cleaned)

def _genre_tokens(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    tokens: List[str] = []
    for raw in _GENRE_SPLIT_RE.split(text):
        cleaned = _clean_text_token(raw)
        if not cleaned:
            continue
        tokens.append(cleaned)
        parts = [p for p in cleaned.split(" ") if len(p) > 2 or "#" in p or "+" in p]
        tokens.extend(parts)
    # Preserve order but drop duplicates while keeping first occurrence
    seen = set()
    deduped = []
    for t in tokens:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped

def _genre_category_score(categories_a: Iterable[str], categories_b: Iterable[str]) -> Tuple[float, List[str]]:
    set_a = {str(c).strip() for c in categories_a or [] if str(c).strip()}
    set_b = {str(c).strip() for c in categories_b or [] if str(c).strip()}
    if not set_a or not set_b:
        return 0.5, []
    overlap = sorted(set_a & set_b)
    if overlap:
        if set_a == set_b and len(set_a) == 1:
            return 1.0, overlap
        if set_a == set_b:
            return 0.92, overlap
        return 0.78 + 0.05 * min(len(overlap), 3), overlap
    # no overlap - favour related broad categories (e.g., both not default)
    if DEFAULT_GENRE_CATEGORY in set_a or DEFAULT_GENRE_CATEGORY in set_b:
        base = 0.45
    else:
        base = 0.35
    best = base
    for ca in set_a:
        for cb in set_b:
            weight = GENRE_CATEGORY_ADJACENCY.get(ca, {}).get(cb)
            if weight is None:
                weight = GENRE_CATEGORY_ADJACENCY.get(cb, {}).get(ca)
            if weight:
                best = max(best, base + 0.4 * weight)
    return best, []

def _derive_genre_categories(value: Any) -> List[str]:
    tokens = _genre_tokens(value)
    if not tokens:
        return []
    joined = " ".join(tokens)
    categories: List[str] = []
    for category, keywords in GENRE_CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in joined:
                categories.append(category)
                break
    token_set = set(tokens)
    for token in token_set:
        for category in GENRE_CATEGORY_TOKEN_MAP.get(token, []):
            if category not in categories:
                categories.append(category)
    if not categories:
        categories.append(DEFAULT_GENRE_CATEGORY)
    return categories

def _parse_category_numbers(selection: str, options: List[str]) -> List[str]:
    selection = selection.strip()
    if not selection:
        return []
    parts = re.split(r"[,\s]+", selection)
    picked: List[str] = []
    for part in parts:
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"Invalid selection '{part}'. Use numbers only.")
        idx = int(part)
        if idx < 1 or idx > len(options):
            raise ValueError(f"Selection '{part}' is out of range.")
        category = options[idx - 1]
        if category not in picked:
            picked.append(category)
    return picked

def _genre_score(a: Any, b: Any) -> float:
    tok_list_a = _genre_tokens(a)
    tok_list_b = _genre_tokens(b)
    if not tok_list_a or not tok_list_b:
        return 0.5
    ta, tb = set(tok_list_a), set(tok_list_b)
    overlap = ta & tb
    if not overlap:
        return 0.25
    jaccard = len(overlap) / float(len(ta | tb))
    base = 0.55 + 0.45 * jaccard
    if any(" " in token for token in overlap):
        base = max(base, 0.85)
    primary_a = tok_list_a[0].split(" ")[0] if tok_list_a else ""
    primary_b = tok_list_b[0].split(" ")[0] if tok_list_b else ""
    if primary_a and primary_b and primary_a == primary_b:
        base = max(base, 0.9)
    return min(1.0, base)


# ================================
# Scoring pieces (NaN-safe)
# ================================
def _key_mode_score(key_a: float, mode_a: float, key_b: float, mode_b: float) -> float:
    try:
        k1 = int(round(key_a)) % 12
        k2 = int(round(key_b)) % 12
        m1 = int(round(mode_a)) % 2
        m2 = int(round(mode_b)) % 2
    except Exception:
        return 0.5
    if k1 == k2 and m1 == m2: return 1.0
    if k1 == k2 and m1 != m2: return 0.93
    semi = min((k1 - k2) % 12, (k2 - k1) % 12)
    if semi == 7: return 0.88 if m1 == m2 else 0.83
    if semi in (1,2): return 0.80 if semi == 1 else 0.78
    base = 1.0 - (semi/12.0) * 0.9
    return max(0.0, min(0.95, base))

def _tempo_score(bpm_a: float, bpm_b: float) -> Tuple[float, Optional[float], Optional[float]]:
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.5, None, None

    def score_from_diff(diff: float) -> float:
        if diff <= 1.0:
            return 1.0
        if diff <= 3.0:
            return max(0.9, 0.97 - 0.03 * (diff - 1.0))
        if diff <= 5.0:
            return max(0.8, 0.9 - 0.04 * (diff - 3.0))
        if diff <= 8.0:
            return max(0.6, 0.82 - 0.06 * (diff - 5.0))
        if diff <= 12.0:
            return max(0.4, 0.64 - 0.045 * (diff - 8.0))
        if diff <= 18.0:
            return max(0.25, 0.46 - 0.03 * (diff - 12.0))
        return 0.2

    candidates = [(bpm_b, 1.0), (bpm_b * 2.0, 2.0), (bpm_b / 2.0, 0.5)]
    best_score = -1.0
    best_diff = None
    best_factor = None
    for cand, factor in candidates:
        if cand <= 0:
            continue
        diff = abs(bpm_a - cand)
        score = score_from_diff(diff)
        if score > best_score:
            best_score = score
            best_diff = diff
            best_factor = factor

    if best_score < 0:
        return 0.5, None, None
    return min(1.0, max(0.0, best_score)), best_diff, best_factor

def _bounded_diff_score(a: float, b: float, span: float = 1.0) -> float:
    if _is_missing(a,b): return 0.5
    return max(0.0, min(1.0, 1.0 - abs(a-b)/float(span)))

def _loudness_score(a: float, b: float) -> float:
    if _is_missing(a,b): return 0.5
    diff = abs(a-b)
    if diff <= 1.0: return 0.95
    if diff <= 2.0: return 0.88
    return max(0.0, 0.88 - (diff-2.0)*0.10)

def _bar_length_seconds(tempo: float) -> Optional[float]:
    try:
        tempo = float(tempo)
    except (TypeError, ValueError):
        return None
    if tempo <= 0:
        return None
    beat_length = 60.0 / tempo
    return beat_length * 4.0

def _mod_alignment(delta: float, period: float) -> float:
    if period is None or period <= 0:
        return abs(delta)
    mod = abs(delta) % period
    return min(mod, period - mod)

def _phrase_alignment_score(anchor_meta: Dict[str, Any], cand_meta: Dict[str, Any],
                            anchor_tempo: float, cand_tempo: float) -> Tuple[float, Optional[float], Optional[str]]:
    anchor_cues = {}
    cand_cues = {}
    for key, label in TIMING_DISPLAY_NAMES.items():
        anchor_cues[key] = _parse_time_to_seconds(anchor_meta.get(label))
        cand_cues[key] = _parse_time_to_seconds(cand_meta.get(label))
    anchor_bar = _bar_length_seconds(anchor_tempo)
    cand_bar = _bar_length_seconds(cand_tempo)
    if not anchor_bar and not cand_bar:
        return 0.5, None, None
    use_bar = None
    if anchor_bar and cand_bar:
        use_bar = (anchor_bar + cand_bar) / 2.0
    else:
        use_bar = anchor_bar or cand_bar

    cue_pairs = [
        ("outro_start", "intro_end"),
        ("drop_point", "drop_point"),
        ("fill2", "intro_end"),
        ("outro_start", "drop_point"),
    ]
    best_alignment = None
    best_pair = None
    for anchor_key, cand_key in cue_pairs:
        a_time = anchor_cues.get(anchor_key)
        b_time = cand_cues.get(cand_key)
        if a_time is None or b_time is None:
            continue
        delta = b_time - a_time
        alignment = _mod_alignment(delta, use_bar)
        if best_alignment is None or alignment < best_alignment:
            best_alignment = alignment
            best_pair = (anchor_key, cand_key, delta)

    if best_alignment is None or use_bar is None:
        return 0.55, None, None

    tightness = max(0.0, 1.0 - (best_alignment / max(use_bar, 1e-6)))
    # Encourage tighter bounds (within 0.1 bar) to approach 1.0
    score = 0.6 + 0.4 * tightness
    anchor_label = TIMING_DISPLAY_NAMES.get(best_pair[0], best_pair[0]) if best_pair else None
    cand_label = TIMING_DISPLAY_NAMES.get(best_pair[1], best_pair[1]) if best_pair else None
    return min(1.0, score), best_alignment, f"{anchor_label}→{cand_label}" if anchor_label and cand_label else None

def _vocal_overlap_penalty(speech_a: float, speech_b: float) -> float:
    if _is_missing(speech_a, speech_b):
        return 0.65
    speech_a = max(0.0, min(1.0, float(speech_a)))
    speech_b = max(0.0, min(1.0, float(speech_b)))
    product = speech_a * speech_b
    if product <= 0.05:
        return 0.95
    penalty = max(0.15, 1.0 - 1.2 * math.sqrt(product))
    return penalty

def _energy_trajectory_score(delta: float, intent: str) -> float:
    if delta is None:
        return 0.6
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        return 0.6
    intent = (intent or "maintain").lower()
    if intent not in ENERGY_INTENTS:
        intent = "maintain"
    if intent == "maintain":
        return max(0.2, 1.0 - abs(delta) * 1.5)
    if intent == "rise":
        if delta >= 0.1:
            return min(1.0, 0.7 + delta * 1.5)
        return max(0.25, 0.6 + delta * 2.0)
    if intent == "cool":
        if delta <= -0.1:
            return min(1.0, 0.7 + abs(delta) * 1.5)
        return max(0.25, 0.6 - delta * 2.0)
    return 0.6

def _groove_score(dance_anchor: float, dance_cand: float, tempo_diff: Optional[float], tempo_factor: Optional[float]) -> float:
    if _is_missing(dance_anchor, dance_cand):
        return 0.5
    anchor = max(0.0, min(1.0, float(dance_anchor)))
    cand = max(0.0, min(1.0, float(dance_cand)))
    base = 0.55 + 0.45 * ((anchor + cand) / 2.0)
    if tempo_diff is None:
        return base
    tempo_diff = abs(float(tempo_diff))
    if tempo_diff <= 2.0 and (tempo_factor is None or abs(tempo_factor - 1.0) < 1e-3):
        return min(1.0, base + 0.2)
    if tempo_diff <= 4.0 and (tempo_factor is None or abs(tempo_factor - 1.0) < 1e-3):
        return base
    if tempo_factor and (tempo_factor >= 1.9 or tempo_factor <= 0.55):
        return max(0.3, base - 0.25)
    return max(0.4, base - 0.15)

def _time_signature_alignment(a_ts: float, b_ts: float, anchor_tempo: float, cand_tempo: float) -> float:
    if _is_missing(a_ts, b_ts):
        return 0.6
    try:
        a_val = int(round(float(a_ts)))
        b_val = int(round(float(b_ts)))
    except Exception:
        return 0.6
    if a_val == b_val:
        if a_val == 4:
            bar_a = _bar_length_seconds(anchor_tempo) or 0.0
            bar_b = _bar_length_seconds(cand_tempo) or 0.0
            if bar_a > 0 and bar_b > 0:
                diff = abs(bar_a - bar_b)
                if diff <= 0.15:
                    return 1.0
                if diff <= 0.4:
                    return 0.85
            return 0.75
        return 0.8
    # Creative mixes: allow 3/4 into 4/4 with penalty, others heavy
    pair = {a_val, b_val}
    if pair == {3, 4}:
        return 0.4
    return 0.25

def _camelot_context_multiplier(camelot_relation: str, camelot_rule: str,
                                energy_delta: Optional[float], valence_delta: Optional[float]) -> float:
    relation = (camelot_relation or "").lower()
    rule = (camelot_rule or "").lower()
    delta_e = float(energy_delta) if energy_delta is not None else 0.0
    delta_v = float(valence_delta) if valence_delta is not None else 0.0
    boost = 1.0
    if "adjacent + mode swap" in relation and delta_e > 0.05:
        boost += min(0.25, delta_e * 0.6)
    if "two-step" in relation and delta_v > 0.05:
        boost += min(0.2, delta_v * 0.5)
    if rule == "distant" and delta_e < -0.1:
        boost *= 0.9
    return boost


# ================================
# Weights
# ================================
@dataclass
class Weights:
    camelot: float = 10.0
    camelot_rule: float = 6.0
    tempo_close: float = 15.0
    tempo_mid: float = 2.2
    tempo_far: float = 0.8
    genre: float = 15.0
    genre_category: float = 6.0
    phrase_alignment: float = 7.5
    vocal_overlap: float = 4.0
    energy_trajectory: float = 5.0
    groove: float = 3.5
    camelot_context: float = 2.5
    time_signature_alignment: float = 1.5
    key: float = 8.0
    energy: float = 1.5
    danceability: float = 1.2
    valence: float = 0.8
    loudness: float = 0.6
    speechiness: float = 0.3
    acousticness: float = 0.3
    instrumentalness: float = 0.3
    liveness: float = 0.3


# ================================
# Camelot helpers (NaN-safe)
# ================================
_CAMELOT_MAP_MAJOR = {0:"8B",1:"3B",2:"10B",3:"5B",4:"12B",5:"7B",6:"2B",7:"9B",8:"4B",9:"11B",10:"6B",11:"1B"}
_CAMELOT_MAP_MINOR = {0:"5A",1:"12A",2:"7A",3:"2A",4:"9A",5:"4A",6:"11A",7:"6A",8:"1A",9:"8A",10:"3A",11:"10A"}
_CAMELOT_CODE_RE = re.compile(r"^\s*(1[0-2]|[1-9])\s*([AaBb])\s*$")

def keymode_to_camelot(key: float, mode: float) -> str:
    try:
        k = int(round(key)) % 12
        m = int(round(mode)) % 2
    except Exception:
        return "?"
    return _CAMELOT_MAP_MAJOR[k] if m == 1 else _CAMELOT_MAP_MINOR[k]

def suggest_keys_camelot(key: float, mode: float) -> List[str]:
    base = keymode_to_camelot(key, mode)
    if base == "?": return ["Unknown (missing key/mode)"]
    num, suf = int(base[:-1]), base[-1]
    alt = "B" if suf == "A" else "A"
    suggestions = [f"{num}{suf}", f"{num}{alt}", f"{((num)%12)+1}{suf}", f"{((num-2)%12)+1}{suf}", f"{((num)%12)+1}{alt}"]
    out, seen = [], set()
    for s in suggestions:
        n = int(s[:-1]); x = n if 1 <= n <= 12 else ((n-1) % 12) + 1
        keystr = f"{x}{s[-1]}"
        if keystr not in seen:
            out.append(keystr); seen.add(keystr)
    return out[:5]

def _parse_camelot_code(code: Any) -> Optional[Tuple[int, str]]:
    if code is None:
        return None
    if isinstance(code, float) and np.isnan(code):
        return None
    m = _CAMELOT_CODE_RE.match(str(code))
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper()

def _camelot_relation_and_score(code_a: Any, code_b: Any) -> Tuple[float, str, Optional[int], str]:
    pa = _parse_camelot_code(code_a)
    pb = _parse_camelot_code(code_b)
    if not pa or not pb:
        return 0.5, "Unknown", None, "unknown"
    num_a, ring_a = pa
    num_b, ring_b = pb
    diff = abs(num_a - num_b)
    diff = diff if diff <= 6 else 12 - diff
    same_ring = ring_a == ring_b
    if diff == 0 and same_ring:
        return 1.0, "Same key", 0, "primary"
    if diff == 0:
        return 0.98, "Relative major/minor", 0, "primary"
    if diff == 1 and same_ring:
        return 0.93, "Adjacent key", 1, "primary"
    if diff == 1:
        return 0.82, "Adjacent + mode swap", 1, "secondary"
    if diff == 2 and same_ring:
        return 0.72, "Two-step energy shift", 2, "secondary"
    if diff == 2:
        return 0.65, "Two-step + mode swap", 2, "secondary"
    if diff == 3 and same_ring:
        return 0.55, "Three-step creative", 3, "creative"
    if diff == 3:
        return 0.48, "Three-step + mode swap", 3, "creative"
    score = max(0.15, 0.45 - 0.05 * diff)
    label = "Distant creative" if same_ring else "Distant + mode swap"
    return score, label, diff, "distant"


# ================================
# Main matcher
# ================================
class SongMatcher:
    def __init__(self, df: pd.DataFrame, weights: Optional[Weights] = None, ignore_intros: bool = True):
        self.raw_df = df.copy()
        self.df = _canonicalise_columns(df)
        if ignore_intros and ID_COL_TRACKNAME in self.df.columns:
            mask = ~self.df[ID_COL_TRACKNAME].astype(str).str.lower().apply(_is_intro_like)
            self.df = self.df[mask].reset_index(drop=True)
        computed_camelot = pd.Series(
            [keymode_to_camelot(k, m) if not _is_missing(k, m) else "?" for k, m in zip(self.df["Key"], self.df["Mode"])],
            index=self.df.index
        )
        camelot_col = None
        for c in self.df.columns:
            if str(c).strip().lower() == "camelot":
                camelot_col = c
                break
        if camelot_col is None:
            camelot_col = "Camelot"
            self.df[camelot_col] = computed_camelot
        else:
            existing = self.df[camelot_col].astype(str).str.strip()
            mask = existing.eq("") | existing.eq("nan") | existing.eq("?")
            self.df.loc[mask, camelot_col] = computed_camelot[mask]
        self.camelot_col = camelot_col
        self.genre_col = None
        for c in self.df.columns:
            if "genre" in str(c).strip().lower():
                self.genre_col = c
                break
        if self.genre_col and self.genre_col in self.df.columns:
            genre_source = self.df[self.genre_col]
        elif "Genre" in self.df.columns:
            genre_source = self.df["Genre"]
        else:
            genre_source = pd.Series([None] * len(self.df), index=self.df.index)
        categories_series = genre_source.apply(_derive_genre_categories)
        self.df["GenreCategories"] = categories_series
        self.df["PrimaryGenreCategory"] = categories_series.apply(lambda cats: cats[0] if cats else DEFAULT_GENRE_CATEGORY)
        self.timing_cols: Dict[str, str] = {}
        seen_cols = set()
        for key, patterns in TIMING_COLUMN_PATTERNS.items():
            for col in self.df.columns:
                col_norm = str(col).strip().lower()
                if col in seen_cols:
                    continue
                for pattern in patterns:
                    pattern_norm = pattern.strip().lower()
                    if not pattern_norm:
                        continue
                    if col_norm == pattern_norm or pattern_norm in col_norm:
                        self.timing_cols[key] = col
                        seen_cols.add(col)
                        break
                if key in self.timing_cols:
                    break
        self.weights = weights or Weights()
        self.default_context: Dict[str, Any] = {"energy_intent": "maintain"}
        self.feedback_log_path: Path = Path("transition_feedback.csv")

    @classmethod
    def from_excel(cls, path: Union[str, Path], sheet_name: Optional[Union[str,int]] = 0, **kwargs):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        return cls(df, **kwargs)

    def apply_style_preset(self, name: str):
        preset = DJ_STYLE_PRESETS.get(name)
        if not preset:
            raise ValueError(f"Unknown style preset '{name}'.")
        data = self.weights.__dict__.copy()
        data.update(preset)
        self.weights = Weights(**data)

    def learn_from_feedback(self, path: Optional[Union[str, Path]] = None,
                            lr: float = 0.15, iterations: int = 200):
        feedback_path = Path(path) if path else self.feedback_log_path
        if not feedback_path.exists():
            raise FileNotFoundError(f"No feedback log found at {feedback_path}")
        df = pd.read_csv(feedback_path)
        if "label" not in df.columns:
            raise ValueError("Feedback file must include a 'label' column (1=worked, 0=skip).")
        feature_cols = [
            "CamelotScore",
            "TempoScore",
            "GenreScore",
            "GenreCategoryScore",
            "PhraseAlignScore",
            "VocalOverlapScore",
            "EnergyTrajectoryScore",
            "GrooveScore",
            "LoudnessScore",
            "TimeSignatureScore",
        ]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feedback file missing columns: {', '.join(missing)}")
        data = df[feature_cols].fillna(0.5).to_numpy(dtype=float)
        labels = df["label"].astype(float).to_numpy()
        if len(data) == 0:
            raise ValueError("Feedback file is empty.")
        # normalise
        data = np.clip(data, 0.0, 1.0)
        X = np.hstack([np.ones((len(data), 1)), data])
        weights = np.zeros(X.shape[1])
        for _ in range(max(1, iterations)):
            z = X @ weights
            preds = 1.0 / (1.0 + np.exp(-z))
            gradient = X.T @ (preds - labels) / len(labels)
            weights -= lr * gradient
        adjustments = weights[1:]
        feature_to_weight = {
            "CamelotScore": "camelot",
            "TempoScore": "tempo_close",
            "GenreScore": "genre",
            "GenreCategoryScore": "genre_category",
            "PhraseAlignScore": "phrase_alignment",
            "VocalOverlapScore": "vocal_overlap",
            "EnergyTrajectoryScore": "energy_trajectory",
            "GrooveScore": "groove",
            "LoudnessScore": "loudness",
            "TimeSignatureScore": "time_signature_alignment",
        }
        weights_dict = self.weights.__dict__.copy()
        for coeff, feature in zip(adjustments, feature_cols):
            target_attr = feature_to_weight.get(feature)
            if not target_attr:
                continue
            current = weights_dict.get(target_attr, 0.0)
            scale = 1.0 + np.tanh(coeff) * 0.2
            weights_dict[target_attr] = max(0.0, current * scale)
        self.weights = Weights(**weights_dict)

    def _row_to_features(self, row: pd.Series) -> Dict[str, float]:
        return {k: float(row[k]) for k in REQUIRED_FEATURES}

    def _extract_meta(self, source: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}

        def safe_get(obj, key):
            if obj is None or key is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            if hasattr(obj, "get"):
                try:
                    return obj.get(key, None)
                except TypeError:
                    return obj.get(key)
            return None

        camelot_val = None
        if self.camelot_col:
            camelot_val = safe_get(source, self.camelot_col)
        if camelot_val is None:
            camelot_val = safe_get(source, "Camelot")
        parsed_camelot = _parse_camelot_code(camelot_val)
        if not parsed_camelot:
            key_val = safe_get(source, "Key")
            mode_val = safe_get(source, "Mode")
            if not _is_missing(key_val, mode_val):
                camelot_val = keymode_to_camelot(key_val, mode_val)
            else:
                camelot_val = "?"
        else:
            camelot_val = f"{parsed_camelot[0]}{parsed_camelot[1]}"
        meta["Camelot"] = camelot_val
        if self.camelot_col:
            meta[self.camelot_col] = camelot_val

        genre_val = None
        if self.genre_col:
            genre_val = safe_get(source, self.genre_col)
        if genre_val is None:
            for col in ("Genre", "Genres", "Primary Genre"):
                value = safe_get(source, col)
                if value is not None:
                    genre_val = value
                    break
        if isinstance(genre_val, float) and np.isnan(genre_val):
            genre_val = None
        if isinstance(genre_val, str):
            genre_val = genre_val.strip()
        if genre_val is not None:
            meta["Genre"] = genre_val
            if self.genre_col:
                meta[self.genre_col] = genre_val
        categories_val = safe_get(source, "GenreCategories")
        if isinstance(categories_val, str):
            categories_val = [part.strip() for part in categories_val.split(",") if part.strip()]
        if not categories_val:
            categories_val = _derive_genre_categories(genre_val)
        meta["GenreCategories"] = categories_val
        if categories_val:
            meta["PrimaryGenreCategory"] = categories_val[0]

        for key, col_name in self.timing_cols.items():
            label = TIMING_DISPLAY_NAMES.get(key, col_name)
            value = safe_get(source, col_name)
            if isinstance(value, float) and np.isnan(value):
                value = None
            if isinstance(value, str) and not value.strip():
                value = None
            meta[label] = value
            timing_key = f"Timing_{key}"
            meta[timing_key] = value

        return meta

    def describe_track(self, row: pd.Series) -> str:
        meta = self._extract_meta(row)
        name = _normalize_text_value(row.get(ID_COL_TRACKNAME, "<unknown>")) or "<unknown>"
        artist = _resolve_artist(row)
        header = f"Selected Track: {name}"
        if artist:
            header += f" – {artist}"

        genre = _normalize_text_value(meta.get(self.genre_col)) if self.genre_col else ""
        if not genre:
            genre = _normalize_text_value(meta.get("Genre"))
        if genre.lower() in ("?", "nan"):
            genre = ""
        camelot = _normalize_text_value(meta.get("Camelot"))
        if camelot.lower() in ("", "?", "nan"):
            camelot = ""

        def _fmt_tempo(value: Any) -> str:
            try:
                t = float(value)
            except (TypeError, ValueError):
                return ""
            if np.isnan(t):
                return ""
            return f"{t:.1f} BPM"

        tempo_str = _fmt_tempo(row.get("Tempo"))

        key_desc = ""
        key_val = row.get("Key")
        mode_val = row.get("Mode")
        if not _is_missing(key_val, mode_val):
            try:
                key_idx = int(round(float(key_val))) % 12
                mode_idx = int(round(float(mode_val))) % 2
                mode_name = MODE_NAMES.get(mode_idx, "")
                key_name = KEY_NAMES[key_idx]
                key_desc = f"{key_name} {mode_name}".strip()
            except Exception:
                pass

        info_parts = []
        if genre:
            info_parts.append(f"Genre: {genre}")
        categories_list = meta.get("GenreCategories") or []
        if isinstance(categories_list, str):
            categories_list = [part.strip() for part in categories_list.split(",") if part.strip()]
        if categories_list:
            info_parts.append("Categories: " + ", ".join(categories_list[:3]))
        if camelot:
            info_parts.append(f"Camelot: {camelot}")
        if key_desc:
            info_parts.append(f"Key: {key_desc}")
        if tempo_str:
            info_parts.append(f"Tempo: {tempo_str}")

        energy = _format_metric(row.get("Energy"), "Energy")
        dance = _format_metric(row.get("Danceability"), "Danceability")
        valence = _format_metric(row.get("Valence"), "Valence")
        loud = _format_metric(row.get("Loudness"), "Loudness", decimals=1, unit=" dB")

        metric_parts = [part for part in (energy, dance, valence, loud) if part]

        lines = [header]
        if info_parts:
            lines.append(" | ".join(info_parts))
        if metric_parts:
            lines.append(" | ".join(metric_parts))
        timing_parts = []
        for key, label in TIMING_DISPLAY_NAMES.items():
            formatted = _format_time_value(meta.get(label))
            if formatted:
                timing_parts.append(f"{label}: {formatted}")
        if timing_parts:
            lines.append("Cues: " + " | ".join(timing_parts))
        return "\n".join(lines)

    def explain_weights(self) -> pd.DataFrame:
        w = self.weights
        data = {
            "Factor": [
                "Camelot",
                "Camelot rule priority",
                "Tempo (<=5 BPM diff)",
                "Tempo (6-10 BPM diff)",
                "Tempo (>10 BPM diff)",
                "Genre",
                "Genre category",
                "Phrase & cue alignment",
                "Vocal overlap penalty",
                "Energy trajectory fit",
                "Groove compatibility",
                "Camelot context boost",
                "Time signature alignment",
                "Key (fallback)",
                "Energy",
                "Danceability",
                "Valence",
                "Loudness",
                "Speechiness",
                "Acousticness",
                "Instrumentalness",
                "Liveness",
            ],
            "Weight": [
                w.camelot,
                w.camelot_rule,
                w.tempo_close,
                w.tempo_mid,
                w.tempo_far,
                w.genre,
                w.genre_category,
                w.phrase_alignment,
                w.vocal_overlap,
                w.energy_trajectory,
                w.groove,
                w.camelot_context,
                w.time_signature_alignment,
                w.key,
                w.energy,
                w.danceability,
                w.valence,
                w.loudness,
                w.speechiness,
                w.acousticness,
                w.instrumentalness,
                w.liveness,
            ],
        }
        return pd.DataFrame(data).sort_values("Weight", ascending=False).reset_index(drop=True)

    # ---- dynamic-weight matchscore (skips missing features) ----
    def _matchscore(
        self,
        a: Dict[str, float],
        b: Dict[str, float],
        meta_a: Optional[Dict[str, Any]] = None,
        meta_b: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        parts: Dict[str, Any] = {}
        total_num = 0.0
        total_den = 0.0
        w = self.weights
        meta_a = meta_a or {}
        meta_b = meta_b or {}
        context = context or {}

        def acc(score, weight, label):
            nonlocal total_num, total_den
            if score is None or weight is None or weight <= 0:
                return
            parts[label] = score
            total_num += weight * score
            total_den += weight

        def resolve_camelot(meta: Dict[str, Any], features: Dict[str, float]) -> str:
            camelot_val = meta.get("Camelot")
            parsed = _parse_camelot_code(camelot_val)
            if parsed:
                return f"{parsed[0]}{parsed[1]}"
            if not _is_missing(features.get("Key"), features.get("Mode")):
                return keymode_to_camelot(features["Key"], features["Mode"])
            return "?"

        anchor_camelot = resolve_camelot(meta_a, a)
        cand_camelot = resolve_camelot(meta_b, b)
        camelot_score, camelot_relation, camelot_steps, camelot_rule = _camelot_relation_and_score(anchor_camelot, cand_camelot)
        parts["CamelotAnchor"] = anchor_camelot
        parts["CamelotCandidate"] = cand_camelot
        parts["CamelotRelation"] = camelot_relation
        parts["CamelotRule"] = camelot_rule
        if camelot_steps is not None:
            parts["CamelotSteps"] = camelot_steps
            acc(camelot_score, w.camelot, "CamelotScore")
        else:
            parts["CamelotScore"] = camelot_score

        rule_weight = w.camelot_rule
        energy_delta = None if _is_missing(a.get("Energy"), b.get("Energy")) else float(b["Energy"]) - float(a["Energy"])
        valence_delta = None if _is_missing(a.get("Valence"), b.get("Valence")) else float(b["Valence"]) - float(a["Valence"])
        camelot_context_boost = _camelot_context_multiplier(camelot_relation, camelot_rule, energy_delta, valence_delta)
        parts["CamelotContextBoost"] = camelot_context_boost
        acc(min(1.0, max(0.0, camelot_context_boost)), w.camelot_context, "CamelotContextScore")
        if camelot_rule == "primary":
            acc(min(1.0, 1.0 * camelot_context_boost), rule_weight, "CamelotRuleScore")
        elif camelot_rule == "secondary":
            acc(min(1.0, 0.6 * camelot_context_boost), rule_weight * 0.7, "CamelotRuleScore")
        elif camelot_rule == "creative":
            acc(min(1.0, 0.35 * camelot_context_boost), rule_weight * 0.5, "CamelotRuleScore")
        elif camelot_rule == "distant":
            acc(min(1.0, 0.15 * camelot_context_boost), rule_weight * 0.4, "CamelotRuleScore")

        if not _is_missing(a.get("Key"), a.get("Mode"), b.get("Key"), b.get("Mode")):
            key_weight = w.key * (0.35 if parts.get("CamelotSteps") is not None else 1.0)
            if key_weight > 0:
                acc(_key_mode_score(a["Key"], a["Mode"], b["Key"], b["Mode"]), key_weight, "KeyScore")
        if not _is_missing(a.get("Tempo"), b.get("Tempo")):
            tempo_result = _tempo_score(a["Tempo"], b["Tempo"])
            if isinstance(tempo_result, tuple):
                tempo_score, tempo_diff, tempo_factor = tempo_result
            else:
                tempo_score, tempo_diff, tempo_factor = tempo_result, None, None
            if tempo_diff is not None:
                if tempo_diff <= 5:
                    tempo_weight = w.tempo_close
                elif tempo_diff <= 10:
                    tempo_weight = w.tempo_mid
                else:
                    tempo_weight = w.tempo_far
                if tempo_factor and tempo_factor != 1.0:
                    tempo_weight *= 0.85
            else:
                tempo_weight = w.tempo_mid
            acc(tempo_score, tempo_weight, "TempoScore")
            if tempo_diff is not None:
                parts["TempoDelta"] = float(tempo_diff)
            if tempo_factor:
                parts["TempoMatchFactor"] = tempo_factor
        else:
            tempo_diff = None
            tempo_factor = None

        phrase_score, phrase_alignment, phrase_pair = _phrase_alignment_score(
            meta_a,
            meta_b,
            a.get("Tempo"),
            b.get("Tempo"),
        )
        acc(phrase_score, w.phrase_alignment, "PhraseAlignScore")
        if phrase_alignment is not None:
            parts["PhraseAlignOffset"] = float(phrase_alignment)
        if phrase_pair:
            parts["PhraseAlignPair"] = phrase_pair

        genre_source_key = self.genre_col or "Genre"
        genre_a_val = meta_a.get(genre_source_key) if isinstance(meta_a, dict) else None
        genre_b_val = meta_b.get(genre_source_key) if isinstance(meta_b, dict) else None
        if genre_a_val is None and "Genre" in meta_a:
            genre_a_val = meta_a.get("Genre")
        if genre_b_val is None and "Genre" in meta_b:
            genre_b_val = meta_b.get("Genre")
        genre_tokens_a = _genre_tokens(genre_a_val)
        genre_tokens_b = _genre_tokens(genre_b_val)
        genre_score = _genre_score(genre_a_val, genre_b_val)
        acc(genre_score, w.genre, "GenreScore")

        categories_a = meta_a.get("GenreCategories") or []
        categories_b = meta_b.get("GenreCategories") or []
        if isinstance(categories_a, str):
            categories_a = [part.strip() for part in categories_a.split(",") if part.strip()]
        if isinstance(categories_b, str):
            categories_b = [part.strip() for part in categories_b.split(",") if part.strip()]
        parts["GenreCategoriesAnchor"] = categories_a
        parts["GenreCategoriesCandidate"] = categories_b
        cat_score, cat_overlap = _genre_category_score(categories_a, categories_b)
        if cat_overlap:
            parts["GenreCategoryOverlap"] = cat_overlap
        acc(cat_score, w.genre_category, "GenreCategoryScore")

        vocal_score = _vocal_overlap_penalty(a.get("Speechiness"), b.get("Speechiness"))
        acc(vocal_score, w.vocal_overlap, "VocalOverlapScore")

        if energy_delta is not None:
            parts["EnergyDelta"] = float(energy_delta)
        if valence_delta is not None:
            parts["ValenceDelta"] = float(valence_delta)
        if not _is_missing(a.get("Energy"), b.get("Energy")):
            acc(_bounded_diff_score(a["Energy"], b["Energy"]), w.energy, "EnergyScore")
        if not _is_missing(a.get("Danceability"), b.get("Danceability")):
            dance_score = _bounded_diff_score(a["Danceability"], b["Danceability"])
            acc(dance_score, w.danceability, "DanceabilityScore")
            groove_score = _groove_score(a.get("Danceability"), b.get("Danceability"), tempo_diff, tempo_factor)
            acc(groove_score, w.groove, "GrooveScore")

        energy_intent = str(context.get("energy_intent", "maintain") or "").strip().lower()
        parts["EnergyIntent"] = energy_intent or "maintain"
        if energy_delta is not None:
            energy_traj_score = _energy_trajectory_score(energy_delta, energy_intent)
            acc(energy_traj_score, w.energy_trajectory, "EnergyTrajectoryScore")

        if not _is_missing(a.get("Valence"), b.get("Valence")):
            acc(_bounded_diff_score(a["Valence"], b["Valence"]), w.valence, "ValenceScore")
        if not _is_missing(a.get("Loudness"), b.get("Loudness")):
            loudness_delta = float(b["Loudness"]) - float(a["Loudness"])
            parts["LoudnessDelta"] = loudness_delta
            acc(_loudness_score(a["Loudness"], b["Loudness"]), w.loudness, "LoudnessScore")
            parts["SuggestedGainTrim"] = max(-6.0, min(6.0, -loudness_delta))
        if not _is_missing(a.get("Acousticness"), b.get("Acousticness")):
            acc(_bounded_diff_score(a["Acousticness"], b["Acousticness"]), w.acousticness, "AcousticnessScore")
        if not _is_missing(a.get("Instrumentalness"), b.get("Instrumentalness")):
            acc(_bounded_diff_score(a["Instrumentalness"], b["Instrumentalness"]), w.instrumentalness, "InstrumentalnessScore")
        if not _is_missing(a.get("Liveness"), b.get("Liveness")):
            acc(_bounded_diff_score(a["Liveness"], b["Liveness"]), w.liveness, "LivenessScore")

        ts_score = _time_signature_alignment(
            a.get("Time Signature"),
            b.get("Time Signature"),
            a.get("Tempo"),
            b.get("Tempo"),
        )
        acc(ts_score, w.time_signature_alignment, "TimeSignatureScore")

        if total_den <= 0:
            return 0.0, parts
        overall = total_num / total_den
        overall = max(0.0, min(1.0, overall))
        return overall, parts

    # ---- public APIs ----
    def recommend(self, input_song: Dict[str, float], top_k: int = 20,
                  export_csv: bool = False, mix_mode: str = "conservative",
                  csv_path: Optional[Union[str,Path]] = None,
                  input_meta: Optional[Dict[str, Any]] = None,
                  scoring_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        anchor = {k: float(input_song[k]) for k in REQUIRED_FEATURES}
        anchor_meta = input_meta or self._extract_meta(input_song)
        if scoring_context is None:
            scoring_context = dict(self.default_context)
        else:
            merged = dict(self.default_context)
            merged.update(scoring_context)
            scoring_context = merged
        rows = []
        id_cols = [c for c in OPTIONAL_ID_COLS if c in self.df.columns]

        for idx, row in self.df.iterrows():
            cand = self._row_to_features(row)
            cand_meta = self._extract_meta(row)
            if all(not _is_missing(anchor[k], cand[k]) and abs(anchor[k]-cand[k])<1e-12 for k in REQUIRED_FEATURES):
                continue
            score, parts = self._matchscore(anchor, cand, meta_a=anchor_meta, meta_b=cand_meta, context=scoring_context)
            vol, eq, fx, note, ok = self._mix_recommendations(score, parts, mix_mode)
            out = {"MatchScore": score, **parts, "Volume": vol, "EQ": eq, "FX": fx, "Notes": note, "_row_index": idx}
            candidate_camelot = cand_meta.get("Camelot")
            if candidate_camelot is not None:
                out["Camelot"] = candidate_camelot
                if self.camelot_col and self.camelot_col != "Camelot":
                    out[self.camelot_col] = candidate_camelot
            if self.genre_col:
                out[self.genre_col] = cand_meta.get(self.genre_col)
            elif cand_meta.get("Genre") is not None:
                out["Genre"] = cand_meta.get("Genre")
            for key, col_name in self.timing_cols.items():
                label = TIMING_DISPLAY_NAMES.get(key, col_name)
                timing_val = cand_meta.get(f"Timing_{key}", cand_meta.get(label))
                if timing_val is not None:
                    out[label] = timing_val
            for c in id_cols: out[c] = row[c]
            rows.append(out)

        res = pd.DataFrame(rows).sort_values("MatchScore", ascending=False).reset_index(drop=True)
        if top_k and top_k > 0: res = res.head(top_k)

        if export_csv:
            csv_columns = [
                ID_COL_TRACKNAME,
                "Artist",
                "Camelot",
                "Genre",
                *[TIMING_DISPLAY_NAMES[key] for key in TIMING_DISPLAY_NAMES],
                "MatchScore",
                "CamelotScore",
                "CamelotRule",
                "CamelotRuleScore",
                "TempoScore",
                "GenreScore",
                "EnergyScore",
                "Volume",
                "EQ",
                "FX",
                "Notes",
            ]
            if self.genre_col and self.genre_col not in ("Genre",):
                csv_columns.insert(3, self.genre_col)
            keep = [col for col in csv_columns if col in res.columns]
            csv_df = res[keep].copy()
            if csv_path is None:
                anchor_name = input_song.get(ID_COL_TRACKNAME, None)
                fname = f"transitions_for_{re.sub(r'[^A-Za-z0-9 _.-]+','', str(anchor_name or 'anchor'))}.csv"
                csv_path = Path(fname)
            else:
                csv_path = Path(csv_path)
            print(f"Saving {csv_path.name} ... ", end="")
            csv_df.to_csv(csv_path, index=False)
            print("Done!")
        return res

    def recommend_by_name(self, track_name: str, top_k: int = 20,
                          export_csv: bool = False, mix_mode: str = "conservative",
                          csv_path: Optional[Union[str,Path]] = None,
                          scoring_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if ID_COL_TRACKNAME not in self.df.columns:
            raise ValueError(f"Expected '{ID_COL_TRACKNAME}' column.")
        mask = self.df[ID_COL_TRACKNAME].astype(str).str.lower() == str(track_name).strip().lower()
        if not mask.any():
            raise ValueError(f"Track '{track_name}' not found in '{ID_COL_TRACKNAME}'.")
        row = self.df[mask].iloc[0]
        anchor = self._row_to_features(row)
        anchor[ID_COL_TRACKNAME] = row[ID_COL_TRACKNAME]
        anchor_meta = self._extract_meta(row)
        return self.recommend(anchor, top_k=top_k, export_csv=export_csv, mix_mode=mix_mode,
                              csv_path=csv_path, input_meta=anchor_meta, scoring_context=scoring_context)

    def find_tracks_by_tempo_camelot(
        self,
        tempo: Optional[float] = None,
        camelot: Optional[str] = None,
        tolerance: float = 1.0,
        categories: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return tracks close to the requested tempo and Camelot code."""
        result = self.df.copy()
        mask = pd.Series(True, index=result.index)
        category_filter: Optional[Set[str]] = None
        if categories:
            category_filter = {c.strip() for c in categories if c and str(c).strip()}
            if category_filter:
                mask &= result["GenreCategories"].apply(
                    lambda cats: bool(set(cats if isinstance(cats, list) else [cats]) & category_filter)
                )

        if tempo is not None:
            try:
                tempo_value = float(tempo)
            except (TypeError, ValueError):
                raise ValueError("Tempo must be a number.")
            if tolerance is None:
                tolerance = 0.0
            if tolerance < 0:
                raise ValueError("Tempo tolerance must be zero or greater.")
            tempo_series = pd.to_numeric(result["Tempo"], errors="coerce")
            mask &= tempo_series.notna()
            mask &= tempo_series.sub(tempo_value).abs() <= float(tolerance)
        else:
            tempo_value = None

        if camelot:
            parsed = _parse_camelot_code(camelot)
            if not parsed:
                raise ValueError("Camelot code must look like '8A' or '10B'.")
            camelot_target = f"{parsed[0]}{parsed[1]}"
            camelot_series = result[self.camelot_col].astype(str).str.strip().str.upper()
            mask &= camelot_series == camelot_target
        filtered = result[mask].copy()

        if tempo_value is not None:
            tempo_series = pd.to_numeric(filtered["Tempo"], errors="coerce")
            filtered["TempoDiff"] = tempo_series.sub(tempo_value).abs()
            filtered = filtered.sort_values(["TempoDiff", ID_COL_TRACKNAME])
        else:
            filtered = filtered.sort_values(ID_COL_TRACKNAME)

        return filtered.reset_index(drop=True)

    def filter_by_categories(self, categories: Iterable[str]) -> "SongMatcher":
        category_set = {str(c).strip() for c in categories if c and str(c).strip()}
        if not category_set:
            return self
        mask = self.df["GenreCategories"].apply(
            lambda cats: bool(set(cats if isinstance(cats, list) else [cats]) & category_set)
        )
        filtered = self.df[mask].reset_index(drop=True)
        if filtered.empty:
            raise ValueError("No tracks remain after applying genre filters.")
        clone = SongMatcher(filtered, weights=self.weights, ignore_intros=False)
        clone.default_context = dict(self.default_context)
        return clone

    # ---- mix recommendations (Conservative/Creative) ----
    def _mix_recommendations(self, match: float, parts: Dict[str, Any], mode: str) -> Tuple[str,str,str,str,bool]:
        harmonic = parts.get("CamelotScore", parts.get("KeyScore", 0.5))
        ts = parts.get("TempoScore", 0.5)
        es = parts.get("EnergyScore", 0.5)
        safety = 0.55*harmonic + 0.35*ts + 0.10*es
        conservative = (mode or "conservative").lower().startswith("c")
        if conservative:
            if safety >= 0.90 and match >= 0.90:
                return ("smooth crossfade","none","none","Perfect harmonic flow — blend during breakdown or vocal intro.", True)
            if safety >= 0.82 and match >= 0.82:
                return ("centre out","3-band fade","low-pass filter out","Safe musical blend — bring next track in softly, swap bass near phrase end.", True)
            if safety >= 0.74 and match >= 0.74:
                return ("overlap","quick bass cut","high-pass filter in low-pass filter out","Use during non-vocal sections; manage bass energy to avoid clashes.", True)
            return ("NOT RECOMMENDED","NOT RECOMMENDED","NOT RECOMMENDED","Key/tempo misaligned — skip or hard switch only.", False)
        else:
            if safety >= 0.90 and match >= 0.90:
                return ("centre out","none","none","Butter-smooth. Optionally add echo tail on outgoing vocal for sparkle.", True)
            if safety >= 0.82 and match >= 0.82:
                return ("fade in fade out","centre bass swap","low-pass filter in high-pass filter out","Musical and lively — swap bass at phrase, sweep filters for lift.", True)
            if safety >= 0.74 and match >= 0.74:
                return ("cut in fade out","quick bass cut","high-pass filter in low-pass filter out","Punchy creative switch — avoid vocals; ride filters for impact.", True)
            if safety >= 0.68 and match >= 0.68:
                return ("fade in fast out","long bass cut","high-pass filter in","Edgier move — use during breakdown/build; prepare crowd for energy shift.", True)
            return ("NOT RECOMMENDED","NOT RECOMMENDED","NOT RECOMMENDED","Too far apart — only hard cut or skip.", False)


# ================================
# Pretty console
# ================================
def _colorize_row(row) -> str:
    ms = row.get("MatchScore", 0.0)
    name = row.get(ID_COL_TRACKNAME, "<unknown>")
    artist = _resolve_artist(row)
    genre = _resolve_genre(row)
    categories_raw = row.get("GenreCategoriesCandidate") or row.get("GenreCategories")
    if isinstance(categories_raw, str):
        categories = [part.strip() for part in categories_raw.split(",") if part.strip()]
    elif isinstance(categories_raw, Iterable):
        categories = [str(part).strip() for part in categories_raw if str(part).strip()]
    else:
        categories = []
    vol, eq, fx = row.get("Volume",""), row.get("EQ",""), row.get("FX","")
    notes = row.get("Notes","")
    camelot_rule = row.get("CamelotRule", "")
    camelot_display = f"Camelot rule: {camelot_rule}" if camelot_rule else ""
    if ms >= 0.9: color = "\033[92m"   # green
    elif ms >= 0.8: color = "\033[93m" # yellow
    else: color = "\033[91m"           # red
    reset = "\033[0m"
    header = f"{color}TRACK ▶ {name}{reset}"

    detail_parts = []
    if artist:
        detail_parts.append(f"Artist: {artist}")
    detail_parts.append(f"Match {ms:.2f}")
    if camelot_display:
        detail_parts.append(camelot_display)
    if genre:
        detail_parts.append(f"Genre: {genre}")
    if categories:
        detail_parts.append("Categories: " + ", ".join(categories[:3]))
    gain_trim = row.get("SuggestedGainTrim")
    if isinstance(gain_trim, (int, float)) and not isinstance(gain_trim, bool):
        detail_parts.append(f"Gain trim {gain_trim:+.2f} dB")
    detail_parts.extend([
        f"Vol: {vol}",
        f"EQ: {eq}",
        f"FX: {fx}",
    ])
    if notes:
        detail_parts.append(notes)

    details = " | ".join(part for part in detail_parts if part)

    timing_parts = []
    phrase_pair = row.get("PhraseAlignPair")
    phrase_offset = row.get("PhraseAlignOffset")
    if phrase_pair and not _is_missing(phrase_offset):
        timing_parts.append(f"Phrase align {phrase_pair} (Δ {float(phrase_offset):.2f}s)")
    for key, label in TIMING_DISPLAY_NAMES.items():
        value = row.get(label)
        formatted = _format_time_value(value)
        if formatted:
            timing_parts.append(f"{label}: {formatted}")

    lines = [header]
    if details:
        lines.append(f"    {details}")
    if timing_parts:
        lines.append("    Cues: " + " | ".join(timing_parts))
    return "\n".join(lines)


# ================================
# Name resolver (interactive)
# ================================
def resolve_track_name(df: pd.DataFrame, query: str) -> Optional[str]:
    col = ID_COL_TRACKNAME
    s = (query or "").strip()
    if not s:
        return None
    # exact (case-insensitive)
    mask = df[col].astype(str).str.lower() == s.lower()
    if mask.any(): return df.loc[mask, col].iloc[0]

    # suggestions (substring) including ASCII-normalised variant
    def _norm(x: str) -> str:
        return unicodedata.normalize("NFKD", str(x)).encode("ASCII","ignore").decode()

    near = df[col].astype(str)[df[col].astype(str).str.contains(re.escape(s), case=False, na=False)].unique().tolist()
    if not near:
        near = df[col].astype(str)[df[col].apply(_norm).str.contains(re.escape(_norm(s)), case=False, na=False)].unique().tolist()
    if near:
        print("\nCouldn't find an exact match. Did you mean:")
        for i, n in enumerate(near[:10], 1):
            print(f"  {i}. {n}")
        pick = input("Pick a number (or press Enter to cancel): ").strip()
        if pick.isdigit():
            i = int(pick)
            if 1 <= i <= min(10, len(near)):
                return near[i-1]
    return None


# ================================
# __main__
# ================================
if __name__ == "__main__":
    excel = Path("./🎧.xlsx")
    if not excel.exists():
        print("Put your Excel file named '🎧.xlsx' next to this script to run the demo.")
        sys.exit(0)

    matcher = SongMatcher.from_excel(excel, ignore_intros=True)
    print("Loaded:", len(matcher.df), "tracks.\n")
    print("Weights:\n", matcher.explain_weights().to_string(index=False))

    category_menu = [ALL_GENRES_OPTION] + GENRE_CATEGORY_ORDER + [DEFAULT_GENRE_CATEGORY]
    print("\nSelect genre focus (numbers, comma-separated). Enter handles multiple selections; choose '1' for all genres or press Enter to skip:")
    for idx, label in enumerate(category_menu, 1):
        print(f"  {idx}. {label}")
    genre_selection_raw = input("Chosen genres: ").strip()
    selected_categories: List[str] = []
    if genre_selection_raw:
        try:
            selected_categories = _parse_category_numbers(genre_selection_raw, category_menu)
        except ValueError as exc:
            print(exc)
            sys.exit(1)
    if selected_categories and ALL_GENRES_OPTION in selected_categories:
        selected_categories = []
    if selected_categories:
        try:
            matcher = matcher.filter_by_categories(selected_categories)
        except ValueError as exc:
            print(exc)
            sys.exit(0)
        print(f"\nFiltered library to {len(matcher.df)} tracks across: {', '.join(selected_categories)}")
    else:
        selected_categories = []

    style_names = list(DJ_STYLE_PRESETS.keys())
    print("\nDJ style presets:")
    for idx, name in enumerate(style_names, 1):
        print(f"  {idx}. {name}")
    style_pick = input("Select style preset (Enter=default weights): ").strip()
    chosen_style = None
    if style_pick:
        if style_pick.isdigit() and 1 <= int(style_pick) <= len(style_names):
            chosen_style = style_names[int(style_pick)-1]
        else:
            matches = [s for s in style_names if s.lower().startswith(style_pick.lower())]
            if len(matches) == 1:
                chosen_style = matches[0]
        if chosen_style:
            try:
                matcher.apply_style_preset(chosen_style)
                print(f"Applied preset: {chosen_style}")
            except ValueError as exc:
                print(f"Preset error: {exc}")
                chosen_style = None
        else:
            print("Unknown preset selection; keeping default weights.")

    print("\nEnergy trajectory intent (influences rise/cool scoring):")
    print("  1. Maintain")
    print("  2. Rise / build energy")
    print("  3. Cool down / reduce energy")
    energy_pick = input("Pick 1-3 (Enter=maintain): ").strip()
    energy_map = {"": "maintain", "1": "maintain", "2": "rise", "3": "cool"}
    energy_intent = energy_map.get(energy_pick, "maintain")
    matcher.default_context["energy_intent"] = energy_intent

    print("\nActions:")
    print("  1. Recommend transitions for a track")
    print("  2. List tracks by tempo and Camelot code")
    action = input("Select option [1/2] (Enter=1): ").strip()

    if action == "2":
        tempo_value: Optional[float] = None
        tolerance_value: float = 1.0
        listing_categories = selected_categories.copy()

        print("\nListing genre filters (numbers, comma-separated).")
        if selected_categories:
            print(f"Current selection: {', '.join(selected_categories)}")
        else:
            print("Press Enter to include all genres.")
        for idx, label in enumerate(category_menu, 1):
            print(f"  {idx}. {label}")
        listing_selection_raw = input("Listing genres (Enter=use current selection): ").strip()
        if listing_selection_raw:
            try:
                listing_categories = _parse_category_numbers(listing_selection_raw, category_menu)
            except ValueError as exc:
                print(exc)
                sys.exit(1)
            if ALL_GENRES_OPTION in listing_categories:
                listing_categories = []
        elif not selected_categories:
            listing_categories = []

        tempo_input = input("\nTarget tempo (BPM, leave blank to match any): ").strip()
        if tempo_input:
            try:
                tempo_value = float(tempo_input)
            except (TypeError, ValueError):
                print("Tempo must be a number.")
                sys.exit(1)
            tolerance_input = input("Tempo tolerance (+/- BPM, Enter=1.0): ").strip()
            if tolerance_input:
                try:
                    tolerance_value = float(tolerance_input)
                except (TypeError, ValueError):
                    print("Tempo tolerance must be a number.")
                    sys.exit(1)
            if tolerance_value < 0:
                print("Tempo tolerance must be zero or greater.")
                sys.exit(1)

        camelot_input = input("Camelot code (e.g., 8A, leave blank to match any): ").strip().upper()
        camelot_value = camelot_input if camelot_input else None

        try:
            results = matcher.find_tracks_by_tempo_camelot(
                tempo=tempo_value,
                camelot=camelot_value,
                tolerance=tolerance_value,
                categories=listing_categories or None,
            )
        except ValueError as exc:
            print("Lookup error:", exc)
            sys.exit(1)

        if results.empty:
            print("\nNo tracks found with that tempo and Camelot combination.")
            sys.exit(0)

        print(f"\nFound {len(results)} tracks:\n")
        for idx, row in results.iterrows():
            meta = matcher._extract_meta(row)
            name = _normalize_text_value(row.get(ID_COL_TRACKNAME, "<unknown>")) or "<unknown>"
            artist = _resolve_artist(row)
            tempo_val = row.get("Tempo")
            tempo_str = ""
            if not _is_missing(tempo_val):
                try:
                    tempo_float = float(tempo_val)
                    tempo_str = f"{tempo_float:.1f} BPM"
                except (TypeError, ValueError):
                    tempo_str = ""
            if tempo_value is not None:
                tempo_diff = row.get("TempoDiff")
                if not _is_missing(tempo_diff):
                    if tempo_str:
                        tempo_str = f"{tempo_str} (diff {float(tempo_diff):.2f})"
                    else:
                        tempo_str = f"Diff {float(tempo_diff):.2f}"
            camelot_val = _normalize_text_value(meta.get("Camelot") or row.get(matcher.camelot_col))
            camelot_str = f"Camelot {camelot_val}" if camelot_val and camelot_val.lower() != "nan" else ""
            genre_val = ""
            if matcher.genre_col:
                genre_val = _normalize_text_value(meta.get(matcher.genre_col))
            if not genre_val:
                genre_val = _normalize_text_value(meta.get("Genre"))
            categories_list = meta.get("GenreCategories") or row.get("GenreCategories")
            if isinstance(categories_list, str):
                categories_list = [part.strip() for part in categories_list.split(",") if part.strip()]
            categories_display = ", ".join(categories_list) if categories_list else ""
            details = [part for part in (tempo_str, camelot_str, genre_val, categories_display) if part]
            phrase_pair = row.get("PhraseAlignPair")
            phrase_offset = row.get("PhraseAlignOffset")
            if phrase_pair and not _is_missing(phrase_offset):
                details.append(f"Phrase {phrase_pair} Δ{float(phrase_offset):.2f}s")
            gain_trim = row.get("SuggestedGainTrim")
            if not _is_missing(gain_trim):
                details.append(f"Gain {float(gain_trim):+.2f} dB")
            artist_suffix = f" - {artist}" if artist else ""
            print(f"{idx+1}. {name}{artist_suffix}")
            if details:
                print(f"   {' | '.join(details)}")
        sys.exit(0)

    # Prompt for name (or use first track)
    user_input = input("\nEnter exact Track Name (or press Enter to use first track): ").strip()
    if user_input:
        name = resolve_track_name(matcher.df, user_input)
        if not name:
            print("No matching track. Tip: copy the title exactly from Excel (watch hyphens/plus signs).")
            sys.exit(1)
        anchor_name = name
    else:
        row0 = matcher.df.iloc[0]
        anchor_name = row0.get(ID_COL_TRACKNAME, "<unknown>")

    # Optional heads-up if anchor is missing key/mode
    anchor_row = None
    try:
        mask_anchor = matcher.df[ID_COL_TRACKNAME].astype(str).str.lower() == anchor_name.lower()
        if mask_anchor.any():
            anchor_row = matcher.df[mask_anchor].iloc[0]
        if anchor_row is not None and _is_missing(anchor_row.get("Key"), anchor_row.get("Mode")):
            print("⚠ Selected track is missing Key/Mode – ranking will rely on Tempo/Energy.")
    except Exception:
        anchor_row = None
        pass

    if anchor_row is not None:
        print("\n=== Anchor Track ===")
        print(matcher.describe_track(anchor_row))
        print("\nRecommendations:\n")
    else:
        print(f"\nTesting recommendations for: {anchor_name}\n")
        print("Recommendations:\n")

    # Mode prompt (defaults to conservative)
    mode_in = input("Mix mode [conservative/creative] (Enter=conservative): ").strip().lower()
    mix_mode = "creative" if mode_in.startswith("c") is False and mode_in.startswith("cr") else "conservative"
    # CSV prompt
    csv_in = input("Export clean CSV? [y/N]: ").strip().lower()
    export_csv = csv_in == "y"

    scoring_payload = {"energy_intent": energy_intent}
    try:
        recs = matcher.recommend_by_name(anchor_name, top_k=10, export_csv=export_csv, mix_mode=mix_mode,
                                         scoring_context=scoring_payload)
    except Exception as e:
        print("Lookup error:", e)
        sys.exit(1)

    if recs is not None and not recs.empty:
        for _, r in recs.iterrows():
            print(_colorize_row(r))
    else:
        print("No recommendations found.")

    # Camelot suggestions
    if anchor_row is not None:
        try:
            sk = suggest_keys_camelot(float(anchor_row.get("Key", np.nan)), float(anchor_row.get("Mode", np.nan)))
            if sk and "Unknown" not in sk[0]:
                print("\nSuggested Keys (Camelot):", ", ".join(sk))
            else:
                print("\nSuggested Keys (Camelot): Unknown – missing key/mode in this row.")
        except Exception:
            print("\nSuggested Keys (Camelot): Unknown – could not resolve anchor row.")

    print("\nDone.")
