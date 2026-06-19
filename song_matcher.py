
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import math
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------
# Utility: column handling
# ---------------------------

CANON_FEATURES = [
    "Danceability","Energy","Key","Loudness","Mode","Speechiness",
    "Acousticness","Instrumentalness","Liveness","Valence","Tempo","Time Signature"
]

OPTIONAL_ID_COLS = [
    "Name","Track","Track Name","Title","Song","Artist","Artists","Album"
]


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip spaces and normalise case for matching, but keep originals where possible
    mapping = {}
    for col in df.columns:
        key = col.strip().lower().replace("_"," ").replace("-"," ").replace("  "," ")
        mapping[key] = col

    def resolve(name: str) -> Optional[str]:
        key = name.strip().lower().replace("_"," ").replace("-"," ").replace("  "," ")
        return mapping.get(key, None)

    # Ensure required features exist
    missing = []
    resolved = {}
    for feat in CANON_FEATURES:
        col = resolve(feat)
        if col is None:
            missing.append(feat)
        else:
            resolved[feat] = col

    if missing:
        raise ValueError(
            "Your Excel file is missing required columns: "
            + ", ".join(missing)
            + ".\nMake sure your sheet has the exact feature set:\n"
            + ", ".join(CANON_FEATURES)
        )

    # Bring features in desired order + keep any ID-like columns (if present)
    ordered_cols = [resolved[f] for f in CANON_FEATURES]

    id_cols_present = []
    for cand in OPTIONAL_ID_COLS:
        col = resolve(cand)
        if col is not None and col not in ordered_cols:
            id_cols_present.append(col)

    # Rebuild dataframe with id cols first (if exist), then features (ordered), then the rest
    keep_cols = id_cols_present + ordered_cols
    others = [c for c in df.columns if c not in keep_cols]
    return df[keep_cols + others].copy()


# ---------------------------
# Scoring components
# ---------------------------

def _key_mode_score(key_a: float, mode_a: float, key_b: float, mode_b: float) -> float:
    """
    Returns a score in [0,1] reflecting harmonic compatibility between two songs.

    - Exact same key & mode -> 1.00
    - Same key, different mode (relative major/minor) -> 0.92
    - Perfect 5th (±7 semitones), same mode -> 0.88
    - Perfect 5th, relative mode -> 0.82
    - Camelot-adjacent (±1 on wheel) approximation -> 0.80
    - Otherwise: 1 - (circular semitone distance / 12) * 0.9  (max 0.9 decrement)
    """
    try:
        k1 = int(round(key_a)) % 12
        k2 = int(round(key_b)) % 12
        m1 = int(round(mode_a)) % 2
        m2 = int(round(mode_b)) % 2
    except Exception:
        return 0.5  # fallback

    same_key = (k1 == k2)
    same_mode = (m1 == m2)
    rel_mode = (m1 != m2)

    # Perfect fifth distance in semitones (±7)
    semi_diff = min((k1 - k2) % 12, (k2 - k1) % 12)

    if same_key and same_mode:
        return 1.00
    if same_key and rel_mode:
        return 0.92

    if semi_diff == 7:  # fifth
        return 0.88 if same_mode else 0.82

    # Camelot "adjacent" approximation: semitone distance of 2 (a whole tone) gets a small boost
    if semi_diff in (1, 2):
        # closer gets a slightly higher score
        return 0.80 if semi_diff == 1 else 0.78

    # Generic circular-key closeness
    base = 1.0 - (semi_diff / 12.0) * 0.9
    return max(0.0, min(0.95, base))


def _tempo_score(bpm_a: float, bpm_b: float) -> float:
    """
    Score in [0,1] for tempo closeness, allowing for halving/doubling (mixers often do this).
    We consider the minimum relative error across {b, 2b, b/2} against a.
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.5

    candidates = [bpm_b, bpm_b * 2.0, bpm_b / 2.0]
    best = 0.0
    for cand in candidates:
        rel_err = abs(bpm_a - cand) / max(bpm_a, cand)
        # Map relative error 0 -> 1.0, 10% -> ~0.7, 20% -> ~0.4, 30%+ -> lower
        score = max(0.0, 1.0 - 3.0 * rel_err)
        if score > best:
            best = score
    return best


def _bounded_diff_score(a: float, b: float, span: float = 1.0) -> float:
    """
    Generic similarity for features expected in [0,1] ranges (Energy, Danceability, etc.).
    score = 1 - |a - b| / span, clamped to [0,1].
    """
    if any(map(lambda x: x is None or np.isnan(x), [a, b])):
        return 0.5
    return max(0.0, min(1.0, 1.0 - abs(a - b) / float(span)))


def _loudness_score(lufs_a: float, lufs_b: float) -> float:
    """
    Loudness in dB, more negative -> quieter.
    Treat +/- 2 dB as very close (~0.85+), then decay.
    """
    if any(map(lambda x: x is None or np.isnan(x), [lufs_a, lufs_b])):
        return 0.5
    diff = abs(lufs_a - lufs_b)
    if diff <= 1.0:
        return 0.95
    if diff <= 2.0:
        return 0.88
    # beyond 6 dB, it's quite different
    return max(0.0, 0.88 - (diff - 2.0) * 0.10)


# ---------------------------
# Main Recommender
# ---------------------------

@dataclass
class Weights:
    key: float = 3.0
    tempo: float = 2.5
    energy: float = 2.0
    danceability: float = 1.5
    valence: float = 1.0
    loudness: float = 0.8
    speechiness: float = 0.5
    acousticness: float = 0.5
    instrumentalness: float = 0.5
    liveness: float = 0.5
    time_signature: float = 0.3


class SongMatcher:
    def __init__(self, df: pd.DataFrame, weights: Optional[Weights] = None):
        self.df = _canonicalise_columns(df)
        self.weights = weights or Weights()

        # Cache feature matrix
        self.features = self.df[[
            "Danceability","Energy","Key","Loudness","Mode","Speechiness",
            "Acousticness","Instrumentalness","Liveness","Valence","Tempo","Time Signature"
        ]].astype(float)

    @classmethod
    def from_excel(cls, path: Union[str, Path], sheet_name: Optional[Union[str,int]] = 0) -> "SongMatcher":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        # openpyxl supports .xlsm as well
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        return cls(df)

    def _row_to_feature_dict(self, row: pd.Series) -> Dict[str, float]:
        return {k: float(row[k]) for k in [
            "Danceability","Energy","Key","Loudness","Mode","Speechiness",
            "Acousticness","Instrumentalness","Liveness","Valence","Tempo","Time Signature"
        ]}

    def score_against(self, anchor: Dict[str, float], candidate: Dict[str, float]) -> float:
        # Individual component scores
        s_key   = _key_mode_score(anchor["Key"], anchor["Mode"], candidate["Key"], candidate["Mode"])
        s_tempo = _tempo_score(anchor["Tempo"], candidate["Tempo"])
        s_energy= _bounded_diff_score(anchor["Energy"], candidate["Energy"])
        s_dance = _bounded_diff_score(anchor["Danceability"], candidate["Danceability"])
        s_val   = _bounded_diff_score(anchor["Valence"], candidate["Valence"])
        s_loud  = _loudness_score(anchor["Loudness"], candidate["Loudness"])
        s_speech= _bounded_diff_score(anchor["Speechiness"], candidate["Speechiness"])
        s_acous = _bounded_diff_score(anchor["Acousticness"], candidate["Acousticness"])
        s_instr = _bounded_diff_score(anchor["Instrumentalness"], candidate["Instrumentalness"])
        s_live  = _bounded_diff_score(anchor["Liveness"], candidate["Liveness"])
        s_ts    = 1.0 if int(round(anchor["Time Signature"])) == int(round(candidate["Time Signature"])) else 0.6

        w = self.weights
        num = (
            w.key * s_key +
            w.tempo * s_tempo +
            w.energy * s_energy +
            w.danceability * s_dance +
            w.valence * s_val +
            w.loudness * s_loud +
            w.speechiness * s_speech +
            w.acousticness * s_acous +
            w.instrumentalness * s_instr +
            w.liveness * s_live +
            w.time_signature * s_ts
        )
        den = (
            w.key + w.tempo + w.energy + w.danceability + w.valence + w.loudness +
            w.speechiness + w.acousticness + w.instrumentalness + w.liveness + w.time_signature
        )
        return float(num / den)  # 0..1

    def recommend(
        self,
        input_song: Dict[str, float],
        top_k: int = 20,
        exclude_exact_match: bool = True,
        id_columns_preference: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
        [<ID cols...>, 'MatchScore', 'KeyScore','TempoScore','EnergyScore', ...] sorted by MatchScore desc.
        """
        # Prepare anchor vector with all required fields
        missing = [f for f in [
            "Danceability","Energy","Key","Loudness","Mode","Speechiness",
            "Acousticness","Instrumentalness","Liveness","Valence","Tempo","Time Signature"
        ] if f not in input_song]
        if missing:
            raise ValueError("Input song missing fields: " + ", ".join(missing))

        anchor = {k: float(input_song[k]) for k in [
            "Danceability","Energy","Key","Loudness","Mode","Speechiness",
            "Acousticness","Instrumentalness","Liveness","Valence","Tempo","Time Signature"
        ]}

        # Build per-candidate scores
        rows = []
        for idx, row in self.df.iterrows():
            cand = self._row_to_feature_dict(row)

            # Optionally skip if identical across all features (likely the same track)
            if exclude_exact_match:
                if all(abs(anchor[k] - cand[k]) < 1e-9 for k in anchor.keys()):
                    continue

            key_s   = _key_mode_score(anchor["Key"], anchor["Mode"], cand["Key"], cand["Mode"])
            tempo_s = _tempo_score(anchor["Tempo"], cand["Tempo"])
            energy_s= _bounded_diff_score(anchor["Energy"], cand["Energy"])
            dance_s = _bounded_diff_score(anchor["Danceability"], cand["Danceability"])
            val_s   = _bounded_diff_score(anchor["Valence"], cand["Valence"])
            loud_s  = _loudness_score(anchor["Loudness"], cand["Loudness"])
            speech_s= _bounded_diff_score(anchor["Speechiness"], cand["Speechiness"])
            acous_s = _bounded_diff_score(anchor["Acousticness"], cand["Acousticness"])
            instr_s = _bounded_diff_score(anchor["Instrumentalness"], cand["Instrumentalness"])
            live_s  = _bounded_diff_score(anchor["Liveness"], cand["Liveness"])
            ts_s    = 1.0 if int(round(anchor["Time Signature"])) == int(round(cand["Time Signature"])) else 0.6

            match = self.score_against(anchor, cand)

            # Collect ID columns (if present)
            id_cols = []
            pref = id_columns_preference or ["Name","Track","Track Name","Title","Song","Artist","Album"]
            for c in pref:
                if c in self.df.columns:
                    id_cols.append(c)

            out = {
                "MatchScore": match,
                "KeyScore": key_s,
                "TempoScore": tempo_s,
                "EnergyScore": energy_s,
                "DanceabilityScore": dance_s,
                "ValenceScore": val_s,
                "LoudnessScore": loud_s,
                "SpeechinessScore": speech_s,
                "AcousticnessScore": acous_s,
                "InstrumentalnessScore": instr_s,
                "LivenessScore": live_s,
                "TimeSignatureScore": ts_s,
                "_row_index": idx
            }
            for col in id_cols:
                out[col] = row[col]
            rows.append(out)

        res = pd.DataFrame(rows)
        res = res.sort_values("MatchScore", ascending=False).reset_index(drop=True)
        if top_k is not None and top_k > 0:
            res = res.head(top_k)
        return res

    def explain_weights(self) -> pd.DataFrame:
        """Return a tidy table of current weights for display."""
        w = self.weights
        data = {
            "Factor": ["Key","Tempo","Energy","Danceability","Valence","Loudness",
                       "Speechiness","Acousticness","Instrumentalness","Liveness","Time Signature"],
            "Weight": [w.key, w.tempo, w.energy, w.danceability, w.valence, w.loudness,
                       w.speechiness, w.acousticness, w.instrumentalness, w.liveness, w.time_signature]
        }
        return pd.DataFrame(data).sort_values("Weight", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Example quick start: read ./🎧.xlsm and run a demo using the first track as the anchor.
    excel_path = Path("./🎧.xlsm")
    if not excel_path.exists():
        print("Put your Excel file named '🎧.xlsm' next to this script to run the demo.")
    else:
        matcher = SongMatcher.from_excel(excel_path)
        print("Loaded sheet with", len(matcher.df), "rows. Current weights:")
        print(matcher.explain_weights())

        # Use the first row as the anchor example
        row0 = matcher.df.iloc[0]
        anchor = {
            "Danceability": float(row0["Danceability"]),
            "Energy": float(row0["Energy"]),
            "Key": float(row0["Key"]),
            "Loudness": float(row0["Loudness"]),
            "Mode": float(row0["Mode"]),
            "Speechiness": float(row0["Speechiness"]),
            "Acousticness": float(row0["Acousticness"]),
            "Instrumentalness": float(row0["Instrumentalness"]),
            "Liveness": float(row0["Liveness"]),
            "Valence": float(row0["Valence"]),
            "Tempo": float(row0["Tempo"]),
            "Time Signature": float(row0["Time Signature"]),
        }

        recs = matcher.recommend(anchor, top_k=15)
        # Display selected columns nicely if present
        display_cols = [c for c in ["Name","Artist","Album","MatchScore","KeyScore","TempoScore","EnergyScore"] if c in recs.columns]
        if display_cols:
            print(recs[display_cols].to_string(index=False))
        else:
            print(recs.head(15).to_string(index=False))
