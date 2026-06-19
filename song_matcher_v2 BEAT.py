# song_matcher_v3.py
from __future__ import annotations

import sys, re, unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
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

def _tempo_score(bpm_a: float, bpm_b: float) -> float:
    if bpm_a <= 0 or bpm_b <= 0: return 0.5
    cands = [bpm_b, bpm_b*2.0, bpm_b/2.0]
    best = 0.0
    for c in cands:
        rel = abs(bpm_a - c) / max(bpm_a, c)
        score = max(0.0, 1.0 - 4.0*rel)  # ~10% -> 0.6
        if score > best: best = score
    return best

def _bounded_diff_score(a: float, b: float, span: float = 1.0) -> float:
    if _is_missing(a,b): return 0.5
    return max(0.0, min(1.0, 1.0 - abs(a-b)/float(span)))

def _loudness_score(a: float, b: float) -> float:
    if _is_missing(a,b): return 0.5
    diff = abs(a-b)
    if diff <= 1.0: return 0.95
    if diff <= 2.0: return 0.88
    return max(0.0, 0.88 - (diff-2.0)*0.10)


# ================================
# Weights
# ================================
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


# ================================
# Camelot helpers (NaN-safe)
# ================================
_CAMELOT_MAP_MAJOR = {0:"8B",1:"3B",2:"10B",3:"5B",4:"12B",5:"7B",6:"2B",7:"9B",8:"4B",9:"11B",10:"6B",11:"1B"}
_CAMELOT_MAP_MINOR = {0:"5A",1:"12A",2:"7A",3:"2A",4:"9A",5:"4A",6:"11A",7:"6A",8:"1A",9:"8A",10:"3A",11:"10A"}

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
        self.weights = weights or Weights()

    @classmethod
    def from_excel(cls, path: Union[str, Path], sheet_name: Optional[Union[str,int]] = 0, **kwargs):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        return cls(df, **kwargs)

    def _row_to_features(self, row: pd.Series) -> Dict[str, float]:
        return {k: float(row[k]) for k in REQUIRED_FEATURES}

    def explain_weights(self) -> pd.DataFrame:
        w = self.weights
        data = {"Factor":["Key","Tempo","Energy","Danceability","Valence","Loudness",
                          "Speechiness","Acousticness","Instrumentalness","Liveness","Time Signature"],
                "Weight":[w.key,w.tempo,w.energy,w.danceability,w.valence,w.loudness,
                          w.speechiness,w.acousticness,w.instrumentalness,w.liveness,w.time_signature]}
        return pd.DataFrame(data).sort_values("Weight", ascending=False).reset_index(drop=True)

    # ---- dynamic-weight matchscore (skips missing features) ----
    def _matchscore(self, a: Dict[str,float], b: Dict[str,float]) -> Tuple[float, Dict[str,float]]:
        parts: Dict[str, float] = {}
        total_num = 0.0
        total_den = 0.0
        w = self.weights

        def acc(score, weight, label):
            nonlocal total_num, total_den
            parts[label] = score
            total_num += weight * score
            total_den += weight

        if not _is_missing(a.get("Key"), a.get("Mode"), b.get("Key"), b.get("Mode")):
            acc(_key_mode_score(a["Key"], a["Mode"], b["Key"], b["Mode"]), w.key, "KeyScore")
        if not _is_missing(a.get("Tempo"), b.get("Tempo")):
            acc(_tempo_score(a["Tempo"], b["Tempo"]), w.tempo, "TempoScore")
        if not _is_missing(a.get("Energy"), b.get("Energy")):
            acc(_bounded_diff_score(a["Energy"], b["Energy"]), w.energy, "EnergyScore")
        if not _is_missing(a.get("Danceability"), b.get("Danceability")):
            acc(_bounded_diff_score(a["Danceability"], b["Danceability"]), w.danceability, "DanceabilityScore")
        if not _is_missing(a.get("Valence"), b.get("Valence")):
            acc(_bounded_diff_score(a["Valence"], b["Valence"]), w.valence, "ValenceScore")
        if not _is_missing(a.get("Loudness"), b.get("Loudness")):
            acc(_loudness_score(a["Loudness"], b["Loudness"]), w.loudness, "LoudnessScore")
        if not _is_missing(a.get("Speechiness"), b.get("Speechiness")):
            acc(_bounded_diff_score(a["Speechiness"], b["Speechiness"]), w.speechiness, "SpeechinessScore")
        if not _is_missing(a.get("Acousticness"), b.get("Acousticness")):
            acc(_bounded_diff_score(a["Acousticness"], b["Acousticness"]), w.acousticness, "AcousticnessScore")
        if not _is_missing(a.get("Instrumentalness"), b.get("Instrumentalness")):
            acc(_bounded_diff_score(a["Instrumentalness"], b["Instrumentalness"]), w.instrumentalness, "InstrumentalnessScore")
        if not _is_missing(a.get("Liveness"), b.get("Liveness")):
            acc(_bounded_diff_score(a["Liveness"], b["Liveness"]), w.liveness, "LivenessScore")
        if not _is_missing(a.get("Time Signature"), b.get("Time Signature")):
            try:
                tsig = 1.0 if int(round(a["Time Signature"])) == int(round(b["Time Signature"])) else 0.6
            except Exception:
                tsig = 0.6
            acc(tsig, w.time_signature, "TimeSignatureScore")

        overall = float(total_num/total_den) if total_den > 0 else 0.0
        return overall, parts

    # ---- public APIs ----
    def recommend(self, input_song: Dict[str,float], top_k: int = 20,
                  export_csv: bool = False, mix_mode: str = "conservative",
                  csv_path: Optional[Union[str,Path]] = None) -> pd.DataFrame:
        anchor = {k: float(input_song[k]) for k in REQUIRED_FEATURES}
        rows = []
        id_cols = [c for c in OPTIONAL_ID_COLS if c in self.df.columns]

        for idx, row in self.df.iterrows():
            cand = self._row_to_features(row)
            if all(not _is_missing(anchor[k], cand[k]) and abs(anchor[k]-cand[k])<1e-12 for k in REQUIRED_FEATURES):
                continue
            score, parts = self._matchscore(anchor, cand)
            vol, eq, fx, note, ok = self._mix_recommendations(score, parts, mix_mode)
            out = {"MatchScore": score, **parts, "Volume": vol, "EQ": eq, "FX": fx, "Notes": note, "_row_index": idx}
            for c in id_cols: out[c] = row[c]
            rows.append(out)

        res = pd.DataFrame(rows).sort_values("MatchScore", ascending=False).reset_index(drop=True)
        if top_k and top_k > 0: res = res.head(top_k)

        if export_csv:
            keep = [col for col in [ID_COL_TRACKNAME,"Artist","MatchScore","KeyScore","TempoScore","EnergyScore","Volume","EQ","FX","Notes"] if col in res.columns]
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
                          csv_path: Optional[Union[str,Path]] = None) -> pd.DataFrame:
        if ID_COL_TRACKNAME not in self.df.columns:
            raise ValueError(f"Expected '{ID_COL_TRACKNAME}' column.")
        mask = self.df[ID_COL_TRACKNAME].astype(str).str.lower() == str(track_name).strip().lower()
        if not mask.any():
            raise ValueError(f"Track '{track_name}' not found in '{ID_COL_TRACKNAME}'.")
        row = self.df[mask].iloc[0]
        anchor = self._row_to_features(row)
        anchor[ID_COL_TRACKNAME] = row[ID_COL_TRACKNAME]
        return self.recommend(anchor, top_k=top_k, export_csv=export_csv, mix_mode=mix_mode, csv_path=csv_path)

    # ---- mix recommendations (Conservative/Creative) ----
    def _mix_recommendations(self, match: float, parts: Dict[str,float], mode: str) -> Tuple[str,str,str,str,bool]:
        ks, ts, es = parts.get("KeyScore",0.5), parts.get("TempoScore",0.5), parts.get("EnergyScore",0.5)
        safety = 0.5*ks + 0.4*ts + 0.1*es
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
    artist = row.get("Artist", "")
    vol, eq, fx = row.get("Volume",""), row.get("EQ",""), row.get("FX","")
    notes = row.get("Notes","")
    if ms >= 0.9: color = "\033[92m"   # green
    elif ms >= 0.8: color = "\033[93m" # yellow
    else: color = "\033[91m"           # red
    reset = "\033[0m"
    return f"{color}{name} — {artist} | Match {ms:.2f} | Vol: {vol} | EQ: {eq} | FX: {fx} | {notes}{reset}"


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
    excel = Path("./🎧.xlsm")
    if not excel.exists():
        print("Put your Excel file named '🎧.xlsm' next to this script to run the demo.")
        sys.exit(0)

    matcher = SongMatcher.from_excel(excel, ignore_intros=True)
    print("Loaded:", len(matcher.df), "tracks.\n")
    print("Weights:\n", matcher.explain_weights().to_string(index=False))

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
    try:
        ar = matcher.df[matcher.df[ID_COL_TRACKNAME].astype(str).str.lower() == anchor_name.lower()].iloc[0]
        if _is_missing(ar.get("Key"), ar.get("Mode")):
            print("⚠ Selected track is missing Key/Mode — ranking will rely on Tempo/Energy.")
    except Exception:
        pass

    print(f"\nTesting recommendations for: {anchor_name}\n")

    # Mode prompt (defaults to conservative)
    mode_in = input("Mix mode [conservative/creative] (Enter=conservative): ").strip().lower()
    mix_mode = "creative" if mode_in.startswith("c") is False and mode_in.startswith("cr") else "conservative"
    # CSV prompt
    csv_in = input("Export clean CSV? [y/N]: ").strip().lower()
    export_csv = csv_in == "y"

    try:
        recs = matcher.recommend_by_name(anchor_name, top_k=10, export_csv=export_csv, mix_mode=mix_mode)
    except Exception as e:
        print("Lookup error:", e)
        sys.exit(1)

    if recs is not None and not recs.empty:
        for _, r in recs.iterrows():
            print(_colorize_row(r))
    else:
        print("No recommendations found.")

    # Camelot suggestions
    try:
        sk = suggest_keys_camelot(float(ar.get("Key", np.nan)), float(ar.get("Mode", np.nan)))
        if sk and "Unknown" not in sk[0]:
            print("\nSuggested Keys (Camelot):", ", ".join(sk))
        else:
            print("\nSuggested Keys (Camelot): Unknown — missing key/mode in this row.")
    except Exception:
        print("\nSuggested Keys (Camelot): Unknown — could not resolve anchor row.")

    print("\nDone.")
