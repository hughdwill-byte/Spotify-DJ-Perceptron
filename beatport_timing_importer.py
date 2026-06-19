"""
beatport_timing_importer.py
---------------------------------
Create a NEW workbook with three timing columns placed at fixed positions:
  - Column Y = IntroEnd
  - Column Z = DropPoint
  - Column AA = OutroStart

It reads your source Excel (default: 🎧.xlsm), computes timing using genre/BPM
templates (no Spotify), and writes a copy:
  <original_name>_TIMED.xlsx

ENV options (PowerShell examples):
  setx DJ_EXCEL  "C:\path\to\🎧.xlsm"   # input file
  setx DJ_SHEET  "0"                    # sheet index OR exact name
  setx DJ_LIMIT  "0"                    # process first N rows (0 = all)
  setx DJ_FAST_ONLY "1"                 # skip web lookups; use only local BPM/Genres
"""

import os, re, sys, time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests

# ------------ Config / ENV ------------
EXCEL_FILE = os.environ.get("DJ_EXCEL", "🎧.xlsm")
SHEET_ENV  = os.environ.get("DJ_SHEET", "0")
SHEET_NAME = int(SHEET_ENV) if SHEET_ENV.isdigit() else SHEET_ENV

DJ_LIMIT      = int(os.environ.get("DJ_LIMIT", "0"))               # 0 = no cap
DJ_FAST_ONLY  = os.environ.get("DJ_FAST_ONLY", "1") == "1"         # default fast/no web

COL_TRACK  = "Track Name"
COL_ARTIST = "Artist"
COL_TEMPO  = "Tempo"
COL_GENRES = "Genres"

# EXACT output positions: Y(24), Z(25), AA(26) – zero-indexed
Y_IDX, Z_IDX, AA_IDX = 24, 25, 26

OUT_COLS = ["IntroEnd", "DropPoint", "OutroStart"]

# ------------ Genre families ------------
FAM_MAP = {
    "house": {"house","tech house","deep house","progressive house","electro house",
              "funky house","minimal house","organic house","bass house","tribal house"},
    "techno": {"techno","melodic techno","hard techno","peak time techno","minimal techno","raw techno"},
    "trance": {"trance","uplifting trance","progressive trance","psytrance"},
    "dnb": {"drum & bass","dnb","liquid dnb","neurofunk","jump up","drum and bass"},
    "hiphop": {"hip hop","hip-hop","rap","trap"},
    "pop": {"pop","dance pop","synthpop","indie pop"},
    "breaks": {"breaks","breakbeat"},
    "edm": {"big room","electro","mainstage","edm"},
}

# ------------ Templates (bars at 4/4) ------------
TEMPLATE = {
    "house":  {"default_bpm":124, "intro_bars":32, "drop_bars":64, "outro_bars":32},
    "techno": {"default_bpm":132, "intro_bars":32, "drop_bars":64, "outro_bars":32},
    "trance": {"default_bpm":136, "intro_bars":64, "drop_bars":96, "outro_bars":64},
    "dnb":    {"default_bpm":174, "intro_bars":16, "drop_bars":48, "outro_bars":16},
    "hiphop": {"default_bpm":94,  "intro_bars":8,  "drop_bars":24, "outro_bars":8},
    "pop":    {"default_bpm":110, "intro_bars":8,  "drop_bars":24, "outro_bars":8},
    "breaks": {"default_bpm":132, "intro_bars":16, "drop_bars":48, "outro_bars":16},
    "edm":    {"default_bpm":128, "intro_bars":16, "drop_bars":32, "outro_bars":16},
    "generic":{"default_bpm":124, "intro_bars":16, "drop_bars":48, "outro_bars":16},
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ------------ Helpers ------------
def normalise_genres(s: str) -> set:
    toks = re.split(r"[,/;|]+", (s or "").lower())
    return {t.strip() for t in toks if t.strip()}

def family_of(genres: set) -> str:
    for fam, members in FAM_MAP.items():
        if genres & members:
            return fam
    joined = " ".join(genres)
    for fam, members in FAM_MAP.items():
        if any(m in joined for m in members):
            return fam
    return "generic"

def bars_to_seconds(bars: int, bpm: float) -> float:
    # 1 bar = 4 beats; seconds/beat = 60/bpm
    return max(0.0, 4.0 * bars * 60.0 / max(1.0, bpm))

def clamp_time(t: float, duration: Optional[float]) -> float:
    if duration and np.isfinite(duration):
        return float(min(max(0.0, t), max(0.0, duration - 1.0)))
    return float(max(0.0, t))

def mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s//60}:{s%60:02d}"

def infer_timings(bpm: Optional[float], family: str, duration: Optional[float]) -> tuple[str, str, str]:
    cfg = TEMPLATE.get(family, TEMPLATE["generic"])
    b = float(bpm if bpm and bpm > 0 else cfg["default_bpm"])

    intro_s = bars_to_seconds(cfg["intro_bars"], b)
    drop_s  = bars_to_seconds(cfg["drop_bars"], b)
    # If we know duration, put outro start 'outro_bars' before end; else estimate after drop+intro
    outro_s = (duration - bars_to_seconds(cfg["outro_bars"], b)) if (duration and duration > 0) \
              else bars_to_seconds(cfg["drop_bars"] + cfg["intro_bars"], b)

    intro_s = clamp_time(intro_s, duration)
    drop_s  = clamp_time(drop_s, duration)
    outro_s = clamp_time(outro_s, duration)
    return mmss(intro_s), mmss(drop_s), mmss(outro_s)

def beatport_search(track: str, artist: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort public lookup for BPM/genre/duration. Not required in FAST mode."""
    q = track if not artist else f"{track} {artist}"
    url = "https://www.beatport.com/search"
    try:
        r = requests.get(url, params={"q": q}, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception:
        return None
    best = {"bpm": None, "genres": None, "duration_s": None}
    try:
        bpm_m = re.findall(r'"bpm"\s*:\s*"?([0-9]{2,3})"?', r.text, flags=re.IGNORECASE)
        if bpm_m: best["bpm"] = float(bpm_m[0])
        dur_m = re.findall(r'"duration"\s*:\s*([0-9]{2,5})', r.text, flags=re.IGNORECASE)
        if dur_m: best["duration_s"] = float(dur_m[0])
        g_m = re.findall(r'"genre"\s*:\s*"\s*([^"]+)\s*"', r.text, flags=re.IGNORECASE)
        if g_m: best["genres"] = ", ".join(sorted(set(g_m[:3])))
    except Exception:
        pass
    return best

def ensure_positions_with_placeholders(df: pd.DataFrame, target_indices: list[int]) -> pd.DataFrame:
    """
    Ensures the DataFrame has at least (max(target_indices)+1) columns by adding placeholder
    columns if needed, so we can reliably insert at Y,Z,AA. Placeholders are named _FILL_x.
    """
    needed_cols = max(target_indices) + 1
    current = len(df.columns)
    if current < needed_cols:
        for k in range(needed_cols - current):
            df[f"_FILL_{k+1}"] = ""
    return df

def insert_at_positions(df: pd.DataFrame, pairs: list[tuple[int, str, list[str]]]) -> pd.DataFrame:
    """
    Insert new columns at fixed positions (0-indexed). To keep positions correct, insert
    in descending order of target index (so earlier inserts don't shift later targets).
    pairs: list of (index, column_name, values_list)
    """
    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
    for pos, col_name, col_values in pairs_sorted:
        df.insert(loc=pos, column=col_name, value=col_values)
    return df

# ------------ Main ------------
def main():
    xfile = Path(EXCEL_FILE)
    if not xfile.exists():
        print(f"Input Excel not found: {xfile.resolve()}")
        sys.exit(1)

    print(f"Loading {xfile.name} ...")
    xls = pd.ExcelFile(xfile, engine="openpyxl")

    # Resolve sheet name/index
    if isinstance(SHEET_NAME, int):
        try:
            sheet_name_str = xls.sheet_names[SHEET_NAME]
        except Exception:
            print("Invalid DJ_SHEET index; defaulting to first sheet.")
            sheet_name_str = xls.sheet_names[0]
    else:
        sheet_name_str = SHEET_NAME

    df = pd.read_excel(xls, sheet_name=sheet_name_str)

    # Keep only real rows (non-empty Track Name)
    if COL_TRACK in df.columns:
        df = df[df[COL_TRACK].astype(str).str.strip().ne("")].copy()
        df.reset_index(drop=True, inplace=True)

    total_rows = len(df)
    if total_rows == 0:
        print("No rows with a Track Name found. Exiting.")
        return

    # Limit rows if requested
    rows_to_process = total_rows if DJ_LIMIT <= 0 else min(DJ_LIMIT, total_rows)
    print(f"Found {total_rows} real tracks in sheet: {sheet_name_str}")
    print(f"Processing {rows_to_process} rows (limit={DJ_LIMIT or 'none'}, fast_only={DJ_FAST_ONLY})")

    # Compute timings
    intro_vals, drop_vals, outro_vals = [], [], []
    for i, (_, row) in enumerate(df.iloc[:rows_to_process].iterrows(), start=1):
        track  = str(row.get(COL_TRACK, "")).strip()
        artist = str(row.get(COL_ARTIST, "")).strip() if pd.notna(row.get(COL_ARTIST, np.nan)) else None

        genres_str = str(row.get(COL_GENRES, "")).strip() if pd.notna(row.get(COL_GENRES, np.nan)) else ""
        genres = normalise_genres(genres_str)
        family = family_of(genres)

        bpm = None
        duration_s = None

        if pd.notna(row.get(COL_TEMPO, np.nan)):
            try:
                bpm = float(row.get(COL_TEMPO))
            except Exception:
                bpm = None

        if (not DJ_FAST_ONLY) and (bpm is None or family == "generic"):
            hit = beatport_search(track, artist)
            if hit:
                if bpm is None and hit.get("bpm"): bpm = float(hit["bpm"])
                if not genres and hit.get("genres"):
                    genres = normalise_genres(hit["genres"])
                    family = family_of(genres)
                if duration_s is None and hit.get("duration_s"):
                    duration_s = float(hit["duration_s"])

        i_s, d_s, o_s = infer_timings(bpm, family, duration_s)
        intro_vals.append(i_s)
        drop_vals.append(d_s)
        outro_vals.append(o_s)

        if i % 25 == 0 or i == rows_to_process:
            print(f"  ... computed {i}/{rows_to_process}")

        if not DJ_FAST_ONLY:
            time.sleep(0.05)

    # If we processed fewer than total_rows, pad remaining rows with blanks (to keep length aligned)
    pad = total_rows - rows_to_process
    if pad > 0:
        intro_vals += [""] * pad
        drop_vals  += [""] * pad
        outro_vals += [""] * pad

    # Prepare output frame (copy input)
    out_df = df.copy()

    # Ensure there are enough columns to allow Y/Z/AA insertion
    out_df = ensure_positions_with_placeholders(out_df, [Y_IDX, Z_IDX, AA_IDX])

    # Insert in descending order to keep absolute positions stable
    out_df = insert_at_positions(
        out_df,
        [
            (AA_IDX, "OutroStart", outro_vals),
            (Z_IDX,  "DropPoint",  drop_vals),
            (Y_IDX,  "IntroEnd",   intro_vals),
        ]
    )

    # Always write a NEW workbook
    out_copy = xfile.with_name(f"{xfile.stem}_TIMED.xlsx")
    with pd.ExcelWriter(out_copy, engine="openpyxl") as w:
        out_df.to_excel(w, sheet_name=str(sheet_name_str) if isinstance(sheet_name_str, str) else "Sheet1", index=False)

    print(f"Saved timings to: {out_copy.name}")
    print("Timing columns placed at: Y=IntroEnd, Z=DropPoint, AA=OutroStart")

if __name__ == "__main__":
    main()
