
"""
spotify_timing_importer.py
---------------------------------
Lightweight tool to fetch transition timings (IntroEnd, DropPoint, OutroStart) from Spotify's
Audio Analysis API for all tracks in your Excel crate, without downloading audio files.

Usage
-----
1) Set environment variables (recommended):
   - SPOTIFY_CLIENT_ID
   - SPOTIFY_CLIENT_SECRET
   - SPOTIFY_REDIRECT_URI   (use: http://localhost:8888/callback)

   Or edit the CONFIG block below to hardcode them (less secure).

2) Place your Excel next to this script (default: 🎧.xlsm). Your sheet should include at least:
   - "Track Name" (exact or close)
   - "Artist" (recommended)
   Optional: "Duration (ms)" if you have it.

3) Run:
   python spotify_timing_importer.py

Output
------
- Adds three columns to your main sheet (if missing): IntroEnd, DropPoint, OutroStart (mm:ss)
- Also writes/updates a cache CSV: spotify_timing_cache.csv for faster re-runs
"""

import os, time, re, math, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import numpy as np

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except Exception as e:
    print("Missing dependency 'spotipy'. Install with: pip install spotipy")
    sys.exit(1)

# ------------------ CONFIG ------------------
EXCEL_FILE = os.environ.get("DJ_EXCEL", "🎧.xlsm")
SHEET_NAME = os.environ.get("DJ_SHEET", 0)  # first sheet by default
CACHE_CSV  = os.environ.get("DJ_TIMING_CACHE", "spotify_timing_cache.csv")

# Read credentials from env (recommended). If not set, you can paste them here.
CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback").strip()

SCOPES = "user-read-private"  # analysis endpoints do not require special scopes

# Columns
COL_TRACK = "Track Name"
COL_ARTIST = "Artist"
OUT_COLS = ["IntroEnd", "DropPoint", "OutroStart"]

# -------------------------------------------

def mmss(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds): return ""
    seconds = max(0, float(seconds))
    m = int(seconds // 60)
    s = int(round(seconds - 60*m))
    return f"{m:01d}:{s:02d}"

def pick_best_search_match(items: List[Dict[str, Any]], track: str, artist: Optional[str]) -> Optional[Dict[str, Any]]:
    if not items: return None
    t_norm = re.sub(r"\s+", " ", (track or "").lower()).strip()
    a_norm = re.sub(r"\s+", " ", (artist or "").lower()).strip() if artist else None

    def score_item(it):
        name = re.sub(r"\s+", " ", it["name"].lower()).strip()
        arts = ", ".join([a["name"] for a in it.get("artists", [])]).lower()
        score = 0
        if name == t_norm: score += 3
        if t_norm in name or name in t_norm: score += 2
        if a_norm:
            if a_norm in arts or arts in a_norm: score += 2
        score += int(it.get("popularity", 0)) / 100.0
        # prefer same duration (+/- 5s) if we have it later
        return score

    items_sorted = sorted(items, key=score_item, reverse=True)
    return items_sorted[0] if items_sorted else None

def infer_timings_from_analysis(analysis: Dict[str, Any], track_duration: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Heuristics using 'sections' from Spotify audio analysis:
    - IntroEnd: first section boundary where loudness rises significantly or within first 45s
    - DropPoint: loudest (or energy-peaking) section center between 25% and 85% of track
    - OutroStart: first boundary in last 25% where loudness decreases or just start of final section
    """
    sections = analysis.get("sections", []) if analysis else []
    if not sections:
        # fallback generic
        intro_end = min(45.0, track_duration * 0.20)
        drop_pt   = track_duration * 0.45
        outro_st  = track_duration * 0.80
        return intro_end, drop_pt, outro_st

    # collect tuples: (start, duration, loudness, loudness_max, tempo_confidence)
    sec = []
    for s in sections:
        start = float(s.get("start", 0.0))
        dur   = float(s.get("duration", 0.0))
        loud  = float(s.get("loudness", -20.0))
        lmax  = float(s.get("loudness_max", loud))
        tconf = float(s.get("tempo_confidence", 0.0))
        sec.append((start, dur, loud, lmax, tconf))

    # IntroEnd: boundary in first 60s where loudness jump > 5dB or loudness > -18dB
    intro_end = None
    for i in range(1, min(len(sec), 6)):
        prev = sec[i-1]; cur = sec[i]
        if cur[0] > 60: break
        if (cur[2] - prev[2]) > 5.0 or cur[2] > -18.0:
            intro_end = cur[0]
            break
    if intro_end is None:
        intro_end = min(45.0, track_duration * 0.20)

    # DropPoint: center of the highest energy section between 25% and 85%
    lo = track_duration * 0.25
    hi = track_duration * 0.85
    best_e = -1e9
    drop_pt = None
    for (start, dur, loud, lmax, tconf) in sec:
        mid = start + dur/2.0
        if lo <= mid <= hi:
            # energy proxy: loudness max + tempo confidence
            e = lmax + 6.0*tconf
            if e > best_e:
                best_e = e
                drop_pt = mid
    if drop_pt is None:
        drop_pt = track_duration * 0.50

    # OutroStart: first boundary in last 30% where loudness < -16 dB or loudness decreases by > 4dB
    outro_st = None
    cutoff = track_duration * 0.70
    for i in range(1, len(sec)):
        prev = sec[i-1]; cur = sec[i]
        if cur[0] >= cutoff:
            if cur[2] < -16.0 or (cur[2] - prev[2]) < -4.0:
                outro_st = cur[0]
                break
    if outro_st is None:
        if sec:
            outro_st = sec[-1][0]
        else:
            outro_st = track_duration * 0.80

    return intro_end, drop_pt, outro_st

def main():
    # OAuth
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET env vars (or edit the script).")
        sys.exit(1)

    auth_mgr = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=".spotify_token_cache"
    )
    sp = spotipy.Spotify(auth_manager=auth_mgr)

    # Load Excel
    xfile = Path(EXCEL_FILE)
    if not xfile.exists():
        print(f"Excel not found: {xfile.resolve()}")
        sys.exit(1)
    print(f"Loading {xfile.name} ...")
    df = pd.read_excel(xfile, sheet_name=SHEET_NAME, engine="openpyxl")

    # Optional cache
    cache_df = None
    cache_path = Path(CACHE_CSV)
    if cache_path.exists():
        try:
            cache_df = pd.read_csv(cache_path)
        except Exception:
            cache_df = None

    # Ensure output columns
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = ""

    updated = 0
    rows = []
    for idx, row in df.iterrows():
        track = str(row.get(COL_TRACK, "")).strip()
        artist = str(row.get(COL_ARTIST, "")).strip() if not pd.isna(row.get(COL_ARTIST, np.nan)) else None
        if not track:
            continue

        # Skip existing timings
        if all(str(row.get(c,"")).strip() for c in OUT_COLS):
            continue

        # Cache lookup
        cache_hit = None
        if cache_df is not None:
            mask = (cache_df["track"].str.lower() == track.lower())
            if artist:
                mask = mask & (cache_df["artist"].fillna("").str.lower() == artist.lower())
            hit = cache_df[mask]
            if not hit.empty:
                cache_hit = hit.iloc[0]

        if cache_hit is not None:
            intro_end = float(cache_hit.get("intro_end_s", np.nan))
            drop_pt   = float(cache_hit.get("drop_point_s", np.nan))
            outro_st  = float(cache_hit.get("outro_start_s", np.nan))
            dur_s     = float(cache_hit.get("duration_s", np.nan))
        else:
            # Search track
            q = f"track:{track}"
            if artist: q += f" artist:{artist}"
            res = sp.search(q=q, type="track", limit=5)
            items = res.get("tracks", {}).get("items", [])
            pick = pick_best_search_match(items, track, artist)
            if not pick:
                print(f"[skip] Not found on Spotify: {track} — {artist or ''}")
                continue

            track_id = pick["id"]
            dur_s = pick["duration_ms"] / 1000.0 if pick.get("duration_ms") else None

            # Fetch analysis
            try:
                analysis = sp.audio_analysis(track_id)
            except spotipy.exceptions.SpotifyException as e:
                print(f"[warn] analysis error for {track}: {e}")
                time.sleep(0.5)
                continue

            intro_end, drop_pt, outro_st = infer_timings_from_analysis(analysis, dur_s or 240.0)

            # write to cache list
            rows.append({
                "track": track,
                "artist": artist or "",
                "spotify_id": track_id,
                "duration_s": dur_s or "",
                "intro_end_s": intro_end or "",
                "drop_point_s": drop_pt or "",
                "outro_start_s": outro_st or ""
            })

        # Assign to df
        df.at[idx, "IntroEnd"]   = mmss(intro_end) if intro_end is not None else ""
        df.at[idx, "DropPoint"]  = mmss(drop_pt) if drop_pt is not None else ""
        df.at[idx, "OutroStart"] = mmss(outro_st) if outro_st is not None else ""
        updated += 1

        if updated % 25 == 0:
            print(f" ... updated {updated} rows")

        time.sleep(0.1)  # be gentle

    # Save cache
    if rows:
        new_cache = pd.DataFrame(rows)
        if cache_df is not None and not cache_df.empty:
            cache_df = pd.concat([cache_df, new_cache], ignore_index=True)
            # drop dups
            cache_df = cache_df.drop_duplicates(subset=["track","artist"], keep="last")
        else:
            cache_df = new_cache
        cache_df.to_csv(cache_path, index=False)

    # Save Excel (preserve macros if xlsm)
    print(f"Writing timings back to {xfile.name} ...")
    try:
        # keep_vba preserves macros for .xlsm
        with pd.ExcelWriter(xfile, engine="openpyxl", mode="a", if_sheet_exists="overlay", keep_vba=True) as w:
            df.to_excel(w, sheet_name=xfile.stem if isinstance(SHEET_NAME, int) else SHEET_NAME, index=False)
    except Exception:
        # Fallback: write to a new file
        out_alt = xfile.with_name(f"{xfile.stem}_with_timings.xlsx")
        with pd.ExcelWriter(out_alt, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Sheet1", index=False)
        print(f"Saved as {out_alt.name} (fallback)")

    print(f"Done. Updated rows: {updated}")
    print("Columns added/updated: IntroEnd, DropPoint, OutroStart")

if __name__ == "__main__":
    main()
