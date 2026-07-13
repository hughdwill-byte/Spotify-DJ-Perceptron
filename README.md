# Spotify DJ Perceptron

A DJ transition engine: given a track library in Excel with Spotify-style audio features, it ranks the best next songs to mix into and recommends concrete mix settings (volume curve, EQ moves, effects) for **Conservative** and **Creative** transition styles — including Camelot key compatibility and BPM matching with 2×/½× equivalence.

## How it works

Each candidate track is scored against the anchor track on key compatibility (Camelot wheel), tempo (±10% tolerance, double/half-time aware), and energy/feature similarity across the 12 Spotify audio features (Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature). Scores combine into a MatchScore; below ~0.74 in conservative mode a transition is flagged **NOT RECOMMENDED**.

## Files

| File | Purpose |
|---|---|
| `song_matcher_v2.py` | Main matcher — ranking, mix-mode recommendations, CSV export |
| `song_matcher_v2 BEAT.py` | Variant with beat/timing-aware scoring |
| `song_matcher.py` | Original matcher |
| `songspecs.py` | Example: recommend from manually supplied features |
| `spotify_timing_importer.py` | Fetch IntroEnd / DropPoint / OutroStart per track from Spotify's Audio Analysis API |
| `beatport_timing_importer.py` | Estimate the same timings offline from genre/BPM templates (no API needed) |
| `🎧.xlsm` | The track crate — one row per song with audio features |
| `DJ_Transition_QuickStart.md` | Full quick-start guide |

## Install

```bash
pip install pandas openpyxl numpy
```

Python 3.9+. Drop the scripts next to your `🎧.xlsm` crate.

## Quick start

```bash
python song_matcher_v2.py
```

Uses the first track as anchor and prints the top 10 transitions with colour-coded scores.

```python
from song_matcher_v2 import SongMatcher

matcher = SongMatcher.from_excel("🎧.xlsm", ignore_intros=True)
recs = matcher.recommend_by_name("Blinding Lights", top_k=20, mix_mode="conservative", export_csv=True)
```

### Timing importers

For transition timing columns (IntroEnd, DropPoint, OutroStart):

- **Spotify** — set `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`, then run `python spotify_timing_importer.py`.
- **Offline** — `python beatport_timing_importer.py` writes a `_TIMED.xlsx` copy using genre/BPM heuristics.

See `DJ_Transition_QuickStart.md` for the full API, mix-mode details, and thresholds.
