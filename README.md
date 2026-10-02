# Spotify DJ Perceptron

A DJ transition engine. Given a track library in Excel with Spotify-style audio features, it ranks the best songs to mix into next. It also recommends concrete mix settings (volume curve, EQ moves, effects) for two transition styles, **Conservative** and **Creative**, using Camelot key compatibility and BPM matching that treats double and half tempo (2×/½×) as equivalent.

## How it works

Each candidate track is scored against the anchor track on key compatibility (Camelot wheel), tempo (±10% tolerance, double/half-time aware), and energy/feature similarity across the 12 Spotify audio features (Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature). These scores combine into a MatchScore. In conservative mode, a transition scoring below about 0.74 is flagged **NOT RECOMMENDED**.

## Files

| File | Purpose |
|---|---|
| `song_matcher_v2.py` | Main matcher — ranking, mix-mode recommendations, CSV export |
| `song_matcher_v2 BEAT.py` | Variant with beat/timing-aware scoring |
| `song_matcher.py` | Original matcher (v1) |
| `songspecs.py` | Example that gets recommendations from manually supplied features (uses `song_matcher.py`) |
| `spotify_timing_importer.py` | Fetch IntroEnd / DropPoint / OutroStart per track from Spotify's Audio Analysis API |
| `beatport_timing_importer.py` | Estimate the same timings offline from genre/BPM templates (no API needed) |
| `🎧.xlsm` | The track crate — one row per song with audio features |
| `DJ_Transition_QuickStart.md` | Full quick-start guide |

## Install

```bash
pip install pandas openpyxl numpy
pip install spotipy requests   # only for spotify_timing_importer.py
```

Requires Python 3.9+. Keep the scripts in the same folder as your `🎧.xlsm` crate.

## Quick start

```bash
python song_matcher_v2.py
```

This uses the first track as the anchor and prints the top 10 transitions with colour-coded scores.

```python
from song_matcher_v2 import SongMatcher

matcher = SongMatcher.from_excel("🎧.xlsm", ignore_intros=True)
recs = matcher.recommend_by_name("Blinding Lights", top_k=20, mix_mode="conservative", export_csv=True)
```

### Timing importers

Two scripts fill in the transition timing columns (IntroEnd, DropPoint, OutroStart):

- **Spotify**: set `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` and `SPOTIFY_REDIRECT_URI` (default `http://localhost:8888/callback`), then run `python spotify_timing_importer.py`.
- **Offline**: `python beatport_timing_importer.py` estimates the timings from genre/BPM heuristics and writes them to a `_TIMED.xlsx` copy.

See `DJ_Transition_QuickStart.md` for the full API, mix-mode details and thresholds.
