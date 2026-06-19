# DJ Transition Engine — Quick Start

## What this is
A Python module that ranks the best transition songs from your Excel (`🎧.xlsm`) and recommends mix settings (Volume, EQ, Effects) for **Conservative** and **Creative** styles. Includes Camelot key tips and optional CSV export.

## File
- `song_matcher_v2.py` (drop next to `🎧.xlsm`)

## Expected columns in your Excel
`Track Name`, `Artist` *(optional)*, and these features:  
`Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature`

## Install
- Python 3.9+
- `pip install pandas openpyxl numpy`

## Run quick test
```bash
python song_matcher_v2.py
```
It will use the first track as an anchor and print the top 10 transitions with coloured scores.

## API
```python
from song_matcher_v2 import SongMatcher

matcher = SongMatcher.from_excel("🎧.xlsm", ignore_intros=True)

# A) Recommend by exact track name
recs = matcher.recommend_by_name("Blinding Lights", top_k=20, export_csv=True, mix_mode="conservative")

# B) Recommend by manual features
anchor = {  # fill values from your source
  "Danceability": 0.72, "Energy": 0.83, "Key": 9, "Loudness": -6.3, "Mode": 1,
  "Speechiness": 0.04, "Acousticness": 0.18, "Instrumentalness": 0.00,
  "Liveness": 0.12, "Valence": 0.52, "Tempo": 124, "Time Signature": 4
}
recs = matcher.recommend(anchor, top_k=20, export_csv=False, mix_mode="creative")
```

## Mix modes
- **Conservative**: safe, smooth transitions (centre-out, 3-band fade, quick bass cut)
- **Creative**: expressive, energetic switches using filter sweeps and bass swaps

## CSV export
Pass `export_csv=True` to save a clean CSV containing: Track Name, Artist, MatchScore, KeyScore, TempoScore, EnergyScore, Volume, EQ, FX, Notes.

## Notes & thresholds
- If MatchScore < ~0.74 (conservative) it will mark **NOT RECOMMENDED**.
- Tempo scoring allows ±10% tolerance and 2×/½× BPM equivalence.
- Intros/skits are filtered by default; disable with `ignore_intros=False`.

## Camelot helper
```python
from song_matcher_v2 import suggest_keys_camelot
print(suggest_keys_camelot(key=9, mode=1))  # -> e.g., ['8B','8A','9B','7B','9A']
```