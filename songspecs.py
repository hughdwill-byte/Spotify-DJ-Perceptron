from song_matcher import SongMatcher

matcher = SongMatcher.from_excel("🎧.xlsm")
input_song = {
    "Danceability": 0.72,
    "Energy": 0.83,
    "Key": 9,
    "Loudness": -6.3,
    "Mode": 1,
    "Speechiness": 0.04,
    "Acousticness": 0.18,
    "Instrumentalness": 0.00,
    "Liveness": 0.12,
    "Valence": 0.52,
    "Tempo": 124,
    "Time Signature": 4
}
recs = matcher.recommend(input_song, top_k=20)
print(recs[ [c for c in ("Name","Artist","MatchScore","KeyScore","TempoScore","EnergyScore") if c in recs.columns] ].head(20))
