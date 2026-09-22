import pandas as pd
import numpy as np

# Generate realistic data for 3 clusters
np.random.seed(42)

# Casual listeners (approx 70 rows)
casual_hours = np.random.normal(loc=5, scale=2, size=70)
casual_songs = np.random.normal(loc=15, scale=5, size=70)
casual_skip = np.random.normal(loc=0.5, scale=0.1, size=70)
casual_playlists = np.random.poisson(lam=2, size=70)

# Music explorers (approx 70 rows)
explorer_hours = np.random.normal(loc=15, scale=4, size=70)
explorer_songs = np.random.normal(loc=50, scale=10, size=70)
explorer_skip = np.random.normal(loc=0.3, scale=0.1, size=70)
explorer_playlists = np.random.poisson(lam=8, size=70)

# Heavy listeners (approx 60 rows)
heavy_hours = np.random.normal(loc=30, scale=5, size=60)
heavy_songs = np.random.normal(loc=100, scale=20, size=60)
heavy_skip = np.random.normal(loc=0.1, scale=0.05, size=60)
heavy_playlists = np.random.poisson(lam=20, size=60)

hours = np.concatenate([casual_hours, explorer_hours, heavy_hours])
songs = np.concatenate([casual_songs, explorer_songs, heavy_songs])
skips = np.concatenate([casual_skip, explorer_skip, heavy_skip])
playlists = np.concatenate([casual_playlists, explorer_playlists, heavy_playlists])

# clip appropriately
hours = np.clip(hours, 0, 168)
songs = np.clip(songs, 0, 500)
skips = np.clip(skips, 0, 1.0)
playlists = np.clip(playlists, 0, None)

df = pd.DataFrame({
    'listening_hours_per_week': hours,
    'songs_per_day': songs.astype(int),
    'skip_rate': np.round(skips, 2),
    'playlist_count': playlists
})

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('music_listeners.csv', index=False)
print(f"Generated {len(df)} rows in music_listeners.csv")
#This module was developed and completed by Shamili V
