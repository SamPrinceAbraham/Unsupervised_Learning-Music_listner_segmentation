# Music Listener Segmentation — Antigravity Project Prompt

## Project Overview

Build a complete **Unsupervised Machine Learning** project that segments music listeners into behavioural clusters using K-Means clustering. The project includes a training pipeline and a CLI application that classifies a new listener in real time.

---

## Project Name

`music-listener-segmentation`

---

## Tech Stack

- **Language:** Python
- **Core Libraries:** pandas, scikit-learn, joblib, Flask (optional for API extension)
- **Model:** K-Means Clustering (unsupervised)
- **Preprocessing:** StandardScaler (scikit-learn)

---

## File Structure

```
music-listener-segmentation/
├── requirements.txt       # Dependency manifest
├── music_listeners.csv    # Training dataset
├── train_model.py         # ML training script
├── app.py                 # CLI prediction application
├── model.pkl              # Saved trained K-Means model (generated after training)
└── scaler.pkl             # Saved StandardScaler object (generated after training)
```

---

## Dataset: `music_listeners.csv`

A CSV file containing listener behaviour data with **no target/label column**.

### Required Columns (Features Only)

| Column                     | Description                          |
|----------------------------|--------------------------------------|
| `listening_hours_per_week` | Total hours the listener plays music per week |
| `songs_per_day`            | Average number of songs played per day |
| `skip_rate`                | Proportion of songs skipped (0.0 – 1.0) |
| `playlist_count`           | Number of playlists the listener has created |

> **Note:** There is no `target` or `label` column. This is unsupervised learning.

Generate **synthetic but realistic** sample data with at least 200 rows covering a natural spread of casual, moderate, and heavy listener behaviours.

---

## File 1: `requirements.txt`

```
pandas
scikit-learn
joblib
flask
```

---

## File 2: `train_model.py`

### Behaviour

1. Load `music_listeners.csv` using pandas.
2. Select the four feature columns: `listening_hours_per_week`, `songs_per_day`, `skip_rate`, `playlist_count`.
3. Scale the features using `StandardScaler` — use `fit_transform()` during training only.
4. Create a `KMeans` model with `n_clusters=3` and a fixed `random_state`.
5. Train the model using `fit()`.
6. Print the cluster centres to the console.
7. Interpret the cluster centres to assign meaningful labels:
   - **Cluster with lowest overall activity** → `Casual Listener`
   - **Cluster with medium activity** → `Music Explorer`
   - **Cluster with highest overall activity** → `Heavy Listener`
8. Print the cluster label mapping to the console.
9. Save the trained model to `model.pkl` using joblib.
10. Save the fitted scaler to `scaler.pkl` using joblib.
11. Confirm both files have been saved with a printed message.

### Important Constraints

- Use `fit_transform()` on the training data — never `fit_transform()` at prediction time.
- Store and print the mapping of cluster numbers (0, 1, 2) to human-readable labels.
- Do not perform a train/test split — the entire dataset is used for clustering.

---

## File 3: `app.py`

### Behaviour

1. Load `model.pkl` and `scaler.pkl` using joblib.
2. Define the same cluster-label mapping used in `train_model.py`.
3. Prompt the user via the CLI to enter the following values:
   - Listening hours per week
   - Songs per day
   - Skip rate (e.g. 0.3 for 30%)
   - Playlist count
4. Convert the inputs into a 2D numpy array matching the training feature order.
5. Scale the input using `scaler.transform()` — **do not call** `fit()` or `fit_transform()`.
6. Predict the cluster using `model.predict()`.
7. Map the predicted cluster number to the human-readable label.
8. Display the result clearly, for example:

```
Listener Segment: Music Explorer
```

### Important Constraints

- Feature order must exactly match training: `[listening_hours_per_week, songs_per_day, skip_rate, playlist_count]`.
- Use `transform()` only — never refit the scaler on new input.
- Handle invalid input gracefully with a clear error message.

---

## Model Flow Summary

```
music_listeners.csv
       │
       ▼
  Load with pandas
       │
       ▼
  Select 4 features (X)
       │
       ▼
  StandardScaler.fit_transform(X)
       │
       ▼
  KMeans(n_clusters=3).fit(X_scaled)
       │
       ▼
  Inspect cluster centres → assign labels
       │
       ▼
  Save model.pkl + scaler.pkl
       │
       ▼
  New listener input (CLI)
       │
       ▼
  scaler.transform(new_input)
       │
       ▼
  model.predict(new_input_scaled)
       │
       ▼
  Display listener segment label
```

---

## Cluster Label Logic

After training, sort clusters by overall scaled activity (sum of centre values across features) to assign labels deterministically:

| Activity Level | Label            |
|----------------|------------------|
| Lowest         | Casual Listener  |
| Medium         | Music Explorer   |
| Highest        | Heavy Listener   |

---

## What This Project Uses vs. Skips

| ML Concept           | Used? | Reason                                      |
|----------------------|-------|---------------------------------------------|
| Features (X)         | ✅    | Four listener behaviour columns              |
| Target (y)           | ❌    | No label column — unsupervised learning      |
| StandardScaler       | ✅    | Required before K-Means                      |
| fit()                | ✅    | Model learns from training data              |
| predict()            | ✅    | Assigns new listener to a cluster            |
| Train/Test Split     | ❌    | Not applicable for clustering                |
| Accuracy / Precision | ❌    | No ground truth labels to evaluate against  |
| Joblib               | ✅    | Saves and loads model and scaler             |
| Hyperparameter Tuning| ❌    | Fixed K=3 for simplicity                     |

---

## Run Instructions

```bash
# Step 1 — Install dependencies
python -m pip install -r requirements.txt

# Step 2 — Train the model
python train_model.py

# Step 3 — Run the application
python app.py
```

---

## Constraints & Rules

1. Always use `fit_transform()` during training and `transform()` only during prediction.
2. Feature order must be identical in both `train_model.py` and `app.py`.
3. Cluster numbers (0, 1, 2) are not inherently meaningful — assign labels after inspecting centres.
4. `model.pkl` stores the trained K-Means model; `scaler.pkl` stores the fitted StandardScaler.
5. The model must be trained before `app.py` is run for the first time.
6. Do not use `random_state=None` — set a fixed seed for reproducibility.
