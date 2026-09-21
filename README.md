# Music Listener Segmentation 🎧

Welcome to the **Music Listener Segmentation** project! This repository contains a complete Unsupervised Machine Learning pipeline that uses K-Means clustering to group music listeners into distinct behavioral segments based on their listening habits.

The project features both a seamless **Command-Line Interface (CLI)** and a beautifully designed **Web Application (Flask + Vanilla CSS)** for real-time predictions.

## 🚀 Features

- **Unsupervised ML Pipeline:** Uses Scikit-learn's StandardScaler and KMeans to automatically find clusters in unlabelled listener data.
- **Dynamic Segment Labeling:** Clusters are deterministically labeled as *Casual Listener*, *Music Explorer*, or *Heavy Listener* based on their overall engagement metrics.
- **Modern Web Interface:** A sleek, fully-responsive, and animated dark-mode UI built with Vanilla HTML/CSS and powered by Flask.
- **Real-Time Prediction:** Enter your listening habits into the web app or CLI to instantly discover your vibe!

## 🛠️ Tech Stack

- **Language:** Python 3
- **Machine Learning:** scikit-learn, pandas, 
umpy, joblib
- **Backend / Web Server:** Flask
- **Frontend:** HTML5, CSS3 (Custom Glassmorphism Design)

## 📊 The Data

The model is trained on a synthetic dataset representing realistic listening behaviors. The four core features analyzed are:
1. **Listening Hours Per Week:** Total hours spent actively or passively listening.
2. **Songs Per Day:** Average daily track consumption.
3. **Skip Rate:** The proportion of songs skipped before finishing (0.0 to 1.0).
4. **Playlist Count:** The number of custom playlists actively curated by the user.

## ⚙️ Setup & Installation

**1. Clone the repository:**
`ash
git clone https://github.com/your-username/music-listener-segmentation.git
cd music-listener-segmentation
`

**2. Install dependencies:**
`ash
pip install -r requirements.txt
`

**3. Generate data & train the model:**
The model must be trained at least once before predictions can be made.
`ash
python generate_data.py
python train_model.py
`
*(This will generate model.pkl, scaler.pkl, and label_map.pkl)*

## 💻 Usage

### Web Application (Recommended)
To launch the interactive frontend:
`ash
python app.py
`
Then, open your browser and navigate to http://localhost:5000. 

### CLI Application
If you prefer the terminal, you can run the prediction script directly:
`ash
python cli.py
`
Simply answer the prompts, and the CLI will instantly output your listener segment!

## 🧩 Listener Segments

Depending on your habits, the model will classify you into one of three distinct clusters:
- **Casual Listener:** Selective and occasional listener, relying on familiar favorites.
- **Music Explorer:** Active curator with a balanced and healthy relationship with music discovery.
- **Heavy Listener:** Deeply immersed in audio, with a constant stream of music shaping their day.

TEAM 2
---
*Developed for advanced behavioral clustering and seamless UX integration.*
