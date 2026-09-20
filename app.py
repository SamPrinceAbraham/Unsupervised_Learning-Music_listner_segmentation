from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load model, scaler, and label_map
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_map = joblib.load('label_map.pkl')

@app.route('/')
def index():
    return render_template('index.html')

segment_details = {
    "Casual Listener": """
        <strong>Your Vibe: The Casual Listener</strong><br><br>
        Music is a complementary part of your day, not the main event. You likely turn to music for specific situations—like during a daily commute, while working out, or when unwinding in the evening. Because you don't spend hours actively seeking out new tracks, the music you do listen to is likely highly curated or consists of familiar favorites.<br><br>
        <em>Key Characteristics:</em><br>
        • <strong>Selective Listening:</strong> You prioritize quality over quantity, only hitting play when it truly suits your mood.<br>
        • <strong>Familiarity:</strong> You tend to have a small handful of go-to playlists rather than creating a new one every week.<br>
        • <strong>Low Skip Rate:</strong> Because you know what you like, you rarely find yourself skipping through tracks searching for the perfect vibe.
    """,
    "Music Explorer": """
        <strong>Your Vibe: The Music Explorer</strong><br><br>
        You have a very healthy, balanced, and active relationship with music! You don't just put music on in the background; you actively engage with it. You are open to algorithmic recommendations and enjoy dipping your toes into new genres or discovering emerging artists.<br><br>
        <em>Key Characteristics:</em><br>
        • <strong>Curiosity:</strong> You frequently try out new playlists, albums, and artist radios.<br>
        • <strong>Active Curation:</strong> You maintain several playlists tailored to different moods (e.g., studying, party, road trip).<br>
        • <strong>Moderate Skips:</strong> You aren't afraid to skip a track if it doesn't match your current vibe, showing an active listening habit.
    """,
    "Heavy Listener": """
        <strong>Your Vibe: The Heavy Listener</strong><br><br>
        Music is your lifeblood. You are almost always plugged in, with a soundtrack accompanying every single moment of your day. You consume music at an incredible rate and are likely the person your friends go to for music recommendations. The sheer volume of songs you process daily puts you in the top percentile of listeners.<br><br>
        <em>Key Characteristics:</em><br>
        • <strong>Constant Stream:</strong> Whether you're working, sleeping, or socializing, music is playing.<br>
        • <strong>Playlist Architect:</strong> You have dozens of hyper-specific playlists for every conceivable mood, genre, and tempo.<br>
        • <strong>High Volume:</strong> Your listening hours and daily song counts show a deep, immersive dedication to the audio world.
    """
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form.get('hours'))
        songs = float(request.form.get('songs'))
        skip_rate = float(request.form.get('skip_rate'))
        playlists = float(request.form.get('playlists'))
        
        # Format the input exactly as the model expects
        user_input = np.array([[hours, songs, skip_rate, playlists]])
        
        # Scale the input
        scaled_input = scaler.transform(user_input)
        
        # Predict the cluster
        cluster = model.predict(scaled_input)[0]
        
        # Look up the label
        label = label_map.get(cluster, "Unknown Segment")
        explanation = segment_details.get(label, "Your habits are unique!")
        
        inputs = {
            'hours': hours,
            'songs': songs,
            'skip_rate': skip_rate,
            'playlists': playlists
        }
        
        return render_template('result.html', segment=label, explanation=explanation, inputs=inputs)
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
