import joblib
import numpy as np
import warnings

# Suppress warnings about missing feature names on prediction
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_map = joblib.load('label_map.pkl')
    except Exception as e:
        print("Error loading model files. Please run train_model.py first.")
        return

    print("--- Music Listener Segmentation ---")
    try:
        hours = float(input("Listening hours per week: "))
        songs = float(input("Songs per day: "))
        skip_rate = float(input("Skip rate (e.g., 0.3 for 30%): "))
        playlists = float(input("Playlist count: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Convert the inputs into a 2D numpy array matching the training feature order
    user_input = np.array([[hours, songs, skip_rate, playlists]])

    # Scale the input using scaler.transform()
    scaled_input = scaler.transform(user_input)

    # Predict the cluster
    cluster = model.predict(scaled_input)[0]

    # Map the predicted cluster number to the human-readable label
    label = label_map.get(cluster, "Unknown Segment")

    # Display the result
    print(f"\nListener Segment: {label}")

if __name__ == '__main__':
    main()
