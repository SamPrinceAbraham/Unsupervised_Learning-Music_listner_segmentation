import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

def main():
    # 1. Load music_listeners.csv
    df = pd.read_csv('music_listeners.csv')
    
    # 2. Select the four feature columns
    features = ['listening_hours_per_week', 'songs_per_day', 'skip_rate', 'playlist_count']
    X = df[features]
    
    # 3. Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4 & 5. Create and train a KMeans model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    
    # 6. Print the cluster centres
    print("Cluster Centres (Scaled):")
    print(pd.DataFrame(kmeans.cluster_centers_, columns=features))
    print("-" * 30)
    
    # 7. Interpret cluster centres to assign meaningful labels
    # We sum the scaled cluster center coordinates to gauge overall activity level
    activity_levels = kmeans.cluster_centers_.sum(axis=1)
    
    # Get indices of sorted activity levels
    sorted_indices = np.argsort(activity_levels)
    
    # Determine the mapping from cluster index to label
    # Lowest -> Casual, Medium -> Explorer, Highest -> Heavy
    label_map = {}
    label_map[sorted_indices[0]] = "Casual Listener"
    label_map[sorted_indices[1]] = "Music Explorer"
    label_map[sorted_indices[2]] = "Heavy Listener"
    
    # 8. Print the cluster label mapping
    print("Cluster Label Mapping:")
    for cluster_id, label in label_map.items():
        print(f"Cluster {cluster_id}: {label}")
    print("-" * 30)
    
    # Add a global dictionary for prediction mappings we can import later (optional)
    # joblib will just save the model. We can store the dictionary in a file or just
    # replicate the logic in app.py. The prompt says: 
    # "Define the same cluster-label mapping used in train_model.py."
    # To keep things simple and matching the prompt, we'll save the map as well or just hardcode it in app if deterministic.
    # We will save the mapping itself using joblib to ensure absolute consistency.
    joblib.dump(label_map, 'label_map.pkl')
    
    # 9. Save the trained model to model.pkl
    joblib.dump(kmeans, 'model.pkl')
    
    # 10. Save the fitted scaler to scaler.pkl
    joblib.dump(scaler, 'scaler.pkl')
    
    # 11. Confirm both files have been saved
    print("Successfully saved model.pkl, scaler.pkl, and label_map.pkl")

if __name__ == '__main__':
    main()
