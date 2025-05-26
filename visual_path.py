import pandas as pd
import folium
from pyproj import Transformer
import random
import webbrowser
import os
import time

# --- Read paths CSV directly ---
input_csv = "paths.csv"
paths_df = pd.read_csv(input_csv)

# --- Coordinate Transformation ---
transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

def transform_coords(x, y):
    lon, lat = transformer.transform(x, y)
    return lat, lon  # folium wants (lat, lon)

# --- Scoring Output ---
scores = []

# --- Process Each Path Individually ---
path_ids = paths_df["path_id"].unique()

for pid in path_ids:
    df_pid = paths_df[paths_df["path_id"] == pid]
    
    # Get center from first point
    first = df_pid.iloc[0]
    center_lat, center_lon = transform_coords(first.start_x, first.start_y)
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Build coordinates for the path
    coords = []
    for _, row in df_pid.iterrows():
        lat1, lon1 = transform_coords(row.start_x, row.start_y)
        lat2, lon2 = transform_coords(row.end_x, row.end_y)
        coords.append((lat1, lon1))
        coords.append((lat2, lon2))

    # Remove duplicates while preserving order
    unique_coords = []
    for c in coords:
        if not unique_coords or unique_coords[-1] != c:
            unique_coords.append(c)

    # Draw polyline
    color = f"#{random.randint(0, 0xFFFFFF):06x}"
    folium.PolyLine(unique_coords, color=color, weight=5, opacity=0.8).add_to(map_obj)

    # Save and open map
    output_html = f"temp_path_{pid}.html"
    map_obj.save(output_html)
    webbrowser.open(f"file://{os.path.abspath(output_html)}")

    # Ask user for score
    while True:
        try:
            score = int(input(f"Enter score for path {pid} (0–100): "))
            if 0 <= score <= 100:
                break
            else:
                print("Score must be between 0 and 100.")
        except ValueError:
            print("Please enter a valid integer.")

    # Append to scores
    scores.append({"path_id": pid, "score": score})

    # Optional: allow user to see the map for a few seconds before continuing
    time.sleep(2)

    # Optionally delete temp file:
    os.remove(output_html)

# --- Save scores ---
scores_df = pd.DataFrame(scores)
scores_df.to_csv("score.csv", index=False)
print("Scores saved to score.csv")
