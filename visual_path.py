import pandas as pd
import folium
from pyproj import Transformer
import random
import webbrowser
import os
import time
import subprocess

# --- Read paths CSV directly ---
input_csv = "paths.csv"
paths_df = pd.read_csv(input_csv)

firefox_path = "/usr/bin/firefox"  # adjust path if needed

# --- Coordinate Transformation ---
transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

def transform_coords(x, y):
    lon, lat = transformer.transform(x, y)
    return lat, lon  # folium wants (lat, lon)

# --- Scoring Output ---
scores = []

# --- Process Each Path Individually ---
path_ids = paths_df["path_id"].unique()

def ratio_to_color(green, pavement, shade):
    max_val = max(green, pavement, shade)
    if max_val == 0:
        return "#000000"  # Black for no features
    elif max_val == green:
        return f"#{int(0):02x}{int(255 * green):02x}{int(0):02x}"
    elif max_val == pavement:
        return f"#{int(255 * pavement):02x}{int(0):02x}{int(0):02x}"
    elif max_val == shade:
        return f"#{int(0):02x}{int(0):02x}{int(255 * shade):02x}"

for pid in path_ids:
    df_pid = paths_df[paths_df["path_id"] == pid]

    # --- Compute all coordinates ---
    all_coords = []
    for _, row in df_pid.iterrows():
        lat1, lon1 = transform_coords(row.start_x, row.start_y)
        lat2, lon2 = transform_coords(row.end_x, row.end_y)
        all_coords.append((lat1, lon1))
        all_coords.append((lat2, lon2))

    lats, lons = zip(*all_coords)
    sw = [min(lats), min(lons)]
    ne = [max(lats), max(lons)]

    # --- Create map with fit_bounds to show full path ---
    map_obj = folium.Map()
    map_obj.fit_bounds([sw, ne])

    # --- Draw segments ---
    for _, row in df_pid.iterrows():
        lat1, lon1 = transform_coords(row.start_x, row.start_y)
        lat2, lon2 = transform_coords(row.end_x, row.end_y)

        color = ratio_to_color(row.green_ratio, row.pavement_ratio, row.shade)

        folium.PolyLine(
            [(lat1, lon1), (lat2, lon2)],
            color=color,
            weight=6,
            opacity=0.9
        ).add_to(map_obj)

    # Save and show map
    output_html = f"temp_path_{pid}.html"
    map_obj.save(output_html)
    # webbrowser.open(f"file://{os.path.abspath(output_html)}")
    subprocess.Popen([firefox_path, "--new-tab", os.path.abspath(output_html)])

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

    scores.append({"path_id": pid, "score": score})

    time.sleep(2)
    os.remove(output_html)

with open("path_scores.csv", "w", newline='') as f:
    f.write("path_id,score\n")
    for entry in scores:
        f.write(f"{entry['path_id']},{entry['score']}\n")
    f.write("path_id,score\n")
