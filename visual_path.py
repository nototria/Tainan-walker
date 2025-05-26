import pandas as pd
import folium
from pyproj import Transformer
import webbrowser
import os

# --- Read paths CSV ---
input_csv = "paths.csv"
paths_df = pd.read_csv(input_csv)

# --- Coordinate transformer from EPSG:32651 to EPSG:4326 (WGS84) ---
transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

def transform_coords(x, y):
    lon, lat = transformer.transform(x, y)
    return lat, lon  # folium expects (lat, lon)

# --- Color logic ---
def ratio_to_color(green, pavement, shade):
    max_val = max(green, pavement, shade)
    if max_val == 0:
        return "#000000"  # Black for no features
    elif max_val == green:
        # Green channel saturation
        g = int(255 * green)
        return f"#{0:02x}{g:02x}{0:02x}"
    elif max_val == pavement:
        # Red channel saturation
        r = int(255 * pavement)
        return f"#{r:02x}{0:02x}{0:02x}"
    else:
        # Blue channel saturation (shade)
        b = int(255 * shade)
        return f"#{0:02x}{0:02x}{b:02x}"

# --- Store scores here ---
scores = []

# --- Process each path ---
path_ids = paths_df["path_id"].unique()

for pid in path_ids:
    df_pid = paths_df[paths_df["path_id"] == pid]

    # Collect all coordinates for fit_bounds
    all_coords = []
    for _, row in df_pid.iterrows():
        lat1, lon1 = transform_coords(row.start_x, row.start_y)
        lat2, lon2 = transform_coords(row.end_x, row.end_y)
        all_coords.append((lat1, lon1))
        all_coords.append((lat2, lon2))

    lats, lons = zip(*all_coords)
    sw = [min(lats), min(lons)]
    ne = [max(lats), max(lons)]

    # Create map and fit bounds
    map_obj = folium.Map()
    map_obj.fit_bounds([sw, ne])

    # Draw all segments with color based on ratios
    for _, row in df_pid.iterrows():
        lat1, lon1 = transform_coords(row.start_x, row.start_y)
        lat2, lon2 = transform_coords(row.end_x, row.end_y)
        color = ratio_to_color(row.green_ratio, row.pavement_ratio, row.shade)
        folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                        color=color, weight=6, opacity=0.9).add_to(map_obj)

    # Save HTML file
    output_html = f"temp_path_{pid}.html"
    map_obj.save(output_html)

    # Open in default browser (reuse tab if Firefox)
    webbrowser.open_new_tab(f"file://{os.path.abspath(output_html)}")

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

    # Remove HTML file after scoring
    os.remove(output_html)

# Save all scores to CSV
score_csv = "path_scores.csv"
with open(score_csv, "w") as f:
    f.write("path_id,score\n")
    for entry in scores:
        f.write(f"{entry['path_id']},{entry['score']}\n")

print(f"All scores saved to {score_csv}")
