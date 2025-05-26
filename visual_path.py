import pandas as pd
import folium
from pyproj import Transformer
import random

# --- Read paths CSV directly ---
# CSV format expected:
# path_id,start_x,start_y,end_x,end_y,distance,green_ratio,shade,pavement_ratio
input_csv = "paths.csv"
paths_df = pd.read_csv(input_csv)

# --- Coordinate Transformation ---
# Assuming source coords are in TWD97 TM2 (EPSG:3826); converting to WGS84 (EPSG:4326)
transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

def transform_coords(x, y):
    lon, lat = transformer.transform(x, y)
    return lat, lon  # folium wants (lat, lon)

# --- Create Folium Map ---
# Center map on first start point
first = paths_df.iloc[0]
center_lat, center_lon = transform_coords(first.start_x, first.start_y)
map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# --- Draw Paths ---
path_ids = paths_df["path_id"].unique()
colors = {pid: f"#{random.randint(0, 0xFFFFFF):06x}" for pid in path_ids}

for pid in path_ids:
    df_pid = paths_df[paths_df["path_id"] == pid]
    # build coordinate list for full path
    coords = []
    for _, row in df_pid.iterrows():
        lat1, lon1 = transform_coords(row.start_x, row.start_y)
        lat2, lon2 = transform_coords(row.end_x, row.end_y)
        coords.append((lat1, lon1))
        # ensure last segment end
        coords.append((lat2, lon2))
    # remove duplicates preserving order
    unique_coords = []
    for c in coords:
        if not unique_coords or unique_coords[-1] != c:
            unique_coords.append(c)
    folium.PolyLine(unique_coords, color=colors[pid], weight=5, opacity=0.8).add_to(map_obj)

# --- Save Map ---
output_html = "paths_on_map.html"
map_obj.save(output_html)
print(f"Map saved to {output_html}")
