from flask import Flask, render_template, request, jsonify
import sys, os, traceback
from pyproj import Transformer
import webbrowser, threading
import time
from geopy.distance import geodesic

# ----- PATH SETUP -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTIL_PATH = os.path.join(BASE_DIR, "util")
sys.path.append(UTIL_PATH)
from utils import read_path_from_csv, build_graph, load_edges, sanity_check_graph, find_k_path

# ----- FILE PATHS -----
edgefile_path = os.path.join(BASE_DIR, "tainan_edges.csv")
output_csv = os.path.join(BASE_DIR, "paths.csv")

# ----- FLASK SETUP -----
app = Flask(__name__)

# ----- GLOBALS -----
G = None  # graph will be built on startup
selected_points = {}

# ----- BUILD GRAPH ON SERVER START -----
def initialize_graph():
    global G
    try:
        print("Loading edge data...")
        edges = load_edges(edgefile_path)
        print(f"Loaded {len(edges)} edges. Building graph...")
        G = build_graph(edges)
        print("Graph successfully built.")
    except Exception as e:
        print("Error during graph initialization:")
        traceback.print_exc()
        sys.exit(1)  # Exit if graph can't be built

# ----- ROUTES -----
@app.route('/')
def index():
    return render_template("map.html")

@app.route('/submit_points', methods=['POST'])
def submit_points():
    global selected_points
    selected_points = request.json
    print("Received points:", selected_points)
    return jsonify({"status": "success"})

@app.route('/process_points', methods=['POST'])
def process_points():
    data = request.get_json()
    start = data['start']
    end = data['end']
    desired_time = data.get('desired_time_min', 30)
    walk_speed_m_per_min = 83.3
    expected_distance = desired_time * walk_speed_m_per_min
    try:
        # latitude first
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
        src_xy = transformer.transform(start['lng'],start['lat']) 
        dst_xy = transformer.transform(end['lng'],end['lat'])

        src_checked,dst_checked = sanity_check_graph(G, src_xy, dst_xy, expected_distance, 0.05)
        print("Sanity check passed, generating paths...")

        find_k_path(
            G_base=G,
            source=src_checked,
            target=dst_checked,
            k=1,
            target_distance=expected_distance,
            output_csv=output_csv,
            middle_edges_ratio=0.02,
            path_SN={'value': 0}
        )

        # Temporary: Return dummy path until CSV reading implemented
        # path = [
        #     {"lat": start['lat'], "lng": start['lng']},
        #     {"lat": (start['lat'] + end['lat']) / 2, "lng": (start['lng'] + end['lng']) / 2},
        #     {"lat": end['lat'], "lng": end['lng']}
        # ]
        path = read_path_from_csv(output_csv)
        
        print("Path generation finished.")
        return jsonify({"status": "success", "path": path})

    except Exception as e:
        print("Error during path generation:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# ----- RUN SERVER -----
def run_app():
    initialize_graph()
    app.run(debug=False, port=5000)

threading.Thread(target=run_app).start()
time.sleep(7)
webbrowser.open("http://127.0.0.1:5000/")
