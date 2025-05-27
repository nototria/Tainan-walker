from flask import Flask, render_template, request, jsonify
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTIL_PATH = os.path.join(BASE_DIR, "util")
sys.path.append(UTIL_PATH)
from utils import generate_paths_from_points

#from model import somefunc
from pyproj import Transformer
import webbrowser
import threading

app = Flask(__name__)

selected_points = {}

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
    
    try:
        # Convert to UTM EPSG:32651
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
        src_xy = transformer.transform(start['lng'], start['lat']) 
        dst_xy = transformer.transform(end['lng'], end['lat'])

        # Generate paths from ../util/utils.py
        generate_paths_from_points(source=src_xy,target=dst_xy)

        # read the path.csv, wait chli
        # input_csv = "../CSV_file/paths.csv"
        # path = somefunc(input_csv)

        # test_path
        path = [
            {"lat": start['lat'], "lng": start['lng']},
            {"lat": (start['lat'] + end['lat']) / 2, "lng": (start['lng'] + end['lng']) / 2},
            {"lat": end['lat'], "lng": end['lng']}
        ]
        return jsonify({"status": "success", "path": path})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
def run_app():
    app.run(debug=False, port=5000)

# Launch browser
threading.Thread(target=run_app).start()
webbrowser.open("http://127.0.0.1:5000/")