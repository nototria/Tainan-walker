from flask import Flask, render_template, request, jsonify
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
    
    # structure holder
    path = [
        {"lat": start['lat'], "lng": start['lng']},
        {"lat": (start['lat'] + end['lat']) / 2, "lng": (start['lng'] + end['lng']) / 2},
        {"lat": end['lat'], "lng": end['lng']}
    ]

    return jsonify({"path": path})

def run_app():
    app.run(debug=False, port=5000)

# Launch browser
threading.Thread(target=run_app).start()
webbrowser.open("http://127.0.0.1:5000/")