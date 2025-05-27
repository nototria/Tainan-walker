import os
import webbrowser

center_lat, center_lon = 23.147305, 120.305786
zoom = 11
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Point Selection Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            height: 100vh;
            width: 100%;
        }}
        #controls {{
            position: fixed;
            top: 10px;
            left: 10px;
            width: 350px;
            background-color: white;
            border: 2px solid grey;
            z-index: 9999;
            font-size: 14px;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        #controls h4 {{
            margin-top: 0;
            color: #333;
        }}
        .btn {{
            padding: 8px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            margin-right: 10px;
            margin-top: 5px;
        }}
        .btn-primary {{ background-color: #2196F3; color: white; }}
        .btn-danger {{ background-color: #f44336; color: white; }}
        .btn-success {{ background-color: #4CAF50; color: white; }}
        #modeStatus {{
            margin-top: 8px;
            font-style: italic;
            padding: 5px;
            border-radius: 3px;
        }}
        .status-disabled {{ background-color: #f0f0f0; color: #666; }}
        .status-enabled {{ background-color: #e8f5e8; color: #2e7d32; }}
    </style>
</head>
<body>
    <div id="controls">
        <h4>Point Selection Tool</h4>
        <button id="toggleMode" class="btn btn-primary" onclick="toggleSelectionMode()">
            Enable Selection Mode
        </button>
        <div id="modeStatus" class="status-disabled">
            Selection mode disabled - map navigation active
        </div>
        <div id="info" style="margin-top: 10px;"></div>
        <div style="margin-top: 10px;">
            <button class="btn btn-success" onclick="sendPointsToServer()">
                Send to Python
            </button>
            <button id="reset" class="btn btn-danger" onclick="resetSelection()" style="display:none;">
                Reset Selection
            </button>
        </div>
    </div>
    
    <div id="map"></div>

    <script>
        // Initialize map
        var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});
        
        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        let points = [];
        let markers = [];
        let selectionMode = false;

        function updateInfo() {{
            const info = document.getElementById('info');
            if (points.length === 1) {{
                info.innerHTML = `<div style="padding: 8px; background-color: #e3f2fd; border-radius: 4px;">
                    <b>Point A selected:</b><br>
                    ${{points[0].lat.toFixed(6)}}, ${{points[0].lng.toFixed(6)}}<br>
                    <i>Click to select Point B</i></div>`;
            }} else if (points.length === 2) {{
                info.innerHTML = `<div style="padding: 8px; background-color: #e8f5e8; border-radius: 4px;">
                    <b>Both Points Selected:</b><br>
                    Point A: ${{points[0].lat.toFixed(6)}}, ${{points[0].lng.toFixed(6)}}<br>
                    Point B: ${{points[1].lat.toFixed(6)}}, ${{points[1].lng.toFixed(6)}}<br>
                    <i>Points are pinned on the map</i></div>`;
                document.getElementById("download").style.display = "inline-block";
                document.getElementById("reset").style.display = "inline-block";
                // Auto-disable selection mode when done
                if (selectionMode) {{
                    toggleSelectionMode();
                }}
            }}
        }}

        function toggleSelectionMode() {{
            selectionMode = !selectionMode;
            const button = document.getElementById('toggleMode');
            const status = document.getElementById('modeStatus');
            
            if (selectionMode) {{
                button.innerHTML = 'Disable Selection Mode';
                button.className = 'btn btn-danger';
                status.innerHTML = 'Selection mode enabled - click map to select points';
                status.className = 'status-enabled';
                // Change cursor to crosshair when in selection mode
                document.getElementById('map').style.cursor = 'crosshair';
            }} else {{
                button.innerHTML = 'Enable Selection Mode';
                button.className = 'btn btn-primary';
                status.innerHTML = 'Selection mode disabled - map navigation active';
                status.className = 'status-disabled';
                // Restore normal cursor
                document.getElementById('map').style.cursor = '';
            }}
        }}

        function onMapClick(e) {{
            // Only respond to clicks when in selection mode
            if (!selectionMode) {{
                return;
            }}
            
            if (points.length >= 2) {{
                alert("You've already selected 2 points. Use Reset to start over.");
                return;
            }}
            
            // Create custom marker
            const markerIcon = L.divIcon({{
                className: 'custom-marker',
                html: `<div style="
                    background-color: #ff4444; 
                    width: 16px; 
                    height: 16px; 
                    border-radius: 50%; 
                    border: 3px solid white; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                    position: relative;
                "></div>`,
                iconSize: [22, 22],
                iconAnchor: [11, 11]
            }});
            
            const marker = L.marker(e.latlng, {{ icon: markerIcon }}).addTo(map);
            markers.push(marker);
            
            // Add point to array
            points.push(e.latlng);
            updateInfo();
        }}

        // Add click listener to map
        map.on('click', onMapClick);

        function resetSelection() {{
            // Clear existing points
            points = [];
            markers.forEach(marker => {{
                map.removeLayer(marker);
            }});
            markers = [];
            
            // Hide buttons and clear info
            document.getElementById("download").style.display = "none";
            document.getElementById("reset").style.display = "none";
            document.getElementById("info").innerHTML = "";
            
            // Disable selection mode
            if (selectionMode) {{
                toggleSelectionMode();
            }}
        }}

    function sendPointsToServer() {{
        if (points.length < 2) {{
            alert("Select 2 points first.");
            return;
        }}

        const data = {{
            start: {{
                lat: points[0].lat,
                lng: points[0].lng
            }},
            end: {{
                lat: points[1].lat,
                lng: points[1].lng
            }}
        }};

        fetch('/submit_points', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify(data)
        }}).then(response => response.json())
        .then(result => {{
            alert("Points sent to Python backend.");
        }});
    }}
    </script>
</body>
</html>
"""
output_html = "select_points_map.html"
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_template)
try:
    webbrowser.open('file://' + os.path.abspath(output_html))
except:
    print(f"Map saved as {output_html}")
    print(f"Please open the file manually: {os.path.abspath(output_html)}")