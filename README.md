# Tainan-walker

**City Walk Route Recommendation System**

Tainan-walker is a web-based route planning tool designed for pedestrians in Tainan City. It recommends walking routes that prioritize environmental quality, including shade, greenery, and pavement condition, aiming to provide a more comfortable and enjoyable walking experience.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Design Process](#design-process)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Customization](#customization)
* [Contributing](#contributing)
* [License](#license)

## Overview

Tainan-walker addresses the need for pedestrian-friendly navigation in Tainan by calculating walking paths that favor environmental comfort. Users can interactively select a start and end point on a map, and the system will compute a visually-enhanced path based on three core metrics: shade, greenery, and pavement quality.

## Features

* **Map-Based User Interface**: An interactive Leaflet map for selecting start and end points.
* **Environmentally Weighted Routing**: Route scoring based on pre-processed data layers representing shade, green coverage, and walkable pavement.
* **Color Visualization**: Walking paths are color-coded to represent the dominance of each environmental factor using RGB blending.
* **Legend Support**: A floating legend clarifies the color encoding for route interpretation.
* **Local Routing Engine**: Routing is performed using an internally managed graph and environmental scoring algorithms.

## Design Process

1. **Problem Identification**: Walking in urban environments such as Tainan often lacks comfort due to heat and poor pedestrian infrastructure.
2. **Data Collection and Processing**: Spatial data related to greenery, shade (such as tree canopies or building shadows), and pavement conditions were analyzed and embedded into the routing graph.
3. **Routing Algorithm**: A weighted scoring system evaluates each edge in the graph based on user-selected criteria, using a customizable linear combination of the three environmental scores.
4. **Web Application Development**: The interface was developed using Flask (Python) for backend handling and Leaflet.js for map rendering and interaction.
5. **Testing and Refinement**: Feedback and observations led to improved user interface and performance optimizations.

## Project Structure

```text
Tainan-walker/
├── app.py                   # Main Flask application
├── requirements.txt         # Python dependencies
├── model/
│   └── weight_route.py      # Route generation and scoring logic
├── util/
│   ├── coords_convert.py    # Coordinate system conversion utilities
│   ├── file_read.py         # GeoJSON reading and preprocessing
│   └── graph_construct.py   # Graph construction and adjacency logic
├── templates/
│   └── index.html           # HTML template rendered by Flask
├── static/
│   ├── css/
│   │   └── style.css        # Application styles
│   └── js/
│       └── map.js           # Client-side JavaScript for map interactivity
└── static/data/
    └── Tainan_road_network.geojson  # Routing network with environment scores
```

## Installation

To set up the project locally, follow the steps below:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/nototria/Tainan-walker.git
   cd Tainan-walker
   ```

2. **Create a Virtual Environment** (optional but recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   ```bash
   cd UI
   python app.py
   ```

5. **Open in Browser**

   Visit `http://127.0.0.1:5000` in your web browser.

## Usage

1. **Select Points**: Click on the map to set a start point (A) and end point (B).
2. **Route Generation**: The application calculates a walking path between the points using the weighted scoring algorithm.
3. **Color Interpretation**:

   * Red tones indicate higher pavement quality.
   * Green tones indicate more greenery.
   * Blue tones represent better shade coverage.
   * Mixed tones reflect combinations of the three factors.
4. **Legend Reference**: A legend in the bottom-right corner clarifies the color encoding used for each environmental factor.

## Customization

The environmental weightings used for route scoring can be adjusted by modifying the `weight_route.py` script in the `model/` directory. Each ecological factor contributes to the total edge weight according to user-defined parameters, allowing researchers or developers to tailor the route preferences.

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.

2. Create a new feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit:

   ```bash
   git commit -m "Describe your changes"
   ```

4. Push your branch:

   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a pull request describing your modifications and improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
