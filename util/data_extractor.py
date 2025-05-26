import osmnx as ox
import os.path
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import split
from tqdm import tqdm
import folium
import networkx as nx
import branca.colormap as cm

# Target city and coordinate system
city = "Tainan, Taiwan"
# Mapping coordinate system to EPSG32651, 326xx uses meter as unit, suitable for calculating path
projected_epsg = 32651 

# Calculate attribute
def calc_green_ratio(buffer_geom):
    return green_areas.intersection(buffer_geom).area.sum() / buffer_geom.area

def calc_shade(buffer_geom):
    return min(shade.intersects(buffer_geom).sum() / 10, 1.0)

def compute_pavement_ratio(road_geom,sidewalk_geoms):
    intersection_total = 0
    for sw_geom in sidewalk_geoms:
        if road_geom.intersects(sw_geom):
            inter = road_geom.intersection(sw_geom)
            # intersection maybe LineString or MultiLineString
            if inter.length > 0:
                intersection_total += inter.length
    return min(intersection_total / road_geom.length, 1.0)

if __name__ == '__main__':

    ##### Download walkable street map
    street_path = "tainan_street.geojson"
    if os.path.exists(street_path):
        print("Loading cached street data...")
        G = gpd.read_file(street_path).to_crs(projected_epsg)
    else:
        print("Downloading walkable street network...")
        G = ox.graph_from_place(city, network_type='walk')  # load map from OSM for given city
        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)  # load to geo dataformat, nodes is the point, edge connects two nodes, default is 4326, using laititude and altitude as data unit
        edges = edges.to_crs(epsg=projected_epsg)  
    
    ###### Download gree area geom->green ratio
    green_path = "tainan_green.geojson"
    if os.path.exists(green_path):
        print("Loading cached green data...")
        green_areas = gpd.read_file(green_path).to_crs(projected_epsg)
    else:
        print("Downloading green area geometries...")
        # using tags to filter desired data
        # loading green area, used to intersect with city map to find green ratio
        green_tags = {
            'leisure': ['park', 'garden','playground'],
            'landuse': ['grass', 'forest','meadow','orchard','vine','greenery','village_green'],
            'natural': ['wood', 'tree', 'scrub','tree_row']
        }
        green_areas = ox.features_from_place(city, tags=green_tags).to_crs(projected_epsg)
        green_areas.to_file(green_path,driver="GeoJSON") 
   
    ##### Download shade geom->shade
    shade_path = "tainan_shade.geojson"
    if os.path.exists(shade_path):
        print("Loading cached shade data...")
        shade = gpd.read_file(shade_path).to_crs(epsg=projected_epsg)
    else:
        print("Downloading shade data from OSM...")
        # loading trees or tree_row to find shade ratio
        shade_tags = {
            'natural': ['tree_row'],
            'building' : ['apartments','roof']      
        }
        shade = ox.features_from_place(city, tags=shade_tags).to_crs(projected_epsg)
        shade.to_file(shade_path,driver="GeoJSON")
    
    ####### Download sidewalk geom->pavement
    sidewalk_path = "tainan_sidewalks.geojson"
    if os.path.exists(sidewalk_path):
        print("Loading cached sidewalk data...")
        sidewalks = gpd.read_file(sidewalk_path).to_crs(epsg=projected_epsg)
    else:
        print("Downloading sidewalk data from OSM...")
        sidewalk_tags = {
            'highway': ['footway', 'path', 'pedestrian', 'residential', 'service','living_street'],
            'footway': ['sidewalk']                                 
        }
        sidewalks = ox.features_from_place("Tainan, Taiwan", tags=sidewalk_tags)
        sidewalks = sidewalks[sidewalks.geometry.type == 'LineString']
        sidewalks = sidewalks.to_crs(epsg=projected_epsg)
        sidewalks.to_file(sidewalk_path, driver="GeoJSON")

    ###### Calculate attributes
    print("Copmuting green_ratio...")

    # Buffer area, used to calcaulate ratio
    edges['buffer'] = edges.geometry.buffer(15) # broaden road from center, adding buffer with radius 15m
    
    edges['distance'] = edges.geometry.length
    edges['green_ratio'] = edges['buffer'].apply(calc_green_ratio)
    edges['shade'] = edges['buffer'].apply(calc_shade)
    pavement_ratios = []
    for geom in tqdm(edges['geometry'], desc="Computing pavement_ratio"):
        pavement_ratios.append(compute_pavement_ratio(geom,sidewalks.geometry))
    edges['pavement_ratio'] = pavement_ratios

    # Extract coordinate, each data point is coordinate under ESPG 32651
    edges['start_x'] = edges.geometry.apply(lambda line: line.coords[0][0])
    edges['start_y'] = edges.geometry.apply(lambda line: line.coords[0][1])
    edges['end_x'] = edges.geometry.apply(lambda line: line.coords[-1][0])
    edges['end_y'] = edges.geometry.apply(lambda line: line.coords[-1][1])

    # CSV
    output_df = edges[['start_x', 'start_y', 'end_x', 'end_y', 'distance',
                    'green_ratio', 'pavement_ratio', 'shade']]

    output_df.to_csv("tainan_edges.csv", index=False)
    print("CSV exported to tainan_edges.csv")
    
    ##### Render map
    print("Rendering interactive map...")
    # Center city
    center_latlon = ox.geocode(city)
    m = folium.Map(location=center_latlon, zoom_start=14, tiles="CartoDB positron")

    # Establish color mapping according to green ratio
    colormap = cm.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0, vmax=1)
    colormap.caption = 'Green Ratio'

    # Convert edge to WGS84 for folium
    edges_wgs = edges.to_crs(epsg=4326)

    for _, row in edges_wgs.iterrows():
        coords = list(row['geometry'].coords)
        popup_text = (
            f"Green ratio: {row['green_ratio']:.2f}<br>"
            f"Shade ratio: {row['shade']:.2f}<br>"
            f"Pavement ratio: {'is' if row['pavement_ratio'] == 1.0 else 'not'}<br>"
            f"Length: {row['distance']:.1f} m"
        )
        folium.PolyLine(
            locations=[(y, x) for x, y in coords],
            color=colormap(row['green_ratio']),
            weight=3,
            opacity=0.7,
        ).add_to(m)

    colormap.add_to(m)

    m.save("tainan_map.html")
    print("Map saved as tainan_map.html")
