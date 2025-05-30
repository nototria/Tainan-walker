import osmnx as ox
import os.path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
import pandas as pd                             # A data type represent a table
import geopandas as gpd                         # Additional geometry colume and crs operation than pandas dataframe
from shapely.geometry import LineString         # geometry is a geoseries data type, a shapely object:point, linestring, polygon...
from shapely.strtree import STRtree
from tqdm import tqdm
import folium
import networkx as nx
import branca.colormap as cm
import matplotlib.colors as mcolors             # Convert numeric to color
import matplotlib.cm as cmx

# Target city and coordinate system
city = "Tainan, Taiwan"

# Mapping coordinate system to EPSG32651, 326xx uses meter as unit, suitable for calculating path
projected_epsg = 32651 

# broaden the line's, making it detectable with other objects intersection
buffer_size = 15

# Flatten list-type attributes to string for GeoJSON compatibility
# Data obtains from OSM contains list type data, e.g. highway = [path, motorway], but GeoJSON doesn't support!
# So when saving to geojson, list type data will be skipped and be NULL, so convert to string to prevent. 
def flatten_attributes(df):
    for col in df.columns:
        if col == 'geometry':
            continue
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
    return df

# Calculate attribute
# whether there are same nodes in two datatype, ex: 
# line = LineString([(0, 0), (2, 2)])
# poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
# intersection = (1 1, 2 2) ===> return intersection area
# intersects = true === >return boolean

# Linestring + polygon
def compute_green_ratio(buffer_geom):
    area = buffer_geom.area 
    # Polygon -> area
    poly_idxs = poly_green_tree.query(buffer_geom)
    poly_area = sum(poly_green_geoms[i].intersection(buffer_geom).area 
                    for i in poly_idxs 
                    if poly_green_geoms[i].intersects(buffer_geom))

    # Line / Point -> number
    line_idxs = line_green_tree.query(buffer_geom)
    line_hits = sum(line_green_geoms[i].intersection(buffer_geom).area 
                    for i in line_idxs 
                    if line_green_geoms[i].intersects(buffer_geom))
    
    result = min((poly_area / area) + (line_hits / buffer_size), 1.0)
    dbg.write(str(result)+"\n")
    return result

# Linestring intersection + polygon
def compute_shade(buffer_geom):
    area = buffer_geom.area
    poly_total = 0.0
    line_total = 0.0

    poly_idxs = poly_shade_tree.query(buffer_geom)
    poly_total = sum(poly_shade_geoms[i].intersection(buffer_geom).area
                    for i in poly_idxs 
                    if poly_shade_geoms[i].intersects(buffer_geom))

    line_idxs = line_shade_tree.query(buffer_geom)
    line_total = sum(line_shade_geoms[i].intersection(buffer_geom).length 
                    for i in line_idxs 
                    if line_shade_geoms[i].intersects(buffer_geom))

    shade_area = poly_total + line_total  
    result = min(shade_area / area, 1.0) 
    dbs.write(str(result) + "\n")
    return result

# Linestring intersection
def compute_pavement_ratio(buffer_geom):
    intersection_total = 0.0
    idxs = sidewalk_index.query(buffer_geom)

    for i in idxs:
        sw_geom = sidewalk_geoms[i]                
        if sw_geom.intersects(buffer_geom):
            inter = sw_geom.intersection(buffer_geom)
            if inter.length > 0:
                intersection_total += inter.length
    result = min(intersection_total / buffer_geom.length, 1.0)
    dbp.write(str(result) + "\n")
    return result

# Helper function for computation
def normalize_geom(geom):
    if geom.geom_type == "Point":
        return geom.buffer(2).boundary  # Buffer point type data
    else:
        return geom
    
if __name__ == '__main__':

    #=========================# 
    # Download attribute maps #
    #=========================#

    ###### Download walkable street map
    street_path = os.path.join(BASE_DIR,"tainan_street.geojson")
    if os.path.exists(street_path):
        print("Loading cached street data...")
        edges = gpd.read_file(street_path).to_crs(projected_epsg)
    else:
        print("Downloading walkable street network...")
        G = ox.graph_from_place(city, network_type='walk')  # load map from OSM for given city
        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)  # load to geo dataformat, nodes is the point, edge connects two nodes, default is 4326, using laititude and altitude as data unit
        edges = flatten_attributes(edges)
        edges.to_file(street_path ,driver="GeoJSON")    # It's better to save geojson as epsg=4326, complys with RFC 7946
        edges = edges.to_crs(epsg=projected_epsg)  

    
    ###### Download green area geom->green ratio
    green_path = os.path.join(BASE_DIR,"tainan_green.geojson")
    if os.path.exists(green_path):
        print("Loading cached green data...")
        green_areas = gpd.read_file(green_path).to_crs(projected_epsg)
    else:
        print("Downloading green area geometries...")
        # using tags to filter desired data
        # loading green area, used to intersect with city map to find green ratio
        green_tags = {
            'leisure': ['park', 'garden'],
            'landuse': ['grass', 'forest','meadow','orchard','vine','greenery','village_green'],
            'natural': ['wood', 'tree', 'scrub','tree_row'],
            'highway': ['track']
        }
        green_areas = ox.features_from_place(city, tags=green_tags)
        green_areas['geometry'] = green_areas['geometry'].apply(normalize_geom)
        green_areas = green_areas[~green_areas['geometry'].isna()]
        green_areas = flatten_attributes(green_areas)
        green_areas.to_file(green_path,driver="GeoJSON") 
        green_areas = green_areas.to_crs(epsg=projected_epsg)

    print("==========Green_Areas==========")
    print(green_areas['geometry'].geom_type.value_counts())
    print("===============================\n")


   
    ###### Download shade geom->shade
    shade_path = os.path.join(BASE_DIR,"tainan_shade.geojson")
    if os.path.exists(shade_path):
        print("Loading cached shade data...")
        shade = gpd.read_file(shade_path).to_crs(epsg=projected_epsg)
    else:
        print("Downloading shade data from OSM...")
        shade_tags = {
            'natural': ['tree_row','tree'],
            'building': ['apartments','roof'],
            'highway': ['corridor'],
            'amenity': ['shelter'],
            'man_made': ['canopy','awning'],
            'covered': ['yes'] 
        }
        shade = ox.features_from_place(city, tags=shade_tags)
        shade['geometry'] = shade['geometry'].apply(normalize_geom)
        shade = shade[~shade['geometry'].isna()]    # filter data whose geometry colume is NaN
        shade = flatten_attributes(shade)
        shade.to_file(shade_path, driver="GeoJSON")
        shade = shade.to_crs(epsg=projected_epsg)

    print("==========Shade_Areas==========")
    print(shade['geometry'].geom_type.value_counts())
    print("===============================\n")

    
    ####### Download sidewalk geom->pavement
    sidewalk_path = os.path.join(BASE_DIR,"tainan_pavement.geojson")
    if os.path.exists(sidewalk_path):
        print("Loading cached sidewalk data...")
        sidewalks = gpd.read_file(sidewalk_path).to_crs(epsg=projected_epsg)
    else:
        print("Downloading sidewalk data from OSM...")
        sidewalk_tags = {
            'highway': ['footway', 'path', 'pedestrian', 'residential', 'service','living_street','bridleway'],
            'footway': ['sidewalk','crossing'],
        }
        sidewalks = ox.features_from_place(city, tags=sidewalk_tags)
        # only want Linestring type data to calculate ratio, point, polygon will be filtered
        sidewalks = sidewalks[sidewalks.geometry.type == 'LineString']
        sidewalks = sidewalks[~sidewalks['geometry'].isna()]
        sidewalks = flatten_attributes(sidewalks)
        sidewalks.to_file(sidewalk_path, driver="GeoJSON")
        sidewalks = sidewalks.to_crs(epsg=projected_epsg)

    print("==========Sidewalks==========")
    print(sidewalks['geometry'].geom_type.value_counts())
    print("=============================\n")

    #======================# 
    # Calculate attributes #
    #======================#

    # Buffer area, used to calcaulate ratio
    edges['buffer'] = edges.geometry.buffer(buffer_size) # broaden road from center, adding buffer with radius 15m

    edges['distance'] = edges.geometry.length
    
    ##### Compute green ratio
    poly_green_geoms = list(green_areas[green_areas.geometry.type.isin(['Polygon', 'MultiPolygon'])].geometry)
    line_green_geoms = list(green_areas[green_areas.geometry.type.isin(['LineString', 'MultiLineString','Point'])].geometry)
    poly_green_tree = STRtree(poly_green_geoms)
    line_green_tree = STRtree(line_green_geoms)
    
    green_ratios = []
    with open("debug_green.txt", "w") as dbg:
        for geom in tqdm(edges['buffer'],desc="Computing green_ratio"):
            green_ratios.append(compute_green_ratio(geom))
    edges['green_ratio'] = green_ratios
    
    ##### Compute shade
    poly_shade_geoms = list(shade[shade.geometry.type.isin(['Polygon', 'MultiPolygon'])].geometry)
    line_shade_geoms = list(shade[shade.geometry.type.isin(['LineString', 'MultiLineString','Point'])].geometry)
    poly_shade_tree = STRtree(poly_shade_geoms)
    line_shade_tree = STRtree(line_shade_geoms)
    shade_ratio = []
    with open("debug_shade.txt", "w") as dbs:        
        for geom in tqdm(edges['buffer'],desc="Computing shade"):
            shade_ratio.append(compute_shade(geom))
    edges['shade'] = shade_ratio

    ##### Compute pavement_ratio
    sidewalk_geoms = list(sidewalks.geometry)          
    sidewalk_index = STRtree(sidewalk_geoms)
    pavement_ratios = []
    with open("debug_pavement.txt", "w") as dbp:        
        for geom in tqdm(edges['buffer'], desc="Computing pavement_ratio"):
            pavement_ratios.append(compute_pavement_ratio(geom))
    edges['pavement_ratio'] = pavement_ratios

    #============# 
    # CSV Output #
    #============#
    csv_path = os.path.join(BASE_DIR,"tainan_edges.csv")
    # Extract coordinate, each data point is coordinate under ESPG 32651
    edges['start_x'] = edges.geometry.apply(lambda line: line.coords[0][0])
    edges['start_y'] = edges.geometry.apply(lambda line: line.coords[0][1])
    edges['end_x'] = edges.geometry.apply(lambda line: line.coords[-1][0])
    edges['end_y'] = edges.geometry.apply(lambda line: line.coords[-1][1])

    # Output to CSV
    output_df = edges[['start_x', 'start_y', 'end_x', 'end_y', 'distance',
                    'green_ratio', 'pavement_ratio', 'shade']]

    output_df.to_csv(csv_path, index=False)
    print("CSV exported to", csv_path)
    
    #======================# 
    #      Render map      #
    #======================#

    map_path = os.path.join(BASE_DIR,"tainan_map.html")
    print("Rendering interactive map...")
    center_latlon = ox.geocode(city)
    m = folium.Map(location=center_latlon, zoom_start=14, tiles="CartoDB positron")

    edges_wgs = edges.to_crs(epsg=4326)

    def compute_score(row):
        return (
            1.0 * row['green_ratio'] +
            1.0 * row['shade'] +
            1.0 * row['pavement_ratio']
        )

    edges_wgs['score'] = edges_wgs.apply(compute_score, axis=1)
    score_min = edges_wgs['score'].min()
    score_max = edges_wgs['score'].max()

    # Configure gradient (plasma, viridis, inferno, etc.)
    colormap = cmx.ScalarMappable(norm=mcolors.Normalize(vmin=score_min, vmax=score_max), cmap='viridis')

    for _, row in edges_wgs.iterrows():
        coords = list(row['geometry'].coords)
        popup_text = (
            f"<b>Green ratio</b>: {row['green_ratio']:.2f}<br>"
            f"<b>Shade ratio</b>: {row['shade']:.2f}<br>"
            f"<b>Pavement ratio</b>: {row['pavement_ratio']:.2f}<br>"
            f"<b>Length</b>: {row['distance']:.1f} m<br>"
            f"<b>Score</b>: {row['score']:.2f}"
        )

        color = mcolors.to_hex(colormap.to_rgba(row['score']))

        folium.PolyLine(
            locations=[(y, x) for x, y in coords],
            color=color,
            weight=4,
            opacity=0.85,
            popup=folium.Popup(popup_text, max_width=250),
        ).add_to(m)

    # legend 
    colormap._A = []  # trick to make ScalarMappable printable
    colormap_caption = cm.LinearColormap(['#440154', '#21918c', '#fde725'], vmin=score_min, vmax=score_max)
    colormap_caption.caption = 'Path Quality Score'
    colormap_caption.add_to(m)

    m.save(map_path)
    print(f"Map saved as {map_path}")

