import pandas as pd
import networkx as nx
import csv
import math
import time
import random
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Helper
def load_edges(file_path):
    return pd.read_csv(file_path)
def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        u = (float(row.start_x), float(row.start_y))
        v = (float(row.end_x),   float(row.end_y))
        attrs = {
            'distance':       float(row.distance),
            'green_ratio':    float(row.green_ratio),
            'shade':          float(row.shade),
            'pavement_ratio': float(row.pavement_ratio)
        }
        G.add_edge(u, v, **attrs)
    return G
def find_nearest_node(coord, nodes, tol=1e-3):
    x0, y0 = coord
    # nearest, min_dist = None, float('inf')
    # for node in nodes:
    #     d = math.hypot(node[0] - x0, node[1] - y0)
    #     if d < min_dist:
    #         nearest, min_dist = node, d
    # return nearest if min_dist <= tol else None
    return min(nodes, key=lambda node: math.hypot(node[0] - x0, node[1] - y0))
def cumulative_exceed_edge(G, path, max_dist):
    cum = 0.0
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        d = G[u][v]['distance']
        cum += d
        if cum > max_dist:
            return (u, v), cum
    return None, cum
def path_length(G, path):
    return sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
def path_edges(path):
    return [(path[i], path[i+1]) for i in range(len(path)-1)]
def sanity_check_graph(G,source, target, target_distance, tolerance):
    if not isinstance(G, nx.Graph):
        raise TypeError("Input must be a NetworkX graph.")
    if len(G.edges()) == 0:
        raise ValueError("Graph has no edges.")
    if any('distance' not in data for u, v, data in G.edges(data=True)):
        raise ValueError("All edges must have 'distance' attribute.")
    
    if source not in G.nodes():
        alt = find_nearest_node(source, G.nodes())
        if alt:
            print(f"Source {source} not found; using nearest node {alt} instead.")
            source = alt
        else:
            raise ValueError(f"Source node {source} not in graph.")
    if target not in G.nodes():
        alt = find_nearest_node(target, G.nodes())
        if alt:
            print(f"Target {target} not found; using nearest node {alt} instead.")
            target = alt
        else:
            raise ValueError(f"Target node {target} not in graph.")
        
    min_len = nx.shortest_path_length(G, source, target, weight='distance')
    if min_len > target_distance * (1 + tolerance):
        raise ValueError(f"Shortest path length {min_len} is larger than {target_distance * (1 + tolerance)}. Choose two closer points or increase time.")
    return source, target

def find_k_path(G_base, source, target, output_csv, path_SN, k=3, target_distance=200.0, tolerance=0.05, max_time=30, middle_edges_ratio=0.02):
    min_dist = 0
    max_dist = target_distance * (1 + tolerance)
    results = []
    G_work  = G_base.copy()
    start_time = time.time()
    while path_SN['value'] < k:
        # time cutoff
        if time.time() - start_time > max_time:
            print(f"Time limit of {max_time}s reached; stopping search.")
            break

        try:
            paths_iter = nx.shortest_simple_paths(G_work, source, target, weight='distance')
            chosen = None
            for path in paths_iter:
                if path_SN['value'] >= k:
                    print(f"Reached target of {k} paths; stopping search.")
                    break
                # early-stop if cumulative exceeds max_dist
                if any(not G_work.has_edge(u,v) for u,v in path_edges(path)): continue
                exceeded_edge, cum = cumulative_exceed_edge(G_work, path, max_dist)
                if exceeded_edge:
                    u, v = exceeded_edge
                    print(f"Founded path exceeds max distance at edge {u}-{v}, stopping.")
                    break

                total = cum
                if total < min_dist or total > max_dist:
                    u, v = path[-2], path[-1]
                    if G_work.has_edge(u, v):
                        G_work.remove_edge(u, v)
                    continue

                chosen = path
                print(f"Found path {path_SN} with length {total:.2f}.")
                break
        except nx.NetworkXNoPath:
            print(f"No path available at iteration {path_SN}.")
            break


        if not chosen:
            print(f"No more paths within tolerance after {path_SN['value']} paths.")
            break
        
        # record edges
        true_min_dist = target_distance * (1 - tolerance)
        true_max_dist = target_distance * (1 + tolerance)
        valid = True
        if path_length(G_work, chosen) < true_min_dist or path_length(G_work, chosen) > true_max_dist:
            print(f"Chosen path {path_SN} length {path_length(G_work, chosen):.2f} is out of bounds.")
            valid = False
        if valid and path_SN['value'] < k:
            for u, v in path_edges(chosen):
                data = G_work[u][v]
                results.append({
                    'path_id':         path_SN['value'] + 1,
                    'start_x':         u[0],  'start_y': u[1],
                    'end_x':           v[0],  'end_y':   v[1],
                    'distance':        data['distance'],
                    'green_ratio':     data['green_ratio'],
                    'shade':           data['shade'],
                    'pavement_ratio':  data['pavement_ratio']
                })
            path_SN['value'] += 1

        # randomly remove edges in the middle 2% of the path
        edges_list = path_edges(chosen)
        m = len(edges_list)
        start_idx = int(m * (0.5 - middle_edges_ratio/2.0))
        end_idx   = int(m * (0.5 + middle_edges_ratio/2.0))

        if start_idx == end_idx:
            middle_edges = edges_list[start_idx]
            edges_to_remove = [middle_edges]
        else:
            middle_edges = edges_list[start_idx:end_idx]
            kaggle = random.randint(1, len(middle_edges))
            edges_to_remove = random.sample(middle_edges, kaggle)
        
        for u, v in edges_to_remove:
            if G_work.has_edge(u, v):
                G_work.remove_edge(u, v)
    # write CSV
    keys = ['path_id','start_x','start_y','end_x','end_y','distance',
            'green_ratio','shade','pavement_ratio']
    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        # write header only if file is empty
        if f.tell() == 0:
            writer.writeheader()
        
        writer.writerows(results)

    print(f"Exported {len(results)} edges across {path_SN} paths to '{output_csv}'.")

def chlee_test_file_what_the_fuck(source, target):
    load_path = os.path.join(BASE_DIR, '..', 'CSV_file', 'tainan_edges.csv')
    df = load_edges(load_path)
    print("Building graph from edges...")
    G = build_graph(df)

    points_df = pd.read_csv('tainan_random_routes.csv')
    # testcase
    # source = (214348.3195031177,2547868.730116239)
    # target = (215172.271623499,2545335.762676438)
    distance = 60000.0
    t = 1
    k_length = 10
    maximum_time = 60
    output_csv = os.path.join(BASE_DIR, '..', 'CSV_file', 'paths.csv')
    mer = 0.005
    PSN = {'value': 1}
    # testcase


    for idx, row in points_df.iterrows():
        source = (row['start_x'], row['start_y'])
        target = (row['end_x'], row['end_y'])
        output_csv = 'paths.csv'

        print("Performing sanity checks on graph...")
        source_checked, target_checked = sanity_check_graph(G, source, target, distance, t)
        print(f"Sanity check passed for pair {idx}...\nFinding paths...")
        find_k_path(
            G_base         = G,
            source         = source_checked,
            target         = target_checked,
            k              = k_length,
            target_distance= distance,
            tolerance      = t,
            max_time       = maximum_time,
            output_csv     = output_csv,
            middle_edges_ratio= mer,
            path_SN          = PSN
        )

def generate_paths_from_points(edgefile_path, source, target, output_csv, k=3, target_distance=20000.0, tolerance=0.05, max_time=30, middle_edges_ratio=0.02):
    #tolerance = 0.05
    """
    Generate paths from source to target using the specified parameters.
    
    :param edgefile: Path to the CSV file containing edges.
    :param source: Tuple (x, y) representing the source coordinates.
    :param target: Tuple (x, y) representing the target coordinates.
    :param k: Number of paths to find.
    :param target_distance: Target distance for the paths.
    :param tolerance: Tolerance for the path length.
    :param max_time: Maximum time allowed for finding paths.
    :param output_csv: Output CSV file to save the paths.
    :param middle_edges_ratio: Ratio of edges in the middle of the path to randomly remove.
    """
    print("enter_generate_path_func")
    df = load_edges(edgefile_path)
    print("Building graph from edges...")
    G = build_graph(df)
    print("Performing sanity checks on graph...")
    source_checked, target_checked = sanity_check_graph(G, source, target, target_distance, tolerance)
    
    print("Finding paths...")
    PSN = {'value': 1}
    find_k_path(
        G_base         = G,
        source         = source_checked,
        target         = target_checked,
        k              = k,
        target_distance= target_distance,
        tolerance      = tolerance,
        max_time       = max_time,
        output_csv     = output_csv,
        middle_edges_ratio= middle_edges_ratio,
        path_SN          = PSN
    )

def read_path_from_csv(csv_file_path):
    """
    Reads the CSV and returns a list of edges with start/end lat/lng and ratios.
    Each edge is a dict: {
        "start": {"lat": float, "lng": float},
        "end":   {"lat": float, "lng": float},
        "green_ratio": float,
        "shade": float,
        "pavement_ratio": float
    }
    """
    edges = []

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge = {
                "start": {
                    "lat": float(row["start_y"]),
                    "lng": float(row["start_x"])
                },
                "end": {
                    "lat": float(row["end_y"]),
                    "lng": float(row["end_x"])
                },
                "green_ratio": float(row["green_ratio"]),
                "shade": float(row["shade"]),
                "pavement_ratio": float(row["pavement_ratio"])
            }
            edges.append(edge)
    return edges
