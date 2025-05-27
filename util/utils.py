import pandas as pd
import networkx as nx
import csv
import math
import time
import random
import os

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
    nearest, min_dist = None, float('inf')
    for node in nodes:
        d = math.hypot(node[0] - x0, node[1] - y0)
        if d < min_dist:
            nearest, min_dist = node, d
    return nearest if min_dist <= tol else None
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
        raise ValueError(f"Shortest path length {min_len} is out of bounds {target_distance * (1 + tolerance)}.")    
    return source, target

def find_k_paths_with_distance_and_tolerance(G_base,source,target,found,k=3,target_distance=200.0,tolerance=0.05,max_time=30,output_csv='paths.csv', middle_edges_ratio=0.02):
    min_dist = 0
    max_dist = target_distance * (1 + tolerance)
    results = []
    G_work  = G_base.copy()
    start_time = time.time()
    for path_id in range(1, k+1):
        # time cutoff
        if time.time() - start_time > max_time:
            print(f"Time limit of {max_time}s reached; stopping search.")
            break

        try:
            paths_iter = nx.shortest_simple_paths(G_work, source, target, weight='distance')
            chosen = None
            for path in paths_iter:
                # early-stop if cumulative exceeds max_dist
                if any(not G_work.has_edge(u,v) for u,v in path_edges(path)): continue
                exceeded_edge, cum = cumulative_exceed_edge(G_work, path, max_dist)
                if exceeded_edge:
                    u, v = exceeded_edge
                    print(f"Path {path_id} exceeds max distance at edge {u}-{v}, stopping.")
                    break

                total = cum
                if total < min_dist or total > max_dist:
                    u, v = path[-2], path[-1]
                    if G_work.has_edge(u, v):
                        G_work.remove_edge(u, v)
                    continue

                chosen = path
                print(f"Found path {path_id} with length {total:.2f}.")
                break
        except nx.NetworkXNoPath:
            print(f"No path available at iteration {path_id}.")
            break


        if not chosen:
            print(f"No more paths within tolerance after {found['value']} paths.")
            break
        
        # record edges
        true_min_dist = target_distance * (1 - tolerance)
        true_max_dist = target_distance * (1 + tolerance)
        valid = True
        if path_length(G_work, chosen) < true_min_dist or path_length(G_work, chosen) > true_max_dist:
            print(f"Chosen path {path_id} length {path_length(G_work, chosen):.2f} is out of bounds.")
            valid = False
        if valid:
            for u, v in path_edges(chosen):
                data = G_work[u][v]
                results.append({
                    'path_id':         found['value'] + 1,
                    'start_x':         u[0],  'start_y': u[1],
                    'end_x':           v[0],  'end_y':   v[1],
                    'distance':        data['distance'],
                    'green_ratio':     data['green_ratio'],
                    'shade':           data['shade'],
                    'pavement_ratio':  data['pavement_ratio']
                })
            found['value'] += 1

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
            k = random.randint(1, len(middle_edges))
            edges_to_remove = random.sample(middle_edges, k)
        
        for u, v in edges_to_remove:
            if G_work.has_edge(u, v):
                G_work.remove_edge(u, v)
    # write CSV
    keys = ['path_id','start_x','start_y','end_x','end_y','distance',
            'green_ratio','shade','pavement_ratio']
    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        
        # 如果檔案是空的，再寫入 header
        if f.tell() == 0:
            writer.writeheader()
        
        writer.writerows(results)

    print(f"Exported {len(results)} edges across {found} paths to '{output_csv}'.")

if __name__ == '__main__':
    if os.path.exists('paths.csv'):
        os.remove('paths.csv')
    df = load_edges('tainan_edges.csv')
    print("Building graph from edges...")
    G = build_graph(df)

    points_df = pd.read_csv('tainan_random_routes.csv')
    # testcase
    founded = {'value': 0}
    distance = 500000.0
    t = 1
    k_length = 10
    maximum_time = 60
    output_csv = 'paths.csv'
    mer = 0.01
    # testcase


    for idx, row in points_df.iterrows():
        source = (row['start_x'], row['start_y'])
        target = (row['end_x'], row['end_y'])
        output_csv = 'paths.csv'

        print("Performing sanity checks on graph...")
        source_checked, target_checked = sanity_check_graph(G, source, target, distance, t)
        print(f"Sanity check passed for pair {idx}...\nFinding paths...")
        find_k_paths_with_distance_and_tolerance(
            G_base         = G,
            source         = source_checked,
            target         = target_checked,
            k              = k_length,
            target_distance= distance,
            tolerance      = t,
            max_time       = maximum_time,
            output_csv     = output_csv,
            found          = founded
        )
