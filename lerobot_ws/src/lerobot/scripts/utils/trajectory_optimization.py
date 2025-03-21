import numpy as np
import pandas as pd
from glob import glob

def nearest_neighbor_tsp_clusters(cluster_info_df, start_cluster=0):
    n = len(cluster_info_df)
    
    if n == 0:
        raise ValueError("No clusters found. Check your input data.")
    visited = [False] * n
    
    route = []
    
    current = start_cluster
    
    route.append(current)
    
    visited[current] = True
    
    while len(route) < n:
        current_end_x = cluster_info_df.loc[current, 'end_x']
        current_end_y = cluster_info_df.loc[current, 'end_y']
        next_cluster = None
        min_dist = float('inf')
        for i in range(n):
            if not visited[i]:
                candidate_start_x = cluster_info_df.loc[i, 'start_x']
                candidate_start_y = cluster_info_df.loc[i, 'start_y']
                dist = np.sqrt((current_end_x - candidate_start_x)**2 +
                               (current_end_y - candidate_start_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    next_cluster = i
        if next_cluster is None:
            # This should not happen if there are unvisited clusters.
            break
        route.append(next_cluster)
        visited[next_cluster] = True
        current = next_cluster
    return route


# 5. Calculate the total travel distance for a given set of waypoints.
def total_distance(df):
    xs = df['x'].values
    ys = df['y'].values
    dx = np.diff(xs)
    dy = np.diff(ys)
    distances = np.sqrt(dx**2 + dy**2)
    return np.sum(distances)
