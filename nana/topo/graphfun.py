
#import glob
#import sys
#from os import listdir
#from collections import namedtuple
import itertools as itertools

import numpy             as np
npa    = np.array

#import pandas            as pd
import matplotlib.pyplot as plt

#from   scipy             import stats
#from   scipy             import optimize
import networkx as nx

#import hipy.utils        as ut
#import hipy.pltext       as pltext
#from hipy.styles import style
#style()

def convert_to_graph(coors, bins, ene, ene_threshold = 0, nsides = 1):
    """Convert binned data to a graph representation:
    arguments:
        coors: list of 3 arrays with the x, y, z coordinates of the points
        bins:  list of 3 arrays with the bin edges for x, y, z
        ene:   array with the energy values for each point
        ene_threshold: minimum energy to consider a voxel as occupied
        nsides: number of sides to consider for connectivity (1: 6-connectivity,
    returns:
        graph: networkx Graph object representing the 3D structure
    """

    h3d, _ = np.histogramdd(coors, bins = bins, weights = ene)

    voxels  = h3d > ene_threshold
    indices = np.argwhere(voxels > ene_threshold)
    nx_bins, ny_bins, nz_bins = h3d.shape

    cube_index = list(itertools.product([-1, 0, 1], repeat=3))
    cube_index = [cube_id for cube_id in cube_index if np.sum(np.abs(cube_id)) <= nsides]
    print(cube_index)

    def in_range(i, j, k):
        return  (0 <= i < nx_bins) and (0 <= j < ny_bins) and (0 <= k < nz_bins)
 
    graph = nx.Graph()

    for i, j, k in indices:
        graph.add_node((i, j, k), weight = h3d[i, j, k])
        
        for di, dj, dk in cube_index:
            ni, nj, nk = i + di, j + dj, k + dk
            if in_range(ni, nj, nk):
                if voxels[ni, nj, nk]:
                    graph.add_edge((i, j, k), (ni, nj, nk))

    return graph

def graph_extremes(graph):
    """
    Find the two farthest nodes in the graph using double BFS.
    inputs
        graph: networkx Graph object
    returns
        u, v: the two farthest nodes
        dist: the distance between them
    """
    # Start from an arbitrary node
    start = next(iter(graph.nodes()))
    
    # 1) find the farthest node from start
    dist_start = nx.single_source_shortest_path_length(graph, start)
    u = max(dist_start, key=dist_start.get)
    
    # 2) from that farthest node, find its farthest partner
    dist_u = nx.single_source_shortest_path_length(graph, u)
    v = max(dist_u, key=dist_u.get)
    
    # Return the extremes and the distance
    return u, v, dist_u[v]

def graph_connectivity(graph):
    """
    Analyze the connectivity of the graph.
    inputs
        graph: networkx Graph object
    returns
        components: list of sets, each set contains the nodes in a connected component
        largest_component: subgraph corresponding to the largest connected component
    """

    print("Is graph conected? ", nx.is_connected(graph))
    components = list(nx.connected_components(graph))
    print(f"Number of components: {len(components)}")
    for i, comp in enumerate(components, 1):
        print(f"Component {i}: {comp}")

    largest_component_nodes = max(components, key=len)
    largest_component = graph.subgraph(largest_component_nodes).copy()
    return components, largest_component


def subgraph_within_distance(graph, start, n = 1):
    # nodes at distance <= n from start
    nodes = nx.single_source_shortest_path_length(graph, start, cutoff = n).keys()
    # subgrapgh with those nodes
    return graph.subgraph(nodes).copy()


def graph_total_weight(graph):
    """
    Compute the total weight of the graph based on node weights.
    """

    weights   = npa([data["weight"] for _, data in graph.nodes(data = True)])
    total_w = weights.sum()
    return total_w


def graph_barycenter(graph):
    """
    Compute the barycenter of the graph based on node positions and weights.
    """

    positions = npa([npa(node)      for node, _ in graph.nodes(data = True)])
    weights   = npa([data["weight"] for _, data in graph.nodes(data = True)])

    total_w = weights.sum()
    if total_w == 0: raise ValueError("Total weight is zero — the barycenter would be undefined!")

    bcenter = np.sum(positions * weights[:, None], axis = 0) / total_w
    
    return bcenter

def graph_total_weight(graph):
    """
    Compute the total weight of the graph based on node weights.
    """

    weights   = npa([data["weight"] for _, data in graph.nodes(data = True)])
    total_w = weights.sum()
    return total_w

#---------------
#   blobs
#---------------

def get_blobs(graph, extremes, distance):
    blobs     = [subgraph_within_distance(graph, x, distance) for x in extremes]
    blobs_ene = [graph_total_weight(blob) for blob in blobs]
    if blobs_ene[0] < blobs_ene[1]:
        blobs     = blobs[::-1]
        blobs_ene = blobs_ene[::-1]
    blobs_bc  = [graph_barycenter(blob) for blob in blobs]
    return blobs, blobs_ene, blobs_bc

#---------------


def display_graph(graph, extremes = [], blobs = []):
    """
    Display the graph using a 2D layout.
    inputs
        graph: networkx Graph object
    returns
        extremes: tuple of the two farthest nodes
        dist: distance between the two farthest nodes
    """
    cmap = 'plasma' # 'viridis', 'cividis', 'plasma'
    #plt.figure(figsize=(6,6))

    weights = nx.get_node_attributes(graph, "weight")
    node_colors = list(weights.values())
    node_sizes  = [w * 4e3 for w in node_colors]  # escala ajustable
    #node_sizes =  200 * node_sizes/np.max(node_sizes) 

    # Posiciones del grafo
    pos = nx.kamada_kawai_layout(graph)
    #pos = nx.spring_layout(graph, seed = 0)


    # Draw edges with alpha
    nx.draw_networkx_edges(graph, pos, alpha = 0.1, width = 1, edge_color="black")
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size = node_sizes, node_color =node_colors, cmap=plt.cm.viridis, alpha = 0.5)

    if (len(blobs) > 1):
        for blob in blobs:
            xs = [pos[n][0] for n in blob.nodes()]
            ys = [pos[n][1] for n in blob.nodes()]
            plt.scatter(xs, ys, s = 200, c = 'green', marker = '+', linewidths = 3, label = 'blob')

    if len(extremes) == 2:
        xs = [pos[n][0] for n in extremes]
        ys = [pos[n][1] for n in extremes]
        plt.scatter(xs, ys, s = 200, c = 'red', marker = 'x', linewidths = 3, label = 'Extremos')

    #plt.gca().set_aspect("equal")
    #plt.show()

    return