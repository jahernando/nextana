
#import glob
#import sys
#from os import listdir
#from collections import namedtuple
import itertools as itertools

import numpy             as np
npa  = np.array
norm = np.linalg.norm

#import pandas            as pd
import matplotlib.pyplot as plt

#from   scipy             import stats
#from   scipy             import optimize
import networkx as nx

#import hipy.utils        as ut
import hipy.pltext       as pltext
from hipy.styles import style
style()

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
    #print(cube_index)

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

    #print("Is graph conected? ", nx.is_connected(graph))
    components = list(nx.connected_components(graph))
    #print(f"Number of components: {len(components)}")
    #for i, comp in enumerate(components, 1):
    #    print(f"Component {i}: {comp}")

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

def summary_graph(graphs, longest_graph):
    """
    return the summary of the graphs and the loguest graph
    """
    ngraphs       = len(graphs)
    lgraph_nhits  = longest_graph.number_of_nodes()
    lgraph_nlinks = longest_graph.number_of_edges()
    lgraph_energy = graph_total_weight(longest_graph) 
    df = {'ngraphs' : ngraphs, 'lgraph_nodes' : lgraph_nhits, 'lgraph_edges' : lgraph_nlinks,
          'lgraph_energy' : lgraph_energy}
    return df


#---------------
#   blobs
#---------------

def get_blobs(graph, extremes, distance = 3):
    """
    return the blobs (subgraph at a gimen distance)
    """
    blobs     = [subgraph_within_distance(graph, x, distance) for x in extremes]
    blobs_ene = [graph_total_weight(blob) for blob in blobs]
    if blobs_ene[0] < blobs_ene[1]:
        blobs     = blobs[::-1]
    ##    blobs_ene = blobs_ene[::-1]
        extremes  = extremes[::-1] 
    blobs_bc  = [graph_barycenter(blob) for blob in blobs]
    return blobs, extremes


def summary_blobs(graph, extremes, distance, blobs):
    """
    return the summary of the blobs
    """

    ext_ene    = [graph.nodes[n]["weight"] for n in extremes]
    ext_degree = [graph.degree[n]          for n in extremes]
    ext_dist   = [distance                 for n in extremes]

    blob_id     = [i for i, _ in enumerate(blobs)]
    blob_ene    = [graph_total_weight(blob)  for blob in blobs]
    blob_nodes  = [blob.number_of_nodes()    for blob in blobs]
    blob_edges  = [blob.number_of_edges()    for blob in blobs]

    dfe = {'ext_ene' : ext_ene, 'ext_degree' : ext_degree, 'ext_dist' : ext_dist}
    dfb = {'blob_id' : blob_id, 'blob_ene' : blob_ene, 'blob_nodes' : blob_nodes, 'blob_edges' : blob_edges}
    dfe.update(dfb)
    return dfe

#------ MC


def summary_blobs_mc(evt, coors, bins, longest_graph, blobs):
    """
    return the summary of the mc information of the blobs
    """
    h3d, _ = np.histogramdd(coors, bins = bins, weights = evt.extlabel)
    voxels  = h3d > 0
    mcextremes = np.argwhere(voxels > 0)
    if (len(mcextremes) <= 0): 
        print('No extremes ', evt.dataset_id.unique(), mcextremes)
        return {'blob_mcext_dist' : [-1., 1.], 'blob_mcext_closenode' : [False, False],
              'blob_mcseg' : [-1., -1.]}


    extnodes = []
    for mcext in mcextremes:
        dd = sorted([(norm(mcext-np.array(node)), node) for node in longest_graph.nodes()])[0]
        extnodes.append(dd[1])
    #print(extnodes)

    blob_mcext_dis          = []
    blob_mcext_closenode_in = []
    for blob in blobs:
        #print('blob > ', blob.nodes())
        dd   = sorted([(norm(mcext-np.array(node)), node) for node in blob.nodes() for mcext in mcextremes])[0]
        isin = np.any([(tuple(mcext) in blob.nodes()) for mcext in extnodes])
        blob_mcext_dis.append(dd[0])
        blob_mcext_closenode_in.append(isin)

    h3d, _ = np.histogramdd(coors, bins = bins, weights = evt.segclass)
    voxels  = h3d >= 3
    labelblobs = np.argwhere(voxels > 0)
    labelblobs = [tuple(n) for n in labelblobs]
    blob_mcsegment = [np.sum([node in labelblobs for node in blob.nodes()]) for blob in blobs]

    
    df = {'blob_mcext_dist' : blob_mcext_dis, 'blob_mcext_closenode' : blob_mcext_closenode_in,
          'blob_mcseg' : blob_mcsegment}
    return df


#---------------
# Display
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