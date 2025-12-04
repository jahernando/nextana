
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

def convert_to_graph_direct(evt, bin_info, ename = 'energy', nsides = 3):

    ## get the information of the nodes
    enes = evt[ename].values
    sel  = enes > 0
    evt_ = evt[sel]
    icoors = [evt_[x+'bin'].values for x in ('x', 'y', 'z')]
    xcoors = [bin_info['min_'+x].values + evt_[x+'bin'].values * bin_info['size_'+x].values for x in ('x', 'y', 'z')] 
    enes   = evt_[ename].values 

    ## create the graph
    graph = nx.Graph()
    nnodes = len(enes)
    for ii in range(nnodes):
        icoor = [icoors[k][ii] for k in range(3)]
        xcoor = [xcoors[k][ii] for k in range(3)]
        ene   = enes[ii]
        graph.add_node(tuple(icoor), energy = ene, position = npa(xcoor))

    ## create the links
    cube_index = list(itertools.product([-1, 0, 1], repeat=3))
    cube_index = [cube_id for cube_id in cube_index if np.sum(np.abs(cube_id)) <= nsides]
    nodes_labels = graph.nodes()
    for i, j, k in nodes_labels:        
        for di, dj, dk in cube_index:
            ni, nj, nk = i + di, j + dj, k + dk
            if (ni, nj, nk) in nodes_labels:
                graph.add_edge((i, j, k), (ni, nj, nk))

    return graph


def get_nodes_variable(graph, varname):
    weights   = npa([data[varname] for _, data in graph.nodes(data = True)])
    return weights

def graph_energy(graph):
    return np.sum(get_nodes_variable(graph, "energy"))

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
    return (u, v), dist_u[v]

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
    subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    #components = list(nx.connected_components(graph))
    #print(f"Number of components: {len(components)}")
    #for i, comp in enumerate(components, 1):
    #    print(f"Component {i}: {comp}")

    largest_component_nodes = max(subgraphs, key=len)
    largest_component = graph.subgraph(largest_component_nodes).copy()
    return subgraphs, largest_component


def graph_barycenter(graph):
    """
    Compute the barycenter of the graph based on node positions and weights.
    """

    positions = get_nodes_variable(graph, 'position')
    weights   = get_nodes_variable(graph, 'energy')

    total_w = weights.sum()
    if total_w == 0: raise ValueError("Total weight is zero — the barycenter would be undefined!")

    bcenter = np.sum(positions * weights[:, None], axis = 0) / total_w
    
    return bcenter

def subgraph_inradius(graph, pos, radius = 21.):

    in_radius = lambda data : norm(data['position'] - pos) < radius

    selected_nodes = [node for node, data in graph.nodes(data = True) if in_radius(data)]
    return graph.subgraph(selected_nodes).copy()


def subgraph_atdistance(graph, node, distance = 3):
    """
    return the blobs (subgraph at a gimen distance)
    """
    nodes = nx.single_source_shortest_path_length(graph, node, cutoff = distance).keys()
    return graph.subgraph(nodes).copy()


def closest_node(graph, pos):
    dis = lambda data : norm(data['position'] - pos)
    dd = sorted([(dis(data), node) for node, data in graph.nodes(data = True)])
    return dd[0]


#----------
# Data
#----------

def get_graphs(evt, bin_info, ename = 'energy'):

    sel   = evt[ename].values > 0
    graph                  = convert_to_graph_direct(evt[sel], bin_info, ename = ename)
    graphs, longest_graph  = graph_connectivity(graph)

    return graph, graphs, longest_graph


def summary_graph(graphs, longest_graph):
    """
    return the summary of the graphs and the loguest graph
    """
    ngraphs       = len(graphs)
    lgraph_nhits  = longest_graph.number_of_nodes()
    lgraph_nlinks = longest_graph.number_of_edges()
    lgraph_energy = np.sum(get_nodes_variable(longest_graph, 'energy')) 
    df = {'ngraphs' : ngraphs, 'lgraph_nodes' : lgraph_nhits, 'lgraph_edges' : lgraph_nlinks,
          'lgraph_energy' : lgraph_energy}
    return df

#------------
# Blobs
#------------

def get_blobs(graph, extremes):

    xpos      = [graph.nodes[extreme]["position"] for extreme in extremes]
    blobs     = [subgraph_inradius(graph, x)    for x in xpos] 
    blobs_ene = [graph_energy(b) for b in blobs]
    if blobs_ene[0] < blobs_ene[1]:
        blobs_ene = blobs_ene[::-1]
        blobs     = blobs[::-1]
        extremes  = extremes[::-1] 

    return blobs, extremes, blobs_ene


def summary_blobs(graph, extremes, distance, blobs, blobs_ene):
    """
    return the summary of the blobs
    """

    ext_ene    = [graph.nodes[n]["energy"] for n in extremes]
    ext_degree = [graph.degree[n]          for n in extremes]
    ext_dist   = [distance                 for n in extremes]

    blob_id     = [i for i, _ in enumerate(blobs)]
    blob_nodes  = [blob.number_of_nodes()    for blob in blobs]
    blob_edges  = [blob.number_of_edges()    for blob in blobs]

    dfe = {'ext_ene' : ext_ene, 'ext_degree' : ext_degree, 'ext_dist' : ext_dist}
    dfb = {'blob_id' : blob_id, 'blob_ene' : blobs_ene, 'blob_nodes' : blob_nodes, 'blob_edges' : blob_edges}
    dfe.update(dfb)
    return dfe


#-------------
# MC
#--------------

def get_mcgraphs(evt, bin_info):

    # track
    sel = evt.segclass > 1
    tgraph      = convert_to_graph_direct(evt[sel], bin_info, ename = 'energy')
    tgraphs, longest_tgraph  = graph_connectivity(tgraph)

    #blobs
    sel = evt.segclass == 3
    bgraph      = convert_to_graph_direct(evt[sel], bin_info, ename = 'energy')
    bgraphs, _  = graph_connectivity(bgraph)

    # extremes
    sel = evt.extlabel > 0
    egraph      = convert_to_graph_direct(evt[sel], bin_info, ename = 'energy')
    egraphs, _  = graph_connectivity(egraph)

    return tgraph, tgraphs, longest_tgraph, bgraphs, egraphs


def summary_mcext(graph, extremes, blobs, ext_mcgraphs):
    """
    return the summary of the mc information of the blobs
    """

    xmcpos = [graph_barycenter(gext) for gext in ext_mcgraphs]

    xpos   = [graph.nodes[ext]['position'] for ext in extremes]
    bpos   = [graph_barycenter(blob) for blob in blobs]

    ddext  = [np.min([norm(x - xmc) for xmc in xmcpos]) for x in xpos]
    ddblob = [np.min([norm(x - xmc) for xmc in xmcpos]) for x in bpos]

    df = {'mcext_ext' : ddext , 'mcext_blob': ddblob}
    return df


def summary_mcblobs(blobs, blob_mcgraphs):

    bene = [np.max([np.sum([data['energy'] for node, data in mcblob.nodes() if node in blob.nodes()]) for mcblob in blob_mcgraphs]) for blob in blobs]

    df = {'mcblob_ene' : bene}
    return df

#     h3d, _ = np.histogramdd(coors, bins = bins, weights = evt.extlabel)
#     voxels  = h3d > 0
#     mcextremes = np.argwhere(voxels > 0)
#     if (len(mcextremes) <= 0): 
#         print('No extremes ', evt.dataset_id.unique(), mcextremes)
#         return {'blob_mcext_dist' : [-1., 1.], 'blob_mcext_closenode' : [False, False],
#               'blob_mcseg' : [-1., -1.]}


#     extnodes = []
#     for mcext in mcextremes:
#         dd = sorted([(norm(mcext-np.array(node)), node) for node in longest_graph.nodes()])[0]
#         extnodes.append(dd[1])
#     #print(extnodes)

#     blob_mcext_dis          = []
#     blob_mcext_closenode_in = []
#     for blob in blobs:
#         #print('blob > ', blob.nodes())
#         dd   = sorted([(norm(mcext-np.array(node)), node) for node in blob.nodes() for mcext in mcextremes])[0]
#         isin = np.any([(tuple(mcext) in blob.nodes()) for mcext in extnodes])
#         blob_mcext_dis.append(dd[0])
#         blob_mcext_closenode_in.append(isin)

#     h3d, _ = np.histogramdd(coors, bins = bins, weights = evt.segclass)
#     voxels  = h3d >= 3
#     labelblobs = np.argwhere(voxels > 0)
#     labelblobs = [tuple(n) for n in labelblobs]
#     blob_mcsegment = [np.sum([node in labelblobs for node in blob.nodes()]) for blob in blobs]

    
#     df = {'blob_mcext_dist' : blob_mcext_dis, 'blob_mcext_closenode' : blob_mcext_closenode_in,
#           'blob_mcseg' : blob_mcsegment}
#     return df




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

    weights = nx.get_node_attributes(graph, "energy")
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

    if len(extremes) > 1:
        cc = ['red', 'green']
        for i, ext in enumerate(extremes):
            xs = [pos[n][0] for n in ext]
            ys = [pos[n][1] for n in ext]
            plt.scatter(xs, ys, s = 200, c = cc[i], marker = 'x', linewidths = 3, label = 'Extremos')

    plt.tight_layout()
    #plt.gca().set_aspect("equal")
    #plt.show()

    return

def display_graph_event(graph, extremes = [], blobs = []):
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

    weights = nx.get_node_attributes(graph, "energy")
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

def display_graph_scatter(graph):    
    cmap = 'plasma'
    cv = pltext.canvas(4, 2)
    pos = get_nodes_variable(graph, 'position')
    ene = get_nodes_variable(graph, 'energy')
    labels = ['X', 'Y', 'Z']
    xs = [[x[i] for x in pos] for i in range(3)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cv(i+1)
        plt.scatter(xs[i], xs[j], alpha = 0.5, c = ene, s = 2e3*ene, cmap = cmap, marker = 's')
        xl, yl = labels[i], labels[j]
        plt.xlabel(xl); plt.ylabel(yl); plt.title(xl+yl+' projection'); plt.colorbar(label = 'energy')
    plt.tight_layout()
    return cv
