
#import glob
#import sys
#from os import listdir
#from collections import namedtuple
import itertools as itertools

import numpy             as np
npa    = np.array

import pandas            as pd
import matplotlib.pyplot as plt

#from   scipy             import stats
#from   scipy             import optimize
import networkx as nx

#import hipy.utils        as ut
#import hipy.pltext       as pltext
#from hipy.styles import style
#style()

import nana.topo.evtfun    as evtfun
import nana.topo.graphfun  as graphfun

datafilename = 'dataset_15607_DEP_deco.h5'
mcfilename   = 'dataset_4bar_MC_DEP_deco.h5'
datapath     = '/mnt/netapp1/Store_next_data/NEXT100/deco_seg/'


def dicts_to_dataframe(long_dict, *short_dicts):    
    # number of files
    n = len(next(iter(long_dict.values())))
    
    data = {}

    # Fill columns to the large dictionary (they are list of n-length)
    for key, value_list in long_dict.items():
        if len(value_list) != n:
            raise ValueError(f"La clave '{key}' no tiene exactamente {n} elementos.")
        data[key] = value_list

    # Expand and add the short dictionaries
    for d in short_dicts:
        for key, value in d.items():
            data[key] = [value] * n

    # Convert to DataFrame
    return pd.DataFrame(data)


def zora_event(evt, bin_info, bin_widths):

    sumevt = evtfun.summary_event(evt)

    evt   = evt[evt.decopred == 1]
    coors = evtfun.get_coors(evt, bin_info)
    ene   = evt.enecn.values
    bins  = evtfun.get_bins(coors, bin_widths)

    graph                      = graphfun.convert_to_graph(coors, bins, ene, nsides = 3)
    components, longest_graph  = graphfun.graph_connectivity(graph)

    sumgraph = graphfun.summary_graph(components, longest_graph)

    u, v, dist = graphfun.graph_extremes(longest_graph)
    # print('extremes distance:', dist, ", extrenes nodes:", u, v)
    
    blobs, extremes = graphfun.get_blobs(longest_graph, (u, v), distance = 3)
    sumblobs = graphfun.summary_blobs(longest_graph, extremes, dist, blobs)

    df = dicts_to_dataframe(sumblobs,  sumevt, sumgraph)

    odata = {'graph': graph, 'longest_graph' : longest_graph, 'extremes' : extremes, 'distance' : dist,
             'blobs' : blobs, 'summary' : df}

    return odata





def zora(datapath = datapath, ifilename = datafilename, ofilename = 'zora_output.h5',  nevents = 10):

    ifilename = datapath + ifilename
    print(f'loading {ifilename}')
    data     =  pd.read_hdf(ifilename, 'DATASET/Voxels');
    bin_info =  pd.read_hdf(ifilename, 'DATASET/BinsInfo');
    bin_widths  = evtfun.get_bin_widths(bin_info)

    ndata  = len(data.dataset_id.unique()) - 1
    print(f"Number of events in input file {ndata + 1}")

    nevents = min(nevents, ndata) if nevents > 0 else ndata
    print(f"Number of events to process {nevents}")
    data    = evtfun.get_deconvoluted_data(data[data.dataset_id <= nevents])

    groupid = 'dataset_id'
    dfout = None
    for i, evt in data.groupby(groupid):
        if i % 100 == 0: print(f"Event {i}")
        if (i >= nevents): break
        odata = zora_event(evt, bin_info, bin_widths)
        df = odata['summary']
        dfout = df if i == 1 else pd.concat([dfout, df], ignore_index=True)

    ndata = len(dfout.dataset_id.unique())
    #ofilename = datapath + ofilename
    print(f'saving {ofilename}')
    print(f"Number of events in output file {ndata}")
    dfout.to_hdf(ofilename, key = "zora", mode = "w")
    return dfout