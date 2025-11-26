
import glob
import sys
from os import listdir
from collections import namedtuple
import itertools as itertools

import numpy             as np
#import pandas            as pd
import matplotlib.pyplot as plt

#import hipy.utils        as ut
import hipy.pltext       as pltext
#from hipy.styles import style
#style()


def get_coors(data, bin_info):
    """Get the coordinates from binned data and bin info."""
    coors = [bin_info['min_'+x].values + data[x+'bin'].values * bin_info['size_'+x].values for x in ('x', 'y', 'z')] 
    return coors

def get_bin_widths(bin_info):
    """Get the bin widths from bin info"""
    return [bin_info['size_'+x].values for x in ('x', 'y', 'z')]

def get_bins(coors, bin_widths = (1, 1, 1)):
    """Get the bins for each coordinate given the bin widths."""
    bins = [np.arange(np.min(coor) - 3*width/2, np.max(coor) + 3*width/2, width) for coor, width in zip(coors, bin_widths)]
    return bins

def event_display(coors, bins, ene = None):
    """Display a 2D histogram of the event given the coordinates and bin width."""
    cmap = 'plasma' # 'viridis', 'cividis', 'plasma'
    cv   = pltext.canvas(4, 2)
    clabel = 'counts' if ene is None else 'energy (keV)'
    ene    = ene if ene is not None else np.ones_like(coors[0])
    labels = ['X', 'Y', 'Z']
    for k, coor in enumerate(((0, 1), (2, 1), (0, 2))):
        i, j = coor
        cv(k+1); plt.hist2d(coors[i], coors[j], weights = ene, bins = (bins[i], bins[j]), cmap = cmap);
        xl, yl = labels[i], labels[j]
        plt.xlabel(xl); plt.ylabel(yl); plt.title(xl+yl+' projection'); plt.colorbar(label = clabel)
    plt.tight_layout()
    return cv
