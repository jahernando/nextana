
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

# def get_deconvoluted_event(data):
#     """Get the deconvoluted data with coordinates and normalized energy."""
#     data['enecor'] = data.energy.values * data.decopred.values
#     emissing  = 0.
#     df["enecor"] = df.groupby("zbin")["energy"].transform(lambda x: x * k)

#         zslice.enecor = zslice.enecor.values * ene0/ene1
#         if ene1 == 0: emissing += ene0
#     print(f"  Missing energy in deconvoluted event: {emissing:.2f} (out of {data.energy.sum():.2f})")
#     return data

def get_deconvoluted_data(data):
    """
    reasigh the energy of the slice to the spine voxels according to the fraction of the energy they have in the slice
    ensure that the total energy of the hit on the spine is one.
    inputs:
        data: dataframe with the event data including the decopred column
    outputs:
        data: dataframe with two new columns enecc (corrected energy per slice) and enecn (and normalized energy)
    """
    ## share the energy per slice
    data['enecc'] = data.energy.values * data.decopred.values
    idgroup = ['dataset_id', 'zbin']
    ene  = data.groupby(idgroup).energy.sum()
    enec = data.groupby(idgroup).enecc.sum()
    data['enecc']  = data.groupby(idgroup)['enecc'].transform(lambda x: x * ene[x.name]/enec[x.name] if enec[x.name]>0 else 0)
    ## share the total energy for empty decovoluted slices
    iddata = 'dataset_id'
    etot = data.groupby(iddata).enecc.sum()
    data['enecn'] = data.groupby(iddata) ['enecc'].transform(lambda x: x * (1./etot[x.name]))
    return data

def test_get_deconvoluted_data(data):
    """
    Test that the deconvoluted data has correct energy assignments.
    inputs:
        data: dataframe with the event data including the decopred column with 1/0 values
    """
    data_ = get_deconvoluted_data(data)

    # test that the total energy in the slice is preserved for enecc
    idgroup = ['dataset_id', 'zbin']    
    ene   = data_.groupby(idgroup).energy.sum()
    enecc = data_.groupby(idgroup).enecc.sum()
    enecn = data_.groupby(idgroup).enecn.sum()
    iddata    = 'dataset_id'
    enecc_sum = data_.groupby(iddata) .enecc.sum()

    for (idd, zbin), ecc in enecc.items():
        e00     = ene  .get((idd, zbin), 0.)
        ecn     = enecn.get((idd, zbin), 0.) 
        sumecc  = enecc_sum.get(idd, 0.)

        ## energy per slice should be preserved after selection of devonvoluted voxels
        assert np.isclose(ecc, e00) or np.isclose(ecc, 0.), f"Energy mismatch for id {idd} zbin {zbin}: {ecc} != {e00}"

        ## noralized energy per slice should be correctly computed
        ecc2  = ecn * (sumecc) 
        assert np.isclose(ecc, ecc2), f"corrected normalize energy mismatch for id {idd} {zbin}: {ecc} != {ecc2}"

    return

def event_display(coors, bins, ene = None):
    """Display a 2D histogram of the event given the coordinates and bin width."""
    cmap = 'plasma' # 'viridis', 'cividis', 'plasma'
    cv   = pltext.canvas(4, 2)
    clabel = 'counts' if ene is None else 'energy (normalized)'
    ene    = ene if ene is not None else np.ones_like(coors[0])
    labels = ['X', 'Y', 'Z']
    for k, coor in enumerate(((0, 1), (2, 1), (0, 2))):
        i, j = coor
        cv(k+1); plt.hist2d(coors[i], coors[j], weights = ene, bins = (bins[i], bins[j]), cmap = cmap);
        xl, yl = labels[i], labels[j]
        plt.xlabel(xl); plt.ylabel(yl); plt.title(xl+yl+' projection'); plt.colorbar(label = clabel)
    plt.tight_layout()
    return cv
