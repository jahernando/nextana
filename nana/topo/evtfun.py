
import glob
import sys
from os import listdir
from collections import namedtuple
import itertools as itertools

import numpy             as np
import pandas            as pd
from   scipy             import stats
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

#import hipy.utils        as ut
import hipy.pltext       as pltext
from hipy.styles import style
style()


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

#--------------

def summary_event(evt):
    idd    = evt.dataset_id.unique()[0]
    nhits  = evt.energy.count()
    etot   = evt.energy.sum()
    bclass = evt.binclass.unique()[0] if "binclass" in evt.columns else -1
    devt   = evt[evt.decopred == 1]
    spine_nhits = devt.energy.count()
    spine_ene   = devt.energy.sum()
    df = {'dataset_id' : idd, 'binclass' : bclass,
          'nhits': nhits, 'energy': etot, 
          'spine_nhits': spine_nhits, 'spine_energy': spine_ene}
    return df


def summary_data(data):
    idd    = data.dataset_id.unique()
    evts   = data.groupby('dataset_id')
    nhits  = evts.energy.count()
    etot   = evts.energy.sum()
    dcov   = data[data.decopred == 1]
    devts  = dcov.groupby('dataset_id')
    spine_nhits = devts.energy.count()
    spine_ene   = devts.energy.sum()
    df = {'dataset_id' : idd, 'nhits': nhits, 'energy': etot, 'spine_nhits': spine_nhits, 'spine_energy': spine_ene}
    return df


#--------------


def display_event(evt, bin_info, ename = 'energy'):

    # sel the voxels with some energy
    ene = evt[ename].values
    sel = ene > 0
    evt_ = evt[sel]

    # get the coordenates
    coors      = get_coors(evt_, bin_info)
    bin_widths = get_bin_widths(bin_info)
    bins       = get_bins(coors, bin_widths)
    ene        = evt_[ename].values

    # display event
    cv = display_coors(coors, bins, ene, 'energy')

    # display mc
    hasmc = 'extlabel' in list(evt.columns)
    if not hasmc: return cv

    cv1 = display_coors_max(coors, bins, evt_.segclass.values, 'segment')

    trklabel = evt_.extlabel.values + 1
    cv2 = display_coors_max(coors, bins, trklabel, 'track')

    return (cv, cv1, cv2)


def _cmap_white(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Copiamos la LUT completa
    lut = cmap(np.linspace(0, 1, cmap.N))

    # El primer color (índice 0) corresponde a valor 0 → lo pintamos de blanco
    lut[0] = [1, 1, 1, 1]  # RGBA blanco puro

    # Crear un nuevo colormap modificado
    cmap_white = pltcolors.ListedColormap(lut)
    return cmap_white
    

def display_coors(coors, bins, value, name):
    """Display a 2D histogram of the event given the coordinates and bin width."""
    cmap_name   = 'plasma' # 'viridis', 'cividis', 'plasma'
    cmap = _cmap_white(cmap_name)

    cmap.set_under('white')
    cv     = pltext.canvas(3, 3)
    clabel = name
    ene    = value
    labels = ['X', 'Y', 'Z']
    for k, coor in enumerate(((0, 1), (2, 1), (0, 2))):
        i, j = coor
        cv(k+1); plt.hist2d(coors[i], coors[j], weights = ene, bins = (bins[i], bins[j]), cmap = cmap);
        xl, yl = labels[i], labels[j]
        plt.xlabel(xl); plt.ylabel(yl); plt.title(xl+yl+' projection'); plt.colorbar(label = clabel)
    plt.tight_layout()
    return cv

def display_coors_max(coors, bins, value, name):

    cmap_name   = 'plasma' # 'viridis', 'cividis', 'plasma'
    cmap = plt.get_cmap(cmap_name)

    cv     = pltext.canvas(3, 3)
    clabel = name
    labels = ['X', 'Y', 'Z']

    for k, coor in enumerate(((0, 1), (2, 1), (0, 2))):
        i, j = coor
        xbins = (bins[i], bins[j])
        histo, xedges, yedges, _ = stats.binned_statistic_2d(coors[i], coors[j], value, statistic = 'max', bins = xbins)
        cv(k+1); plt.pcolormesh(xedges, yedges, histo.T, cmap = cmap)
        xl, yl = labels[i], labels[j]
        plt.xlabel(xl); plt.ylabel(yl); plt.title(xl+yl+' projection'); plt.colorbar(label = clabel)
    plt.tight_layout()
    return cv

#------------------
