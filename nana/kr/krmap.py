#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 13:37:09 2022

@author: hernando
"""


import numpy  as np
import pandas as pd

from   collections import namedtuple
from   scipy       import stats
from   scipy       import optimize



import matplotlib.pyplot as plt
import hipy.utils        as ut
import hipy.histos       as histos
import hipy.pltext       as pltext
import hipy.profile      as prof


#-------------------
#  Data Generator
#-------------------

# parameters
length, width           = 100., 100.
e0, tau0, beta, sigma   = 41.5, 0.1, 0.2, 0.05
wi                      = 41.5/2000.

def generate_kr_toy(size = 100000, 
                    length = length,
                    width = length,
                    e0 = e0,
                    tau = 10*length,
                    beta = beta,
                    sigma = sigma,
                    x0 = 0.,
                    y0 = 0.):
    """
    Generate Toy Kr sample:

    Inputs:
        size    : (int)    size of the sample
        length  : (float)  length of the chamber (mm)
        width   : (float)  witdh of the chamber (mm)
        e0      : (float)  energy at zero drift-time  (keV)
        tau     : (float)  life-time in (mm)
        beta    : (float)  radial distortion (parabolic)
        sigma   : (float)  (%) of the fluctiation of e0
        x0      : (float)  displacement respect the origin, x-coordinate
        y0      : (float)  displacement respect the origin, y-coordinate

    Returns:
        df      : (DataFrame) ['x', 'y', 'dtime', 'r', 'phi', 'energy']
    """
    
    ts = stats.uniform.rvs(0, length, size = size)
    xs = stats.uniform.rvs(0, width, size = size) - 0.5 * width
    ys = stats.uniform.rvs(0, width, size = size) - 0.5 * width
    es  = np.exp(-ts/tau) * stats.norm.rvs(loc = e0, scale = e0 * sigma, size = size)
    #es = (1. - ts/tau) * stats.norm.rvs(loc = e0, scale = e0 * sigma, size = size)
    r0s = np.sqrt((xs - x0)** 2 + (ys - y0)** 2)
    er = es * (1 - beta * (2 * r0s / width) ** 2)

    rs  = np.sqrt(xs**2 + ys**2)
    sel = rs < 0.5 * width

    df = {'dtime' : ts[sel],
          'x'     : xs[sel],
          'y'     : ys[sel],
          'r'     : rs[sel],
          'phi'   : np.arcsin(xs[sel]/rs[sel]),
          'energy': er[sel]}
    
    return pd.DataFrame(df)
                               


#-----------------------------------
#  Kr maps from fitting
#-----------------------------------

def residuals_(ts, es, par, cov):
    
    xv    = np.ones(shape = (2, len(ts)))
    xv[1] = -ts
    res   = np.dot(par, xv) - es

    var   = np.sum(xv * np.matmul(cov, xv), axis = 0)
    sig   = np.sqrt(var)
    
    sigma = np.sqrt(np.sum(res * res)/ (len(ts) - 2))
    return res, sig, sigma 


krmap_names = ('counts', 'eref', 'dedt', 'dtref', 'ueref', 'udedt', 'cov',
               'chi2', 'pvalue', 'sigma', 'success',
               'bin_centers', 'bin_edges')

KrMap = namedtuple('KrMap', krmap_names)


def krmap(coors, dtime, energy, bins = (36, 36), counts_min = 40, dt0 = 0):
    
    
    counts, ebins, ibins = stats.binned_statistic_dd(coors, energy, 
                                                     bins = bins, statistic = 'count',
                                                     expand_binnumbers = True)    
    ibins = [b-1 for b in ibins]
    cbins = [0.5 * (x[1:] + x[:-1]) for x in ebins]

    ref     = 1000
    indices =  ref * ibins[1] + ibins[0]
    #indices0 = np.array([ref * i1 + i0 for i0 in range(bins[0]) for i1 in range(bins[1])], int)

    eref  = np.zeros(shape = counts.shape)
    dedt  = np.zeros(shape = counts.shape)
    dtref = np.zeros(shape = counts.shape)
    ueref = np.zeros(shape = counts.shape)
    udedt = np.zeros(shape = counts.shape)
    cov   = np.zeros(shape = counts.shape)
    chi2  = np.zeros(shape = counts.shape)
    sig   = np.zeros(shape = counts.shape)
    pval  = np.zeros(shape = counts.shape)
    
    success   = counts > counts_min  
    residuals = np.nan * np.ones(len(energy))
    
    for i0, i1 in np.argwhere(success == True):
        ijsel = indices == int(ref * i1 + i0)
        ts, enes = dtime[ijsel], energy[ijsel]
        #print(len(ts), len(enes), counts[i0, i1], np.sum(ijsel))
        tij = np.mean(ts) if dt0 is None else dt0
        st_fun = lambda ts, a, b : a - b * (ts - tij)
        par, var = optimize.curve_fit(st_fun, ts, enes)
        eref [i0, i1] = par[0]
        dedt [i0, i1] = par[1]
        dtref[i0, i1] = tij
        ueref[i0, i1] = np.sqrt(var[0, 0])
        udedt[i0, i1] = np.sqrt(var[1, 1])
        cov  [i0, i1] = var[0, 1]
        
        res, _ , ijsig = residuals_(ts - tij, enes, par, var)
        residuals[ijsel] = res/ijsig
        ijchi2 = np.sum(res * res)/(len(res) - 2)
        ijpval = stats.shapiro(res)[1] if (len(res) > 3) else 0.
        chi2  [i0, i1] = ijchi2
        pval  [i0, i1] = ijpval
        sig   [i0, i1] = ijsig
        
    ikrmap = KrMap(counts, eref, dedt, dtref, ueref, udedt, cov,
                   chi2, pval, sig, success,
                   cbins, ebins)
    
    return ikrmap, residuals


def krmap_scale(coors, dtime, energy, krmap, scale = 1., mask = None):
    
    
    ndim      = len(coors)
    bin_edges = krmap.bin_edges
    
    idx = [np.digitize(coors[i], bin_edges[i])-1          for i in range(ndim)]
    sel = [(idx[i] >= 0) & (idx[i] < len(bin_edges[i])-1) for i in range(ndim)]
    sel = np.logical_and(*sel) if ndim >1 else sel[0]

    idx    = tuple([idx[i][sel] for i in range(ndim)])
    dt     = dtime[sel]
    ene    = energy[sel] 
    
    eref   = krmap.eref
    dedt   = krmap.dedt
    dtref  = krmap.dtref 
    mask   = krmap.success if mask == None else mask
    
    eref[~mask] = np.nan
    
    eref   = eref[idx]
    dedt   = dedt[idx]
    dtref  = dtref[idx]

    vals   = scale * ene / (eref - dedt * (dt - dtref)) 
    
    cene   = np.nan * np.ones(len(energy))
    cene[sel == True] = vals

    return cene


#--- Conversion to LT

def krmap_lifetime(krmap):
    dedt, udedt = krmap.dedt, krmap.udedt
    eref, ueref = krmap.eref, krmap.ueref
    lt  = eref/dedt
    ult = np.sqrt((ueref/dedt)**2 + (lt * udedt/dedt)**2)
    return lt, ult


# coordinates and index utilities
#-------------------------------

def coors_map(coors, bins):
    """ returns the map of the coordinates
    """
    cmaps = []
    for coor in coors:
        cmap,  _ , ids  = stats.binned_statistic_dd(coors, coor, 
                                                  bins = bins, statistic = 'mean',
                                                  expand_binnumbers = True)
        cmaps.append(cmap)
    return cmaps

def coors_index(coors, bins):
    """ returns the index of the coordinates in the bins
    """
    return np.array([np.digitize(coor, bin) for coor, bin in zip(coors, bins)])

def coors_mask(icoors, mask, ref = 1000):
    """ provided the index of the coordenates (icoors) and a mask of a map (mask),
    returns a mask on the coordenates whose indices are in the map mask
    """
    ids    = np.argwhere(mask == True)
    index_ = ref * icoors[1, :] + icoors[0, :]
    ids_   = ref * ids[:, 1] + ids[:, 0]
    #print(ids_)
    cmask   = np.isin(index_, ids_)
    return cmask


#--- Save and Load into/from h5
#---------------------------------
    

save = prof.save

load = lambda key, ifile : prof.load(key, ifile, KrMap)

#------------------------------------
#    Accept Residuals - Clean data
#------------------------------------

def accept_residuals(residuals, 
                     nbins = 100, range = (-5, 5.),
                     fun = 'gaus',
                     nsigma = 3.5, 
                     min_sigma = 0.9, 
                     plot = False):
    
    xsel       = ~np.isnan(residuals)
    _, _, _, pars, _, _  = histos.hfit(residuals[xsel], nbins, range = range, fun = fun);
    sigma      = pars[2]
    
    if (plot):
        canvas = pltext.canvas(2, 2)
        canvas(1)
        pltext.hist(residuals[xsel], nbins);
        plt.yscale('log');
        plt.xlabel('normalized residuals');
        canvas(2)
        pltext.hfit(residuals[xsel], nbins, range = range, fun = fun);
        plt.yscale('log'); plt.ylim((1, 1e5));
        plt.xlabel('normalized residuals');
        plt.show()
        
    done  = sigma > min_sigma
    sel   = xsel if done else np.abs(residuals) <= nsigma * sigma
    if (np.sum(sel) == np.sum(xsel)): done = True
    eff = 100 * np.sum(sel)/len(residuals)
    print('sigma {:4.2f}'.format(sigma), 'done ', done, ' eff {:4.2f}'.format(eff))    

    return done, sel


def clean_krmap(coors, dtime, energy, min_sigma = 0.9, bins = 50, counts_min = 40, dt0 = 0, plot = True):
    sel     = np.ones(len(energy), bool)
    clean   = np.copy(sel)
    done    = False
    while (not done):
        icoors  = [coor[clean] for coor in coors]
        idtime  = dtime[clean]
        iene    = energy[clean]
        krmap_, residuals   = krmap(icoors, idtime, iene,
                                    bins = bins, counts_min = counts_min, dt0 = dt0)
        done, usel = accept_residuals(residuals, range = (-5.,5.), 
                                    fun = 'gaus', nsigma = 4., 
                                    min_sigma = min_sigma, plot = plot)
        clean[clean == True] = usel
    return krmap_, clean



#------------------------
#    Voxing
#
#  This produce a class with the mean and std information in each voxel
#-------------------------


krvoxels_names = ('counts', 'emean', 'estd', 'coors_means', 'coors_stds',
                  'success', 'bin_centers', 'bin_edges')

KrVoxels = namedtuple('KrVoxels', krvoxels_names)


def krvoxels(coors, energy, bins = (30, 30, 30), counts_min = 40):

    counts, ebins, ibins = stats.binned_statistic_dd(coors, energy, 
                                                     bins = bins, statistic = 'count',
                                                     expand_binnumbers = True)    
    
    mask = counts > counts_min

    emean, _, _ = stats.binned_statistic_dd(coors, energy, 
                                                     bins = bins, statistic = 'mean',
                                                     expand_binnumbers = True)    

    estd, _, _ = stats.binned_statistic_dd(coors, energy, 
                                                     bins = bins, statistic = 'std',
                                                     expand_binnumbers = True)    
    
    cmeans, cstds = [], []
    for coor in coors:
        cmean, _, _ = stats.binned_statistic_dd(coors, coor, 
                                                     bins = bins, statistic = 'mean',
                                                     expand_binnumbers = True)    
        cstd, _,  _ = stats.binned_statistic_dd(coors, coor, 
                                                     bins = bins, statistic = 'std',
                                                     expand_binnumbers = True)
        cmeans.append(cmean)
        cstds.append (cstd)

    cbins = [0.5 * (x[1:] + x[:-1]) for x in ebins]

    krvoxels = KrVoxels(counts, emean, estd, cmeans, cstds, mask, ebins, cbins)
    return krvoxels

#-----------------------
#    Plotting
#------------------------


def plot_kr_data(df, bins):
    """
    Plot Kr Data
    """
    canvas = pltext.canvas(6, 2)
    canvas(1)
    pltext.hist(df.dtime, 100);
    plt.xlabel('drift time (ms)')
    canvas(2)
    pltext.hist(df.x, 100);
    plt.xlabel('x (mm)')
    canvas(3)
    pltext.hist(df.y, 100);
    plt.xlabel('y (mm)')
    canvas(4)
    pltext.hist(df.energy, 100);
    plt.xlabel('energy (keV)')
    canvas(5)
    plt.hist2d(df.dtime, df.energy, bins) 
    plt.xlabel('drift time (ms)'); plt.ylabel('energy (keV)')
    plt.colorbar();
    canvas(6)
    mean, ebins, _  = stats.binned_statistic_dd((df.x, df.y), df.energy,
                                                bins = bins , statistic = 'mean')
    cbins = [0.5 * (b[1:] + b[:-1]) for b in ebins]
    mesh = np.meshgrid(*cbins)
    plt.hist2d(mesh[0].ravel(), mesh[1].ravel(), bins = ebins, weights = mean.T.ravel())
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('energy (keV)')
    plt.colorbar();
    plt.tight_layout();
    
plot_data = plot_kr_data


def plot_xyvar(var, bins = None, title = '', mask = None, nbins = 100, range = None):
    
    mask   = var != np.nan if mask is None else mask 
    nx, ny = var.shape
    bins   = (np.arange(nx+1), np.arange(ny+1)) if bins == None else bins
    cbins  = [0.5 * (x[1:] + x[:-1]) for x in bins]
    mesh   = np.meshgrid(cbins[0], cbins[1])
    canvas = pltext.canvas(2, 2)
    canvas(1)
    uvar   = np.copy(var)
    if (var.dtype != bool):
        rmask  = mask if range is None else ut.in_range(var, range)
        mask   = np.logical_and(mask, rmask)
        uvar[~mask] = np.nan
    plt.hist2d(mesh[0].ravel(), mesh[1].ravel(), bins = bins, range = range,
               weights = uvar.T.ravel());
    #plt.hist2d(mesh[0][mask].ravel(), mesh[1][mask].ravel(), bins = bins,
    #           weights = var[mask].T.ravel());

    plt.xlabel('x'); plt.ylabel('y'); plt.title(title);
    plt.colorbar();
    canvas(2)
    #xsel = var != np.nan
    pltext.hist(var[mask].ravel(), bins = nbins, range = range);
    plt.xlabel(title)
    plt.tight_layout();
    return


def plot_xydt_energy_profiles(xdf, nbins = 100, names = ('dtime', 'x', 'y')):
    for name in names:
        zprof, _  = prof.profile((xdf[name],), xdf.energy, nbins)
        prof.plot_profile(zprof, nbins = nbins, stats = ('mean',), coornames = (name,))
    return
    