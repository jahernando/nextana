#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:22:49 2022

@author: hernando
"""


import numpy  as np
import pandas as pd

from   collections import namedtuple
from   scipy       import stats
from   scipy       import optimize


import matplotlib.pyplot as plt
import hipy.histos       as histos
import hipy.pltext       as pltext
import hipy.profile      as prof

#-------------------
#  Data Generator
#-------------------

# parameters
size                    = 100
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
                               

#---------------------------------
#  (x, y) data fit in binned coordinates
#-----------------------------------


def residuals_fun(fun, xs, ys, pars):
    """ compute the residuals of ys - fun(xs, *pars)

    Inputs:
        fun  : (function) fun(xs, *pars) i.e for straight line fun(xs, a, b) =  a + b * xs
        xs   : (np.array) x values
        ys   : (np.array) y values
        pars : (np.array) parameter values, arguments of the function  

    Returns:
        res   : (np.array) residuals ys - fun(xs, *pars)
        sigma : (float)    std of the residuals
        chi2  : (float)    chi2/ndf of the residuals
    """

    vals  = fun(xs, *pars)
    res   = vals - ys
    nsize = len(res)
    sigma = np.sqrt(np.sum(res*res)/(nsize-1))
    chi2  = (np.sum(res * res)/(sigma**2))/(nsize-2)

    return res, sigma, chi2

# profile_fit_map_fields = ('counts', 'success', 'par', 'cov', 'sigma', 'chi2', 'pvalue',
#                           'center_bins', 'edges_bins', 'function')

# ProfileFitMap = namedtuple('ProfileFitMap', profile_fit_map_fields)

# def profile_fit_map(coors, bins, x, y, function, counts_min):
#     """
     
#     place data (time, energy) into bins in (coors) (i.e x, y) and fit then to a function
#     return result

#     Inputs
#         coors      : (tuple(np.array)), tuple with the coordinates i.e (u, v)
#         bins       : (int, tuple, or bins) i.e (20, 20), or ([0, 1., 2, 3], 1)
#         data       : (tuple(np.array)) tuple with the data to fit, i.e (time, energy)   
#         func       : function to fit y = function(x, *pars)
#         counts_min : (int) minimum number of events in bin to fit

#     Returns
#         map     : (NamedTuple) see fitmap for the attributes of this named tuple
#         residuals : (np.array), normalized residuals, same size (y - fun(x, *pars))
#     """

#     counts, ebins, ibins = stats.binned_statistic_dd(coors, y, 
#                                                     bins = bins, statistic = 'count',
#                                                     expand_binnumbers = True)    
#     ibins = [b-1 for b in ibins]
#     cbins = [0.5 * (x[1:] + x[:-1]) for x in ebins]

#     ref     =  1000 
#     if (len(ibins[0]) > ref): ref = 10 * len(ibins[0])

#     indices =  ref * ibins[1] + ibins[0]
#     #indices0 = np.array([ref * i1 + i0 for i0 in range(bins[0]) for i1 in range(bins[1])], int)

#     dmap = {}
    
#     success = counts > counts_min
#     res     = np.zeros(shape = counts.shape)
#     chi2    = np.zeros(shape = counts.shape)
#     pvalue  = np.zeros(shape = counts.shape)
#     sigma   = np.zeros(shape = counts.shape)
        
#     nu, nv = counts.shape 
#     pars = nu*[nv*[None]]
#     covs = nu*[nv*[None]]

#     residuals = np.nan * np.ones(len(y))

#     for i0, i1 in np.argwhere(success == True):
            
#         sel = indices == int(ref * i1 + i0)

#         ix, iy     = x[sel], y[sel]
#         ipar, icov = optimize.curve_fit(function, ix, iy)

#         pars [i0, i1] = ipar
#         covs [i0, i1] = icov

#         ires, isigma, ichi2 = residuals_fun(function, ix, iy, ipar)
#         ipvalue             = stats.shapiro(res)[1] if (len(res) > 3) else 0.

#         chi2  [i0, i1] = ichi2
#         pvalue[i0, i1] = ipvalue
#         sigma [i0, i1] = isigma
#         residuals[sel] = ires/isigma
    
#     map = ProfileFitMap(counts, success, pars, covs, chi2, sigma, pvalue, cbins, ebins, function)
#     return map, residuals

# def correction_from_profile_fit_map(coors, x, y, profile_map):

#     return

#---------------------

krmap_fields = ['counts', 'e0', 'lt', 'ue0', 'ult', 'cov',
                'chi2', 'pvalue', 'sigma', 'success',
                'coordenates', 'kr_fit_type',
                'bin_centers', 'bin_edges']

def sline_to_exp(par, var):
    opar = np.array((par[0], par[0]/par[1]))
    return (opar, var)

def loge_to_exp(par, var):
    opar = np.array((np.exp(par[0]), par[1]))
    return (opar, var)

KrFitType = namedtuple('KrFitType', ('par_names', 'function', 'input_data', 'output_pars'))

krfit_exp = KrFitType(('e0', 'lt'), 
                      lambda ts, a, b: a* np.exp(-ts/b), 
                      lambda ts, es: (ts, es),
                      lambda par, var: (par,var))

krfit_sline = KrFitType(('eref', 'dedt'), lambda ts, a, b: a - b * ts, 
                      lambda ts, es: (ts, es), sline_to_exp)

krfit_elog = KrFitType(('loge', 'lt'), lambda ts, a, b: a - ts/b, 
                      lambda ts, es: (ts, np.log(es)), loge_to_exp)

krfit_dict_ = {'exp'   : krfit_exp, 
               'sline' : krfit_sline,
               'loge'  : krfit_elog}


def get_coors(df, coordenates):

    if (coordenates not in ['cartesian', 'polar']):
        raise("Not valid type of coordinates " + coordenates)
    
    coors = (df.x, df.y)
    if (coordenates == 'polar'): coors = (df.r, df.phi)

    return coors


def configure_krmap_creator(coordenates, bins, counts_min, kr_fit_type):
    """ Return the function that creates a KrMap for a given fit type

    Inputs:
        counts_min   : (int) minimum number of counts required to fit
        kr_fit_type  : (str) name of the kr fit, i. ('exp', 'loge', 'sline')

    Outputs:
        krmap_creator : (function) creates the krmap 
    """

    if (kr_fit_type not in krfit_dict_.keys()):
        raise("Input not valid: type of Kr fit not accepted " + kr_fit_type)
    
    krfit   = krfit_dict_[kr_fit_type]
    npars   = len(krfit.par_names)
    parname = lambda i    : 'par_'+str(i) # parnames_[i]
    covname = lambda i, j : 'cov_'+str(i)+str(j) # parnames_[i]+'_'+parnames_[j]
    names   = list(krmap_fields)
    names   += [parname(i) for i in range(npars)]
    names   += [covname(i, j) for i in range(npars) for j in range(i+1, npars)]

    KrMapX = namedtuple('KrMapX', names)

    def krmap_creator(df):
        """
        Constructs a KrMap

        Inputs
            df  : (pd.DataFrame) ['x', 'y', 'dtime', 'energy', 'r', 'phy']
        Returns
            krmap     : (NamedTuple) see krmap_fields for the attributes of this named tuple
            residuals : (np.array), normalized residuals, same size (as coordinates)
        """

        coors  = get_coors(df, coordenates)
        dtime  = df.dtime
        energy = df.energy
        return krmap_creator_base(coors, bins, dtime, energy)

    # configurate the map creator function
    def krmap_creator_base(coors, bins, dtime, energy):
        """
        Constructs a KrMap

        Inputs
            coors      : (tuple(np.array)), tuple with the coordinates i.e (x, y)
            bins       : (int, tuple, or bins) i.e (20, 20), or ([0, 1., 2, 3], 1)
            dtime      : (np.array), drift time
            energy     : (np.array), energy

        Returns
            krmap     : (NamedTuple) see krmap_fields for the attributes of this named tuple
            residuals : (np.array), normalized residuals, same size (as coordinates)
        """

        counts, ebins, ibins = stats.binned_statistic_dd(coors, energy, 
                                                         bins = bins, statistic = 'count',
                                                         expand_binnumbers = True)    
        ibins = [b-1 for b in ibins]
        cbins = [0.5 * (x[1:] + x[:-1]) for x in ebins]

        ref     =  1000 
        if (len(ibins[0]) > ref): ref = 10 * len(ibins[0])

        indices =  ref * ibins[1] + ibins[0]
        #indices0 = np.array([ref * i1 + i0 for i0 in range(bins[0]) for i1 in range(bins[1])], int)

        dmap = {}
        for name in names: dmap[name] = np.zeros(shape = counts.shape)
        residuals = np.nan * np.ones(len(energy))
    
        success = counts > counts_min
        dmap['success']  = success
        dmap['counts']   = counts
    
        for i0, i1 in np.argwhere(success == True):
            
            sel = indices == int(ref * i1 + i0)
            ts, enes = dtime[sel], energy[sel]
            xs, ys   = krfit.input_data(ts, enes) 

            ipar, ivar = optimize.curve_fit(krfit.function, xs, ys)
            opar, ovar = krfit.output_pars(ipar, ivar)
            
            dmap['e0'] [i0, i1] = opar[0]
            dmap['lt' ][i0, i1] = opar[1]
            dmap['ue0'][i0, i1] = np.sqrt(ovar[0, 0])
            dmap['ult'][i0, i1] = np.sqrt(ovar[1, 1])
            dmap['cov'][i0, i1] = ovar[0, 1]

            for i in range(npars): 
                dmap[parname(i)][i0, i1] = ipar[i]
            for i in range(npars):
                for j in range(i+1, npars):
                    dmap[covname(i, j)][i0, i1] = ivar[i, j]
        
            res, sigma, chi2 = residuals_fun(krfit.function, xs, ys, ipar)
            pval             = stats.shapiro(res)[1] if (len(res) > 3) else 0.
            dmap['chi2']  [i0, i1] = chi2
            dmap['pvalue'][i0, i1] = pval
            dmap['sigma'] [i0, i1] = sigma
            residuals[sel]         = res/sigma

        dmap['coordenates'] = coordenates
        dmap['kr_fit_type'] = kr_fit_type
        dmap['bin_centers'] = cbins
        dmap['bin_edges']   = ebins        
    
        krmap = KrMapX(**dmap)
        return krmap, residuals
    
    # returns the configurated map creator function
    return krmap_creator


#-----------------
#  Correction
#------------------

def krmap_scale_(df, krmap, scale = 1., mask = None):
    """
    
    correct the energy at a given drift-time by a Krmap

    Inputs:
        coors  : tuple(np.array), tuple with the coordinates, i.e (x, y)
        dtime  : np.array, drift-times
        energy : np.array, energy
        krmap  : KrMap, named tuple
        scale  : float, value to scale, default 1.
        mask   : np.array, shape as the krmap shape, mask given bins of the krmap, default None

    Output:
        cene    : np.array, corrected energy
    """

    coors   = get_coors(df, krmap.coordenates) 
    dtime   = df.dtime
    energy  = df.energy

    ndim      = len(coors)
    bin_edges = krmap.bin_edges
    
    idx  = [np.digitize(coors[i], bin_edges[i])-1          for i in range(ndim)]
    sels = [(idx[i] >= 0) & (idx[i] < len(bin_edges[i])-1) for i in range(ndim)]
    sel  = sels[0]
    for isel in sels[1:]: sel = np.logical_and(sel, isel)

    idx    = tuple([idx[i][sel] for i in range(ndim)])
    dt     = dtime[sel]
    ene    = energy[sel] 
    
    e0     = krmap.e0
    elt    = krmap.lt
    mask   = krmap.success if mask == None else mask
    
    e0[~mask] = np.nan
    
    e0   = e0[idx]
    elt  = elt[idx]

    function = krfit_exp.function

    vals   = scale * ene / function(dt, e0, elt)
    
    cene   = np.nan * np.ones(len(energy))
    cene[sel == True] = vals

    return cene


#--------------------------------
#--- Save and Load into/from h5
#-------------------------------
    
save = prof.save

load = lambda key, ifile, map : prof.load(key, ifile, map)


#---------------------------------------
#    Accept Residuals 
#--------------------------------------

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


#------------------------
#   Plotting
#------------------------

#def plot_xydt_energy_profiles(xdf, nbins = 100, names = ('dtime', 'x', 'y'), ename = 'energy'):

def plot_data(x, y, dtime, energy, bins):
    """
    Plot Kr Data
    """
    canvas = pltext.canvas(6, 2)
    canvas(1)
    pltext.hist(dtime, 100);
    plt.xlabel('drift time (ms)')
    canvas(2)
    pltext.hist(x, 100);
    plt.xlabel('x (mm)')
    canvas(3)
    pltext.hist(y, 100);
    plt.xlabel('y (mm)')
    canvas(4)
    pltext.hist(energy, 100);
    plt.xlabel('energy (keV)')
    canvas(5)
    plt.hist2d(dtime, energy, bins) 
    plt.xlabel('drift time (ms)'); plt.ylabel('energy (keV)')
    plt.colorbar();
    canvas(6)
    mean, ebins, _  = stats.binned_statistic_dd((x, y), energy,
                                                bins = bins , statistic = 'mean')
    cbins = [0.5 * (b[1:] + b[:-1]) for b in ebins]
    mesh = np.meshgrid(*cbins)
    plt.hist2d(mesh[0].ravel(), mesh[1].ravel(), bins = ebins, weights = mean.T.ravel())
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('energy (keV)')
    plt.colorbar();
    plt.tight_layout();
    

def plot_energy_profiles(xdf, nbins = 100, names = ('dtime', 'x', 'y', 'r', 'phi')):
    for name in names:
        zprof, _  = prof.profile((xdf[name],), xdf.energy, nbins)
        prof.plot_profile(zprof, nbins = nbins, stats = ('mean',), coornames = (name,))
    return
    