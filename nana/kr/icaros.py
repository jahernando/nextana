#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:52:32 2022

@author: hernando
"""


import numpy  as np
#import pandas as pd


from invisible_cities.reco       import corrections as cof

#from   collections import namedtuple
#from   scipy       import stats
#from   scipy       import optimize


#import matplotlib.pyplot as plt
#import hipy.pltext       as pltext
#import hipy.profile      as prof


# from invisible_cities.reco.corrections import read_maps, apply_all_correction
# from invisible_cities.types.symbols import NormStrategy

# def get_icaros_correction_function(map_path):
#     #reco = pd.read_hdf(file, 'RECO/Events')

#     map_path = store_path + 'map_MC_4bar_15063.h5'
#     maps = read_maps(map_path)
#     get_coef  = apply_all_correction(maps
#                                     , apply_temp = False
#                                     , norm_strat = NormStrategy.kr)
#     return get_coef


# #corr = get_coef(reco.X, reco.Y, reco.Z, reco.time)
# #reco['Ec_map']  = reco.E * corr

def get_map(map_fname):
    maps      = cof.read_maps(map_fname)
    return maps


def get_correction(maps):
    """ Return correction faction as variables (x, y, z) using the map
    """

    _corrfac  = cof.apply_all_correction(maps, apply_temp = False,
                                          norm_strat = cof.norm_strategy.kr)
    def corrfac(x, y, dt, times):
        return _corrfac(x, y, dt, times)

    return corrfac


def get_corrz(maps):
    """ Return correction faction as variables (x, y, z) using the map
    """

    vdrift    = np.mean(maps.t_evol.dv)
    print('drift velocity ', vdrift)
    _corrfac  = cof.apply_all_correction(maps, apply_temp = True,
                                          norm_strat = cof.norm_strategy.kr)
    def corrfac(x, y, z, times):
        dt = z/vdrift
        return _corrfac(x, y, dt, times)

    return corrfac