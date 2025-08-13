#-----------------------
# Util Functions for HE analysis
#------------------------

from os import listdir

import numpy             as np
import pandas            as pd
import tables            as tb
from   scipy             import stats
from   scipy             import optimize
from   scipy.interpolate import griddata

import hipy.utils        as ut
#import hipy.histos       as histos
#import hipy.pltext       as pltext
#import hipy.profile      as prof

from invisible_cities.io.dst_io import load_dst, load_dsts

from sklearn.cluster import DBSCAN

E_peaks_true = (0.511, 0.583, 0.727, 0.860, 1.592, 2.615)

# Util
#---------------------------------

def get_files(path, token = '.h5'):
    """ get filename in a directory with a given token
    """
    filenames = listdir(path)
    filenames.sort()
    filenames = [file for file in filenames if file.find(token) >0 ]
    #print(filenames)
    return filenames


# # Zirma city
# #----------------------------------

# def zirma(ifilename : str, # input filename
#           path_krmap, # input kr map (3D)
#           ofilename = '', # output filename
#           qthreshold = 7.,  # SiPM threshold (pe)
#           remove_scatter_hits = True): # remove scatter hits

#     # get hits
#     hits = get_hits(ifilename)
#     #print('number of hits ', len(hits))

#     # cluster hits (classify as isolated or connected in clusters)
#     hits = cluster_hits(hits)
    
#     # remove scatter out range hits
#     hits = hits[hits.cluster_id_in >= 0]
#     #print('number of hits in range', len(hits))

#     # remove scatter hits
#     if (remove_scatter_hits):
#         sel = hits.cluster_id < 0
#         hits.Q[sel] = 0.
#         #print('number of isolated hits in range', sum(sel))

#     # recompute the weight of the hits in the same slice
#     hits = hits_redistribute_light(hits, qthreshold)
#     #print('number of hits after q cut ', len(hits))
#     hits['E0'] = hits['E'].values
#     hits['E']  = hits['E_norm'].values

#     # recalibrate energy and replace it
#     Ec = recalibrate_energy(hits, path_krmap)
#     hits['Ec0'] = hits['Ec'].values
#     hits['Ec']  = Ec

#     # summarice event per cluster
#     esum = event_cluster_summary(hits)

#     # save hits and event summaty
#     if (ofilename != ''):
#         print(f'saving hits and event-summary at {ofilename}')
#         #hits.to_hdf(ofilename, 'hits', mode = 'w', complevel = 9, complib = 'zlib')
#         esum.to_hdf(ofilename, 'esum', mode = 'a', complevel = 9, complib = 'zlib')

#     return hits, esum

# Hits
#----------------------------


def get_hits(ifilename : str, 
             with_cluster: bool = False):
    
    print(f'loading {ifilename}')
    if (with_cluster):
        return pd.read_hdf(ifilename, 'hits');
    
    hits  = load_dst(ifilename, "RECO", "Events")
    labels_drop = ['npeak', 'Xpeak', 'Ypeak', 'nsipm', 'Xrms', 'Yrms', 'Qc', 'Ep', 'track_id']
    hits.drop(labels_drop, axis = 1, inplace = True)
    return hits

def select_event_in_energy_range(hits : pd.DataFrame, 
                                 erange : tuple ):

    hsum  = hits.groupby(['event'], as_index = False).sum()
    sel_de = ut.in_range(hsum.Ec.values, erange)
    events_de = hsum.event[sel_de].values
    hits = hits[np.isin(hits.event.values, events_de)].copy()

    return hits

def cluster_hits(hits: pd.DataFrame):
    hits = hits.groupby(['event']).apply(cluster_hits_)
    hits = hits.reset_index(drop = True)
    return hits

def hits_redistribute_light(hits_df: pd.DataFrame, 
                            sipm_threshold: float, 
                            preserve_event_light: bool = True):
    '''
    Redistributes the total energy of each (event, Z) group among the hits
    that survive a charge threshold, proportional to their charge.
    '''
    hits = hits_df.copy()
    # 1. Calcular energía total y carga total por (event, Z) ANTES del corte
    energy_sum = hits.groupby(['event', 'Z'])['E'].sum().rename('E_evz')
    hits = hits.merge(energy_sum, on=['event', 'Z'], how='left')
    # 2. Aplicar el corte
    hits_n = hits[hits.Q > sipm_threshold].copy()
    charge_sum = hits_n.groupby(['event', 'Z'])['Q'].sum().rename('Q_evz')
    # 3. Añadir las columnas globales de suma de energía y carga
    hits_n = hits_n.merge(charge_sum, on=['event', 'Z'], how='left')
    # 4. Redistribuir proporcionalmente
    hits_n['Q_norm'] = hits_n['Q'] / hits_n['Q_evz']
    hits_n['E_norm'] = hits_n['E_evz'] * hits_n['Q_norm']

    if (preserve_event_light):
        e0sum   = hits  .groupby('event')['E'].sum().values
        e1sum   = hits_n.groupby('event')['E_norm'].sum().values
        h1nhits = hits_n.groupby('event')['E_norm'].count().values
        factor  = np.repeat(e0sum/e1sum, h1nhits)
        hits_n['E_norm'] = factor * hits_n['E_norm']

    return hits_n


# def sophronia_extend(ifilename : str, krmap_filename : str, ofilename : str = '',
#                      erange: tuple = (1500, 1800), scale : float = 1.):
#     """ read the sophronia hits, clusterize them, and store them in an outputfile
#     """

#     #ifiles = [ifile.replace('.h5', '_ext.h5') for ifile in filenames]

#     # load hits
#     print(f'loading {ifilename}')
#     hits  = load_dst(ifilename, "RECO", "Events")
#     labels_drop = ['npeak', 'Xpeak', 'Ypeak', 'nsipm', 'Xrms', 'Yrms', 'Qc', 'Ep', 'track_id']
#     hits.drop(labels_drop, axis = 1, inplace = True)

#     # select hits of the events which energy is inside the desired energy range
#     hsum  = hits.groupby(['event'], as_index = False).sum()
#     sel_de = ut.in_range(hsum.Ec.values, erange)
#     events_de = hsum.event[sel_de].values
#     hits = hits[np.isin(hits.event.values, events_de)].copy()

#     # correct the energy using the krmap
#     Ec = recalibrate_energy(hits, krmap_filename, scale = scale)
#     hits['Ec0'] = hits['Ec'].values
#     hits['Ec']  = Ec

#     # label the hits as scatter of in cluster
#     hits = hits.groupby(['event']).apply(label_hits)
#     hits = hits.reset_index(drop = True)

#     # store the hits in the outputfile
#     if (ofilename != ''):
#         print(f'saving hits in {ofilename}')
#         hits.to_hdf(ofilename, 'hits', mode = 'w')
#     return hits

# Labeling Hits 
#------------------------------------------

def cluster_hits_(df):
    """ label hits as scatter or isolated hits (-1) or hit in a cluster (0, 1, 2, ...)
            label is stored as the column 'cluster_id' in the DF
        label hits as in the same z-range of a given cluster 
            i.e. -1 if out of any cluster, 0 in the same z-range of cluster 0, 
            if a hits is in the same z-range of several cluster gets the label of the cluster with the lower index
            label is stored as the column 'cluster_id_in' in the DF
    """ 
    df = label_hits_(df)
    df = label_hits_in_(df)
    return df

def label_hits_(df, scale = (14.55, 15.55, 3.7), eps = 2.3, max_clusters = 5):
    coors = df[['X', 'Y', 'Z']].to_numpy()
        
    coors[:, :2] /= scale[0]
    coors[:, 2]  /= scale[2]
        
    labels = DBSCAN(eps = eps, min_samples = max_clusters).fit_predict(coors)

    df['cluster_id'] = labels
    return df

def label_hits_in_(df):
    cluster_id = df.cluster_id.values
    zs = df.Z.values
    cluster_id_in = -1 * np.ones(len(cluster_id), int)
    df['cluster_id_in'] = -1
    for id in np.arange(max(cluster_id), -1, -1):
        xsel = cluster_id == id
        zmin, zmax = np.min(zs[xsel]), np.max(zs[xsel])
        ysel = ut.in_range(zs, (zmin, zmax))
        cluster_id_in[ysel] = id
    df['cluster_id_in'] = cluster_id_in
    return df

# Energy correction
#--------------------------------------

# def hits_redistribute_light(hits_df: pd.DataFrame, sipm_threshold: float, 
#                             preserve_event_light: bool = True):
#     '''
#     Redistributes the total energy of each (event, Z) group among the hits
#     that survive a charge threshold, proportional to their charge.
#     '''
#     hits = hits_df.copy()
#     # 1. Calcular energía total y carga total por (event, Z) ANTES del corte
#     energy_sum = hits.groupby(['event', 'Z'])['E'].sum().rename('E_evz')
#     hits = hits.merge(energy_sum, on=['event', 'Z'], how='left')
#     # 2. Aplicar el corte
#     hits_n = hits[hits.Q > sipm_threshold].copy()
#     charge_sum = hits_n.groupby(['event', 'Z'])['Q'].sum().rename('Q_evz')
#     # 3. Añadir las columnas globales de suma de energía y carga
#     hits_n = hits_n.merge(charge_sum, on=['event', 'Z'], how='left')
#     # 4. Redistribuir proporcionalmente
#     hits_n['Q_norm'] = hits_n['Q'] / hits_n['Q_evz']
#     hits_n['E_norm'] = hits_n['E_evz'] * hits_n['Q_norm']

#     if (preserve_event_light):
#         e0sum   = hits  .groupby('event')['E'].sum().values
#         e1sum   = hits_n.groupby('event')['E_norm'].sum().values
#         h1nhits = hits_n.groupby('event')['E_norm'].count().values
#         factor  = np.repeat(e0sum/e1sum, h1nhits)
#         hits_n['E_norm'] = factor * hits_n['E_norm']

#     return hits_n

def get_corrector(krmap_filename):
    """ return a function to correct the energy of this hits using (z, x, y) coordinates
        the correction is based on a 3D map (GML)
    """

    krmap = pd.read_hdf(krmap_filename, "/krmap")
    #meta  = pd.read_hdf(pathdata + "GML_krmap_combined.map3d", "/mapmeta")

    dtxy_map   = krmap.loc[:, list("zxy")].values
    factor_map = krmap.factor.values
    def corr(dt, x, y, method="nearest"):
        dtxy_data   = np.stack([dt, x, y], axis=1)
        factor_data = griddata(dtxy_map, factor_map, dtxy_data, method=method)
        return factor_data
    return corr

def recalibrate_energy(hits : pd.DataFrame, krmap_filename : str, scale : float = 1.):
    E, X, Y, Z = hits.E.values, hits.X.values, hits.Y.values, hits.Z.values
    corrector = get_corrector(krmap_filename)
    Ec  = scale * E * corrector(Z, X, Y)
    return Ec

# Event summary
#--------------------------------------    

# def event_summary(hits):
#     """ create an event summary from a DF of sophronia (extended hits) 
#     """

#     nevts = len(np.unique(hits.event))
#     #keys  = ['nevent',
#     #       'Ec0tot', 'Ectot', 'Ec0in', 'Ecin', 'Eccls', 'Eciso', 'Ecisoout', 'Ecisoin',
#     #        'nhits0', 'nhits', 'nhitsclus', 'nhitsiso', 'nhitsout', 'nhitsin',
#     #        'nclus', 'nclus_inz', 
#     #        'zmin', 'zmax', 'xmin', 'xmax', 'ymin', 'ymax', 'zave', 'xave', 'yave',
#     #        'rmin', 'rmax', 'rave']
#     keys  = ['nevent',
#              'Ectot', 
#              'nhits', 
#              'nclus', 
#              'zmin', 'zmax', 'xmin', 'xmax', 'ymin', 'ymax', 'zave', 'xave', 'yave',
#              'rmin', 'rmax', 'rave']

#     df = {}
#     for key in keys:
#         dtype = int if key.find('n') == 0 else float
#         df[key] = np.zeros(nevts, dtype = dtype)
    
#     def _evt(ihits, ii):

#         sel_hits_in     = ihits.cluster_id_in >= 0
#         sel_hits_out    = ihits.cluster_id_in == -1
#         sel_cluster     = ihits.cluster_id >= 0
#         sel_scatter     = ihits.cluster_id == -1
#         sel_scatter_out = np.logical_and(sel_scatter, sel_hits_out)
#         sel_scatter_in  = np.logical_and(sel_scatter, sel_hits_in)

#         df['nevent'][ii]  = ihits.event.max()
#         #df['Ec0tot'][ii]   = ihits.Ec0.sum()
#         df['Ectot'][ii]    = ihits.Ec.sum()
#         #df['Ec0in'][ii]    = ihits.Ec0[sel_hits_in].sum()
#         #df['Ecin'][ii]     = ihits.Ec[sel_hits_in].sum()
#         #df['Eccls'][ii]    = ihits.Ec[sel_cluster].sum()
#         #df['Eciso'][ii]    = ihits.Ec[sel_scatter].sum()
#         #df['Ecisoout'][ii] = ihits.Ec[sel_scatter_out].sum()
#         #df['Ecisoin'][ii]  = ihits.Ec[sel_scatter_in].sum()
#         #df['nhits0'][ii]    = len(ihits)
#         df['nhits'][ii]     = sum(sel_hits_in)
#         #df['nhitsclus'][ii] = sum(sel_cluster)
#         #df['nhitsiso'][ii]  = sum(sel_scatter)
#         #df['nhitsout'][ii]  = sum(sel_scatter_out)
#         #df['nhitsin'][ii]   = sum(sel_scatter_in)
#         df['nclus'][ii]     = ihits.cluster_id.max()
#         #df['nclus_inz'][ii] = ihits.cluster_id_in.max()
#         df['zmin'][ii] = ihits.Z[sel_cluster].min()
#         df['zmax'][ii] = ihits.Z[sel_cluster].max()
#         df['xmin'][ii] = ihits.X[sel_cluster].min()
#         df['xmax'][ii] = ihits.X[sel_cluster].max()
#         df['ymin'][ii] = ihits.Y[sel_cluster].min()
#         df['ymax'][ii] = ihits.Y[sel_cluster].max()
#         df['zave'][ii] = ihits.Z[sel_cluster].mean()
#         df['xave'][ii] = ihits.X[sel_cluster].mean()
#         df['yave'][ii] = ihits.Y[sel_cluster].mean()

#         rs = np.sqrt(ihits.X[sel_cluster]**2 +ihits.Y[sel_cluster]**2)
#         df['rmin'][ii] = float(np.min(rs))
#         df['rmax'][ii] = float(np.max(rs))
#         df['rave'][ii] = float(np.mean(rs))

#     ii = 0
#     for evt, ihits in hits.groupby(['event']):
#         _evt(ihits, ii)
#         ii += 1
   
#     return pd.DataFrame(df)

def event_cluster_summary(hits):
    """ create an event summary from a DF of sophronia (extended hits) 
    """

    ehits = hits.groupby(['event', 'cluster_id'], as_index = False)
    nentries = len(ehits)

    keys  = ['event', 'cluster', 'nhits', 'time', 'energy', 
            'xmin', 'xmax', 'xave', 'ymin', 'ymax', 'yave', 'zmin', 'zmax', 'zave',
            'rmin', 'rmax', 'rave']

    df = {}   
    for i, key in enumerate(keys):
        dtype = float if i >= 3 else int
        df[key] = np.zeros(nentries, dtype = dtype)
 
    def entry_(ihits, i):

        df['event'][i]   = ihits.event.max()
        df['cluster'][i] = ihits.cluster_id.max()
        df['nhits'][i]   = ihits.event.count()
        df['time'][i]    = ihits.time.max()
        df['energy'][i]  = np.sum(ihits.Ec)
        rr = np.sqrt(ihits.X**2 + ihits.Y**2)
        for vn, vv in zip(('x', 'y', 'z', 'r'), (ihits.X, ihits.Y, ihits.Z, rr)):
            df[vn+'min'][i] = np.min(vv)
            df[vn+'max'][i] = np.max(vv)
            df[vn+'ave'][i] = np.mean(vv)

    ii = 0
    for _, ihits in ehits:
        entry_(ihits, ii)
        ii += 1
   
    return pd.DataFrame(df)




#---

# def get_peak_centers(run_numbers, reco_summ, variable, peak_ranges, npeaks = 4, plot = True):
#     peak_nbins = [100, 100, 60, 40, 100, 25]
#     peak_names  = ['e+e- anihilation (511 keV)', r'$^{208}$Tl $\rightarrow$ $^{208}$Pb (583 keV)', r'$^{212}$Bi $\rightarrow$ $^{212}$Po (727 keV)', r'$^{208}$Tl $\rightarrow$ $^{208}$Pb (860 keV)', '1592 keV\nDEP', '2615 keV\nPP']

#     ranges = peak_ranges[:npeaks]
#     nbins  = peak_nbins[:npeaks]
#     names  = peak_names[:npeaks]
    
#     meas_peaks = {}
#     for run_n in run_numbers:
#         run_reco = reco_summ[reco_summ.run_n == str(run_n)]
#         meas_val, meas_err = [], []
#         for rang, nbin, name in zip(ranges, nbins, names):
#             par, res = fit_whole_peaks(run_reco, variable, peak_fit, rang, nbin, title = 'Run {} - '.format(run_n) + name, plot = plot)
#             meas_val.append(par[0][1])
#             meas_err.append(par[1][1])
#         meas_peaks[run_n] = [meas_val, meas_err]
#     return meas_peaks


# def energy_scale_fit(meas_peaks, run_numbers, plot_colors, npeaks = 4):
#     real_peaks = [0.511, 0.583, 0.727, 0.86, 1.592, 2.615]
#     color_peak = ['r', 'orange', 'y', 'g', 'b', 'm']
#     real_peaks = real_peaks[:npeaks]
#     color_peak = color_peak[:npeaks]
    
#     parameters = {}
#     for run_n, c in zip(run_numbers, plot_colors):
#         if run_n not in meas_peaks.keys(): continue
#         if len(meas_peaks[run_n][0]) == 0: continue
#         for m_, r_, c_ in zip(meas_peaks[run_n][0], real_peaks, color_peak):
#             plt.plot(m_, r_, '.', color = c_)
#         par, err  = curve_fit(lineal, meas_peaks[run_n][0], real_peaks, p0 = (1, 0))
#         plt.plot(meas_peaks[run_n][0], lineal(np.array(meas_peaks[run_n][0]), *par), color = c, label = str(run_n) + '\n m = {:.2e}'.format(par[0]) + '\n n = {:.2e}'.format(par[1]))
#         parameters[run_n] = par
#     plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
#     plt.grid()
#     plt.xlabel('Measured peak energy (MeV)')
#     plt.ylabel('Real peak energy (MeV)')
#     return parameters


# def plot_real_vs_meas_peaks(meas_peaks, run_numbers, plot_colors, npeak = 4):
#     real_peaks = [0.511, 0.583, 0.727, 0.86, 1.592, 2.615]
#     real_peaks = real_peaks[:npeak]
#     for run_n, c in zip(run_numbers, plot_colors):
#         if run_n not in meas_peaks.keys(): continue
#         if len(meas_peaks[run_n][0]) == 0: continue
#         m_ = np.array(meas_peaks[run_n][0])
#         e_ = np.array(meas_peaks[run_n][1])
#         r_ = np.array(real_peaks)
#         plt.errorbar(r_, ((m_ - r_) / r_) * 100, yerr = e_ * 100, color = c, markersize = 10, label = run_n, marker = '.', capsize=2)
#     plt.legend()
#     plt.grid()
#     plt.xlabel('Real peak energy (MeV)')
#     plt.ylabel('(Measured - Real) / Real peak energy (%)')