#----------------------------------------
#
# Functions for Kr analysis of NEXT100
#
#----------------------------------------

import numpy             as np
import pandas            as pd
#import tables            as tb
#from   scipy             import stats
#from   scipy             import optimize

import matplotlib.pyplot as plt

import hipy.utils        as ut
import hipy.histos       as histos
import hipy.pltext       as pltext
#import hipy.profile      as prof
import hipy.styles       as pltsty

pltsty.style('jhep')

#from invisible_cities.io  . pmaps_io        import load_pmaps, load_pmaps_as_df
#from invisible_cities.io.dst_io import load_dst, load_dsts
from invisible_cities.core.core_functions import in_range

def eff_of_sel(sel, name = "", do_print = False):
    nsel, ntot = np.sum(sel), len(sel)
    eff = nsel/float(ntot)
    if do_print:
        seff = "{:.4f}".format(eff)
        print(f"{name} efficiency {seff}, {nsel}/{ntot}.")
    return eff

dtrms2_low = lambda dt: -0.7 + 0.030 * (dt-20) # Gonzalo's
dtrms2_upp = lambda dt: 2.6 + 0.036 * (dt-20) # Gonzalo'2

#dtrms2_low = lambda dt: -0.8 + 0.028 * (dt-20)
#dtrms2_upp = lambda dt:  2.7 + 0.040 * (dt-20)
dtrms2_cen = lambda dt:  1.0 + 0.034 * (dt-20)

def dist_to_bandcenter(df): return df.Zrms**2 - dtrms2_cen(df.DT)

dtime_guess = lambda zrms: (zrms**2 - 1.0)/0.034 + 20.
def df_extend(df):
    """ Extend the Kdst to include the distance to the S2w vs DT band, DT-guess and R
    """

    df['d2band']     = dist_to_bandcenter(df)
    df['abs_d2band'] = np.abs(df['d2band'])
    dtguess          = dtime_guess(df.Zrms)
    df['DTguess']    = np.nan_to_num(dtguess, 1600)
    r2               = df.X**2 + df.Y**2
    df['R2']         = r2
    df['R']          = np.sqrt(r2)
    return df


#---------- Selections

def sel_in_band(df):
    """ selection in DTrms vs DT
    """
    return in_range(df.Zrms**2, dtrms2_low(df.DT), dtrms2_upp(df.DT))

def sel_1S1(df):

    df_ = df.groupby("event").count()
    idx = df_.index[df_.nS1 == 1]

    sel = np.isin(df.event.values, idx)

    return sel


def kdst_select(df):
    """ kdst selection
        - 1S2 
        - S2 in trigger time (1.35e6, 1.46e6) tics in 40 ns
        - S2e in range (3e3, 11e4) pes
        - DTrms vs DT in band 
            low -0.7 + 0.030 * (dt-20) # Gonzalo's
            upp  2.6 + 0.036 * (dt-20) # Gonzalo's
        - 1S1 in band
    """
    sel_1S2 = df.nS2.values == 1

    range_S2t = (1.35e6, 1.46e6)
    sel_S2t   = in_range(df.S2t, *range_S2t)

    range_S2e = (3e3, 11e3)
    sel_S2e   = in_range(df.S2e, *range_S2e)

    sel_inband = sel_in_band(df)

    sel = sel_1S2 & sel_S2t &  sel_inband & sel_S2e    
    df_ = df[sel]
    sel_1s1 = sel_1S1(df_)
    df_ = df_[sel_1s1]

    names = ('1S2', 'S2-trigger', 'S2e', 'S2w-DT band', '1S1 in band')
    sels  = (sel_1S2, sel_S2t, sel_S2e, sel_inband)
    steps = []
    usel  = sel_1S2
    for name, sel in zip(names, sels):
        usel = sel & usel
        eff  = eff_of_sel(usel)
        steps.append((name, eff))
    eff_final = len(df_)/len(df)
    steps.append(('1S1 in band', eff_final))

    return df_, steps

def sel_fidutial_lowenergy(df, range_S2e = (3.0e3, 10.e3), max_radius = 140.):
    sel1 = in_range(df.S2e, *range_S2e)
    eff_of_sel(sel1, 'range S2e')

    r2 = df.X**2 + df.Y**2
    sel2 = r2 < max_radius**2
    eff_of_sel(sel2, 'max radius')

    sel = sel1 & sel2
    eff_of_sel(sel, 'S23 range and max raidus')

    return sel

def df_compact_s1(df):

    #df_["DT"] = df["DTguess"]

    df_ = df.groupby("event s2_peak".split()).first()
    df_["DT"] = df_["DTguess"]

    df_ = df_.reset_index()

    return df_


#
#---- ploting
#

dtbins     = np.linspace(0, 1600, 101)
dtrmsbins  = np.linspace(0, 10, 101)
dtrms2bins = np.linspace(0, 55, 101)
ebins      = np.linspace(0, 15e3, 101)

freq = lambda : plt.ylabel("frecuency")

def monitor_S1S2(df):
    cv   = pltext.canvas(4, 2)

    df_   = df.groupby("event").first()
    cv(1); pltext.hist(df_.nS1, 50, (0, 50), density = True, label = 'initial'); 
    plt.xlabel('number of S1'); freq();

    cv(2); pltext.hist(df_.nS2, 20, (0, 20), density = True); plt.xlabel('number of S2'); freq();

    df_   = df.groupby("event s2_peak".split()).count()
    cv(1); pltext.hist(df_.nS1, 50, (0, 50), density = True, label = 'count'); 
    
    df_ = df.groupby("event s2_peak".split()).mean()
    cv(3); pltext.hist(df_.S2t, 100, density = True); plt.xlabel('S2 wf time'); freq();

    df_ = df.groupby("event s1_peak".split()).mean()
    cv(4); pltext.hist(df.S1t, 100, density = True); plt.xlabel('S1 wf time'); freq();
    
    plt.tight_layout()

def monitor_S2(df):

    cv = pltext.canvas(4, 2)
    
    df_ = df.groupby("event s2_peak".split()).mean()
    cv(1); pltext.hist(df_.S2e, ebins, density = True); plt.xlabel('S2e (pe)'); freq();

    cv(2); pltext.hist(df_.S2h, 100, (0, 1e4), density = True); plt.xlabel('S2h (pe)'); freq();

    cv(3); pltext.hist(df_.S2q, 100, (0, 4e3), density = True); plt.xlabel('S2q (pes)'); freq();

    cv(4); pltext.hist(df_.qmax, 100, (0, 1e3), density = True); plt.xlabel('qmax (pes)'); freq();

    plt.tight_layout()

def monitor_S2s_dtime(df):

    cv = pltext.canvas(4, 2)
    
    cv(1); plt.hist2d(df.DT, df.Zrms, (dtbins, dtrmsbins)); 
    plt.xlabel("Drift time ($\mu$s)"); plt.ylabel("DT$_{rms}$ ($\mu$s)")

    cv(2); plt.hist2d(df.DT, df.Zrms**2, (dtbins, dtrms2bins), cmin=1e-3)
    plt.plot(df.DT, dtrms2_low(df.DT), ".r", ms=2)
    plt.plot(df.DT, dtrms2_upp(df.DT), ".r", ms=2)
    plt.plot(df.DT, dtrms2_cen(df.DT), '.g', ms = 2)
    plt.xlabel("Drift time ($\mu$s)"); plt.ylabel("DT$_{rms}^2$ ($\mu$s)")

    #cv(3); pltext.hist(df.DT, dtbins); plt.xlabel("Drift time ($\mu$s)"); freq();

    #cv(4); pltext.hist(df.Zrms**2, 100, (0, 40)); plt.xlabel("DT$_{rms}^2$ ($\mu$s)"); freq();

    plt.tight_layout()

def monitor_lifetime(df):

    krS2erange = (4e3, 1e4)
    dtbins2    = np.arange(0, 1500, 51)
    ebins2     = np.arange(*krS2erange, 51)

    cv = pltext.canvas(4, 2)

    cv(1); plt.hist2d(df.DT, df.S2e, (dtbins2, ebins2));
    plt.xlabel(r"DT ($\mu$s)"); plt.ylabel("S2e (pe)");

    cv(2); plt.scatter(df.DT, df.S2e, alpha = 0.01); plt.xlim(0, 1500); plt.ylim(*krS2erange);
    plt.xlabel(r"DT ($\mu$s)"); plt.ylabel("S2e (pe)");

    plt.tight_layout();


def monitor_kr_distribution(df):

    cv = pltext.canvas(4, 2)

    sel = in_range(df.S2e, 7.5e3, 9.5e3)
    eff_of_sel(sel)

    DT = df.DT[sel]
    R2 = df.X[sel]**2 + df.Y[sel]**2

    cv(1); pltext.hist(DT, dtbins, density = True);
    plt.xlabel(r"DT ($\mu$s)"); freq();

    cv(2); pltext.hist(R2, 100, density = True); plt.xlabel("R$^2$ (mm$^2$)"); freq();

    cv(3); plt.hist2d(DT, R2, (30, 30));
    plt.xlabel("DT ($\mu$s)"); plt.ylabel("R$^2$ (mm$^2$)");

    plt.tight_layout();


#-------------------
# Energy resolution
#-------------------

def energy_resolution(energy, plot = False):
    xsel = ~np.isnan(energy)
    nbins, erange = 100, (38., 45)
    if (plot):
        pltext.hist(energy[xsel], nbins, erange);
    xfun = pltext.hfit if plot else histos.hfit
    cc = xfun(energy[xsel], nbins, range = erange, fun = 'gaus');
    pars = cc[3]
    sigma, mu = pars[2], pars[1]
    fwhm = 235.5 * sigma/mu
    if (plot):
        print(' Resolution {:6.2f} % FWHM'.format(fwhm))
    return fwhm

def eres_in_bins(eres, values, bins):
    dt0 = bins[0]
    ress = []
    for dt in bins[1:]:
        isel = ut.in_range(values, (dt0, dt))
        ires = energy_resolution(eres[isel]) 
        dt0 = dt
        ress.append(ires)
    return ress

def plot_eres_in_regions(cdf, nbins = 20):

    dtbins = np.linspace(0, 1400, nbins)
    ress = []
    for rmax in (200, 400, 450):
        usel = ut.in_range(cdf.r, (0, rmax))
        ires = eres_in_bins(cdf.energy[usel], cdf.dtime[usel], dtbins)
        ress.append(ires)

    rads = np.linspace(0, 450, nbins)

    cv = pltext.canvas(2, 2)
    cv(1)
    plt.plot(ut.centers(dtbins), ress[0], marker = 'o', linestyle = 'None', label = r"R $<$ 200");
    plt.plot(ut.centers(dtbins), ress[0], marker = 'o', linestyle = 'None', label = r"R $<$ 400");
    plt.plot(ut.centers(dtbins), ress[1], marker = 'o', linestyle = 'None', label = r"R $<$ 450");
    plt.legend(); plt.xlabel(r"drift time ($\mu$s)"); plt.ylabel("energy resolution (\% FWHM)");

    rrs   = np.linspace(0, 450, 10)
    ress2 = eres_in_bins(cdf.energy, cdf.r, rrs)
    cv(2)
    plt.plot(ut.centers(rrs), ress2, marker = 'o', linestyle = 'None');
    plt.xlabel(r"radius (mm)"); plt.ylabel("energy resolution (\% FWHM)");


