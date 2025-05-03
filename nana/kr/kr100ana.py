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
import hipy.pltext       as pltext
#import hipy.profile      as prof
import hipy.styles       as pltsty

pltsty.style('jhep')

#from invisible_cities.io  . pmaps_io        import load_pmaps, load_pmaps_as_df
#from invisible_cities.io.dst_io import load_dst, load_dsts
from invisible_cities.core.core_functions import in_range

def eff_of_sel(sel, name = "", do_print = True):
    nsel, ntot = np.sum(sel), len(sel)
    eff = nsel/float(ntot)
    if do_print:
        seff = "{:.4f}".format(eff)
        print(f"{name} efficiency {seff}, {nsel}/{ntot}.")
    return eff

dtrms2_low = lambda dt: -0.8 + 0.028 * (dt-20)
dtrms2_upp = lambda dt:  2.7 + 0.040 * (dt-20)
dtrms2_cen = lambda dt:  1.0 + 0.034 * (dt-20)

def dist_to_bandcenter(df): return df.Zrms**2 - dtrms2_cen(df.DT)

dtime_guess = lambda zrms: (zrms**2 - 1.0)/0.034 + 20.

def df_extend(df):

    df['d2band']     = dist_to_bandcenter(df)
    df['abs_d2band'] = np.abs(df['d2band'])
    dtguess          = dtime_guess(df.Zrms)
    df['DTguess']    = np.nan_to_num(dtguess, 1600)
    return df


#---------- Selections

def sel_in_band(df):
    return in_range(df.Zrms**2, dtrms2_low(df.DT), dtrms2_upp(df.DT))

def sel_1S1(df):

    df_ = df.groupby("event").count()
    idx = df_.index[df_.nS1 == 1]

    sel = np.isin(df.event.values, idx)

    return sel


def df_apply_selection(df, one_s1 = True):
    sel_1S2   = df.nS2.values == 1
    eff_of_sel(sel_1S2, "1S2")

    range_S2t = (1.35e6, 1.46e6)
    sel_S2t   = in_range(df.S2t, *range_S2t)
    eff_of_sel(sel_S2t, "S2 in trigger time")

    #range_S2e = (3e3, 11e4)
    #sel_S2e   = in_range(df.S2t, *range_S2e)
    #eff_of_sel(sel_S2e, "S2e")

    sel_inband = sel_in_band(df)
    eff_of_sel(sel_S2t, "S1 in band")

    sel = sel_1S2 & sel_S2t &  sel_inband
    eff_of_sel(sel, "1S2 & S2 in trigger time & some S1 in band")

    df_ = df[sel]

    if (one_s1):
        sel_1s1 = sel_1S1(df_)
        eff_of_sel(sel_1s1, "1S1 in band") 
        df_ = df_[sel_1s1]

    return df_

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
