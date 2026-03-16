# heana.py - Energy scale analysis for NEXT-100 HE calibration data
#
# Functions for fitting calibration peaks (Th-228 source) and determining
# the energy scale (measured vs true peak positions).
#
# Main functions:
#   - get_peak_centers : fit Gaussian peaks to the energy spectrum for
#                        each run and return measured peak positions
#   - energy_scale_fit : perform a linear fit of measured vs true peak
#                        energies to determine the energy calibration
#   - plot_real_vs_meas_peaks : plot residuals (meas - true) / true
#
# Note: these functions require `fit_whole_peaks`, `peak_fit`, `lineal`,
#       and `curve_fit` to be defined or imported in the calling scope.


def get_peak_centers(run_numbers, reco_summ, variable, peak_ranges, npeaks = 4, plot = True):
    """ Fit Gaussian peaks in the energy spectrum for each run.

    Parameters
    ----------
    run_numbers  : list[int]       - run numbers to process
    reco_summ    : pd.DataFrame    - reconstruction summary with 'run_n' column
    variable     : str             - column name to fit (e.g. 'total_energy_d')
    peak_ranges  : list[tuple]     - (Emin, Emax) ranges for each peak
    npeaks       : int             - number of peaks to fit (up to 6)
    plot         : bool            - whether to plot the fits

    Returns
    -------
    dict[run_n] -> [[peak_values], [peak_errors]]
    """
    peak_nbins = [100, 100, 60, 40, 100, 25]
    peak_names  = ['e+e- anihilation (511 keV)', r'$^{208}$Tl $\rightarrow$ $^{208}$Pb (583 keV)', r'$^{212}$Bi $\rightarrow$ $^{212}$Po (727 keV)', r'$^{208}$Tl $\rightarrow$ $^{208}$Pb (860 keV)', '1592 keV\nDEP', '2615 keV\nPP']

    ranges = peak_ranges[:npeaks]
    nbins  = peak_nbins[:npeaks]
    names  = peak_names[:npeaks]

    meas_peaks = {}
    for run_n in run_numbers:
        run_reco = reco_summ[reco_summ.run_n == str(run_n)]
        meas_val, meas_err = [], []
        for rang, nbin, name in zip(ranges, nbins, names):
            par, res = fit_whole_peaks(run_reco, variable, peak_fit, rang, nbin, title = 'Run {} - '.format(run_n) + name, plot = plot)
            meas_val.append(par[0][1])
            meas_err.append(par[1][1])
        meas_peaks[run_n] = [meas_val, meas_err]
    return meas_peaks


def energy_scale_fit(meas_peaks, run_numbers, plot_colors, npeaks = 4):
    """ Fit a linear relation E_true = m * E_meas + n for each run.

    Plots the measured vs true peak energies and the linear fit.

    Returns
    -------
    dict[run_n] -> (m, n) linear fit parameters
    """
    real_peaks = [0.511, 0.583, 0.727, 0.86, 1.592, 2.615]
    color_peak = ['r', 'orange', 'y', 'g', 'b', 'm']
    real_peaks = real_peaks[:npeaks]
    color_peak = color_peak[:npeaks]
    
    parameters = {}
    for run_n, c in zip(run_numbers, plot_colors):
        if run_n not in meas_peaks.keys(): continue
        if len(meas_peaks[run_n][0]) == 0: continue
        for m_, r_, c_ in zip(meas_peaks[run_n][0], real_peaks, color_peak):
            plt.plot(m_, r_, '.', color = c_)
        par, err  = curve_fit(lineal, meas_peaks[run_n][0], real_peaks, p0 = (1, 0))
        plt.plot(meas_peaks[run_n][0], lineal(np.array(meas_peaks[run_n][0]), *par), color = c, label = str(run_n) + '\n m = {:.2e}'.format(par[0]) + '\n n = {:.2e}'.format(par[1]))
        parameters[run_n] = par
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.grid()
    plt.xlabel('Measured peak energy (MeV)')
    plt.ylabel('Real peak energy (MeV)')
    return parameters


def plot_real_vs_meas_peaks(meas_peaks, run_numbers, plot_colors, npeak = 4):
    """ Plot relative residuals (meas - true) / true (%) for each run. """
    real_peaks = [0.511, 0.583, 0.727, 0.86, 1.592, 2.615]
    real_peaks = real_peaks[:npeak]
    for run_n, c in zip(run_numbers, plot_colors):
        if run_n not in meas_peaks.keys(): continue
        if len(meas_peaks[run_n][0]) == 0: continue
        m_ = np.array(meas_peaks[run_n][0])
        e_ = np.array(meas_peaks[run_n][1])
        r_ = np.array(real_peaks)
        plt.errorbar(r_, ((m_ - r_) / r_) * 100, yerr = e_ * 100, color = c, markersize = 10, label = run_n, marker = '.', capsize=2)
    plt.legend()
    plt.grid()
    plt.xlabel('Real peak energy (MeV)')
    plt.ylabel('(Measured - Real) / Real peak energy (%)')