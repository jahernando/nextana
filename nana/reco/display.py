import numpy             as np
#import pandas            as pd
#import tables            as tb
import matplotlib.pyplot as plt

import hipy.pltext      as pltext
import hipy.utils       as ut


from mpl_toolkits.mplot3d    import Axes3D

import numpy as np
import matplotlib.pyplot as plt

def plot_energy_projections(
    x, y, z, ene,
    bins=50,
    range_xy=None,
    range_yz=None,
    range_zx=None
):
    """
    Plot 2D histograms of energy accumulation for the projections:
    (x, y), (y, z), and (z, x).

    Parameters
    ----------
    x, y, z : array-like
        Coordinates of the points.
    ene : array-like
        Energy associated to each point.
    bins : int or tuple
        Number of bins for the histograms.
    range_xy, range_yz, range_zx : tuple or None
        Ranges for the histograms, e.g. ((xmin, xmax), (ymin, ymax)).
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (x, y)
    hxy = axes[0].hist2d(
        x, y,
        bins=bins,
        range=range_xy,
        weights=ene
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Energy projection (x, y)")
    plt.colorbar(hxy[3], ax=axes[0], label="Accumulated energy")

    # (y, z)
    hyz = axes[1].hist2d(
        y, z,
        bins=bins,
        range=range_yz,
        weights=ene
    )
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("z")
    axes[1].set_title("Energy projection (y, z)")
    plt.colorbar(hyz[3], ax=axes[1], label="Accumulated energy")

    # (z, x)
    hzx = axes[2].hist2d(
        z, x,
        bins=bins,
        range=range_zx,
        weights=ene
    )
    axes[2].set_xlabel("z")
    axes[2].set_ylabel("x")
    axes[2].set_title("Energy projection (z, x)")
    plt.colorbar(hzx[3], ax=axes[2], label="Accumulated energy")

    plt.tight_layout()
    plt.show()


def track(x       : np.array,
          y       : np.array, 
          z       : np.array,
          ene     : np.array,
          scale   : float  = 10.,
          rscale  : float = 9.,
          chamber : bool = False,
          #ax = None,
          **kargs):

    #plt.subplots(2
    rene = np.copy(ene)/np.max(ene)
    ax3D = plt.gca(projection='3d')
    #ax3D = plt.gca(projection='3d') if ax is None else ax
    size   = scale       * (1. + rscale * rene)
    color  = np.max(ene) * rene
    #kargs['c'] = color if 'color' not in kargs.keys() else kargs['color']
    ax3D.scatter(z, x, y, s = size, c = color, **kargs)
    ax3D.set_xlabel('z (mm)')
    ax3D.set_ylabel('x (mm)')
    ax3D.set_zlabel('y (mm)')
    #if chamber:
    #    ax3D.set_xlim(zsize)
    #    ax3D.set_ylim(xysize)
    #    ax3D.set_zlim(xysize)
    return ax3D

# def track(x, y, z, ene, scale = 10., title = '', cmap = 'magma'):
#
#     rene = ut.arscale(ene, scale)
#
#     ax   = plt.subplot(111, projection = '3d')
#     ax.scatter(x, y, z, c = rene, s = rene, alpha = 0.2, cmap = cmap)
#     ax.set_xlabel('X (mm)');
#     ax.set_ylabel('Y (mm)');
#     ax.set_zlabel('Z (mm)');
#     plt.gcf().colorbar();
#     #ax.colorbar()
#
#     return
#

def event(x       : np.array,
          y       : np.array,
          z       : np.array,
          ene     : np.array,
          scale   : float = 10.,
          rscale  : float = 9.,
          chamber : bool = False,
          **kargs):
    """ Draw an event with hits x, y, z, ene
    inputs:
        x   : np.array, x-hit positions
        y   : np.array, y-hit positions
        z   : np.array, z-hit positions
        ene : np.array, energy or intensity of the hits
        scale  : float, scale factor of the markers
        rscale : float, scale factor of the size with energy/intensity
        chamber: bool (false), in NEW x,y,z frame
    """

    if (not 'alpha'  in kargs.keys()): kargs['alpha']  = 0.5
    if (not 'cmap'   in kargs.keys()): kargs['cmap']   = 'Greys'
    if (not 'marker' in kargs.keys()): kargs['marker'] = 's'

    rene = ut.arscale(ene)

    zsize, xysize = (0., 500.), (-200., 200)

    fig  = plt.figure(figsize=(12, 9));
    #plt.subplots(2
    ax3D = fig.add_subplot(221, projection='3d')
    size   = scale       * (1. + rscale * rene)
    color  = np.max(ene) * rene
    #kargs['c'] = color if 'color' not in kargs.keys() else kargs['color']
    ax3D.scatter(z, x, y, s = size, c = color, **kargs)
    ax3D.set_xlabel('z (mm)')
    ax3D.set_ylabel('x (mm)')
    ax3D.set_zlabel('y (mm)')
    if chamber:
        ax3D.set_xlim(zsize)
        ax3D.set_ylim(xysize)
        ax3D.set_zlim(xysize)

    plt.subplot(2, 2, 2)
    plt.scatter(x, z, s = size, c = color, **kargs)
    ax = plt.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    if chamber:
        plt.xlim(xysize); plt.ylim(zsize)
    plt.colorbar();

    plt.subplot(2, 2, 3)
    plt.scatter(z, y, s = size, c = color, **kargs)
    ax = plt.gca()
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('y (mm)')
    if chamber:
        plt.xlim(zsize); plt.ylim(xysize)
    plt.colorbar();

    plt.subplot(2, 2, 4)
    plt.scatter(x, y, s = size, c = color, **kargs)
    ax = plt.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    if chamber:
        plt.xlim(xysize); plt.ylim(xysize)
    plt.colorbar();
    plt.tight_layout()
    return

#--- WFs

def wf(x, y, z, ene, zstep = 2., xystep = 10.,
       elabel = 'Energy (keV)', **kargs):
    """ Draw the x- waveform and (x, y) energy plane
    inputs:
        x      : np.array, x-hit positions
        y      : np.array, y-hit positions
        z      : np.array, z-hit positions
        ene    : np.array, energy or intensity of the hits
        zstep  : float (2,), z wf-step
        xystep : float (10.), xy pitch
        elabel : str ('Energy KeV'), energy label
    """

    xlabel, ylabel, zlabel = 'x (mm)', 'y (mm)', 'z (mm)'
    subplot = pltext.canvas(2)

    xbins = ut.arstep(z, zstep)

    subplot(1)
    pltext.hist(z, xbins, weights = ene, stats = False)
    plt.xlabel(zlabel); plt.ylabel(elabel)


    subplot(2)
    xybins = (ut.arstep(x, xystep), ut.arstep(y, xystep))
    plt.hist2d(x, y, xybins, weights = ene, **kargs);
    plt.xlabel(xlabel); plt.ylabel(ylabel);
    cbar = plt.colorbar(); cbar.set_label(elabel)


    plt.tight_layout()

    return

# def wfcalib(x, y, z, erec, eraw, zstep = 2, xystep = 10.,
#             elabels = ('Energy (keV)', 'Energy (adc)'), **kargs):
#     """ Draw calibration factor erec/eraw in wf and (x,y) plane
#     inputs:
#         x        : np.array, x-hit positions
#         y        : np.array, y-hit positions
#         x        : np.array, z-positions
#         erec     : np.array, energy or intensity of the hits
#         eraw     : np.array (optional), energy raw of the hits (optional)
#         xsize    : float (2.), z-width of wf
#         xystep   : float (10), xy-step pitch
#         xylabels :  tuple(str), labes of energy erec, eraw
#     """

#     xlabel, ylabel, zlabel = 'x (mm)', 'y (mm)', 'z (mm)'
#     elabel, e2label = elabels
#     subplot = pltext.canvas(2)

#     subplot(1)

#     zbins = ut.arstep(z, zstep)

#     wf_rec, wf_zs = np.histogram(z, zbins, weights = erec)
#     wf_raw, wf_zs = np.histogram(z, zbins, weights = eraw)

#     wf_fc  = wf_rec/wf_raw
#     wf_zcs = ut.centers(wf_zs)

#     #pltext.hist(wf_zcs, zbins, weights = wf_rec, stats = False)
#     #plt.gca().twinx()
#     #plt.xlabel(xlabel); plt.ylabel(elabel)
#     pltext.hist(wf_zcs, zbins, weights = wf_fc, stats = False);
#     plt.xlabel(zlabel); plt.ylabel(elabel + '/' + e2label)

#     subplot(2)

#     xybins = (ut.arstep(x, xystep), ut.arstep(y, xystep))
#     qrecs, xs, ys = np.histogram2d(x, y, xybins, weights = erec);
#     qraws, xs, ys = np.histogram2d(x, y, xybins, weights = eraw);
#     fc = qrecs/qraws
#     fc[np.isnan(fc)] = 0.
#     xms, yms = np.meshgrid(ut.centers(xs), ut.centers(ys))
#     plt.hist2d(xms.flatten(), yms.flatten(), xybins, weights = fc.T.flatten(), **kargs);
#     plt.xlabel(xlabel); plt.ylabel(ylabel);
#     cbar = plt.colorbar(); cbar.set_label(elabel + '/' + e2label)

#     plt.tight_layout()

#     return
