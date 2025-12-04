import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy.stats import binned_statistic_2d   # does the heavy lifting


def get_voxel_info(voxel_info):
    sizes = [voxel_info['size_x'].values[0], voxel_info['size_y'].values[0], voxel_info['size_z'].values[0]]
    mins = [voxel_info['min_x'].values[0], voxel_info['min_y'].values[0], voxel_info['min_z'].values[0]]
    return sizes, mins

def transform_voxels(df, voxel_info, coords = ['xbin', 'ybin', 'zbin']):
    sizes, mins = get_voxel_info(voxel_info)
    df = df.copy()
    for coor, size, min_val in zip(coords, sizes, mins):
        df[coor] = df[coor] * size + min_val + size / 2   # centers
    return df, sizes, mins

def create_xy_bins(df, coords = ['xbin', 'ybin', 'zbin'], pitch = [15.55, 15.55, 10]):
    x_min, x_max = df[coords[0]].min() - pitch[0]/2, df[coords[0]].max() + pitch[0]/2
    y_min, y_max = df[coords[1]].min() - pitch[1]/2, df[coords[1]].max() + pitch[1]/2
    z_min, z_max = df[coords[2]].min() - pitch[2]/2, df[coords[2]].max() + pitch[2]/2
    x_bins = np.arange(x_min, x_max + pitch[0], pitch[0])
    y_bins = np.arange(y_min, y_max + pitch[1], pitch[1])
    z_bins = np.arange(z_min, z_max + pitch[2], pitch[2])
    return x_bins, y_bins, z_bins

def plot_2d_bins(event,
                 bins_info,
                 nexus_event = pd.DataFrame([]),
                 value_col: str = 'energy',
                 statistic: str = 'sum',
                 colorbar_label: str = 'Energy (MeV)', 
                 dropped_voxels = False, 
                 center_plots = True, 
                 cbar_loc = 'right'):
    """
    Plot XY, XZ & YZ 2-D maps of *any* voxel attribute, using a chosen
    aggregation per bin (sum, max, mean, …), with one shared colour bar
    and equal-sized square sub-plots.

    Parameters
    ----------
    event         : object
        Passed to `transform_voxels`; needs attributes `.x`, `.y`, `.z`
        for MC hit draws.
    bins_info     : any
        Passed through to `transform_voxels`.
    value_col     : str, default 'energy'
        Column in the voxel DataFrame to visualise.
    statistic     : str | callable, default 'sum'
        Aggregation method per bin.  Anything accepted by
        `binned_statistic_2d` ('sum', 'max', 'mean', 'count', …) or a
        custom function.
        *Example:*  statistic='max' together with value_col='label'
        turns every bin that contains at least one "1" into 1.
    colorbar_label: str | None
        Label for the colour bar.  If None, a sensible default is built
        from `value_col` and `statistic`.
    dropped_voxels: bool | False
        Flag to plot voxels with deco prediction 0 as white boxes
    """
    if value_col == 'label' or value_col == 'decolabel': eps = 1e-10
    else: eps = 0
    mc = nexus_event.copy()
    if mc.empty:
        mc = pd.DataFrame(columns = ['x','y', 'z'])
    event = event[event.decopred == 1]
    drop_vox = event[event.decopred == 0]
    # ------------------------------------------------------------------
    # 1 )  Voxelise the event  -----------------------------------------
    # ------------------------------------------------------------------
    df, _, _ = transform_voxels(event, bins_info)

    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in voxel DataFrame.")
    # ------------------------------------------------------------------
    # 2)  Centering: compute event midpoint and shift coordinates
    # ------------------------------------------------------------------
    x_center = 0.5 * (df.xbin.min() + df.xbin.max())
    y_center = 0.5 * (df.ybin.min() + df.ybin.max())
    z_center = 0.5 * (df.zbin.min() + df.zbin.max())

    # ------------------------------------------------------------------
    # 3)  Bin edges (shifted to match centered event)
    # ------------------------------------------------------------------
    pitch = [bins_info.size_x.values[0],
             bins_info.size_y.values[0],
             bins_info.size_z.values[0]]
    x_bins, y_bins, z_bins = create_xy_bins(df, pitch=pitch)

    # ------------------------------------------------------------------
    # 4 )  Projection definitions  -------------------------------------
    # ------------------------------------------------------------------
    #   (x-array, y-array, xedges, yedges, mc_x, mc_y, title, labels)
    projections = [
        (df.xbin.values, df.ybin.values, x_bins, y_bins,
         mc.x,  mc.y,  'XY', ('X (mm)', 'Y (mm)')),

        (df.xbin.values, df.zbin.values, x_bins, z_bins,
         mc.x,  mc.z,  'XZ', ('X (mm)', 'Z (mm)')),

        (df.ybin.values, df.zbin.values, y_bins, z_bins,
         mc.y,  mc.z,  'YZ', ('Y (mm)', 'Z (mm)')),
    ]

    # ------------------------------------------------------------------
    # 5 )  Compute each binned statistic --------------------------------
    # ------------------------------------------------------------------
    Hs = []
    for x, y, xe, ye, *_ in projections:
        H, *_ = binned_statistic_2d(
            x, y, df[value_col].values + eps,
            statistic=statistic, bins=[xe, ye])
        Hs.append(np.nan_to_num(H, nan=0.0))

    # Shared colour limits
    vmin = 1e-10 #min(H.min() for H in Hs)
    vmax = max(H.max() for H in Hs)
    # Keep empty bins white (works even if vmin==0)
    cmap = plt.get_cmap("viridis")
    # plt.cm.viridis.copy()
    cmap.set_under('white')

    # ------------------------------------------------------------------
    # 6 )  Figure & axes layout  ---------------------------------------
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False, constrained_layout = True, gridspec_kw={'wspace': 0.05})
    for ax in axes:
        ax.set_aspect('equal', adjustable='datalim')

    # ------------------------------------------------------------------
    # 7 )  Draw projections  -------------------------------------------
    # ------------------------------------------------------------------
    im_for_cbar = None
    first_plot = True
    for ax, H, (x, y, xe, ye, mc_x, mc_y, title, (xlbl, ylbl)) in zip(
            axes, Hs, projections):
        if center_plots:
            # Shift bin edges for plotting only
            xe = xe - (x_center if title in ("XY", "XZ") else y_center)
            ye = ye - (y_center if title == "XY" else z_center)

            # Shift MC hits
            mc_x = mc_x - (x_center if title in ("XY", "XZ") else y_center)
            mc_y = mc_y - (y_center if title == "XY" else z_center)

        dx = np.diff(xe)[0]
        dy = np.diff(ye)[0]
        x_all, y_all = [xe[0] - dx/2, xe[-1] + dx/2], [ye[0] - dy/2, ye[-1] + dy/2]
        # plot first the dropped voxels if any
        if dropped_voxels:
            df_drop, _, _ = transform_voxels(drop_vox, bins_info)
            if center_plots:
                df_drop.xbin = df_drop.xbin - x_center
                df_drop.ybin = df_drop.ybin - y_center
                df_drop.zbin = df_drop.zbin - z_center
            # pick correct projection coordinates
            if title == "XY":
                coord = ['xbin', 'ybin']
            elif title == "XZ":
                coord = ['xbin', 'zbin']
            elif title == "YZ":
                coord = ['ybin', 'zbin']
                
            a = df[coord].drop_duplicates()
            b = df_drop[coord].drop_duplicates()
            df_res = b.merge(a, on=coord, how='left', indicator=True)
            xv, yv = df_res[df_res['_merge']=='left_only'][coord].values.T

            x_all = [min(x_all[0], xv.min()-dx), max(x_all[1], xv.max()+dx)]
            y_all = [min(y_all[0], yv.min()-dy), max(y_all[1], yv.max()+dy)]
            first_rect = True
            for xx, yy in zip(xv, yv):
                x0 = xx - dx/2
                y0 = yy - dy/2
                rect = Rectangle((x0, y0), dx, dy,
                                facecolor='none',
                                edgecolor='grey',
                                linewidth=0.8,
                                # zorder=5,
                                label="Dropped" if first_rect else None)
                ax.add_patch(rect)
                first_rect = False

        X, Y = np.meshgrid(xe, ye)
        im = ax.pcolormesh(X, Y, H.T,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto') # ensures no gaps
        im_for_cbar = im

        # MC truth hits
        if not nexus_event.empty:
            ax.scatter(mc_x, mc_y, marker='x', c='r', s=5, label='MC hits')



        # Labels & cosmetics
        ax.set_xlim(*x_all)
        ax.set_ylim(*y_all)
        ax.set_xlabel(xlbl, size = 25) #20
        ax.set_ylabel(ylbl, size = 25, labelpad = -10) #20
        ax.tick_params('both', size = 13, labelsize = 27) #8, 22
        # ax.set_title(title)
        if first_plot:
            ax.legend(loc='best', markerscale=5, fontsize = 23) #20
            first_plot = False

    # Figure coordinates [x0, y0, width, height], all between 0 and 1
    if cbar_loc == 'right':
        cbar_x = 1.01
        lblpad = 20
    if cbar_loc == 'left':
        cbar_x = -0.1
        lblpad = -130
    
    cbar_width = 0.03
    cbar_height = 0.79 # 0.82
    cbar_y = 0.205 #0.165

    cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar = fig.colorbar(im_for_cbar, cax=cax)
    if colorbar_label is None:
        colorbar_label = f"{value_col.capitalize()} ({statistic})"
    cbar.set_label(colorbar_label, size=27, labelpad = lblpad)
    cbar.ax.tick_params(labelsize=25) # 22




