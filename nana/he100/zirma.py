# zirma.py - "Zirma" city: hit filtering and event summary pipeline
#
# Zirma is a processing step that takes Sophronia reconstructed hits and:
#   1. Clusters them with DBSCAN to identify tracks vs scatter
#   2. Optionally zeroes the charge of scatter (isolated) hits
#   3. Redistributes SiPM light above a charge threshold
#   4. Recalibrates energy using a 3D Krypton correction map
#   5. Produces per-cluster event summaries
#
# Can be run as a script: python zirma.py --run_number 15589 --qthreshold 7
#
# The name "Zirma" follows the NEXT convention of naming processing
# stages after Italo Calvino's "Invisible Cities".

import sys
import argparse

pypaths = ('/Users/hernando/work/investigacion/NEXT/software/nextana',
           '/Users/hernando/work/investigacion/NEXT/software/hipy',
           '/Users/hernando/work/investigacion/NEXT/software/IC')
for ipypath in pypaths:
       sys.path.insert(0,ipypath)

pypaths = ('/Users/hernando/miniconda/envs/IC-3.8-2024-06-08/lib/python38.zip',
           '/Users/hernando/miniconda/envs/IC-3.8-2024-06-08/lib/python3.8',
           '/Users/hernando/miniconda/envs/IC-3.8-2024-06-08/lib/python3.8/lib-dynload',
            '/Users/hernando/miniconda/envs/IC-3.8-2024-06-08/lib/python3.8/site-packages')
for ipypath in pypaths:
       sys.path.insert(0,ipypath)

import nana.he100.hefunc as he100


def zirma(ifilename : str,
          path_krmap,
          ofilename = '',
          qthreshold = 7.,
          remove_scatter_hits = True):
    """ Full Zirma processing pipeline for a single input file.

    Parameters
    ----------
    ifilename           : str   - input HDF5 file with Sophronia hits
    path_krmap          : str   - path to the 3D Krypton correction map
    ofilename           : str   - output file path ('' = no output saved)
    qthreshold          : float - SiPM charge threshold in photoelectrons
    remove_scatter_hits : bool  - if True, zero the charge of isolated hits

    Returns
    -------
    (hits, esum) : tuple of DataFrames
        hits - processed hit table
        esum - per-event, per-cluster summary
    """

    # get hits
    hits = he100.get_hits(ifilename)
    #print('number of hits ', len(hits))

    # cluster hits (classify as isolated or connected in clusters)
    hits = he100.cluster_hits(hits)
    
    # remove scatter out range hits
    hits = hits[hits.cluster_id_in >= 0]
    #print('number of hits in range', len(hits))

    # remove scatter hits
    if (remove_scatter_hits):
        sel = hits.cluster_id < 0
        hits.Q[sel] = 0.
        #print('number of isolated hits in range', sum(sel))

    # recompute the weight of the hits in the same slice
    hits = he100.hits_redistribute_light(hits, qthreshold)
    #print('number of hits after q cut ', len(hits))
    hits['E0'] = hits['E'].values
    hits['E']  = hits['E_norm'].values

    # recalibrate energy and replace it
    Ec = he100.recalibrate_energy(hits, path_krmap)
    hits['Ec0'] = hits['Ec'].values
    hits['Ec']  = Ec

    # summarice event per cluster
    esum = he100.event_cluster_summary(hits)

    # save hits and event summaty
    if (ofilename != ''):
        print(f'saving hits and event-summary at {ofilename}')
        #hits.to_hdf(ofilename, 'hits', mode = 'w', complevel = 9, complib = 'zlib')
        esum.to_hdf(ofilename, 'esum', mode = 'a', complevel = 9, complib = 'zlib')

    return hits, esum

def run_zirma(run_number, qthreshold=7, remove_scatter_hits=True):
    """ Run Zirma on all Sophronia hit files for a given run number.

    Reads from: .../sophronia_selected/hits/
    Writes to:  .../sophronia_selected/zirma/

    Output filenames encode the processing parameters (qthreshold, remove_scatter_hits).
    """
    path_data  = f"/Users/hernando/work/investigacion/NEXT/data/NEXT100/data/{run_number}/sophronia_selected/"
    path_krmap = "/Users/hernando/work/investigacion/NEXT/data/NEXT100/data/krmaps/"
    file_krmap = "GML_krmap_combined.map3d"

    files = he100.get_files(path_data + 'hits/')
    for file in files:
        ifile = path_data + 'hits/' + file
        ss = '_q'+str(int(qthreshold))+'_rmscatterhit'+str(remove_scatter_hits)
        ofile = path_data + 'zirma/' + file.replace('.h5', ss + '.h5')
        ofile = ofile.replace('selected_hits_', 'esum_')
        zirma(ifile, path_krmap + file_krmap, ofile,
              qthreshold=qthreshold, remove_scatter_hits=remove_scatter_hits)
    return

#
# Zirma script
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zirma city.")
    parser.add_argument("--run_number", type=int, help="run number")
    parser.add_argument("--qthreshold", type=float, default = 7, help="SiPM pe threshold")
    parser.add_argument("--remove_scatter_hits", type=bool, default = True, help="remove scatter hits")
    args = parser.parse_args()

    run_zirma(args.run_number, args.qthreshold, args.remove_scatter_hits)

