# Tutorial: High Energy (HE) Analysis in NEXT-100

## Package structure `nana/he100`

```
nana/he100/
  hefunc.py      -- Core functions: hit loading, clustering, energy correction, event summary
  heana.py       -- Energy scale analysis (calibration peak fitting)
  zirma.py       -- "Zirma" pipeline: full Sophronia hit processing
  job_zirma.py   -- Batch script to run Zirma with multiple parameters
```

## Notebooks in `NB/he100`

| Notebook                 | Description                                                      | Status |
|--------------------------|------------------------------------------------------------------|--------|
| `he100_escale.ipynb`     | Energy scale calibration: fit Th-228 peaks, linear calibration, FWHM/E extrapolation to Qbb | Clean |
| `he100_peaks_hits.ipynb` | Hit-level analysis: DBSCAN clustering dev, Kr correction, DE event selection, peak fitting | Draft |
| `he100_samsum_de.ipynb`  | Single-cluster event summary: energy resolution, scatter hit fractions | Draft |
| `he100_sophrext.ipynb`   | Sophronia extension: clustering, Kr correction, event statistics development | Draft |
| `he100_zirma.ipynb`      | Post-Zirma analysis: energy resolution vs qthreshold and scatter hit removal | Mixed |

### Notebook details

#### `he100_escale.ipynb` -- Energy Scale Calibration
**Input**: Reconstruction summaries from 8 runs (15589-15597)
**Workflow**:
1. Load DST, RECO, and efficiency tables for each run
2. Correct energy: `total_energy_d = total_energy - Ec_drop_out`
3. Plot full energy spectrum (0.1-3.5 MeV)
4. Fit 5 Th-228 peaks with Gaussian + exponential: 511, 583, 727, 1592, 2615 keV
5. Linear fit of measured vs true peak energies
6. Compute FWHM/E and extrapolate to Q_betabeta (2458 keV)

**Key result**: Energy resolution as a function of energy, extrapolated to the double-beta decay endpoint.

#### `he100_peaks_hits.ipynb` -- Peaks and Hits Analysis (development)
**Input**: Sophronia hits from run 15604
**Workflow**:
1. Load hits from HDF5 files
2. Define and test `label_hits()` (DBSCAN clustering) -- multiple versions
3. Apply Kr energy correction via 3D interpolation
4. Select double-escape (DE) events (1500-1800 keV)
5. Analyze scatter vs cluster hit distributions (energy, position, multiplicity)
6. Fit calibration peaks and compare measured vs true positions

**Note**: Contains duplicate function definitions and exploratory code that was later consolidated into `hefunc.py`.

#### `he100_samsum_de.ipynb` -- Event Cluster Summary (development)
**Input**: CSV event summaries from run 15607
**Workflow**:
1. Load event summaries and filter to single-cluster events (cluster_max == 0)
2. Group by Z-slices, fit energy distributions
3. Define clustering and energy correction functions
4. Select DE events (1600-1750 keV)
5. Study scatter hit fractions and their effect on energy
6. Fit calibration peaks

**Note**: Development notebook. Functions later moved to `hefunc.py`.

#### `he100_sophrext.ipynb` -- Sophronia Extension (development)
**Input**: Sophronia hits and event summaries from run 15604
**Workflow**:
1. Load event summaries, compute energy correction factors
2. Develop DBSCAN clustering functions (`label_hits`, `label_hits_in_cluster_zrange`)
3. Apply Kr energy correction
4. Select DE events (1600-1750 keV)
5. Compute per-event summary statistics
6. Fit energy peaks with Gaussian + polynomial

**Note**: Primary development notebook for the pipeline. Contains many duplicate
function definitions and exploratory cells. The final versions of all functions
were consolidated into `hefunc.py`.

#### `he100_zirma.ipynb` -- Zirma Analysis
**Input**: Zirma output (per-cluster event summaries) from runs 15589, 15590, 15604
**Workflow**:
1. Load `esum` HDF5 files from Zirma processing
2. Study correlations: event energy vs nhits, scatter energy, track length (dz)
3. Fit energy peaks at 511, 583, 1592, 2615 keV
4. Systematic comparison of FWHM for different qthreshold (5, 7, 10, 12 pe) and scatter hit options
5. 2D histograms of energy vs spatial variables (z, dz, R)

**Key finding**: Energy resolution depends weakly on scatter hit removal; the charge threshold has a larger effect.

**Note**: Lower part contains development/exploratory cells with different parameter combinations.

---

## Module `hefunc.py` -- Core functions

### 1. File loading

```python
import nana.he100.hefunc as he100

# List .h5 files in a directory
files = he100.get_files("/path/to/data/", token='.h5')

# Load hits from a Sophronia file
hits = he100.get_hits("sophronia_file.h5")

# Load already-clustered hits
hits = he100.get_hits("processed_file.h5", with_cluster=True)
```

The `hits` DataFrame contains columns: `event, time, X, Y, Z, Q, E, Ec`
- `Q`: SiPM charge (photoelectrons)
- `E`: energy reconstructed by Sophronia
- `Ec`: lifetime-corrected energy

### 2. Event selection by energy

```python
# Keep only events whose total corrected energy falls in a range
hits = he100.select_event_in_energy_range(hits, erange=(1500, 1800))
```

### 3. DBSCAN clustering

```python
# Classify hits as scatter (-1) or belonging to a cluster (0, 1, ...)
hits = he100.cluster_hits(hits)
```

Adds two columns:
- `cluster_id`: DBSCAN label (-1 = scatter/isolated, >= 0 = cluster index)
- `cluster_id_in`: cluster whose z-range contains the hit (-1 if outside all)

DBSCAN parameters are tuned for the NEXT-100 geometry:
- XY scaling: 14.55 mm (SiPM pitch)
- Z scaling: 3.7 mm (longitudinal pitch)
- eps = 2.3, min_samples = 5

### 4. Light redistribution

```python
# Redistribute energy among hits that pass a charge threshold
hits = he100.hits_redistribute_light(hits, sipm_threshold=7.)
```

Internal steps:
1. Compute total energy per (event, Z) slice before the cut
2. Remove hits with Q < threshold
3. Redistribute slice energy proportionally to Q
4. Optionally rescale to conserve total event energy

Added columns: `E_evz, Q_evz, Q_norm, E_norm`

### 5. Energy correction with Krypton map

```python
# Correct energy using the 3D Kr map
Ec = he100.recalibrate_energy(hits, "/path/to/GML_krmap_combined.map3d")
hits['Ec'] = Ec
```

The Kr map provides a multiplicative correction factor f(Z, X, Y) that compensates for:
- Electron attachment losses (Z dependence)
- Geometric effects (X, Y dependence)

### 6. Per-cluster event summary

```python
# Create summary table per (event, cluster)
esum = he100.event_cluster_summary(hits)
```

Summary columns: `event, cluster, nhits, time, energy, xmin, xmax, xave, ymin, ymax, yave, zmin, zmax, zave, rmin, rmax, rave`

Cluster `-1` corresponds to scatter/isolated hits.

---

## Module `heana.py` -- Energy scale

Functions for fitting Th-228 calibration peaks and determining the energy scale.

```python
import nana.he100.heana as heana

# Fit peaks for each run
meas_peaks = heana.get_peak_centers(run_numbers, reco_summ, 'total_energy_d', peak_ranges)

# Linear fit E_true = m * E_meas + n
parameters = heana.energy_scale_fit(meas_peaks, run_numbers, plot_colors)

# Plot residuals
heana.plot_real_vs_meas_peaks(meas_peaks, run_numbers, plot_colors)
```

Reference peaks (Th-228): 511, 583, 727, 860, 1592, 2615 keV

---

## Module `zirma.py` -- Full pipeline

"Zirma" is a processing stage that applies the full chain to Sophronia hits:

```python
from nana.he100.zirma import zirma, run_zirma

# Process a single file
hits, esum = zirma("input.h5", "/path/to/krmap.map3d",
                   ofilename="output.h5",
                   qthreshold=7., remove_scatter_hits=True)

# Process all files for a run
run_zirma(run_number=15589, qthreshold=7, remove_scatter_hits=True)
```

### Internal pipeline of `zirma()`:
1. Load hits with `get_hits()`
2. DBSCAN clustering with `cluster_hits()`
3. Remove hits outside the z-range of any cluster
4. (Optional) Zero the charge of scatter hits
5. Redistribute light with SiPM threshold
6. Recalibrate energy with Kr map
7. Generate per-cluster summary

### Command-line execution:
```bash
python zirma.py --run_number 15589 --qthreshold 7 --remove_scatter_hits True
```

---

## Script `job_zirma.py` -- Parameter scan

Runs Zirma with multiple combinations of `qthreshold` and `remove_scatter_hits`:

```bash
python job_zirma.py
```

Current configuration:
- Runs: [15589]
- Thresholds: [7, 10, 12] pe
- Scatter removal: [True, False]

Total: 6 combinations per run.

---

## Typical workflow

```
Sophronia (.h5)         Kr map (.map3d)
       |                      |
       v                      v
   get_hits()          get_corrector()
       |                      |
       v                      |
  cluster_hits()              |
       |                      |
       v                      |
hits_redistribute_light()     |
       |                      |
       v                      v
         recalibrate_energy()
                |
                v
     event_cluster_summary()
                |
                v
           esum (.h5)
```

### Input data structure

Input files are HDF5 produced by Sophronia (IC):
- Table `RECO/Events`: reconstructed hits with (event, time, X, Y, Z, Q, E, Ec, ...)
- Typical location: `/path/to/data/NEXT100/data/{run_number}/sophronia_selected/hits/`

### Output data structure

Zirma output files contain:
- Table `esum`: per-(event, cluster) summary with energy, position, nhits

---

## Dependencies

- **numpy, pandas, scipy**: numerical computation
- **tables (PyTables)**: HDF5 reading
- **hipy**: internal histogram and utility library (`hipy.utils`, `hipy.histos`, `hipy.pltext`)
- **invisible_cities (IC)**: NEXT reconstruction framework (`load_dst`)
- **scikit-learn**: DBSCAN clustering (used in `label_hits_()`)
- **matplotlib**: visualization (in notebooks)
