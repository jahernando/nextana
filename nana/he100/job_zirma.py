# job_zirma.py - Batch job to run Zirma with multiple parameter combinations
#
# Scans over qthreshold values and scatter-hit removal options
# to produce event summaries for systematic studies.

import sys

from nana.he100.zirma import run_zirma
sys.path.insert(0, '/Users/hernando/work/investigacion/NEXT/software/nextana/nana/he100')

# Configuration: runs, thresholds and scatter-hit options to process
run_numbers = [15589]
qthresholds = [7, 10, 12]
remove_scatter_hits = [True, False]

for run_number in run_numbers:
    for qthreshold in qthresholds:
        for remove_scatter_hit in remove_scatter_hits:
            print(f'Run {run_number}, qthreshold {qthreshold}, remove_scatter_hits {remove_scatter_hit}')
            run_zirma(run_number, qthreshold, remove_scatter_hit)





