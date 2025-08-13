
import sys

from nana.he100.zirma import run_zirma
sys.path.insert(0, '/Users/hernando/work/investigacion/NEXT/software/nextana/nana/he100')


run_numbers = [15589]
qthresholds = [7, 10, 12]
remove_scatter_hits = [True, False]

for run_number in run_numbers:
    for qthreshold in qthresholds:
        for remove_scatter_hit in remove_scatter_hits:  
            print(f'Run {run_number}, qthreshold {qthreshold}, remove_scatter_hits {remove_scatter_hit}')
            run_zirma(run_number, qthreshold, remove_scatter_hit)   





