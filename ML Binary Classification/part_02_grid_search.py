
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: [Your NAME]
# ===============================================
"""

import sys
import itertools

from auxiliary_functions import *
from part_01_cross_validation import cross_validation, custom_metric_print

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_config in_raw_data n_folds")
        return

    in_config_filename = sys.argv[1]
    in_raw_data_filename = sys.argv[2]
    try:
        n_folds = int(sys.argv[3])
        if n_folds < 2:
            print("Invalid value for n_folds. Must be an integer >= 2")
            return
    except:
        print("invalid value for n_folds. Must be an integer >= 2")
        return

    # TODO: 1) Load your (grid-search) hyper-parameters

    # TODO: 2) Load your data

    # TODO: 3) generate a combination of parameters (check itertools.product)
    #          https://docs.python.org/3/library/itertools.html#itertools.product

    # TODO: 4) Use the combinations of parameters to run a grid search
    #       For each combination of parameters
    #        3.1) create a custom config for this combination
    #        3.2) run full cross-validation for this combination of parameters (re-use function from part 01)
    #        3.3) collect results

    # TODO: 4) print the best parameters found (based on highest validation macro f-1 score)
    print("\n\nBest parameters found")
    print(f"\t-> Best configuration: ???")
    print(f"\t-> Best configuration Results (Training):")
    print("\t\t YOUR RESULTS HERE")
    print(f"\t-> Best configuration Results (Validation):")
    print("\t\t YOUR RESULTS HERE")
    # FINISHED!


if __name__ == "__main__":
    main()
