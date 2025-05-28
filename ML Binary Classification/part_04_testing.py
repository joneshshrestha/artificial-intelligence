
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: [Your NAME]
# ===============================================
"""

import sys
import pickle

from auxiliary_functions import *

from sklearn.metrics import classification_report

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_raw_data in_normalizer in_classifier")
        return

    in_raw_data_filename = sys.argv[1]
    in_normalizer_filename = sys.argv[2]
    in_classifier_filename = sys.argv[3]

    # TODO: 1) Load your data

    # TODO: 2) Load the normalizer

    # TODO: 3) Load the classifier

    # TODO: 4) normalize the data ...

    # TODO: 5) evaluate your classifier on the testing dataset (compute and print metrics)

    # FINISHED!


if __name__ == "__main__":
    main()
