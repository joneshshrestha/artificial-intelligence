
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
    if len(sys.argv) < 5:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_config in_raw_data out_normalizer out_classifier")
        return

    in_config_filename = sys.argv[1]
    in_raw_data_filename = sys.argv[2]
    out_normalizer_filename = sys.argv[3]
    out_classifier_filename = sys.argv[4]

    # TODO: 1) Load your (training) hyper-parameters

    # TODO: 2) Load your data

    # TODO: 3) normalize the data ...

    # TODO: 4) train your classifier on the training split

    # TODO: 5) evaluate your classifier on the training dataset (compute and print metrics)

    # TODO: 6) save the classifier and the standard scaler ... (pickle library is fine)

    # FINISHED!


if __name__ == "__main__":
    main()
