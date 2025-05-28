
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: [Your NAME]
# ===============================================
"""

import sys
import numpy as np
import pickle

from auxiliary_functions import *

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# =======================================================================
#    This is an optional function that you can use to print your metrics
#    in a format that makes it easier to tabulate
#
#    This one is up to you. It won't be evaluated!
# =======================================================================
def custom_metric_print(metrics_dict):
    pass


# =======================================================================================================
#   This function runs cross-validation using the provided
#     - raw dataset: a dataset which has not been normalized,
#          represented by raw X and raw Y
#     - n_folds:   number of data splits to use during the process
#     - classifier_name: name of the classifier algorithm to use
#     - hyperparameter: configuration for the classifier to use
#
#   Keep in mind that dataset normalization should be applied independently
#   for each dataset split. That is, create the merged training set and
#   then normalize this data, and then use the same normalization parameters
#   to normalize the corresponding test split.
#
#   For the evaluation part, you should use classification_report from scikit-learn:
#      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
#
#   The function runs the cross-validation step, computes the metrics per split, aggregates them and
#   returns a dictionary with the aggregated metrics. This dictionary should follow this format:
#     {
#         "train": sub_metrics_dict,
#         "validation": sub_metrics_dict
#     }
#
#   Where each sub_metrics_dict follows this format:
#     {
#           "class 1 name": group_metrics,
#           "class 2 name": group_metrics,
#              ...
#           "class N name": group_metrics,
#           "accuracy": [AVG accuracy value],
#           "macro avg": group_metrics,
#           "weighted avg": group_metrics
#     }
#
#   Where each group_metrics follows this format:
#     {
#           "precision": [AVG value for this metric],
#           "recall":    [AVG value for this metric],
#           "f1-score":  [AVG value for this metric],
#           "support":   [SUM value for this metric]
#     }
#
#   Note that this is a modified of the dictionary returned by classification_report
# =======================================================================================================
def cross_validation(raw_x: np.ndarray, raw_y: np.ndarray, n_folds: int, classifier_name: str, hyper_params: Dict) -> Dict:
    # TODO: 1) split the dataset ....

    # TODO: 2) for each split ...
    #         2.1) prepare the split training dataset (concatenate and normalize)
    #         2.2) prepare the split validation dataset (normalize)
    #         2.3) train your classifier on the training split
    #         2.4) evaluate your classifier on the training split (compute and print metrics)
    #         2.5) evaluate your classifier on the validation split (compute and print metrics)
    #         2.6) collect your metrics

    # TODO: 3) compute the averaged metrics
    #          5.1) compute and print training metrics
    #          5.2) compute and print validation metrics
    final_metrics = {
        "train": {
           "java": {"precision": None, "recall": None, "f1-score": None, "support": None},
           "python": {"precision": None, "recall": None, "f1-score": None, "support": None},
           "accuracy": None,
           "macro avg": {"precision": None, "recall": None, "f1-score": None, "support": None},
           "weighted avg": {"precision": None, "recall": None, "f1-score": None, "support": None}
        },
        "validation": {
            "java": {"precision": None, "recall": None, "f1-score": None, "support": None},
            "python": {"precision": None, "recall": None, "f1-score": None, "support": None},
            "accuracy": None,
            "macro avg": {"precision": None, "recall": None, "f1-score": None, "support": None},
            "weighted avg": {"precision": None, "recall": None, "f1-score": None, "support": None}
        }
    }
    # TODO: 4) return your metrics
    return final_metrics

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

    # TODO: 1) Load your (cross-validation) hyper-parameters

    # TODO: 2) Load your data

    # TODO: 3) Run cross-validation

    # FINISHED!


if __name__ == "__main__":
    main()
