
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: [Your NAME]
# ===============================================
"""

import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from typing import List, Tuple, Dict


# =========================================================================
#    This function takes as input a filename (a file in JSON format) and
#    loads the file into memory.
#
#    It returns a tuple of the form (classifier_name, class_config), where:
#      - classifier_name refers to the active classifier in the config
#          it is the value for the "active_classifier" field.
#      - class_config is a python dictionary with the contents of the
#          configuration file for the selected stage ("cross_validation" or
#          "training"), and the active classifier.
# =========================================================================
def load_hyperparameters(config_filename: str, stage: str) -> Tuple[str, Dict]:
    # TODO: 1) load the JSON file with the configuration ...
    # TODO: 2) find the active classifier
    # TODO: 3) return the hyperparameters for the selected stage and classifier
    invalid_config = {}
    return "[Invalid Classifier Name]", invalid_config


# =========================================================================
#    This function takes as input a filename (a file in CVS format) and
#    loads the file into memory. It applies the initial codification
#    that converts categorical attributes into values (No=0.0, Yes=1.0).
#    Other numerical attributes are converted from string value to their
#    floating point values.
#
#    It returns a tuple (X, Y) with two numpy arrays X and Y.
#    Keep in mind that the input file has headers that name the columns
#    Also, keep in mind that one row represents one example
#    The last column always represents the y value for that example
# =========================================================================
def load_raw_dataset(dataset_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: there are many ways to achieve this goal, here I will provide recommendations
    #       for basic built-in functions (no libraries required)
    #       however, feel free to implement this using the CSV or Pandas Library

    # TODO: 1) load the raw data from file in CSV format
    #         (input: filename, output: list of strings (one per line)  loaded from file)

    # TODO: 2) convert the lines of text into a dataset using:
    #             a single list for Y
    #             a List of Lists for X
    #          every line in the file is a data row in the dataset
    #          except for the first row that has the header, which you can ignore
    #          For N columns, the first 1 to N-1 are the attribute values (x), and
    #             the last one (N) is the label value or class (y).
    #          Remember to convert everything to float. This also requires codification
    #             of some categorical values
    #         (input: list of strings (one per line of original file)
    #         (outputs: a list of lists for X, a list for Y)

    # TODO: 3) convert your lists for X and Y into numpy arrays
    # (THIS IS AN EXAMPLE, YOU MUST CHANGE THIS
    e1_x = [0, 0, 0]
    e1_y = ["python"]
    e2_x = [0, 0, 1]
    e2_y = ["java"]
    dataset_x = np.array([e1_x, e2_x])
    dataset_y = np.array([e1_y, e2_y])

    # TODO: 4) return the numpy arrays as a tuple: (x, y)
    return dataset_x, dataset_y


# =========================================================================
#    This function takes as input a raw dataset (a numpy array) and
#    applies normalization, to adjust the values assuming that they come from
#    a normal distribution.
#
#    The function takes as parameter an optional StandardScaler
#       if provided, it is assumed that it has been fitted before (on training data)
#          and this function should simply use it on the given dataset
#          the function will return the SAME StandardScaler object it received
#       if NOT provided, then a new StandardScaler object must be created
#          the function should fit the new scaler on the given dataset
#          the function will return the NEW StandardScaler object it created
#
#    It returns a tuple (new_X, Scaler)
#
#    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# =========================================================================
def apply_normalization(raw_dataset: np.ndarray, scaler: StandardScaler | None) -> Tuple[np.ndarray, StandardScaler | None]:
    # TODO: 1) use or create standard scaler to normalize data
    # CHANGE OR REMOVE THIS
    invalid_dataset = np.zeros((10, 20))
    invalid_scaler = None

    # TODO: 2) return the normalized data AND the scaler
    return invalid_dataset, invalid_scaler


# =========================================================================
#    This function takes as input a dataset and splits it into
#    n equal-size DISJOINT partitions for cross validation.
#    if the original dataset cannot be divided into equally sized partitions
#    the extra elements should be distributed on the first so many partitions
#        For example, if the dataset has 43 elements, and n=5
#        then the function needs to split them into partitions of sizes
#            9, 9, 9, 8, 8 (= 43)
#        No elements will be missing or repeated, and the largest partitions
#        can have at MOST ONE more element than the smallest partitions
#
#    Here the dataset is represented by two numpy arrays,
#       the first one for X values (the attributes)
#       the second one for Y values (he labels or classes)
#
#   BEFORE SPLITTING, THIS FUNCTION SHOULD SHUFFLE THE DATASET. THIS REQUIRES
#   THE SAME RE-ORDERING TO BE APPLIED TO BOTH X AND Y. FOR THIS, CREATE AND
#   USE AN ARRAY OF SHUFFLED INDEXES.
#
#   Each partition is represent by a tuple of numpy arrays representing
#        the X and Y for that partition
#   The function returns a list of tuples representing all partitions
#
#   HINT: some functions on Scikit-learn might be useful here, but this
#   functionality is very easy to implement with array slicing
#   as provided by the ndarray class of numpy
# =========================================================================
def split_dataset(dataset_X: np.ndarray, dataset_Y: np.ndarray, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # TODO 1) create a copy of the dataset

    # TODO 2) shuffle the copy of the dataset
    # TODO:  2.1) create random order for all elements (Check: np.random.shuffle)
    # TODO:  2.2) apply this random order to X
    #  (Check: advanced indexing: https://numpy.org/doc/stable/user/basics.indexing.html)
    # TODO:  2.3) apply this random order to Y
    #  (Check: advanced indexing: https://numpy.org/doc/stable/user/basics.indexing.html)

    # TODO: 3) compute partition sizes

    # TODO: 4) compute the partitions using the SHUFFLED COPY
    #          also, don't forget to split both x and y

    # TODO: 5) return the partitions of the SHUFFLED COPY
    return []


# ==================================================================================
#   This function takes the name of the classifier and the given hyperparameters
#   and creates a Classifier of the corresponding type, with those hyperparameters
#
#   Then, the function trains the new classifier with the given data
#   and returns it
#
#   Types of classifiers supported:
#              classifier_name == "decision_tree"       -> DecisionTreeClassifier
#              classifier_name == "random_forest"       -> RandomForestClassifier
#              classifier_name == "logistic_classifier" -> LogisticRegression
# ==================================================================================
def train_classifier(classifier_name: str, hyper_params: dict, train_split_X: np.ndarray, train_split_Y: np.ndarray):
    # TODO: 1) Create a new classifier
    # TODO: 2) Train this classifier with the given data
    # TODO: 3) Return the trained classifier
    return None

