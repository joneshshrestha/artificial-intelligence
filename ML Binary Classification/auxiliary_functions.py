"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: Jonesh Shrestha
# ===============================================
"""

import json
import numpy as np
import pandas as pd

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
    # 1) load the JSON file with the configuration ...
    with open(config_filename, 'r') as file:
        config = json.load(file)
    
    # 2) find the active classifier
    classifier_name = config["active_classifier"]
    
    # 3) return the hyperparameters for the selected stage and classifier
    class_config = config["hyperparameters"][stage][classifier_name]
    
    return classifier_name, class_config


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
    # 1) load the raw data from file in CSV format
    #         (input: filename, output: pandas DataFrame loaded from file)
    df = pd.read_csv(dataset_filename)
    
    # 2) convert the DataFrame into dataset arrays:
    #             a single array for Y
    #             a 2D array for X
    #          For N columns, the first 1 to N-1 are the attribute values (x), and
    #             the last one (N) is the label value or class (y).
    #          Remember to convert everything to float. This also requires codification
    #             of some categorical values (Yes=1.0, No=0.0)
    
    # Separate features (X) and labels (y)
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]  # Last column
    
    # 3) convert to numpy arrays with categorical mapping
    # Convert categorical values: Yes->1, No->0, keep numeric as-is
    dataset_x = X.replace({'Yes': 1, 'No': 0}).astype(float).to_numpy()
    dataset_y = y.to_numpy()
    
    # 4) return the numpy arrays as a tuple: (x, y)
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
    # use or create standard scaler to normalize data
    if scaler is None:
        # Create new StandardScaler and fit it on the dataset
        scaler = StandardScaler()
        new_X = scaler.fit_transform(raw_dataset)
    else:
        # Use existing fitted scaler to transform the dataset
        new_X = scaler.transform(raw_dataset)
    
    # 2) return the normalized data AND the scaler
    return new_X, scaler


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
    # 1) create a copy of the dataset
    X_copy = dataset_X.copy()
    Y_copy = dataset_Y.copy()
    
    # 2) shuffle the copy of the dataset
    #  2.1) create random order for all elements
    samples = len(dataset_X)
    # create common indices for both X and Y
    indices = np.arange(samples)
    np.random.shuffle(indices)
    
    #  2.2) apply this random order to X
    X_shuffled = X_copy[indices]
    
    #  2.3) apply this random order to Y
    Y_shuffled = Y_copy[indices]
    
    # 3) compute partition sizes
    # compute base size and remainder
    base_size = samples // n
    remainder = samples % n
    
    # create list to store partition sizes
    partition_sizes = []
    # for each partition, add the base size + 1 if the remainder is not 0 else, add the base size
    for i in range(n):
        if i < remainder:
            partition_sizes.append(base_size + 1)
        else:
            partition_sizes.append(base_size)
    
    # 4) compute the partitions using the SHUFFLED COPY
    partitions = []
    # initialize start index
    start_index = 0
    
    for size in partition_sizes:
        end_index = start_index + size
        
        # split both X and Y for this partition
        X_partition = X_shuffled[start_index:end_index]
        Y_partition = Y_shuffled[start_index:end_index]
        
        partitions.append((X_partition, Y_partition))
        # update start index
        start_index = end_index
    
    # 5) return the partitions of the SHUFFLED COPY
    return partitions


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
    # 1) Create a new classifier
    if classifier_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=hyper_params.get('max_depth'),
            criterion=hyper_params.get('criterion')
        )
    elif classifier_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=hyper_params.get('n_trees'),
            max_depth=hyper_params.get('max_depth')
        )
    elif classifier_name == "logistic_classifier":
        # Use 'saga' solver for all penalties
        classifier = LogisticRegression(
            penalty=hyper_params.get('penalty'),
            C=hyper_params.get('C'),
            solver='saga' 
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")
    
    # 2) Train this classifier with the given data
    classifier.fit(train_split_X, train_split_Y)
    
    # 3) Return the trained classifier
    return classifier

