import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from typing import List, Tuple, Dict


def load_hyperparameters(config_filename: str, stage: str) -> Tuple[str, Dict]:
    # 1) load the JSON file with the configuration ...
    with open(config_filename, "r") as file:
        config = json.load(file)

    # 2) find the active classifier
    classifier_name = config["active_classifier"]

    # 3) return the hyperparameters for the selected stage and classifier
    class_config = config["hyperparameters"][stage][classifier_name]

    return classifier_name, class_config


def load_raw_dataset(dataset_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    # 1) load the raw data from file in CSV format
    #         (input: filename, output: pandas DataFrame loaded from file)
    df = pd.read_csv(dataset_filename)

    # 2) convert the DataFrame into dataset arrays:
    #             a single array for Y
    #             a 2D array for X
    #          For N columns, the first 1 to N-1 are the attribute values (x), and
    #             the last one (N) is the label value or class (y).
    #          convert everything to float.

    # Separate features (X) and labels (y)
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]  # Last column

    # 3) convert to numpy arrays with categorical mapping
    # Convert categorical values: Yes->1, No->0, keep numeric as-is
    dataset_x = X.replace({"Yes": 1, "No": 0}).astype(float).to_numpy()
    dataset_y = y.to_numpy()

    # 4) return the numpy arrays as a tuple: (x, y)
    return dataset_x, dataset_y


def apply_normalization(
    raw_dataset: np.ndarray, scaler: StandardScaler | None
) -> Tuple[np.ndarray, StandardScaler | None]:
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


def split_dataset(
    dataset_X: np.ndarray, dataset_Y: np.ndarray, n: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
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


def train_classifier(
    classifier_name: str,
    hyper_params: dict,
    train_split_X: np.ndarray,
    train_split_Y: np.ndarray,
):
    # 1) Create a new classifier
    if classifier_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=hyper_params.get("max_depth"),
            criterion=hyper_params.get("criterion"),
        )
    elif classifier_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=hyper_params.get("n_trees"),
            max_depth=hyper_params.get("max_depth"),
        )
    elif classifier_name == "logistic_classifier":
        # use 'saga' solver for all penalties
        classifier = LogisticRegression(
            penalty=hyper_params.get("penalty"), C=hyper_params.get("C"), solver="saga"
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    # 2) Train this classifier with the given data
    classifier.fit(train_split_X, train_split_Y)

    # 3) Return the trained classifier
    return classifier
