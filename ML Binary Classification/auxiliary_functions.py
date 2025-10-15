import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from typing import List, Tuple, Dict


def load_hyperparameters(config_filename: str, stage: str) -> Tuple[str, Dict]:
    """Load hyperparameters from configuration file for the specified stage."""
    with open(config_filename, "r") as file:
        config = json.load(file)

    classifier_name = config["active_classifier"]
    class_config = config["hyperparameters"][stage][classifier_name]

    return classifier_name, class_config


def load_raw_dataset(dataset_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from CSV file and convert to numpy arrays.

    Separates features (X) and labels (y), converts categorical values
    (Yes->1, No->0), and returns as numpy arrays.
    """
    df = pd.read_csv(dataset_filename)

    # Separate features (X) and labels (y)
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]  # Last column

    # Convert categorical values: Yes->1, No->0, keep numeric as-is
    dataset_x = X.replace({"Yes": 1, "No": 0}).astype(float).to_numpy()
    dataset_y = y.to_numpy()

    return dataset_x, dataset_y


def apply_normalization(
    raw_dataset: np.ndarray, scaler: StandardScaler | None
) -> Tuple[np.ndarray, StandardScaler | None]:
    """Apply standardization to dataset using StandardScaler.

    If scaler is None, creates and fits a new scaler. Otherwise, uses the
    existing fitted scaler for transformation.
    """
    if scaler is None:
        # Create new StandardScaler and fit it on the dataset
        scaler = StandardScaler()
        new_X = scaler.fit_transform(raw_dataset)
    else:
        # Use existing fitted scaler to transform the dataset
        new_X = scaler.transform(raw_dataset)

    return new_X, scaler


def split_dataset(
    dataset_X: np.ndarray, dataset_Y: np.ndarray, n: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split dataset into n random partitions for cross-validation.

    Creates shuffled copies of the dataset and divides them into n partitions
    with balanced sizes.
    """
    X_copy = dataset_X.copy()
    Y_copy = dataset_Y.copy()

    # Shuffle the dataset
    samples = len(dataset_X)
    indices = np.arange(samples)
    np.random.shuffle(indices)

    X_shuffled = X_copy[indices]
    Y_shuffled = Y_copy[indices]

    # Compute partition sizes
    base_size = samples // n
    remainder = samples % n

    partition_sizes = []
    for i in range(n):
        if i < remainder:
            partition_sizes.append(base_size + 1)
        else:
            partition_sizes.append(base_size)

    # Create partitions
    partitions = []
    start_index = 0

    for size in partition_sizes:
        end_index = start_index + size

        X_partition = X_shuffled[start_index:end_index]
        Y_partition = Y_shuffled[start_index:end_index]

        partitions.append((X_partition, Y_partition))
        start_index = end_index

    return partitions


def train_classifier(
    classifier_name: str,
    hyper_params: dict,
    train_split_X: np.ndarray,
    train_split_Y: np.ndarray,
):
    """Create and train a classifier with specified hyperparameters.

    Supports Decision Tree, Random Forest, and Logistic Regression classifiers.
    """
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
        classifier = LogisticRegression(
            penalty=hyper_params.get("penalty"), C=hyper_params.get("C"), solver="saga"
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    classifier.fit(train_split_X, train_split_Y)

    return classifier
