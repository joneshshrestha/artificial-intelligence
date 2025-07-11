import sys
import numpy as np
import pickle
import time

from auxiliary_functions import *

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Function used to print the metrics in a format that makes it easier to tabulate


def custom_metric_print(metrics_dict):
    if not metrics_dict:
        print("No metrics to display.")
        return

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS - CUSTOM FORMATTED")
    print("=" * 80)

    # print header for the table
    print(
        f"{'Dataset':<12} {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
    )
    print("-" * 80)

    for dataset_type in ["train", "validation"]:
        if dataset_type not in metrics_dict:
            continue

        dataset_metrics = metrics_dict[dataset_type]
        dataset_label = dataset_type.capitalize()

        # print class-specific metrics
        for class_name in sorted(dataset_metrics.keys()):
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                class_metrics = dataset_metrics[class_name]
                print(
                    f"{dataset_label:<12} {class_name:<12} "
                    f"{class_metrics['precision']:<10.4f} "
                    f"{class_metrics['recall']:<10.4f} "
                    f"{class_metrics['f1-score']:<10.4f} "
                    f"{int(class_metrics['support']):<10}"
                )
                dataset_label = ""  # Only show dataset label once per dataset

        # print accuracy
        if "accuracy" in dataset_metrics:
            print(
                f"{dataset_label:<12} {'Accuracy':<12} "
                f"{'':<10} {'':<10} "
                f"{dataset_metrics['accuracy']:<10.4f} "
                f"{'':<10}"
            )

        # print macro average
        if "macro avg" in dataset_metrics:
            macro_metrics = dataset_metrics["macro avg"]
            print(
                f"{'':<12} {'Macro Avg':<12} "
                f"{macro_metrics['precision']:<10.4f} "
                f"{macro_metrics['recall']:<10.4f} "
                f"{macro_metrics['f1-score']:<10.4f} "
                f"{int(macro_metrics['support']):<10}"
            )

        # print weighted average
        if "weighted avg" in dataset_metrics:
            weighted_metrics = dataset_metrics["weighted avg"]
            print(
                f"{'':<12} {'Weighted Avg':<12} "
                f"{weighted_metrics['precision']:<10.4f} "
                f"{weighted_metrics['recall']:<10.4f} "
                f"{weighted_metrics['f1-score']:<10.4f} "
                f"{int(weighted_metrics['support']):<10}"
            )

        if dataset_type == "train":
            print("-" * 80)  # separator between train and validation

    print("=" * 80)

    # summary comparison
    if "train" in metrics_dict and "validation" in metrics_dict:
        print("\nSUMMARY COMPARISON:")
        print(f"{'Metric':<20} {'Training':<15} {'Validation':<15} {'Difference':<15}")
        print("-" * 65)

        train_acc = metrics_dict["train"]["accuracy"]
        val_acc = metrics_dict["validation"]["accuracy"]
        print(
            f"{'Accuracy':<20} {train_acc:<15.4f} {val_acc:<15.4f} {train_acc - val_acc:<15.4f}"
        )

        for avg_type in ["macro avg", "weighted avg"]:
            if (
                avg_type in metrics_dict["train"]
                and avg_type in metrics_dict["validation"]
            ):
                for metric in ["precision", "recall", "f1-score"]:
                    train_val = metrics_dict["train"][avg_type][metric]
                    val_val = metrics_dict["validation"][avg_type][metric]
                    metric_name = f"{avg_type.title()} {metric.title()}"
                    print(
                        f"{metric_name:<20} {train_val:<15.4f} {val_val:<15.4f} {train_val - val_val:<15.4f}"
                    )

        print("-" * 65)

    print("\n")


def cross_validation(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    n_folds: int,
    classifier_name: str,
    hyper_params: Dict,
) -> Dict:
    # 1) split the dataset ....
    splits = split_dataset(raw_x, raw_y, n_folds)

    # initialize lists to store metrics for each fold
    train_metrics_list = []
    validation_metrics_list = []

    # initialize timing tracking
    total_train_time = 0
    total_validation_time = 0

    # 2) for each split ...
    for i in range(n_folds):
        print(f"\n *** Fold {i+1}/{n_folds} ***")

        # validation split (current fold)
        validation_x, validation_y = splits[i]

        # combine all other splits for training
        train_x_list = []
        train_y_list = []
        # append to train_x_list and train_y_list all the splits except the current one (i)
        for j in range(n_folds):
            if j != i:
                train_x_list.append(splits[j][0])
                train_y_list.append(splits[j][1])

        #         2.1) prepare the split training dataset (concatenate and normalize)
        train_x = np.concatenate(train_x_list)
        train_y = np.concatenate(train_y_list)

        # normalize training data (fit new scaler)
        normalized_train_x, scaler = apply_normalization(train_x, None)

        #         2.2) prepare the split validation dataset (normalize)
        # use the same scaler fitted on training data
        normalized_val_x, _ = apply_normalization(validation_x, scaler)

        #         2.3) train your classifier on the training split
        train_start_time = time.time()
        classifier = train_classifier(
            classifier_name, hyper_params, normalized_train_x, train_y
        )
        train_end_time = time.time()
        fold_train_time = train_end_time - train_start_time
        total_train_time += fold_train_time

        #         2.4) evaluate your classifier on the training split (compute and print metrics)
        train_predictions = classifier.predict(normalized_train_x)
        train_report = classification_report(
            train_y, train_predictions, output_dict=True
        )
        print("Training Metrics:")
        print(classification_report(train_y, train_predictions))

        #         2.5) evaluate your classifier on the validation split (compute and print metrics)
        val_start_time = time.time()
        val_predictions = classifier.predict(normalized_val_x)
        val_end_time = time.time()
        fold_val_time = val_end_time - val_start_time
        total_validation_time += fold_val_time

        val_report = classification_report(
            validation_y, val_predictions, output_dict=True
        )
        print("Validation Metrics:")
        print(classification_report(validation_y, val_predictions))

        #         2.6) collect your metrics
        train_metrics_list.append(train_report)
        validation_metrics_list.append(val_report)

    # 3) compute the averaged metrics
    #          5.1) compute and print training metrics
    #          5.2) compute and print validation metrics
    final_metrics = {
        "train": {},
        "validation": {},
        "time": {
            "train_time": total_train_time,
            "validation_time": total_validation_time,
        },
    }
    metrics_data = {"train": train_metrics_list, "validation": validation_metrics_list}

    # iterate through the key value pairs in metrics_data (train and validation)
    for dataset_type, classification_reports_list in metrics_data.items():
        # get the first classification_report as a template for the structure
        first_classification_report = classification_reports_list[0]
        sub_metrics_dict = {}

        # iterate through the key value pairs in first_classification_report
        for key in first_classification_report.keys():
            if key == "accuracy":
                # average accuracy across all folds
                sub_metrics_dict[key] = np.mean(
                    [report[key] for report in classification_reports_list]
                )
            else:
                # create a group_metrics for each class and evaluation metric
                group_metrics = {}
                for metric in first_classification_report[key].keys():
                    # if the metric is support, sum the support across all folds
                    if metric == "support":
                        group_metrics[metric] = np.sum(
                            [
                                report[key][metric]
                                for report in classification_reports_list
                            ]
                        )
                    # average precision, recall, f1-score across all folds
                    else:
                        group_metrics[metric] = np.mean(
                            [
                                report[key][metric]
                                for report in classification_reports_list
                            ]
                        )
                sub_metrics_dict[key] = group_metrics
        # add the sub_metrics_dict to the final_metrics dictionary
        final_metrics[dataset_type] = sub_metrics_dict

    print("\n" + "=" * 50)
    print("FINAL AGGREGATED METRICS")
    print("=" * 50)
    print("\nTraining Set - Aggregated Metrics:")
    print(f"Accuracy: {final_metrics['train']['accuracy']:.4f}")
    for class_name in sorted(final_metrics["train"].keys()):
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            print(
                f"{class_name}: precision={final_metrics['train'][class_name]['precision']:.4f}, "
                f"recall={final_metrics['train'][class_name]['recall']:.4f}, "
                f"f1-score={final_metrics['train'][class_name]['f1-score']:.4f}, "
                f"support={final_metrics['train'][class_name]['support']}"
            )

    print(f"\nValidation Set - Aggregated Metrics:")
    print(f"Accuracy: {final_metrics['validation']['accuracy']:.4f}")
    for class_name in sorted(final_metrics["validation"].keys()):
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            print(
                f"{class_name}: precision={final_metrics['validation'][class_name]['precision']:.4f}, "
                f"recall={final_metrics['validation'][class_name]['recall']:.4f}, "
                f"f1-score={final_metrics['validation'][class_name]['f1-score']:.4f}, "
                f"support={final_metrics['validation'][class_name]['support']}"
            )

    # print results using the custom formatting function
    custom_metric_print(final_metrics)

    # 4) return metrics
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

    # 1) Load (cross-validation) hyper-parameters
    classifier_name, class_config = load_hyperparameters(
        in_config_filename, "cross_validation"
    )
    print(f"Using classifier: {classifier_name}")
    print(f"Hyperparameters: {class_config}")

    # 2) Load data
    dataset_x, dataset_y = load_raw_dataset(in_raw_data_filename)
    print(
        f"Dataset loaded: {dataset_x.shape[0]} samples, {dataset_x.shape[1]} features"
    )
    print(f"Classes: {np.unique(dataset_y)}")

    # 3) Run cross-validation
    print(f"\nRunning {n_folds}-fold cross-validation...")
    cross_validation(dataset_x, dataset_y, n_folds, classifier_name, class_config)


if __name__ == "__main__":
    main()
