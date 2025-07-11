import sys
import pickle
import time

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

    # 1) Load your data
    print(f"Loading (testing) data from: {in_raw_data_filename}")
    dataset_x, dataset_y = load_raw_dataset(in_raw_data_filename)

    # 2) Load the normalizer
    print(f"Loading trained normalizer from: {in_normalizer_filename}")
    with open(in_normalizer_filename, "rb") as f:
        normalizer = pickle.load(f)

    # 3) Load the classifier
    print(f"Loading trained classifier from: {in_classifier_filename}")
    with open(in_classifier_filename, "rb") as f:
        classifier = pickle.load(f)

    # 4) normalize the data ...
    print("Normalizing test data using trained normalizer...")
    normalized_x, _ = apply_normalization(dataset_x, normalizer)

    # 5) evaluate classifier on the testing dataset (compute and print metrics)
    print("\nEvaluating classifier on test data...")
    inference_start = time.time()
    test_predictions = classifier.predict(normalized_x)
    inference_end = time.time()
    inference_time = inference_end - inference_start

    # print detailed classification report
    print("Test Set Classification Report:")
    print(classification_report(dataset_y, test_predictions))

    # get classification report as dictionary
    test_report = classification_report(dataset_y, test_predictions, output_dict=True)

    # print metrics summary
    print(f"\nTest Set Summary:")
    print(f"Accuracy: {test_report['accuracy']:.4f}")
    print(
        f"Macro Avg - Precision: {test_report['macro avg']['precision']:.4f}, "
        f"Recall: {test_report['macro avg']['recall']:.4f}, "
        f"F1-Score: {test_report['macro avg']['f1-score']:.4f}"
    )
    print(
        f"Weighted Avg - Precision: {test_report['weighted avg']['precision']:.4f}, "
        f"Recall: {test_report['weighted avg']['recall']:.4f}, "
        f"F1-Score: {test_report['weighted avg']['f1-score']:.4f}"
    )
    print(f"Test Inference Time: {inference_time:.4f} seconds")

    print("\nTesting process completed successfully!")


if __name__ == "__main__":
    main()
