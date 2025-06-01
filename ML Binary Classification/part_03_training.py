
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: Jonesh Shrestha
# ===============================================
"""

import sys
import pickle
import time

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

    # 1) Load your (training) hyper-parameters
    print("Loading (training) hyper-parameters...")
    classifier_name, class_config = load_hyperparameters(in_config_filename, "training")
    print(f"Using classifier: {classifier_name}")
    print(f"Hyperparameters: {class_config}")

    # 2) Load your data
    print(f"Loading training data from: {in_raw_data_filename}")
    dataset_x, dataset_y = load_raw_dataset(in_raw_data_filename)

    # 3) normalize the data ...
    print("Normalizing data...")
    new_X, scaler = apply_normalization(dataset_x, None)

    # 4) train your classifier on the training split
    print(f"Training {classifier_name} classifier...")
    start_time = time.time()
    classifier = train_classifier(classifier_name, class_config, new_X, dataset_y)
    end_time = time.time()
    training_time = end_time - start_time
    print("Training completed.")
    print(f"Training Time: {training_time:.4f} seconds")

    # 5) evaluate your classifier on the training dataset (compute and print metrics)
    print("\nEvaluating classifier on training data...")
    inference_start = time.time()
    train_predictions = classifier.predict(new_X)
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    # print metrics using classification report
    print("Training Set Classification Report:")
    print(classification_report(dataset_y, train_predictions))
    
    # get classification report as dictionary
    train_report = classification_report(dataset_y, train_predictions, output_dict=True)
    
    # print metrics summary 
    print(f"\nTraining Set Summary:")
    print(f"Accuracy: {train_report['accuracy']:.4f}")
    print(f"Macro Avg - Precision: {train_report['macro avg']['precision']:.4f}, "
          f"Recall: {train_report['macro avg']['recall']:.4f}, "
          f"F1-Score: {train_report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg - Precision: {train_report['weighted avg']['precision']:.4f}, "
          f"Recall: {train_report['weighted avg']['recall']:.4f}, "
          f"F1-Score: {train_report['weighted avg']['f1-score']:.4f}")
    print(f"Training Inference Time: {inference_time:.4f} seconds")

    # 6) save the classifier and the standard scaler ... (pickle library is fine)
    print(f"Saving trained classifier to: {out_classifier_filename}")
    with open(out_classifier_filename, 'wb') as f:
        pickle.dump(classifier, f)
        
    print(f"\nSaving trained standard scaler to: {out_normalizer_filename}")
    with open(out_normalizer_filename, 'wb') as f:
        pickle.dump(scaler, f)

    print("\nTraining process completed successfully!")
    print(f"Trained {classifier_name} model and scaler saved.")

    # FINISHED!


if __name__ == "__main__":
    main()
