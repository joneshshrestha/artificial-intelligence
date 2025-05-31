"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: Jonesh Shrestha
# ===============================================
"""

import sys
import itertools

from auxiliary_functions import *
from part_01_cross_validation import cross_validation, custom_metric_print

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

    # 1) Load your (grid-search) hyper-parameters
    # we need to test all classifiers to find the best overall configuration
    with open(in_config_filename, 'r') as file:
        config = json.load(file)
    
    grid_search_hparams = config["hyperparameters"]["grid_search"]
    
    # 2) Load your data
    filename = in_raw_data_filename  # e.g. "training_data_{}.csv"
    
    datasets = {}
    for name in ["small", "medium", "large", "very_large"]:
        datasets[name] = load_raw_dataset(filename.format(name))

    # 3) generate a combination of parameters (check itertools.product)
    all_combinations = []
    # loops over every dataset and every hyperparameter combination
    # decision Tree combinations
    for dataset_name in datasets.keys():
        for params in itertools.product(grid_search_hparams["decision_tree"]["criterion"], 
        grid_search_hparams["decision_tree"]["max_depth"]):
            all_combinations.append({
                "classifier": "decision_tree",
                "dataset": dataset_name,
                "config": {"criterion": params[0], "max_depth": params[1]}
            })
    
    # random Forest combinations
    for dataset_name in datasets.keys():
        for params in itertools.product(grid_search_hparams["random_forest"]["n_trees"], 
        grid_search_hparams["random_forest"]["max_depth"]):
            all_combinations.append({
                "classifier": "random_forest", 
                "dataset": dataset_name,
                "config": {"n_trees": params[0], "max_depth": params[1]}
            })
    
    # logistic Regression combinations
    for dataset_name in datasets.keys():
        for params in itertools.product(grid_search_hparams["logistic_classifier"]["penalty"], 
        grid_search_hparams["logistic_classifier"]["C"]):
            all_combinations.append({
                "classifier": "logistic_classifier",
                "dataset": dataset_name, 
                "config": {"penalty": params[0], "C": params[1]}
            })

    print(f"Total: {len(all_combinations)} configurations")

    # 4) Use the combinations of parameters to run a grid search
    best_config = None
    best_f1_score = -1
    best_results = None
    all_results = []

    # loop over every combination
    for i, combination in enumerate(all_combinations):
        print(f"\n*** Configuration {i+1}/{len(all_combinations)} ***")
        print(f"Classifier: {combination['classifier']}")
        print(f"Dataset: {combination['dataset']}")
        print(f"Parameters: {combination['config']}")
        
        # get dataset from datasets dictionary
        dataset_x, dataset_y = datasets[combination['dataset']]
        
        # run cross-validation
        cv_results = cross_validation(dataset_x, dataset_y, n_folds, combination['classifier'], combination['config'])
        
        # extract metrics and timing
        val_macro_f1 = cv_results['validation']['macro avg']['f1-score']
        train_time = cv_results['time']['train_time']
        validation_time = cv_results['time']['validation_time']
        total_time = train_time + validation_time 
        
        # store results
        result_entry = {
            'combination': combination,
            'train_accuracy': cv_results['train']['accuracy'],
            'val_accuracy': cv_results['validation']['accuracy'], 
            'val_macro_recall': cv_results['validation']['macro avg']['recall'],
            'val_macro_precision': cv_results['validation']['macro avg']['precision'],
            'val_macro_f1': val_macro_f1,
            'total_time': total_time,
            'train_time': train_time,
            'validation_time': validation_time,
            'cv_results': cv_results
        }
        all_results.append(result_entry)
        
        # print results using custom function
        print(f"\nDetailed Results for Configuration {i+1}:")
        custom_metric_print(cv_results)
        print(f"Summary - Validation Macro F1: {val_macro_f1:.4f}, Total Time: {total_time:.2f}s")
        
        # check if this is the best configuration
        if val_macro_f1 > best_f1_score:
            best_f1_score = val_macro_f1
            best_config = combination
            best_results = result_entry

    # 5) print the best parameters found (based on highest validation macro f-1 score)
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETED")
    print("="*80)
    
    print("\n\nBest parameters found")
    print(f"\tBest configuration: {best_config['classifier']} on {best_config['dataset']} dataset")
    print(f"\tParameters: {best_config['config']}")
    print(f"\tBest Validation Macro F1-Score: {best_f1_score:.4f}")
    
    print(f"\t Best configuration Results (Training):")
    print(f"\t\t Accuracy: {best_results['train_accuracy']:.4f}")
    train_results = best_results['cv_results']['train']
    print(f"\t\t Macro Avg Recall: {train_results['macro avg']['recall']:.4f}")
    print(f"\t\t Macro Avg Precision: {train_results['macro avg']['precision']:.4f}")
    print(f"\t\t Macro Avg F1-Score: {train_results['macro avg']['f1-score']:.4f}")
    
    print(f"\t Best configuration Results (Validation):")
    print(f"\t\t Accuracy: {best_results['val_accuracy']:.4f}")
    print(f"\t\t Macro Avg Recall: {best_results['val_macro_recall']:.4f}")
    print(f"\t\t Macro Avg Precision: {best_results['val_macro_precision']:.4f}")
    print(f"\t\t Macro Avg F1-Score: {best_results['val_macro_f1']:.4f}")
    print(f"\t\t Total Time: {best_results['total_time']:.2f}s")
    
    # print summary table for each classifier type
    print("\n" + "="*100)
    print("RESULTS SUMMARY TABLES")
    print("="*100)
    
    # Decision Trees Table
    print("\nTable 1. Results Table for Decision Trees")
    print("-" * 120)
    print(f"{'Train Dataset':<17}{'Criterion':<13}{'Depth':<9}{'Train Acc.':<14}{'Val. Acc.':<13}{'Val. Avg Rec.':<17}{'Val. Avg Prec.':<18}{'Val. Avg F1':<15}{'Time Train':<14}{'Time Val.':<13}")
    print("-" * 120)
    
    for result in all_results:
        if result['combination']['classifier'] == 'decision_tree':
            comb = result['combination']
            train_time_str = f"{result['train_time']:.4f} s"
            val_time_str = f"{result['validation_time']:.4f} s"
            print(f"{comb['dataset']:<17}{comb['config']['criterion']:<13}{str(comb['config']['max_depth']):<9}"
                  f"{result['train_accuracy']:<14.4f}{result['val_accuracy']:<13.4f}"
                  f"{result['val_macro_recall']:<17.4f}{result['val_macro_precision']:<18.4f}"
                  f"{result['val_macro_f1']:<15.4f}{train_time_str:<14}{val_time_str:<13}")

    # Random Forest Table  
    print("\nTable 2. Results Table for Random Forest")
    print("-" * 120)
    print(f"{'Train Dataset':<17}{'N Trees':<13}{'Depth':<9}{'Train Acc.':<14}{'Val. Acc.':<13}{'Val. Avg Rec.':<17}{'Val. Avg Prec.':<18}{'Val. Avg F1':<15}{'Time Train':<14}{'Time Val.':<13}")
    print("-" * 120)
    
    for result in all_results:
        if result['combination']['classifier'] == 'random_forest':
            comb = result['combination']
            train_time_str = f"{result['train_time']:.4f} s"
            val_time_str = f"{result['validation_time']:.4f} s"
            print(f"{comb['dataset']:<17}{comb['config']['n_trees']:<13}{comb['config']['max_depth']:<9}"
                  f"{result['train_accuracy']:<14.4f}{result['val_accuracy']:<13.4f}"
                  f"{result['val_macro_recall']:<17.4f}{result['val_macro_precision']:<18.4f}"
                  f"{result['val_macro_f1']:<15.4f}{train_time_str:<14}{val_time_str:<13}")

    # Logistic Regression Table
    print("\nTable 3. Results Table for Logistic Regression")
    print("-" * 120)
    print(f"{'Train Dataset':<17}{'Penalty':<13}{'C':<9}{'Train Acc.':<14}{'Val. Acc.':<13}{'Val. Avg Rec.':<17}{'Val. Avg Prec.':<18}{'Val. Avg F1':<15}{'Time Train':<14}{'Time Val.':<13}")
    print("-" * 120)
    
    for result in all_results:
        if result['combination']['classifier'] == 'logistic_classifier':
            comb = result['combination']
            penalty_str = str(comb['config']['penalty']) if comb['config']['penalty'] is not None else "None"
            train_time_str = f"{result['train_time']:.4f} s"
            val_time_str = f"{result['validation_time']:.4f} s"
            print(f"{comb['dataset']:<17}{penalty_str:<13}{comb['config']['C']:<9}"
                  f"{result['train_accuracy']:<14.4f}{result['val_accuracy']:<13.4f}"
                  f"{result['val_macro_recall']:<17.4f}{result['val_macro_precision']:<18.4f}"
                  f"{result['val_macro_f1']:<15.4f}{train_time_str:<14}{val_time_str:<13}")

    # FINISHED!


if __name__ == "__main__":
    main()
