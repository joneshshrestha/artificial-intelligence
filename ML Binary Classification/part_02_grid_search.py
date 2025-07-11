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
    all_results = []
    
    # Track best configuration for each classifier type
    best_configs_by_classifier = {
        'decision_tree': {'config': None, 'f1_score': -1, 'results': None, 'dataset': None},
        'random_forest': {'config': None, 'f1_score': -1, 'results': None, 'dataset': None},
        'logistic_classifier': {'config': None, 'f1_score': -1, 'results': None, 'dataset': None}
    }

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
        
        # Track best configuration for each classifier type
        classifier_type = combination['classifier']
        if val_macro_f1 > best_configs_by_classifier[classifier_type]['f1_score']:
            best_configs_by_classifier[classifier_type]['config'] = combination['config']
            best_configs_by_classifier[classifier_type]['f1_score'] = val_macro_f1
            best_configs_by_classifier[classifier_type]['results'] = result_entry
            best_configs_by_classifier[classifier_type]['dataset'] = combination['dataset']

    # 5) Print the best parameters found for each classifier
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETED")
    print("="*80)
    
    # Print best configuration for each classifier type
    print("\n" + "="*80)
    print("BEST PARAMETERS FOR EACH CLASSIFIER")
    print("="*80)
    
    for classifier_name, best_info in best_configs_by_classifier.items():
        if best_info['config'] is not None:
            print(f"\n{classifier_name.upper()}:")
            print(f"\tBest Dataset: {best_info['dataset']}")
            print(f"\tBest Parameters: {best_info['config']}")
            print(f"\tBest F1-Score: {best_info['f1_score']:.4f}")

    # print summary table for each classifier type
    print("\n" + "="*100)
    print("RESULTS SUMMARY TABLES")
    print("="*100)
    
    # Decision Trees Table
    print("\nTable 1. Results Table for Decision Trees")
    print("-" * 140)
    print("Train Dataset\tCriterion\tDepth\tTrain Acc.\tVal. Acc.\tVal. Avg Rec.\tVal. Avg Prec.\tVal. Avg F1\tTime Train\tTime Val.")
    print("-" * 140)
    
    for result in all_results:
        if result['combination']['classifier'] == 'decision_tree':
            comb = result['combination']
            train_time_str = f"{result['train_time']:.4f} s"
            val_time_str = f"{result['validation_time']:.4f} s"
            print(f"{comb['dataset']}\t{comb['config']['criterion']}\t{str(comb['config']['max_depth'])}\t"
                  f"{result['train_accuracy']:.4f}\t{result['val_accuracy']:.4f}\t"
                  f"{result['val_macro_recall']:.4f}\t{result['val_macro_precision']:.4f}\t"
                  f"{result['val_macro_f1']:.4f}\t{train_time_str}\t{val_time_str}")

    # Random Forest Table  
    print("\nTable 2. Results Table for Random Forest")
    print("-" * 140)
    print("Train Dataset\tN Trees\tDepth\tTrain Acc.\tVal. Acc.\tVal. Avg Rec.\tVal. Avg Prec.\tVal. Avg F1\tTime Train\tTime Val.")
    print("-" * 140)
    
    for result in all_results:
        if result['combination']['classifier'] == 'random_forest':
            comb = result['combination']
            train_time_str = f"{result['train_time']:.4f} s"
            val_time_str = f"{result['validation_time']:.4f} s"
            print(f"{comb['dataset']}\t{comb['config']['n_trees']}\t{comb['config']['max_depth']}\t"
                  f"{result['train_accuracy']:.4f}\t{result['val_accuracy']:.4f}\t"
                  f"{result['val_macro_recall']:.4f}\t{result['val_macro_precision']:.4f}\t"
                  f"{result['val_macro_f1']:.4f}\t{train_time_str}\t{val_time_str}")

    # Logistic Regression Table
    print("\nTable 3. Results Table for Logistic Regression")
    print("-" * 140)
    print("Train Dataset\tPenalty\tC\tTrain Acc.\tVal. Acc.\tVal. Avg Rec.\tVal. Avg Prec.\tVal. Avg F1\tTime Train\tTime Val.")
    print("-" * 140)
    
    for result in all_results:
        if result['combination']['classifier'] == 'logistic_classifier':
            comb = result['combination']
            penalty_str = str(comb['config']['penalty']) if comb['config']['penalty'] is not None else "None"
            train_time_str = f"{result['train_time']:.4f} s"
            val_time_str = f"{result['validation_time']:.4f} s"
            print(f"{comb['dataset']}\t{penalty_str}\t{comb['config']['C']}\t"
                  f"{result['train_accuracy']:.4f}\t{result['val_accuracy']:.4f}\t"
                  f"{result['val_macro_recall']:.4f}\t{result['val_macro_precision']:.4f}\t"
                  f"{result['val_macro_f1']:.4f}\t{train_time_str}\t{val_time_str}")



if __name__ == "__main__":
    main()
