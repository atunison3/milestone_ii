

if __name__=='__main__':
    import argparse
    import numpy as np
    import os
    import pandas as pd

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    
    from team17.super.functions import DataCleaningFunctions, LoadData
    from team17.super.functions import split_train_test_by_id, get_paths
    from team17.super.functions import DataIndex
    from team17.super.pipeline import full_pipeline
    from team17.super.evaluations import evaluate_model


    # Create argument parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-results', default='results.csv')
    parser.add_argument('--path_to_assets', default='assets')
    args = parser.parse_args()

    try:
        X_train = np.load(os.path.join(args.path_to_assets, 'X_train.npy'))
        y_train = np.load(os.path.join(args.path_to_assets, 'y_train.npy'))
        X_test = np.load(os.path.join(args.path_to_assets, 'X_test.npy'))
        y_test = np.load(os.path.join(args.path_to_assets, 'y_test.npy'))

    except FileNotFoundError:
        # Read in data
        sessions_path, evse_path = get_paths(args.path_to_assets)
        sessions = LoadData.load_sessions()
        evse = LoadData.load_evse()

        # Merge to one DataFrame
        df = LoadData.merge_data(sessions, evse)
        del sessions, evse

        # Split training/testing data by id
        df_train, df_test = split_train_test_by_id(df, 0.2, 'session_id')

        # Perform data clearning
        clean = DataCleaningFunctions() 
        df_train_clean = clean.clean_dataset(df_train)
        df_test_clean = clean.clean_dataset(df_test)
        del df_train, df_test

        # Run pipelines 
        X_train = full_pipeline.fit_transform(df_train_clean.drop(columns=['charge_duration']))
        X_test = full_pipeline.transform(df_test_clean.drop(columns=['charge_duration']))

        # Get label data
        y_train = np.array(df_train_clean['charge_duration'])
        y_test = np.array(df_test_clean['charge_duration'])
        del df_train_clean, df_test_clean

        # Convert label data from hours to minutes
        y_train *= 60
        y_test *= 60

        # Bin the training data 
        y_train = (y_train // 6).astype(int)
        y_test = (y_test // 6).astype(int)

        # Save 
        np.save(os.path.join(args.path_to_assets, 'X_train.npy'), X_train)
        np.save(os.path.join(args.path_to_assets, 'X_test.npy'), X_test)
        np.save(os.path.join(args.path_to_assets, 'y_train.npy'), y_train)
        np.save(os.path.join(args.path_to_assets, 'y_test.npy'), y_test)


    # Create a data index
    X_train_index = DataIndex('train', X_train.shape[0], 800)
    X_test_index = DataIndex('test', X_test.shape[0], 200)

    # Start grid searching 


    # Logistic Regression
    # Get a new index
    n_train = X_train_index.new_subset('Logistic Regression')
    n_test = X_test_index.new_subset('Logistic Regression')

    # Set up model and params
    log_reg = LogisticRegression(multi_class='multinomial', max_iter=200)
    param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],           
        }
    
    # Evaluate
    log_reg_model = evaluate_model(
        'Logistic Regression',
        log_reg, 
        param_grid, 
        X_train[n_train,:], y_train[n_train],
        X_test[n_test,:], y_test[n_test])



    # Random Forest 

    # Get a new index
    n_train = X_train_index.new_subset('Random Forest')
    n_test = X_test_index.new_subset('Random Forest')

    # Set up model and params
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
            'n_estimators': [50, 100, 200],  
            'max_depth': [None, 5, 10, 20], 
            'min_samples_split': [2, 5, 10],     
            'min_samples_leaf': [1, 2, 4],   
            'bootstrap': [True, False] 
        }
    
    # Evaluate
    rf_model = evaluate_model(
        'Random Forest',
        rf, param_grid,
        X_train[n_train,:], y_train[n_train],
        X_test[n_test], y_test[n_test])    



    # K Nearest Neighbors
    # Get a new index
    n_train = X_train_index.new_subset('KNN')
    n_test = X_test_index.new_subset('KNN')

    # Set up KNN model
    knn = KNeighborsClassifier()
    param_grid = {
            'n_neighbors': list(range(1, 31)), 
            'weights': ['uniform', 'distance']
        }
    # Evaluate
    knn_model = evaluate_model(
        'KNN',
        knn, param_grid,
        X_train[n_train,:], y_train[n_train],
        X_test[n_test], y_test[n_test]) 

    
    # Create dataframe with results
    results = [log_reg_model, rf_model, knn_model]
    results_df = pd.DataFrame([vars(model.score.results) for model in results])
    results_df.to_csv(os.path.join(args.path_to_results, 'results.csv'), index=False)

    









