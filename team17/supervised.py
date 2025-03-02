import pandas as pd 

def add_mse(df: pd.DataFrame, mse: float, model: str) -> pd.DataFrame: 
    '''Concantes the regression_results_df'''

    # Create df with new results
    new_df = pd.DataFrame({'mse': [mse], 'model': [model]})

    # Concatenate df's
    df = pd.concat([df, new_df])

    return df

# Define variables for print statements
green_text = "\033[32m"
reset_text = "\033[0m"

if __name__=='__main__':
    # Import packages
    import argparse
    import numpy as np
    import os
    import tensorflow as tf
    import warnings

    warnings.filterwarnings("ignore")

    # Import more packages
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.linear_model import LogisticRegression, LinearRegression
    # from sklearn.linear_model import Lasso, Ridge, ElasticNet
    # from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
    # from sklearn.metrics import mean_squared_error
    # from sklearn.model_selection import KFold, cross_val_score
    # from sklearn.neighbors import KNeighborsClassifier
    
    # Import packages from this project
    from team17.super.functions import DataCleaningFunctions, LoadData
    from team17.super.functions import split_train_test_by_id, get_paths
    from team17.super.functions import DataIndex
    from team17.super.pipeline import full_pipeline, build_pipeline, cat_features, num_features
    from team17.super.evaluations import evaluate_model


    # Set random seed
    np.random.seed(42)

    # Create argument parser 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path_to_results', 
        help='Path to directory to save results to.')
    parser.add_argument('--path_to_assets', default='assets')
    args = parser.parse_args()


    try:
        print(f"\n\n{green_text}Beginning Loading of Data{reset_text}\n\n") 
        X_train = np.load(os.path.join(args.path_to_assets, 'X_train.npy'))
        y_train = np.load(os.path.join(args.path_to_assets, 'y_train.npy'))
        X_test = np.load(os.path.join(args.path_to_assets, 'X_test.npy'))
        y_test = np.load(os.path.join(args.path_to_assets, 'y_test.npy'))

    except FileNotFoundError:
        print(f"\n\n{green_text}Beginning Loading of Data{reset_text}\n\n") 
        # Read in data
        sessions_path, evse_path = get_paths(args.path_to_assets)
        sessions = LoadData.load_sessions()
        evse = LoadData.load_evse()
        connector = LoadData.load_connector()

        # Merge to one DataFrame
        df = LoadData.merge_data(sessions, evse, connector)
        del sessions, evse, connector

        # Split training/testing data by id
        df_train, df_test = split_train_test_by_id(df, 0.2, 'session_id')

        # Perform data clearning
        clean = DataCleaningFunctions() 
        df_train_clean = clean.clean_dataset(df_train)
        df_test_clean = clean.clean_dataset(df_test)

        # Run pipelines 
        X_train = full_pipeline.fit_transform(df_train_clean.drop(columns=['charge_duration']))
        X_test = full_pipeline.transform(df_test_clean.drop(columns=['charge_duration']))

        # Get label data
        y_train = np.array(df_train_clean['charge_duration'])
        y_test = np.array(df_test_clean['charge_duration'])

        # Convert label data from hours to minutes
        y_train *= 60
        y_test *= 60

        # Save the dfs (for ablation testing)
        n_train = np.random.choice(df_train_clean.index, 400)
        n_test = np.random.choice(df_test_clean.index, 100)
        df_train_clean.loc[n_train].to_csv(os.path.join(args.path_to_assets, 'X_train.csv'), index=False)
        df_test_clean.loc[n_test].to_csv(os.path.join(args.path_to_assets, 'X_test.csv'), index=False)
        del df_train, df_test 

        # Save 
        np.save(os.path.join(args.path_to_assets, 'X_train.npy'), X_train)
        np.save(os.path.join(args.path_to_assets, 'X_test.npy'), X_test)
        np.save(os.path.join(args.path_to_assets, 'y_train.npy'), y_train)
        np.save(os.path.join(args.path_to_assets, 'y_test.npy'), y_test)


    # Create a data index
    X_train_index = DataIndex('train', X_train.shape[0], 4000)
    X_test_index = DataIndex('test', X_test.shape[0], 1000)




    # Start grid searching 
    # Bin the training data 
    y_train = (y_train // 6).astype(int)
    y_test = (y_test // 6).astype(int)


    # Logistic Regression
    print(f"\n\n{green_text}Beginning Logistic Regression{reset_text}\n\n") 

    # Get a new index
    n_train = X_train_index.new_subset('Logistic Regression')
    n_test = X_test_index.new_subset('Logistic Regression')

    # Set up model and params
    log_reg = LogisticRegression(multi_class='multinomial', max_iter=200)
    param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],   
            'max_iter': [10, 100, 1000]        
        }
    
    # Evaluate
    log_reg_model = evaluate_model(
        'Logistic Regression',
        log_reg, 
        param_grid, 
        X_train[n_train,:], y_train[n_train],
        X_test[n_test,:], y_test[n_test])

    log_data = []
    for c in param_grid['C']:
        for i in param_grid['max_iter']:
            log_reg = LogisticRegression(C = c, multi_class='multinomial', max_iter=i)
            log_reg.fit(X_train[n_train,:], y_train[n_train])

            y_pred = log_reg.predict(X_test[n_test,:])
            acc = accuracy_score(y_test[n_test], y_pred)
            rec = recall_score(y_test[n_test], y_pred, average='macro')
            pre = precision_score(y_test[n_test], y_pred, average='macro')
            f1 = f1_score(y_test[n_test], y_pred, average='macro')
            log_data.append([c, i, acc, rec, pre, f1])
    
    # Frame it and save it
    df = pd.DataFrame(log_data, columns=['C', 'Max Iter', 'Accuracy', 'Recall', 'Precision', 'F1'])
    df.to_csv(os.path.join(args.path_to_results, 'log_reg.csv'), index=False)



    # Random Forest 
    print(f"\n\n{green_text}Beginning Random Forest{reset_text}\n\n") 

    # Get a new index
    n_train = X_train_index.new_subset('Random Forest')
    n_test = X_test_index.new_subset('Random Forest')

    # Set up model and params
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
            'n_estimators': [50, 100, 200],  
            'max_depth': np.arange(1, 100, 13), 
            'min_samples_split': [2, 5, 10],      
        }
    
    # Evaluate
    rf_model = evaluate_model(
        'Random Forest',
        rf, param_grid,
        X_train[n_train,:], y_train[n_train],
        X_test[n_test], y_test[n_test])    

    rf_data = []
    for e in param_grid['n_estimators']:
        for d in param_grid['max_depth']:
            for s in param_grid['min_samples_split']:
                rf = RandomForestClassifier(
                    n_estimators=e, 
                    max_depth=d, 
                    min_samples_split=s)
                rf.fit(X_train[n_train,:], y_train[n_train])
                y_pred = rf.predict(X_test[n_test,:])
                acc = accuracy_score(y_test[n_test], y_pred)
                rec = recall_score(y_test[n_test], y_pred, average='macro')
                pre = precision_score(y_test[n_test], y_pred, average='macro')
                f1 = f1_score(y_test[n_test], y_pred, average='macro')
                rf_data.append([e, d, s, acc, rec, pre, f1])   

     # Frame it and save it
    df = pd.DataFrame(
        rf_data, 
        columns=[
            'n_estimators', 
            'max_depth',
            'min_samples_split',
            'Accuracy', 'Recall', 'Precision', 'F1'])
    df.to_csv(os.path.join(args.path_to_results, 'random_forest.csv'), index=False)               



    # K Nearest Neighbors
    print(f"\n\n{green_text}Beginning KNN{reset_text}\n\n") 

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

    knn_data = []
    for n in param_grid['n_neighbors']:
        for w in param_grid['weights']:
            knn = KNeighborsClassifier(n_neighbors=n, weights=w)
            knn.fit(X_train[n_train,:], y_train[n_train])

            # Predict
            y_pred = knn.predict(X_test[n_test,:])

            # Score
            acc = accuracy_score(y_test[n_test], y_pred)
            rec = recall_score(y_test[n_test], y_pred, average='macro')
            pre = precision_score(y_test[n_test], y_pred, average='macro')
            f1 = f1_score(y_test[n_test], y_pred, average='macro')
            knn_data.append([n, w, acc, rec, pre, f1])
    
    # Frame it and save it
    df = pd.DataFrame(
        knn_data, 
        columns=[
            'n_neighbors', 
            'weights',
            'Accuracy', 'Recall', 'Precision', 'F1'])
    df.to_csv(os.path.join(args.path_to_results, 'k_neighbors.csv'), index=False)  


    # Sequential
    n_train = X_train_index.new_subset('Sequential')
    n_test = X_test_index.new_subset('Sequential')

    # Set up sequential model
    param_grid = {
            'n_inputs': [16, 64, 256, 1028],
            'alpha': np.linspace(0, 0.9, 3),
            'dropout': np.linspace(0, 0.9, 3)
        }
    
    sequential_data = []
    for n_input in param_grid['n_inputs']:
        green_text = "\033[32m"
        reset_text = "\033[0m"

        print(f"\n\n\n\n{green_text}n_input: {n_input}{reset_text}\n\n\n\n")

        for alpha in param_grid['alpha']:
            for dropout in param_grid['dropout']:

                # Create Neural network model
                clf = tf.keras.Sequential([
                    tf.keras.layers.Dense(n_input, input_shape=(X_train.shape[1],)),
                    tf.keras.layers.LeakyReLU(negative_slope=alpha),
                    tf.keras.layers.Dropout(dropout),
                    tf.keras.layers.Dense(11, activation='softmax')
                ])

                # Compile the model
                clf.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

                # Set up early stop
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=2, 
                    restore_best_weights=True)

                # Call training
                history = clf.fit(X_train[n_train,:], y_train[n_train],
                    epochs=25,            
                    batch_size=32,
                    validation_split=0.2, 
                    callbacks=[early_stop])

                
                # Predict and score
                y_pred = np.argmax(clf.predict(X_test[n_test,:]), axis=1)
                acc = accuracy_score(y_test[n_test], y_pred)
                rec = recall_score(y_test[n_test], y_pred, average='macro')
                pre = precision_score(y_test[n_test], y_pred, average='macro')
                f1 = f1_score(y_test[n_test], y_pred, average='macro')
                sequential_data.append([n_input, alpha, dropout, acc, rec, pre, f1]) 

    df = pd.DataFrame(
        sequential_data, 
        columns=[
            'n_inputs', 
            'alpha',
            'dropout',
            'Accuracy', 'Recall', 'Precision', 'F1'])
    df.to_csv(os.path.join(args.path_to_results, 'sequential.csv'), index=False)  






    # ##### Regression

    print(f"\n\n\n\n{green_text}Beginning Regression Modeling{reset_text}\n\n\n\n")

    # Load in data again
    X_train = np.load(os.path.join(args.path_to_assets, 'X_train.npy'))
    y_train = np.load(os.path.join(args.path_to_assets, 'y_train.npy'))
    X_test = np.load(os.path.join(args.path_to_assets, 'X_test.npy'))
    y_test = np.load(os.path.join(args.path_to_assets, 'y_test.npy'))


    ## Linear Regression
    # Get a new index
    n_train = X_train_index.new_subset('Linear Regression')
    n_test = X_test_index.new_subset('Linear Regression')

    # Create a model
    model = LinearRegression().fit(X_train[n_train,:], y_train[n_train])

    # Predict
    y_pred = model.predict(X_test[n_test,:])

    # Evaluate the model
    mse = mean_squared_error(y_test[n_test], y_pred)
    regression_results_df = pd.DataFrame({
        'model': ['Linear Regression'],
        'mse': [mse]
    })

    # Visualize
    r2_df = pd.DataFrame({
        'Linear Regression Predictions': y_pred, 
        'Linear Regression Actual': y_test[n_test],
        'Linear Regression n': n_test
    })
    rms = np.sqrt((r2_df['Linear Regression Actual'] - r2_df['Linear Regression Predictions'])**2)
    r2_df['Linear Regression RMS'] = rms

    # Get a new index
    n_train = X_train_index.new_subset('Lasso')
    n_test = X_test_index.new_subset('Lasso')

    # Lasso Regression
    model = Lasso(alpha=0.1)
    model.fit(X_train[n_train,:], y_train[n_train])
    y_pred = model.predict(X_test[n_test,:])

    # Update regression results df
    mse = mean_squared_error(y_test[n_test], y_pred)
    regression_results_df = add_mse(regression_results_df, mse, 'Lasso')

    # Update df
    r2_df['Lasso Actual'] = y_test[n_test]
    r2_df['Lasso Predictions'] = y_pred
    r2_df['Lasso n'] = n_test

    # Get a new index
    n_train = X_train_index.new_subset('Ridge')
    n_test = X_test_index.new_subset('Ridge')

    # Ridge Regression
    model = Ridge(alpha=0.9)
    model.fit(X_train[n_train,:], y_train[n_train])
    y_pred = model.predict(X_test[n_test,:])

    # Update regression results df
    mse = mean_squared_error(y_test[n_test], y_pred)
    regression_results_df = add_mse(regression_results_df, mse, 'Ridge')

    # Update df
    r2_df['Ridge Actual'] = y_test[n_test]
    r2_df['Ridge Predictions'] = y_pred
    r2_df['Ridge n'] = n_test

    # Get a new index
    n_train = X_train_index.new_subset('Elastic')
    n_test = X_test_index.new_subset('Elastic')

    # Elastic Net Regression
    model = ElasticNet(alpha=0.5, l1_ratio=0.5)
    model.fit(X_train[n_train,:], y_train[n_train])
    y_pred = model.predict(X_test[n_test,:])

    # Update regression results df
    mse = mean_squared_error(y_test[n_test], y_pred)
    regression_results_df = add_mse(regression_results_df, mse, 'Elastic')

    # Update df
    r2_df['Elastic Actual'] = y_test[n_test]
    r2_df['Elastic Predictions'] = y_pred
    r2_df['Elastic n'] = n_test


    # Cross Validation

    # Get a new index
    n_train = X_train_index.new_subset('KFold')
    n_test = X_test_index.new_subset('KFold')

    # Define the Lasso model
    model = Lasso(alpha=0.1)

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42) 

    # Perform cross-validation and compute MSE for each fold
    mse = -cross_val_score(model, X_train[n_train,:], y_train[n_train], cv=kf, scoring='neg_mean_squared_error')
    regression = add_mse(regression_results_df, mse.mean(), 'KFold')

    
    # Create dataframe with results
    results = [log_reg_model, rf_model, knn_model]
    results_df = pd.DataFrame([vars(model.score) for model in results])

    # Save all the dataframes
    results_df.to_csv(os.path.join(args.path_to_results, 'results.csv'), index=False)
    regression_results_df.to_csv(os.path.join(args.path_to_results, 'regression_results.csv'), index=False)
    r2_df.to_csv(os.path.join(args.path_to_results, 'r2.csv'), index=False)



    
    
    
    # Ablation

    # Load data 
    train_df = pd.read_csv(os.path.join(args.path_to_assets, 'X_train.csv'))
    test_df = pd.read_csv(os.path.join(args.path_to_assets, 'X_test.csv'))
    print(train_df.columns)

    # Get features 
    X_train_df = train_df.drop(columns=['charge_duration'])
    y_train_df = np.array(train_df['charge_duration']) * 60
    X_test_df = test_df.drop(columns=['charge_duration'])
    y_test_df = np.array(test_df['charge_duration']) * 60

    # Get y data
    y_train = np.array(y_train_df)
    y_test = np.array(y_test_df)

    # Declare lists for feature ablation results
    
    ablated_features = []
    n_features = []

    # Build pipeline and initial model
    temp_pipeline = build_pipeline(cat_features, num_features)
    all_features = cat_features + ['start_soc']
    X_train = temp_pipeline.fit_transform(X_train_df[all_features])
    X_test = temp_pipeline.transform(X_test_df[all_features])
    model = Lasso(alpha=0.1).fit(X_train, y_train)
    

    # Get some tracking statistics
    y_pred = model.predict(X_test)
    current_mse = mean_squared_error(y_test, y_pred)
    n_features = [len(all_features)]
    mse_results = [current_mse]

    # Remove one feature at a time and check results
    looking_for_min = True 
    cat_features1 = cat_features[:]  # For features not removed
    while looking_for_min: 
        mse_test_results = []

        for feature in cat_features1:

            # Create temporary features
            temp_cat_features = cat_features1[:]
            _ = temp_cat_features.pop(temp_cat_features.index(feature))

            # All features
            all_features = temp_cat_features + ['start_soc']
            print(all_features, '\n\n')

            # Create pipeline
            temp_pipeline = build_pipeline(temp_cat_features, num_features)
            X_train = temp_pipeline.fit_transform(X_train_df[all_features])
            X_test = temp_pipeline.transform(X_test_df[all_features])

            # Define model
            model = Lasso(alpha=0.1)

            # Set up cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)  

            # Train on full training set
            model.fit(X_train, y_train)

            # Evaluate on separate test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_test_results.append(mse)
        
        if np.min(mse_test_results) < current_mse:
            # Check if any mse with a removed feature improves mse
            worst_n = np.argmin(mse_test_results)  # Index of feature to remove 
            worst_feature = cat_features1[worst_n]
            current_mse = mse_test_results[worst_n]

            # Get the dropped feature
            dropped_feature = cat_features1.pop(worst_n)
            ablated_features.append(dropped_feature)
            n_features.append(len(cat_features))
            mse_results.append(current_mse)
        else:
            ablated_features.append(float('nan'))
            looking_for_min = False



    # Frame it
    ablation_df = pd.DataFrame({
        'Feature': ablated_features, 
        'MSE Test': mse_results
    })

    ablation_df.to_csv(os.path.join(args.path_to_results, 'ablation.csv'), index=False)

    print(f"\n\n\n\n{green_text}Done!{reset_text}\n\n\n\n")






