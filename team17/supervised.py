

if __name__=='__main__':
    import argparse
    import numpy as np
    import os
    import pandas as pd
    import tensorflow as tf

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
    from sklearn.neighbors import KNeighborsClassifier
    
    from team17.super.functions import DataCleaningFunctions, LoadData
    from team17.super.functions import split_train_test_by_id, get_paths
    from team17.super.functions import DataIndex
    from team17.super.pipeline import full_pipeline
    # from team17.super.evaluations import evaluate_model


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
    X_train_index = DataIndex('train', X_train.shape[0], 8000)
    X_test_index = DataIndex('test', X_test.shape[0], 2000)



    # Start grid searching 


    # Logistic Regression
    # Get a new index
    # n_train = X_train_index.new_subset('Logistic Regression')
    # n_test = X_test_index.new_subset('Logistic Regression')

    # # Set up model and params
    # log_reg = LogisticRegression(multi_class='multinomial', max_iter=200)
    # param_grid = {
    #         'C': [0.01, 0.1, 1, 10, 100],
    #         'penalty': ['l1', 'l2'],   
    #         'max_iter': [10, 100, 1000]        
    #     }
    
    # Evaluate
    # log_reg_model = evaluate_model(
    #     'Logistic Regression',
    #     log_reg, 
    #     param_grid, 
    #     X_train[n_train,:], y_train[n_train],
    #     X_test[n_test,:], y_test[n_test])

    # log_data = []
    # for c in param_grid['C']:
    #     for i in param_grid['max_iter']:
    #         log_reg = LogisticRegression(C = c, multi_class='multinomial', max_iter=i)
    #         log_reg.fit(X_train[n_train,:], y_train[n_train])

    #         y_pred = log_reg.predict(X_test[n_test,:])
    #         acc = accuracy_score(y_test[n_test], y_pred)
    #         rec = recall_score(y_test[n_test], y_pred, average='macro')
    #         pre = precision_score(y_test[n_test], y_pred, average='macro')
    #         f1 = f1_score(y_test[n_test], y_pred, average='macro')
    #         log_data.append([c, i, acc, rec, pre, f1])
    
    # # Frame it and save it
    # df = pd.DataFrame(log_data, columns=['C', 'Max Iter', 'Accuracy', 'Recall', 'Precision', 'F1'])
    # df.to_csv(os.path.join(args.path_to_results, 'log_reg.csv'), index=False)



    # Random Forest 

    # Get a new index
    n_train = X_train_index.new_subset('Random Forest')
    n_test = X_test_index.new_subset('Random Forest')

    # Set up model and params
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
            'n_estimators': [50, 100, 200],  
            'max_depth': np.arange(1, 100), 
            'min_samples_split': [2, 5, 10],      
        }
    
    # Evaluate
    # rf_model = evaluate_model(
    #     'Random Forest',
    #     rf, param_grid,
    #     X_train[n_train,:], y_train[n_train],
    #     X_test[n_test], y_test[n_test])    

    # rf_data = []
    # for e in param_grid['n_estimators']:
    #     for d in param_grid['max_depth']:
    #         for s in param_grid['min_samples_split']:
    #             rf = RandomForestClassifier(
    #                 n_estimators=e, 
    #                 max_depth=d, 
    #                 min_samples_split=s)
    #             rf.fit(X_train[n_train,:], y_train[n_train])
    #             y_pred = rf.predict(X_test[n_test,:])
    #             acc = accuracy_score(y_test[n_test], y_pred)
    #             rec = recall_score(y_test[n_test], y_pred, average='macro')
    #             pre = precision_score(y_test[n_test], y_pred, average='macro')
    #             f1 = f1_score(y_test[n_test], y_pred, average='macro')
    #             rf_data.append([e, d, s, acc, rec, pre, f1])   

    #  # Frame it and save it
    # df = pd.DataFrame(
    #     rf_data, 
    #     columns=[
    #         'n_estimators', 
    #         'max_depth',
    #         'min_samples_split',
    #         'Accuracy', 'Recall', 'Precision', 'F1'])
    # df.to_csv(os.path.join(args.path_to_results, 'random_forest.csv'), index=False)               



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
    # knn_model = evaluate_model(
    #     'KNN',
    #     knn, param_grid,
    #     X_train[n_train,:], y_train[n_train],
    #     X_test[n_test], y_test[n_test]) 

    # knn_data = []
    # for n in param_grid['n_neighbors']:
    #     for w in param_grid['weights']:
    #         knn = KNeighborsClassifier(n_neighbors=n, weights=w)
    #         knn.fit(X_train[n_train,:], y_train[n_train])
    #         y_pred = knn.predict(X_test[n_test,:])
    #         acc = accuracy_score(y_test[n_test], y_pred)
    #         rec = recall_score(y_test[n_test], y_pred, average='macro')
    #         pre = precision_score(y_test[n_test], y_pred, average='macro')
    #         f1 = f1_score(y_test[n_test], y_pred, average='macro')
    #         knn_data.append([n, w, acc, rec, pre, f1])
    
    # # Frame it and save it
    # df = pd.DataFrame(
    #     knn_data, 
    #     columns=[
    #         'n_neighbors', 
    #         'weights',
    #         'Accuracy', 'Recall', 'Precision', 'F1'])
    # df.to_csv(os.path.join(args.path_to_results, 'k_neighbors.csv'), index=False) 


    # Sequential
    n_train = X_train_index.new_subset('Sequential')
    n_test = X_test_index.new_subset('Sequential')

    # Set up sequential model
    param_grid = {
            'n_inputs': [16, 64, 256, 1028],
            'alpha': np.linspace(0, 0.9, 10),
            'dropout': np.linspace(0, 0.9, 10)
        }
    
    sequential_data = []
    for n_input in param_grid['n_inputs']:
        green_text = "\033[32m"
        reset_text = "\033[0m"

        print(f"\n\n\n\n{green_text}n_input: {n_input}{reset_text}\n\n\n\n")

        for alpha in param_grid['alpha']:
            for dropout in param_grid['dropout']:


                clf = tf.keras.Sequential([
                    tf.keras.layers.Dense(n_input, input_shape=(X_train.shape[1],)),
                    tf.keras.layers.LeakyReLU(negative_slope=alpha),
                    tf.keras.layers.Dropout(dropout),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])

                clf.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                history = clf.fit(X_train, y_train,
                    epochs=50,            # Maximum epochs
                    batch_size=32,
                    validation_split=0.2,   # 20% of data used for validation
                    callbacks=[early_stop])

                y_pred = clf.predict(X_test[n_test,:], y_test[n_test])
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
    
    # Create dataframe with results
    # results = [log_reg_model, rf_model, knn_model]
    # results_df = pd.DataFrame([vars(model.score) for model in results])
    # results_df.to_csv(os.path.join(args.path_to_results, 'results.csv'), index=False)

    









