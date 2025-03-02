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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    
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

    print(args.path_to_assets)

    print(f"\n\n{green_text}Beginning Loading of Data{reset_text}\n\n") 
    try:
        X_train = np.load(os.path.join(args.path_to_assets, 'X_train.npy'))
        y_train = np.load(os.path.join(args.path_to_assets, 'y_train.npy'))
        X_test = np.load(os.path.join(args.path_to_assets, 'X_test.npy'))
        y_test = np.load(os.path.join(args.path_to_assets, 'y_test.npy'))

    except FileNotFoundError:

        # Read in data
        sessions_path, evse_path, connector_path = get_paths(args.path_to_assets)
        sessions = LoadData.load_sessions(sessions_path)
        evse = LoadData.load_evse(evse_path)
        connector = LoadData.load_connector(connector_path)

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
                    verbose=0, 
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






    




    print(f"\n\n\n\n{green_text}Done!{reset_text}\n\n\n\n")






