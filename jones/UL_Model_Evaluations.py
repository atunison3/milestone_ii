#Import relevent packages
import time 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

#model options: 
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM #very slow
from sklearn.neighbors import LocalOutlierFactor

#model eval packages
from sklearn.model_selection import KFold

#Import pyscripts 
import Prep_data_UL


def evaluate_pca(input_df, variance_retention=0.95, view_plot=True): 
    """
    This function runs PCA analysis on the input dataframe of predictor variables to determine the min
    number of dimensions that should be retained by the dataset for our machine-learning tasks

    INPUTS: 
    input_df: a pandas dataframe of the predictor variables 
    variance_retention: a float value between 0 and 1 of the minimum variance captured by PCA
    view_plot: a boolean variable, that when true, plots the variance vs. the principle components quantity 

    OUTPTUS: 
    optimal_n_components: an integer variable declaring how many principle components should be used
    pca_X: the input_df transformed via PCA into the optimal dimension size 
    """

    #setup PCA 
    pca = PCA() 
    pca.fit(input_df) 

    #find the cumulative explained variance: 
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    #get the optimal number of principle components: 
    optimal_n_components = (cumulative_variance >= 0.95).argmax() + 1

    #transform the data via PCA
    pca = PCA(n_components=optimal_n_components)
    pca_X = pca.fit_transform(input_df)

    #plot the variance vs. PCA num. 
    if view_plot == True: 
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs. Number of PCA Components')
        plt.show()

    #Output desired vars 
    return(optimal_n_components, pca_X)


def apply_PCA(X, num_components): 
    """This function applies PCA to the training and test datasets for the n number of components
    INPUTS: 
    X: the pandas dataset of predictor variables
    n_components: the number of components to reduce to 

    OUTPUTS: 
    PCA_X: the X dataset transformed by PCA
    """

    #setup PCA     
    pca = PCA(n_components=num_components)
    pca_X = pca.fit_transform(X)
    return pca_X


def score_model(pred_normal, pred_anomaly, X_train, X_anomalies): 
    """This function scores the classification model
    INPUTS: 
    input_normal: the normal set of datapoints for training
    input_anomaly: the anomalous set of datapoints for training
    X_train: non-anomalous dataset
    X_anomalies: anomaly dataset
    OUTPUTS: precision, recall, accuracy, F1_score
    """

    #Identify True and False Identifications for Scoring 
    TP = list(pred_normal).count(1)
    FP = len(X_train) - TP 
    TN = list(pred_anomaly).count(1)
    FN = len(X_anomalies) - TN
    
    #Get precision, recall, accuracy, and F1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, F1)


def evaluate_ISF(X_train, X_anomalies, n_samples=100000, n_estimate=100): 
    """
    This function builds an isolation forest model, based on the specified 
    training data, anomaly data, and number of samples. 

    INPUTS: 
    X_train: the training dataset using only non-anomalous data 
    X_anomalies: the anomaly dataset 
    n_samples: the number of samples to consider 
    n_estimate: the number of estimators used in the model (default to 100)

    OUTPUTS: 
    precision: the precision of the isolation forest model 
    recall: the recall of the isolation forest model 
    accuracy: the accuracy of the isolation forest model 
    """
    #Find contam qty: 
    contam = len(X_anomalies) / (len(X_train) + len(X_anomalies))
    
    #Setup the model: 
    ISF = IsolationForest(n_estimators=n_estimate,
                          max_samples=n_samples,
                          contamination= contam,
                          n_jobs=-1,
                          random_state=42
                          )

    #Fit to the non-anomalous data
    ISF.fit(X_train)

    #Score performance               
    pred_normal = ISF.predict(X_train)
    pred_anomaly = ISF.predict(X_anomalies)

    precision, recall, F1 = score_model(pred_normal, pred_anomaly, X_train, X_anomalies)

    
    return precision, recall, F1 


def evaluate_OCSVM(X_train,X_test): #too slow o^2
    start = time.time()
    ocsvm = OneClassSVM()
    ocsvm = ocsvm.fit(pca_X_train)
    pred_normal = ocsvm.predict(pca_X_train)
    pred_anomaly = ocsvm.predict(pca_X_anomalies)

    #get performance: 
    precision, recall, F1 = score_model(pred_normal, pred_anomaly, X_train, X_anomalies)

    end = time.time() 
    duration = np.round((end-start)/60,1)
    print("Total completion time, 1 iter: {duration}min")
    return precision, recall, F1 


def evaluate_LOF(X_train, X_anomalies):
    start = time.time()
    
    # Initialize and fit Local Outlier Factor model
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_train)
    
    # Predict on the training set and anomalies
    pred_normal = lof.predict(X_train)
    pred_anomaly = lof.predict(X_anomalies)

    # Get performance
    precision, recall, F1 = score_model(pred_normal, pred_anomaly, X_train, X_anomalies)

    end = time.time()
    duration = np.round((end - start) / 60, 1)
    print(f"Total completion time, 1 iter: {duration} min")
    
    return precision, recall, F1 


def evaluate_model(input_model, input_X_train, input_X_anomalies):
    
    """This function trains an input model, and then scores its performance for detecting anomalies
    INPUTS: 
    input_model: the sklearn model to train and score
    input_X_train: the training dataset (no anomalies) 
    input_X_anomalies: the dataset for anomaly detection  

    OUTPUS: 
    precision, recall, accuracy, F1_score: float values representing the model performance
    """

    #Fit the model
    input_model.fit(input_X_train)
    
    #Score performance               
    normal_model = input_model.predict(input_X_train)
    anomaly_model = input_model.predict(input_X_anomalies)

    #Identify True and False Identifications for Scoring 
    TP = list(normal_model).count(1)
    FP = len(input_X_train) - TP 
    TN = list(anomaly_model).count(1)
    FN = len(input_X_anomalies) - TN
    
    #Get precision, recall, accuracy, and F1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, F1 


def review_models(input_X_train, input_X_anomalies): 
    """
    Function to assess via K-fold each of the diff. model types: 
    """

    
    #Let's setup the pipeline for evaluation 
    
    #list of models & their names  
    models = [IsolationForest(),OneClassSVM(),LocalOutlierFactor(novelty=True)] 
    names = ["Isolation Forest", "OC-SVM", "LOF"] 
    
    #Setup empty list for model results 
    model_list = []
    precision_list = []
    recall_list = [] 
    F1_list = [] 
    
    #setup cross val: 
    kf = KFold(5) 
    
    for name, model in zip(names, models): 
        print(f'Beginning K-fold training and evaluation for {name}')    
        start = time.time()
        temp_precision_list = []
        temp_recall_list = [] 
        temp_F1_list = [] 
        for i, (train_index, test_index) in enumerate(kf.split(input_X_train)):
        #for train_index, test_index in cv.split(X_train):
            X_train_cv, X_test_cv = input_X_train.iloc[list(train_index)], input_X_train.iloc[list(train_index)]
            precision, recall, F1 = evaluate_model(model, X_train_cv, input_X_anomalies)
            temp_precision_list.append(precision)
            temp_recall_list.append(recall)
            temp_F1_list.append(F1)
    
        precision_list.append(np.mean(temp_precision_list))
        recall_list.append(np.mean(temp_recall_list))
        F1_list.append(np.mean(temp_F1_list))
        model_list.append(name)
    
        stop = time.time()
        duration = np.round((stop - start) / 60,1) 
        print(f'Finished K-fold training and evaluation for {name} in {duration}min')     
    
    evaluation_df = pd.DataFrame({"Model": model_list, "Preciscion": precision_list, 
                                  "Recall": recall_list, "F1": F1_list}) 

    evaluation_df = evaluation_df.sort_values(by="F1", ascending=False).reset_index(drop=True)
    
    return evaluation_df


#Execution: 
if __name__ == "__main__": 
    X_train, X_test, X_anomalies, y_train, y_test = Prep_data_UL.get_UL_data()
    X_train = X_train.reset_index(drop=True)

    #Only down-sampling to enable efficient model review while waiting for HPC access...
    X_train_sample = X_train.sample(5000) 
    X_anomaly_sample = X_anomalies.sample(5000) 

    #Apply PCA to datasets 
    pca_X_train = apply_PCA(X_train_sample, 17)
    pca_X_anomalies = apply_PCA(X_anomaly_sample, 17)

    #Evaluate Models: 
    review_models(X_train_sample, X_anomaly_sample)