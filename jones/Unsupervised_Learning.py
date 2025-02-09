#Import relavent packages
import matplotlib.pyplot as plt
import numpy as np 

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score

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
    normal_isf = ISF.predict(X_train)
    outlier_isf = ISF.predict(X_anomalies)

    #Identify True and False Identifications for Scoring 
    TP = list(normal_isf).count(1)
    FP = len(X_train) - TP 
    TN = list(outlier_isf).count(1)
    FN = len(X_anomalies) - TN
    
    #Get precision, recall, and total accuracy
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / len(X_test) 
    
    return precision, recall, accuracy


if __name__ == "__main__":
    X_train, X_test, X_anomalies, y_train, y_test = Prep_data_UL.get_UL_data()
    optimal_n_components, pca_X = evaluate_pca(X_train, variance_retention=0.95, view_plot=True)
    precision, recall, accuracy = evaluate_ISF(X_train, X_anomalies, n_samples=10000)