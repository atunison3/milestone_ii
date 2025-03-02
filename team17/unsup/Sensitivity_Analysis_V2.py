#!/usr/bin/env python
# coding: utf-8

# ### Sensitivity Analysis
# 
# In this notebook, we'll evaluate how sensitive our anomaly detection model's 
# performance is for test evaluation, when different sample sizes are passed. 
# To do this, we'll call the get_data_pipe, review_models_pipe, and hyper_parameter_pipe 
# in succession, and do this iteratively to understand the impact that data quantity has on 
# our anomaly detection results. 
# 
# Let's begin

# In[9]:


# Import python libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import time 

#Import python scripts
from .Unsupervised_Learning_V3 import load_downsample, evaluate_pca, apply_PCA
from .Unsupervised_Learning_Hyper_param import lof_sensitivity_analysis, IsoF_sensitivity_analysis
from .Unsupervised_Learning_Hyper_param import hyper_parameter_plotting, evaluate_LOF, evaluate_ISF

#Import pyscripts 


def get_data_pipe2(
    path_to_results: str, dropword: str = None, num_sample: int = 5000
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    """This function serves as the data retrieval pipeline for unsupervised learning. It retrieves the scaled, 
    transformed datasets, then identifies optimal principal components for 95% variance retention, and finally 
    transforms the data to be represented by the dimensional feature space of the principal components. 
    
    INPUTS:
    num_sample, an interger which specifies how many records to pull from the original dataset for modeling purposes
    
    OUTPUTS: 
    PCA_X_train, PCA_X_test, y_train, y_test, PCA_X, y. Each are pandas dataframes. X's referring 
    to predictor features, y's to the target feature dataframe.
    """

    X_train, X_test, y_train, y_test = load_downsample(path_to_results, num_sample)
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
    
    X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

    # Remove unawanted features from each (X_train, X_test, X) 
    all_features = list(X_train.columns)
    temp_words = [feature for feature in all_features if dropword not in feature]

    # Remove unwanted features 
    X_train = X_train[temp_words]
    X_test = X_test[temp_words] 
    X = X[temp_words] 
    
    # Find the optimal number of components for dimension reduction
    optimal_n_components, pca_X = evaluate_pca(
        X_train, variance_retention=0.95, view_plot=True)

    # Apply PCA to the 
    pca_X_train, pca_X_test, pca_X = apply_PCA(X_train, X_test, X, optimal_n_components)
    
    return(pca_X_train, pca_X_test, y_train, y_test, pca_X, y)


def hyper_parameter_pipe2(
    path_to_results: str, dropword = None, num_sample: int = 1000
    ) -> list[
        float, float, float, float, float, float, 
        pd.DataFrame, pd.DataFrame, 
        
    ]:
    '''Perform hyper parameter searching on multiple models'''
    
    #Get training data for hyper-param analysis
    pca_X_train, pca_X_test, y_train, y_test, pca_X, y = get_data_pipe2(
        path_to_results, dropword = dropword, num_sample=num_sample)

    #Set LoF hyper params to grid-search evaluate
    neighbor_list = [5, 10, 15, 20, 30, 40]
    algorithm_list = ['ball_tree']
    leaf_list = [5, 10, 15, 20, 30, 40]
    p_list = [1]

    #Set IsoF hyper params to grid-search evaluate
    n_estim_list = [5, 10, 20, 40]
    contam_ratio_list = [0.02, 0.05, 0.1, 0.15]
    
    #start clock (to help with HPC timing)
    start = time.time() 

    #run evaluation
    df_lof = lof_sensitivity_analysis(
        pca_X_train, n_list = neighbor_list, 
        algorithm = algorithm_list, leaf_size = leaf_list, 
        p = p_list)

    df_isoF = IsoF_sensitivity_analysis(
        pca_X_train, n_estim = n_estim_list, contam_ratio = contam_ratio_list)

    #Sort by scores: 
    df_lof = df_lof.sort_values(by='rank_test_score', ascending=True).reset_index(drop=True)
    df_isoF = df_isoF.sort_values(by='rank_test_score', ascending=True).reset_index(drop=True)
        
    #plot results: 
    LoF_plot = hyper_parameter_plotting(
        df_lof, model_type="LoF", col_x="param_leaf_size", col_y="param_n_neighbors", 
        col_z="rank_test_score", x_name = "Leaf Size", y_name = "Number of Neighbors", 
        z_name = "Score Rank", fig_rotation=140)

    IsoF_plot = hyper_parameter_plotting(
        df_isoF, model_type="IsoF", col_x="param_n_estimators", col_y="param_contamination", 
        col_z="rank_test_score", x_name = "Number of Estimators", y_name = "Contamination Ratio", 
        z_name = "Score Rank", fig_rotation=140)

    #Now score the model for anomaly identification: 
    algo = df_lof["param_algorithm"].iloc[0]
    n_leaf = df_lof["param_leaf_size"].iloc[0]
    n_neighbor = df_lof["param_n_neighbors"].iloc[0]
    p_var = df_lof["param_p"].iloc[0]
    
    lof_precision, lof_recall, lof_F1 = evaluate_LOF(
        pca_X_train, pca_X_test, np.ravel(y_train), 
        np.ravel(y_test), n_neighbor= n_neighbor, 
        n_leaf = n_leaf, algorithm = algo, p=p_var)
    
    
    val_contam = df_isoF["param_contamination"].iloc[0]
    num_estimate = df_isoF["param_n_estimators"].iloc[0]
    
    IsoF_precision, IsoFrecall, IsoF1 = evaluate_ISF(
        pca_X_train, pca_X_test, np.ravel(y_train), np.ravel(y_test), 
        n_estimate= num_estimate, contam_ratio = val_contam)
    
    outputs = [
        IsoF_precision, IsoFrecall, IsoF1, 
        lof_precision, lof_recall, lof_F1, 
        df_lof, df_isoF, 
        LoF_plot, IsoF_plot, 
        pca_X_train, pca_X_test, y_train, y_test, pca_X, y]
    
    end = time.time() 
    print("hyper-parameters finished in ", (end-start)/60, "min") 
    
    return outputs


if __name__ == "__main__": 
    #Examine the impact of removing specific features from the analysis: 
    sample_size = 2000
    potential_strings = [
        'power', 'connector', 'pricing', 'region', 'land_use', 
        'metro_area', 'charge_level', 'venue', 'total_duration',
        'charge_duration', 'energy_kwh', 'start_soc', 'end_soc', 'soc_charged', 
        'num_ports', 'connector_number', 'year', 'month', 'day']
    
    #Set lists for caching results
    keyword_list = []
    IsoFrecall_list = []
    lof_recall_list = []
    IsoF_precision_list = []
    lof_precision_list = []
    IsoF1_list = []
    lof_F1_list = []
    sample_size_list = []
    counter = 0 
    
    #Iteratively exclude features from model to see impact on score
    for dropword in potential_strings: 
        t1 = time.time()

        #Get results 
        results = hyper_parameter_pipe2(dropword, num_sample=sample_size)
        IsoF_precision, IsoFrecall, IsoF1, lof_precision, lof_recall, lof_F1, df_lof, df_isoF, LoF_plot, IsoF_plot,pca_X_train, pca_X_test, y_train, y_test, pca_X, y = results  # noqa: E501
        
        #Store the results 
        keyword_list .append(dropword)
        IsoFrecall_list.append(IsoFrecall)
        lof_recall_list.append(lof_recall)
        IsoF_precision_list.append(IsoF_precision)
        lof_precision_list.append(lof_precision)
        IsoF1_list.append(IsoF1)
        lof_F1_list.append(lof_F1)
        sample_size_list.append(sample_size)
    
        counter += 1
        t2 = time.time()
        #Simple print to track progress
        print("==========================")
        print(f"Feature(s) {counter} out of {len(potential_strings)} Reviewed in {np.round((t2-t1)/60,1)}min")
        print("==========================")
    
    
    output_dataframe = pd.DataFrame({"Excluded Feature": keyword_list, 
                                     "Resulting IsoF Recall": IsoFrecall_list, 
                                     "Resulting lof Recall": lof_recall_list, 
                                     "Resulting IsoF precision": IsoF_precision_list, 
                                     "Resulting lof precision": lof_precision_list, 
                                     "Resulting IsoF F1": IsoF1_list, 
                                     "Resulting lof F1": lof_F1_list, 
                                     "sample size": sample_size_list})
    
    output_dataframe.to_csv("Feature_influence_on_Anomaly_Identification.csv", index=False)
    
    X = output_dataframe["Excluded Feature"]
    Y = output_dataframe["Resulting lof F1"]
    
    cmap = plt.get_cmap('Blues_r')
    norm = plt.Normalize(min(Y), max(Y))
    colors = cmap(norm(Y))
    
    plt.bar(X,Y, color=colors, edgecolor='black')
    plt.xticks(rotation=45, ha='right') 
    plt.title("Anomaly Predict Model Sensitivity to Feature Exclusion")
    plt.xlabel("Feature Excluded from Model")
    plt.ylabel("F1 Score of optimal Local Outlier Factor model") 
    plt.show()

