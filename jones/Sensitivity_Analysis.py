#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Import general packages
import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

#Import ML Packages 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import LocalOutlierFactor

#Import py scripts: 
import Prep_data_UL
import UL_Model_Evaluations



def lof_sensitivity_analysis(input_training_data, n_list = [5, 10, 20, 40], 
                             algorithm = ['ball_tree', 'kd_tree', 'brute'], 
                             leaf_size = [10, 30, 50],
                             p = [1,2]):
    
    """
    Custom function to perform sensitivty analysis by grid-search on input training data given a list 
    for each parameter desired to evaluate: 
    
    INPUTS: 
    input_training data: a dataframe of the training data 
    n_list: the list of neighbor values (integers) to evaluate
    algorithm: the list of algorithms to use in the LoF (ball_tree, kd_tree, and/or brute) 
    leaf_size: the list of leaf numbers in the model (integer) 
    p: the distance measurement method (1 for manhattan, 2 for euclidean)
    
    OUTPUTS: 
    results: a pandas dataframe with the score for the different grid-search combinations
    """

    class LOF_grid_search(LocalOutlierFactor):
        """
        This class is used for a custom LoF sensitivity analysis, that allows use to
        score the LoF model since LoF does not have an intrinsic scoring method.
        INPUTS: LocalOutlierFactor, a LoF model
        """
        def __init__(self, n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2):
            super().__init__(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p)
    
        def fit(self, X, y=None):
            self.model = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                metric=self.metric,
                p=self.p
            )
            self.model.fit(X)
            return self
    
        def score(self, X, y=None):
            y_pred = self.model.fit_predict(X)
            return calinski_harabasz_score(X, y_pred)
    
    
    param_grid = {
        'n_neighbors': n_list,
        'algorithm': algorithm,
        'leaf_size': leaf_size,
        'p': p  # Fixed parameter values to avoid 'p'
    }
    
    LOF_grid_search = LOF_grid_search()
    
    grid_search = GridSearchCV(estimator=LOF_grid_search, param_grid=param_grid, cv=5)
    grid_search.fit(input_training_data)
    
    results = pd.DataFrame(grid_search.cv_results_)
    
    #clock performance & return results 
    end = time.time() 
    print(f"Sensitivity Analysis Finished in {np.round((end - start)/60,1)} minutes")
    return results
    

def hyper_parameter_plotting(input_df, model_type, col_x, col_y, col_z, x_name=None, y_name=None, z_name =None, fig_rotation=140): 
    """This function creates a 3D plot showing the sensitivty analysis for a model score against two hyper parameters
    INPUTS: 
    input_df: a pandas dataframe for ploting score vs. hyper params
    model_type: the type of model being evaluated
    col_x: hyper-parameter 1 to display
    col_y: hyper-parameter 2 to display
    col_z: the scoring column
    x_name: the string to name/label X dimension
    y_name: the string to name/label Y dimension
    z_name: the string to name/label Z dimension
    fig_rotation: the azimuth rotation angle for the plot

    OUTPUTS: none. A plt 3d plot is displayed 
    """
    
    
    X = input_df[col_x]
    Y = input_df[col_y]
    Z = input_df[col_z]

    if x_name == None: 
        x_name = col_x
    if y_name == None: 
        y_name = col_y
    if z_name == None: 
        z_name = col_z
    
    #Grid for plot contours: 
    xi = np.linspace(X.min(), X.max(), 100)
    yi = np.linspace(Y.min(), Y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((X, Y), Z, (xi, yi), method='linear')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    norm = plt.Normalize(Z.min(), Z.max())
    colors = plt.cm.RdYlGn(norm(Z))  # Using the 'RdYlGn' colormap
    
    #surface map: 
    surface = ax.plot_surface(xi, yi, zi, cmap='RdYlGn', alpha=0.8)
    
    # Create scatter plot
    scatter = ax.scatter(X, Y, Z, c=colors, marker='o')
    
    # Setting labels
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    ax.set_title(f"Hyperparameter Sensitivity for {model_type}-based Model")
    
    ax.view_init(elev=30, azim=fig_rotation)  # Adjust these values to change the angle
    
    plot_subtext = (f"{x_name} ranges from {X.min()} - {X.max()}, while {y_name} varies from {Y.min()} - {Y.max()}.\
    Ranking is based on the calinski-harabaz score from a 5-fold, cross-validation of a hyper-\nparameter sweep of the {model_type}-\
    based model. Other hyper-parameters, while explored, are not shown.") 
    
    plt.figtext(
        0.15, -0.07, plot_subtext, wrap=True, horizontalalignment='left', fontsize=9, fontstyle='italic',
    )
    
    plt.show()


#Main process: if successful will push to Git.
if __name__ == "__main__":
    down_sample = True
    
    # X_train, X_test, X_anomalies, y_train, y_test = Prep_data_UL.get_UL_data()
    # X_train = X_train.reset_index(drop=True)
    
    if down_sample == True: #Only down-sampling to enable efficient model review while waiting for HPC access...
        X_train_sample = X_train.sample(1000) 
        X_anomaly_sample = X_anomalies.sample(1000) 
    else: 
        X_train_sample = X_train
        X_anomaly_sample = X_anomalies            

    #Apply PCA to datasets 
    pca_X_train = UL_Model_Evaluations.apply_PCA(X_train_sample, 17)
    pca_X_anomalies = UL_Model_Evaluations.apply_PCA(X_anomaly_sample, 17)

    neighbor_list = [5, 10, 15, 20, 30, 40]
    algorithm_list = ['ball_tree']
    leaf_list = [5, 10, 15, 20, 30, 40]
    p_list = [1]

    #start clock
    start = time.time() 

    #run evaluation
    df_performance = lof_sensitivity_analysis(pca_X_train, n_list = neighbor_list, algorithm = algorithm_list, 
                                              leaf_size = leaf_list, p = p_list)

    #plot results: 
    hyper_parameter_plotting(df_performance, model_type="LoF", col_x="param_leaf_size", col_y="param_n_neighbors", 
                             col_z="rank_test_score", x_name = "Leaf Size", y_name = "Number of Neighbors", 
                             z_name = "Score Rank", fig_rotation=140)


# In[19]:


X_train


# In[ ]:




