### Hyperparameter Tuning


# Import general packages
import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Model options: 
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Model eval packages
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import calinski_harabasz_score

# Import pyscripts 
from .Unsupervised_Learning_V3 import get_data_pipe

class Timer:
    def __init__(self):
        self.start_time = time.time() 

    def stop(self):
        self.end_time = time.time()
        self.duration = np.round((self.end_time - self.start_time) / 60, 1)
        print(f"LOF Evaluation Finished in: {self.duration} min")


def evaluate_ISF(
    X_train: pd.DataFrame, X_test: pd.DataFrame, 
    y_train: pd.DataFrame, y_test: pd.DataFrame, 
    n_estimate: int = 50, contam_ratio: float = 0.2
    ) -> tuple[float, float, float]: 
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
    
    start = time.time()
    
    # Setup the model: 
    isf = IsolationForest(n_estimators = n_estimate, contamination = contam_ratio)

    # Fit to the non-anomalous data
    isf.fit(X_train)

    # Score performance               
    # pred_train = isf.predict(X_train)
    pred_test = isf.predict(X_test)

    precision, recall, F1 = get_scores(pred_test, y_test)

    end = time.time()
    duration = np.round((end - start) / 60, 1)
    print(f"IsoF Evaluation Finished in: {duration} min")
    
    return precision, recall, F1 


def evaluate_LOF(
    X_train: pd.DataFrame, X_test: pd.DataFrame, 
    y_train: pd.DataFrame, y_test: pd.DataFrame, 
    n_neighbor: int = 5, n_leaf: int = 20,
    algorithm: str = "ball tree", p: int = 1
    ) -> tuple[float, float, float]:
    '''Perform local outlier factor evaluation'''

    start = time.time()
    
    # Initialize and fit Local Outlier Factor model
    lof = LocalOutlierFactor(
        novelty=True,
        n_neighbors=n_neighbor,
        leaf_size=n_leaf,
        algorithm=algorithm,
        p=p)
    lof.fit(X_train)
    
    # Predict on the training set and anomalies
    #pred_train = lof.predict(X_train)
    pred_test = lof.predict(X_test)

    # Get performance
    precision, recall, F1 = get_scores(pred_test, y_test)
    
    # Print 
    end = time.time()
    duration = np.round((end - start) / 60, 1)
    print(f"LOF Evaluation Finished in: {duration} min")
    
    return precision, recall, F1 


def get_scores(
    input_prediction: np.array, input_data: np.array
    ) -> tuple[float, float, float]: 
    '''Calculate scores based on predictions and truths'''

    # Calculate true/false positives/negatives
    TP = np.sum((input_prediction == 1) & (input_data == 1))
    FP = np.sum((input_prediction == 1) & (input_data == 0))
    FN = np.sum((input_prediction == 0) & (input_data == 1))   

    # Calculate scores
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)  
    
    return precision, recall, F1
    

def lof_sensitivity_analysis(
    input_training_data: pd.DataFrame, 
    n_list: list[int] = [5, 10, 20, 40], 
    algorithm: list[str] = ['ball_tree', 'kd_tree', 'brute'], 
    leaf_size: list[int] = [10, 30, 50],
    p: list[int] = [1,2]
    ) -> pd.DataFrame:
    
    """
    grid-search hyper-param tuning & sensitivty analysis for LoF model 
    
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
        """This class is used for hyper-parameter tuning of the LoF model for cluster analysis
        """
        def __init__(
            self, n_neighbors: int = 20, 
            algorithm: str = 'auto', leaf_size: int = 30, 
            metric: str = 'minkowski', p: int = 2):
            super().__init__(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p)
    
        def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
            self.model = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                metric=self.metric,
                p=self.p
            )

            # Fit the model
            self.model.fit(X)

            return self
    
        def score(self, X: pd.DataFrame, y: pd.DataFrame = None) -> float:
            '''Calculates the Calinski Harabasz score of the model'''

            # Predict 
            y_pred = self.model.fit_predict(X)

            return calinski_harabasz_score(X, y_pred)
    
    # Set up param grid
    param_grid = {
        'n_neighbors': n_list,
        'algorithm': algorithm,
        'leaf_size': leaf_size,
        'p': p  # Fixed parameter values to avoid 'p'
    }
    
    # Start a timer
    start = time.time()

    # Perform grid search
    LOF_grid_search = LOF_grid_search()
    grid_search = GridSearchCV(estimator=LOF_grid_search, param_grid=param_grid, cv=5)
    grid_search.fit(input_training_data)
    
    # Save results to a dataframe
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Clock performance & return results 
    end = time.time() 
    print(f"Sensitivity Analysis Finished in {np.round((end - start)/60, 1)} minutes")

    return results


def IsoF_sensitivity_analysis(
    input_training_data: pd.DataFrame, 
    n_estim: list[int] = [5, 10, 20, 40], 
    contam_ratio: list[int] = [0.02, 0.05, 0.1, 0.15]
    ) -> pd.DataFrame:
    
    """
    grid-search hyper-param tuning & sensitivty analysis for LoF model 
    
    INPUTS: 
    input_training data: a dataframe of the training data 
    n_estim: the list of number of estimators to use
    contam_ratio: the ratios of contamination in the training data 
    
    OUTPUTS: 
    results: a pandas dataframe with the score for the different grid-search combinations
    """


    class IsoF_grid_search(IsolationForest):
        """This class is used for hyper-parameter tuning of the IsoF model for cluster analysis
        """
        def __init__(
            self, n_estimators: int = 100, max_samples: str = 'auto', 
            contamination: str = 'auto', max_features: float = 1.0, bootstrap: bool = False):
            super().__init__(n_estimators=n_estimators,contamination=contamination)
        
        def fit(self, X: pd.DataFrame, y: list = None):
            '''Creates a fits a model'''

            # Initialize model
            self.model = IsolationForest(
                n_estimators = self.n_estimators,
                contamination=self.contamination,
            )

            # Fit the model
            self.model.fit(X)

            return self
        
        def score(self, X: pd.DataFrame, y: pd.DataFrame = None) -> float:
            '''Returns the Calinski Harabasz score'''

            # Predict
            y_pred = self.model.fit_predict(X)

            return calinski_harabasz_score(X, y_pred)

    # Set up param gird
    param_grid = {
        'n_estimators': n_estim,
        'contamination': contam_ratio,
    }
    
    # Start a timer
    start = time.time()

    # Perform grid search
    IsoF_grid_search = IsoF_grid_search()
    grid_search = GridSearchCV(estimator=IsoF_grid_search, param_grid=param_grid, cv=5)
    grid_search.fit(input_training_data)
    
    # Save results in dataframe
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Clock performance & return results 
    end = time.time() 
    print(f"Sensitivity Analysis Finished in {np.round((end - start)/60,1)} minutes")

    return results        
    

def hyper_parameter_plotting(
    input_df: pd.DataFrame, model_type: str, 
    col_x: list[str], col_y: list[str], col_z: list[str], 
    x_name: str = None, y_name: str = None, z_name: str = None, 
    fig_rotation: float = 140) -> None: 
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
    
    # Extract x, y, z data
    x = input_df[col_x]
    y = input_df[col_y]
    z = input_df[col_z]

    if not x_name: 
        x_name = col_x
    if not y_name: 
        y_name = col_y
    if not z_name: 
        z_name = col_z
    
    # Grid for plot contours: 
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize 
    norm = plt.Normalize(z.min(), z.max())
    colors = plt.cm.RdYlGn(norm(z))  # Using the 'RdYlGn' colormap
    
    # Surface map: 
    _ = ax.plot_surface(xi, yi, zi, cmap='RdYlGn', alpha=0.8)
    
    # Create scatter plot
    _ = ax.scatter(x, y, z, c=colors, marker='o')
    
    # Setting labels
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    ax.set_title(f"Hyperparameter Tuning for {model_type}-based Model")
    
    ax.view_init(elev=30, azim=fig_rotation)  # Adjust these values to change the angle
    
    plot_subtext = f"{x_name} ranges from {x.min()} - {x.max()}, while {y_name} varies from {y.min()} - {y.max()}."
    plot_subtext += "Ranking is based on the calinski-harabaz score from a 5-fold, cross-validation of a hyper-\n"
    plot_subtext += f"parameter sweep of the {model_type}-\based model. Other hyper-parameters,"
    plot_subtext += " while explored, are not shown." 
    plot_subtext = (plot_subtext)
    
    plt.figtext(
        0.15, -0.07, plot_subtext, wrap=True, horizontalalignment='left', fontsize=9, fontstyle='italic',
    )
    
    plt.show()


def hyper_parameter_pipe(path_to_results: str, num_sample: int = 1000) -> list:
    '''Perform hyper parameter pipeline'''
    
    # Get training data for hyper-param analysis
    pca_X_train, pca_X_test, y_train, y_test, pca_X, y = get_data_pipe(
        path_to_results, num_sample=num_sample)

    # Set LoF hyper params to grid-search evaluate
    neighbor_list = [5, 10, 15, 20, 30, 40]
    algorithm_list = ['ball_tree']
    leaf_list = [5, 10, 15, 20, 30, 40]
    p_list = [1]

    # Set IsoF hyper params to grid-search evaluate
    n_estim_list = [5, 10, 20, 40]
    contam_ratio_list = [0.02, 0.05, 0.1, 0.15]
    
    # Start clock (to help with HPC timing)
    timer = Timer()

    # Run evaluation
    df_lof = lof_sensitivity_analysis(pca_X_train, n_list = neighbor_list, algorithm = algorithm_list, 
                                              leaf_size = leaf_list, p = p_list)

    df_isoF = IsoF_sensitivity_analysis(pca_X_train, n_estim = n_estim_list, contam_ratio = contam_ratio_list)

    # Sort by scores: 
    df_lof = df_lof.sort_values(by='rank_test_score', ascending=True).reset_index(drop=True)
    df_isoF = df_isoF.sort_values(by='rank_test_score', ascending=True).reset_index(drop=True)
        
    # Plot results: 
    LoF_plot = hyper_parameter_plotting(
        df_lof, model_type="LoF", col_x="param_leaf_size", 
        col_y="param_n_neighbors", col_z="rank_test_score", x_name = "Leaf Size", 
        y_name = "Number of Neighbors", z_name = "Score Rank", fig_rotation=140)

    IsoF_plot = hyper_parameter_plotting(
        df_isoF, model_type="IsoF", col_x="param_n_estimators", 
        col_y="param_contamination", col_z="rank_test_score", 
        x_name = "Number of Estimators", y_name = "Contamination Ratio", 
        z_name = "Score Rank", fig_rotation=140)

    # Now score the model for anomaly identification: 
    algo = df_lof["param_algorithm"].iloc[0]
    n_leaf = df_lof["param_leaf_size"].iloc[0]
    n_neighbor = df_lof["param_n_neighbors"].iloc[0]
    p_var = df_lof["param_p"].iloc[0]
    
    lof_precision, lof_recall, lof_F1 = evaluate_LOF(pca_X_train, 
                                                     pca_X_test, 
                                                     np.ravel(y_train), 
                                                     np.ravel(y_test), 
                                                     n_neighbor= n_neighbor, 
                                                     n_leaf = n_leaf, 
                                                     algorithm = algo, 
                                                     p=p_var,
                                                    )
    
    
    val_contam = df_isoF["param_contamination"].iloc[0]
    num_estimate = df_isoF["param_n_estimators"].iloc[0]
    
    IsoF_precision, IsoFrecall, IsoF1 = evaluate_ISF(pca_X_train, 
                                                     pca_X_test, 
                                                     np.ravel(y_train), 
                                                     np.ravel(y_test), 
                                                     n_estimate= num_estimate, 
                                                     contam_ratio = val_contam, 
                                                     )
    
    outputs = [
        IsoF_precision, IsoFrecall, IsoF1, 
        lof_precision, lof_recall, lof_F1, 
        df_lof, df_isoF, 
        LoF_plot, IsoF_plot, 
        pca_X_train, pca_X_test, y_train, y_test, pca_X, y]

    # Stop the timer
    timer.stop()

    return outputs


if __name__ == "__main__": 
    results = hyper_parameter_pipe(num_sample=2000)
    IsoF_precision, IsoFrecall, IsoF1, lof_precision, lof_recall, lof_F1, df_lof, df_isoF, LoF_plot, IsoF_plot,pca_X_train, pca_X_test, y_train, y_test, pca_X, y = results  # noqa: E501


