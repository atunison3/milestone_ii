

if __name__ == '__main__':
    import argparse 
    import matplotlib.pyplot as plt 
    import numpy as np
    import os
    import pandas as pd
    import time

    from team17.unsup.EDA_part1 import show_EDA1_results
    from team17.unsup.EDA_part2 import combine_charging_data, generate_desired_plots
    from team17.unsup.Prep_data_UL_V2 import main_execution
    from team17.unsup.Unsupervised_Learning_V3 import get_data_pipe, review_models_pipe
    from team17.unsup.Unsupervised_Learning_Hyper_param import hyper_parameter_pipe
    from team17.unsup.Sensitivity_Analysis_V2 import hyper_parameter_pipe2

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_results')
    parser.add_argument('--path_to_assets', default='assets_reduced')
    parser.add_argument('--input_condition1', type=int, default=2)
    parser.add_argument('--input_condition2', type=int, default=1)
    parser.add_argument('--num_sample1', type=int, default=40000)
    parser.add_argument('--num_sample2', type=int, default=2000)
    args = parser.parse_args()

    # Perform EDA
    print("\033[32mPerforming EDA Part 1\033[0m")
    show_EDA1_results(os.path.join(args.path_to_assets, 'evwatts.public.session.csv'))

    # Perform EDA 2
    print("\033[32mPerforming EDA Part 2\033[0m")
    df = combine_charging_data(input_condition=args.input_condition1, assets_path=args.path_to_assets)
    generate_desired_plots(df)

    # Perform Prep data
    print("\033[32mPerforming Prep Data\033[0m")
    X_train, X_test, y_train, y_test = main_execution(
        input_condition=args.input_condition2, path_to_results=args.path_to_results)

    # Perform Unsupervised Learning
    print("\033[32mPerforming Unsupervised Learning\033[0m")
    pca_X_train, pca_X_test, y_train, y_test, pca_X, y = get_data_pipe(
        args.path_to_results, num_sample=args.num_sample1)
    test = review_models_pipe(pca_X, np.ravel(y))

    # Hyper parameter tuning
    print("\033[32mPerforming Unsupervised Learning Hyper Parameter Tuning\033[0m")
    results = hyper_parameter_pipe(args.path_to_results, num_sample=args.num_sample2)
    IsoF_precision, IsoFrecall, IsoF1, lof_precision, lof_recall, lof_F1, df_lof, df_isoF, LoF_plot, IsoF_plot,pca_X_train, pca_X_test, y_train, y_test, pca_X, y = results  # noqa: E501

    # Sensitivity analysis
    print("\033[32mPerforming Sensitivity Analysis\033[0m")

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
        results = hyper_parameter_pipe2(args.path_to_results, dropword, num_sample=sample_size)
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

