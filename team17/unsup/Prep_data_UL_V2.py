#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This script will be used to perform the data cleaning and preparation tasks required for unsupervised learning

# Import necessary packages 
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 

# Import relavent pyscripts: 
from .EDA_part2 import combine_charging_data

def transform_data(
    input_df: pd.DataFrame, remove_cols: list, 
    cat_cols: list, y_col: list) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    """
    This function prepares the dataframe for model training by cleaning the dataframe, 
    standardizing quantitative inputs, and one-Hot encoding the categorical data 
    
    INPUTS: 
    input_df: a pandas dataframe object of the combined charging data 
    remove_cols: a list of columns to remove from the dataset (not relavent to our study) 
    cat_cols: a list of string entries detailing which columns are categorical and should be one-hot encoded
    y_col: a string entry title for the column that we're performing unsupervised learning on

    OUTPUTS: 
    output_df: a pandas dataframe object that is scaled and one-hot encoded with the desired features 
    """

    # First remove columns unwanted: 
    temp_df = input_df.drop(columns=remove_cols)
    
    # Make subset df that does not include the target column(s)
    if isinstance(y_col, list):
        features_df = temp_df.drop(columns=y_col)
        target_df = temp_df[y_col]
    else: 
        features_df = temp_df.drop(columns=[y_col])
        target_df = temp_df[[y_col]]

    # Separate test data from train/val set: 
    X_train, X_test, y_train, y_test = train_test_split(features_df,target_df, test_size=0.2, random_state=42)

    # Find which columns are quantitative by deduction
    all_cols = list(features_df.columns) 
    quant_cols = [col for col in all_cols if col not in cat_cols]
    
    # Set the encoder & scaler 
    encoder = OneHotEncoder(sparse_output=False, drop=None) 
    scaler = StandardScaler() 

    # Fit and transform the columns appropriately 
    X_train_scaled_columns = scaler.fit_transform(X_train[quant_cols]) 
    X_train_encoded_columns = encoder.fit_transform(X_train[cat_cols])
    X_test_scaled_columns = scaler.transform(X_test[quant_cols])
    X_test_encoded_columns = encoder.transform(X_test[cat_cols])
    
    # Now map back to a scaled, encoded dataframe: 
    X_train_scaled_df = pd.DataFrame(X_train_scaled_columns, columns=quant_cols)
    X_train_encoded_df = pd.DataFrame(X_train_encoded_columns, columns=encoder.get_feature_names_out(cat_cols))
    X_test_scaled_df = pd.DataFrame(X_test_scaled_columns, columns=quant_cols)
    X_test_encoded_df = pd.DataFrame(X_test_encoded_columns, columns=encoder.get_feature_names_out(cat_cols))
    
    # Combine the processed data 
    X_train_scale_encoded_df = pd.concat([X_train_encoded_df, X_train_scaled_df], axis=1) 
    X_test_scale_encoded_df = pd.concat([X_test_encoded_df, X_test_scaled_df], axis=1) 

    return (X_train_scale_encoded_df, X_test_scale_encoded_df, y_train, y_test)


def process_datetime(input_df: pd.DataFrame) -> pd.DataFrame: 
    """
    Takes an input_dataframe for the charging events and parses the datetime column to 
    extract the characteristic information from it
    such as day, month, year, day of the week, day of the year, hour, etc. 
    
    INPUTS: 
    input_df: a pandas dataframe object 
    
    OUTPUS: 
    out_df: a pandas dataframe with added columns for the specific date information 
    """

    #Given the start_datetime, extract key features
    input_df["start_datetime"] = pd.to_datetime(input_df["start_datetime"])
    input_df["year"] = input_df["start_datetime"].dt.year.astype(int)
    input_df["month"] = input_df["start_datetime"].dt.month.astype(int)
    input_df["day"] = input_df["start_datetime"].dt.day.astype(int)
    input_df["hour"] = input_df["start_datetime"].dt.hour.astype(int)
    input_df["minute"] = input_df["start_datetime"].dt.minute.astype(int)
    input_df["second"] = input_df["start_datetime"].dt.second.astype(int)
    input_df["time_of_day"] = input_df["hour"] + input_df["minute"]/60 + input_df["second"]/3600
    input_df["weekday"] = input_df["start_datetime"].dt.weekday.astype(int)
    input_df["day_of_year"] = input_df["start_datetime"].dt.dayofyear.astype(int)
    
    return input_df 


def anomaly_tags(
    input_df: pd.DataFrame, anomaly_list: list[int] = None) -> pd.DataFrame: 
    """
    Converts entries from the input_df's "flag_id" column to 0 or 1 based on anomaly prescence 
    INPUTS: 
    input_df: a pandas dataframe with the flag_id column that needs to be updated
    anomaly_list: a list of integer values associated with the flag_id entry code

    OUTPUTS: 
    out_df: a pandas dataframe with the updated flag_id column
    """

    if anomaly_list is None: 
        # Automatically detects anomalies based on flag_id being present or not
        input_df["flag_id"] = input_df["flag_id"].map(lambda x: 1 if x != 0 else 0)
    else: 
        # Convert anomalies based on user defined anomalies
        input_df["flag_id"] = input_df["flag_id"].map(lambda x: 1 if x in anomaly_list else 0)
    
    return input_df


def main_execution(
    path_to_results: str, path_to_assets: str, 
    input_condition: int = 1, test_ratio: float = 0.2,
    anomaly_list: list[int] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    """
    This function runs the main execution to process the data into train, val, and test datasets
    INPUTS: 
    input_condition: An integer value (0, 1, or 2) specifying which condition to apply to combine the data
                     0: do nothing, 1: drop nulls, 2: drops impractical values (used for supervised learning)
    test_ratio: a float value between 0 and 1 determining how much of the input data will be the test dataset
    anomaly_list: a list of integer values associated with the flag_id entry code

    OUTPUTS: 
    X_train, X_val, & X_test: 
    Y_train, y_val, & y_test:    
    """
    # Get the dataframe processed
    merged_df = combine_charging_data(input_condition, assets_path=path_to_assets)
    merged_time_df = process_datetime(merged_df)
    mapped_df = anomaly_tags(merged_time_df, anomaly_list=anomaly_list)
    mapped_df = mapped_df.dropna()

    # Declare the target, categorical, and unwanted columns: 
    y_col = "flag_id"
    cat_cols = ["power_kw", "connector_type", "pricing", "region", "land_use", "metro_area", "charge_level", "venue"]
    remove_cols = ["session_id", "connector_id_x", "evse_id", "connector_id_y", "start_datetime", "end_datetime",
                   "hour", "minute", "second"]
    
    # Get the train, val, and test split of data
    X_train, X_test, y_train, y_test = transform_data(
        mapped_df, remove_cols=remove_cols, cat_cols=cat_cols, y_col=y_col)

    # Remove two binary categoricals: 
    X_train = X_train.drop(columns=["connector_type_Combo", "venue_Undesignated", "charge_level_DCFC"])
    X_test = X_test.drop(columns=["connector_type_Combo", "venue_Undesignated", "charge_level_DCFC"])
    
    # Output to datafile for future calling if not already in cwd: 
    dir_contents = os.listdir()
    outdict = {"UL_Xtrain.csv": X_train, "UL_Xtest.csv": X_test, "UL_ytrain.csv": y_train, "UL_ytest.csv": y_test}
    for key, value in outdict.items(): 
        if key not in dir_contents: 
            path_to_file = os.path.join(path_to_results, key)
            value.to_csv(path_to_file, index=False)

    return (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main_execution(input_condition=1)





