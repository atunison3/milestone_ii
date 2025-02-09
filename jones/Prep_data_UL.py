#This script will be used to perform the data cleaning and preparation tasks required for unsupervised learning

#import necessary packages 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 

#import relavent pyscripts: 
import EDA_part1
import EDA_part2

def transform_data(input_df, remove_cols, cat_cols, y_col): 
    """
    This function prepares the dataframe for model training by cleaning the dataframe, standardizing quantitative inputs, and one-
    Hot encoding the categorical data 
    
    INPUTS: 
    input_df: a pandas dataframe object of the combined charging data 
    remove_cols: a list of columns to remove from the dataset (not relavent to our study) 
    cat_cols: a list of string entries detailing which columns are categorical and should be one-hot encoded
    y_col: a string entry title for the column that we're performing unsupervised learning on

    OUTPUTS: 
    output_df: a pandas dataframe object that is scaled and one-hot encoded with the desired features 
    """

    #first remove columns unwanted: 
    temp_df = input_df.drop(columns=remove_cols)
    
    #Make subset df that does not include the target column(s)
    if isinstance(y_col, list):
        features_df = temp_df.drop(columns=y_col)
        target_df = temp_df[y_col]
    else: 
        features_df = temp_df.drop(columns=[y_col])
        target_df = temp_df[[y_col]]

    #Find which columns are quantitative by deduction
    all_cols = list(features_df.columns) 
    quant_cols = [col for col in all_cols if col not in cat_cols]
    
    #set the encoder & scaler 
    encoder = OneHotEncoder(sparse_output=False, drop=None) 
    scaler = StandardScaler() 

    #Fit and transform the columns appropriately 
    scaled_columns = scaler.fit_transform(features_df[quant_cols]) #scale quant features only 
    encoded_columns = encoder.fit_transform(features_df[cat_cols]) #encode cat features only

    #Now map back to a scaled, encoded dataframe: 
    scaled_df = pd.DataFrame(scaled_columns, columns=quant_cols)
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(cat_cols))

    #Combine the processed data 
    scale_encoded_df = pd.concat([encoded_df, scaled_df], axis=1) 

    #Now add the target variables (in case needed): 
    output_df =  pd.concat([scale_encoded_df, target_df], axis=1)

    #return the preped_df: 
    return output_df


def split_data(input_df, target_cols, test_ratio=0.2, val_ratio=0.25): 
    """
    This function takes the input dataframe, and given the target columns, splits the data into a training, validation, and test dataset
    INPUTS: 
    input_df: a pandas dataframe with the prepared data
    target_cols: a list of string entries identifying the target columns we seek to be predictive to
    test_ratio: a float value between 0 and 1 determining how much of the input data will be the test dataset
    val_ratio: a float value between 0 and 1 determining what proportion of non-test data will be used for validation 

    OUTPTS: 
    X_train, X_val, and X_test: 
    Y_train, y_val, and y_test:
    """
    
    if isinstance(target_cols, list):
        y = input_df[target_cols]
    else: 
        y = input_df[[target_cols]]

    #Split_data between predictor and regressor: 
    X = input_df.drop(columns=target_cols) 
    
    #Separate test data from train/val set: 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,y, test_size=test_ratio, random_state=42)

    #Separate training and validation data: 
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio, random_state = 42) 

    #return the train, val, & test datasets
    return X_train, X_val, X_test, y_train, y_val, y_test


def process_datetime(input_df): 
    """
    Takes an input_dataframe for the charging events and parses the datetime column to extract the characteristic information from it
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

    #Now convert each of these to numeric rather than datetime: 
    
    
    out_df = input_df
    
    return out_df 


def anomaly_tags(input_df, anomaly_list=None): 
    """
    Converts entries from the input_df's "flag_id" column to 0 or 1 based on anomaly prescence 

    INPUTS: 
    input_df: a pandas dataframe with the flag_id column that needs to be updated
    anomaly_list: a list of integer values associated with the flag_id entry code

    OUTPUTS: 
    out_df: a pandas dataframe with the updated flag_id column
    """

    if anomaly_list == None: 
        input_df["flag_id"] = input_df["flag_id"].map(lambda x: 1 if x != 0 else 0)
    else: 
        input_df["flag_id"] = input_df["flag_id"].map(lambda x: 1 if x in anomaly_list else 0)

    out_df = input_df
    
    return out_df


def get_SL_data(input_condition=2, test_ratio=0.2, val_ratio=0.25, anomaly_list=None): 
    """
    This function prepares the data into train, val, and test datasets for supervised learning
    INPUTS: 
    input_condition: An integer value (0, 1, or 2) specifying which condition to apply to combine the data
                     0: do nothing, 1: drop nulls, 2: drops impractical values (used for supervised learning)
    test_ratio: a float value between 0 and 1 determining how much of the input data will be the test dataset
    val_ratio: a float value between 0 and 1 determining what proportion of non-test data will be used for validation
    anomaly_list: a list of integer values associated with the flag_id entry code

    OUTPUTS: 
    X_train, X_val, and X_test: 
    Y_train, y_val, and y_test:    
    """
    #Get the dataframe processed
    merged_df = EDA_part2.combine_charging_data(input_condition=input_condition)
    merged_time_df = process_datetime(merged_df)
    mapped_df = anomaly_tags(merged_time_df, anomaly_list=anomaly_list)
    
    #Declare the target, categorical, and unwanted columns: 
    y_col = "flag_id"
    cat_cols = ["power_kw", "connector_type", "pricing", "region", "land_use", "metro_area", "charge_level", "venue"]
    remove_cols = ["session_id", "connector_id_x", "evse_id", "connector_id_y", "start_datetime", "end_datetime",
                   "minute", "second"]
    
    # Get the train, val, and test split of data
    transformed_df = transform_data(mapped_df, remove_cols=remove_cols, cat_cols=cat_cols, y_col=y_col).dropna()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(transformed_df,
                                                                target_cols=y_col,
                                                                test_ratio=test_ratio,
                                                                val_ratio=val_ratio,
                                                                )

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def get_UL_data(input_condition=2, test_ratio=0.2, val_ratio=0.25, anomaly_list=None): 
    """
    This function prepares the data into train, val, and test datasets for unsupervised learning
    INPUTS: 
    input_condition: An integer value (0, 1, or 2) specifying which condition to apply to combine the data
                     0: do nothing, 1: drop nulls, 2: drops impractical values (used for supervised learning)
    test_ratio: a float value between 0 and 1 determining how much of the input data will be the test dataset
    val_ratio: a float value between 0 and 1 determining what proportion of non-test data will be used for validation
    anomaly_list: a list of integer values associated with the flag_id entry code

    OUTPUTS: 
    X_train, X_test: 
    Y_train, y_test:    
    """
    #Get the dataframe processed
    merged_df = EDA_part2.combine_charging_data(input_condition=input_condition)
    merged_time_df = process_datetime(merged_df)
    mapped_df = anomaly_tags(merged_time_df, anomaly_list=anomaly_list)
    
    #Declare the target, categorical, and unwanted columns: 
    y_col = "flag_id"
    cat_cols = ["power_kw", "connector_type", "pricing", "region", "land_use", "metro_area", "charge_level", "venue"]
    remove_cols = ["session_id", "connector_id_x", "evse_id", "connector_id_y", "start_datetime", "end_datetime",
                   "minute", "second"]
    
    # Get the train, val, and test split of data
    transformed_df = transform_data(mapped_df, remove_cols=remove_cols, cat_cols=cat_cols, y_col=y_col).dropna()
    
    y = transformed_df[[y_col]]
    X = transformed_df.drop(columns=y_col) 
    
    #Separate test data from train/test set: 
    X_train = transformed_df[transformed_df["flag_id"]==0].drop(columns=y_col)
    X_anomalies = transformed_df[transformed_df["flag_id"]==1].drop(columns=y_col)
    X_test = transformed_df.drop(columns=y_col)
    y_train = transformed_df[transformed_df["flag_id"]==0][["flag_id"]]
    y_test = transformed_df[["flag_id"]]

    return (X_train, X_test, X_anomalies, y_train, y_test)


if __name__ == "__main__":
    #X_train, X_val, X_test, y_train, y_val, y_test = get_SL_data()
    X_train, X_test, X_anomalies, y_train, y_test = get_UL_data()