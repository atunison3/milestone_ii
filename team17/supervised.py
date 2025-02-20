from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MultiLabelBinarizer

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names: list):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.attribute_names].values
    
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = MultiLabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)

string_converter = FunctionTransformer(lambda x: x.astype(str), validate=False)

# Select features
num_features = ['start_soc', 'year']
cat_features = [
    'evse_id', 'connector_id', 'metro_area', 'land_use', 'num_ports', 'venue', 
    'pricing', 'month', 'day', 'hour', 'day_of_year', 'quarter', 'weekday', 'week_num'
    ]

# Create pipelines
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_features)),
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_features)),
        ('convert_to_string', string_converter),
        ('label_binarizer', MyLabelBinarizer())
    ])

# Combine to full pipeline
full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
    ])

if __name__=='__main__':
    import argparse
    
    from team17._supervised_functions import DataCleaningFunctions, LoadData
    from team17._supervised_functions import split_train_test_by_id, get_paths

    # Create argument parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_assets', default=None)
    args = parser.parse_args()

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
    X_test = full_pipeline.transform(df_test_cleaned.drop(columns=['charge_duration']))

    # Get label data
    y_train = np.array(df_train_clean['charge_duration'])
    y_test = np.array(df_test_clean['charge_duration'])
    del df_train_clean, df_test_clean

    # Convert label data from hours to minutes
    y_train *= 60
    y_test *= 60









