from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MultiLabelBinarizer

from team17._supervised_conf import cat_features, num_features

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
