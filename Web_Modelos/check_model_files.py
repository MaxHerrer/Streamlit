import joblib
import sys
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Monkey-patch to handle compatibility with older pickle files
if not hasattr(sys.modules['sklearn.compose._column_transformer'], '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

# Custom transformer for loading pickles
class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X

kmeans = joblib.load("models/kmeans_pipeline.pkl")
logistic = joblib.load("models/logistic_regression_pipeline.pkl")
tree = joblib.load("models/decision_tree_pipeline.pkl")

print("OK")
