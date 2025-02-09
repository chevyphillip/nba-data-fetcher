"""Custom model classes for NBA player stats prediction."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

class TargetTransformer:
    """Custom transformer to scale targets and inverse transform predictions."""
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, y):
        self.scaler.fit(y.values.reshape(-1, 1))
        return self
        
    def transform(self, y):
        return self.scaler.transform(y.values.reshape(-1, 1)).ravel()
        
    def inverse_transform(self, y):
        return self.scaler.inverse_transform(y.reshape(-1, 1)).ravel()

class ScaledGradientBoostingRegressor:
    """Wrapper for GradientBoostingRegressor that handles target scaling."""
    def __init__(self, **params):
        self.target_transformer = TargetTransformer()
        self.regressor = GradientBoostingRegressor(**params)
        
    def fit(self, X, y):
        # Scale the target variable
        y_scaled = self.target_transformer.fit(y).transform(y)
        # Fit the regressor on scaled target
        self.regressor.fit(X, y_scaled)
        return self
        
    def predict(self, X):
        # Get predictions in scaled space
        y_pred_scaled = self.regressor.predict(X)
        # Transform back to original space
        return self.target_transformer.inverse_transform(y_pred_scaled)
    
    @property
    def feature_importances_(self):
        """Get feature importances from the underlying regressor."""
        return self.regressor.feature_importances_
