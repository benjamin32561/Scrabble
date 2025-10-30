import sys
from pathlib import Path

# Local imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, Dict, Any

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer



from src.config import RANDOM_STATE, TEST_SIZE


class ModelPipeline:
    """Pipeline for training and evaluating models"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
    
    def prepare_features(self, X: pd.DataFrame, categorical_features: list = None,
                        numerical_features: list = None, fit: bool = True):
        """Prepare features with encoding and scaling"""
        if categorical_features is None:
            categorical_features = ['lexicon', 'time_control_name', 'first', 
                                       'rating_mode', 'game_end_reason']
            categorical_features = [col for col in categorical_features if col in X.columns]
        
        if numerical_features is None:
            numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            numerical_features = [col for col in numerical_features 
                                if col not in ['game_id', 'winner', 'is_rated', 'has_overtime']
                                and not col.endswith('_id')]
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        if fit:
            X_preprocessed = preprocessor.fit_transform(X)
            self.preprocessor = preprocessor
            # Get feature names
            num_features = numerical_features
            if categorical_features:
                ohe = preprocessor.named_transformers_['cat']
                cat_features = ohe.get_feature_names_out(categorical_features).tolist()
                all_features = num_features + cat_features
            else:
                all_features = num_features
            self.feature_names = all_features
        else:
            X_preprocessed = self.preprocessor.transform(X)
        
        return X_preprocessed
    
    def train(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'rf',
              train_size: float = 1 - TEST_SIZE) -> Dict[str, Any]:
        """Train a model and return metrics"""
        
        # Prepare features
        X_prep = self.prepare_features(X, fit=True)
        
        # Split data
        if train_size < 1.0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_prep, y, test_size=1-train_size, random_state=RANDOM_STATE
            )
        else:
            X_train, y_train = X_prep, y
            X_val, y_val = None, None
        
        # Initialize model
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif model_type == 'gbr':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE,
                learning_rate=0.1
            )
        elif model_type == 'lr':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        results = {}
        
        if X_val is not None:
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)
            
            results['train_mse'] = mean_squared_error(y_train, y_pred_train)
            results['train_rmse'] = np.sqrt(results['train_mse'])
            results['train_mae'] = mean_absolute_error(y_train, y_pred_train)
            results['train_r2'] = r2_score(y_train, y_pred_train)
            
            results['val_mse'] = mean_squared_error(y_val, y_pred_val)
            results['val_rmse'] = np.sqrt(results['val_mse'])
            results['val_mae'] = mean_absolute_error(y_val, y_pred_val)
            results['val_r2'] = r2_score(y_val, y_pred_val)
        
        # Cross validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        results['cv_rmse'] = np.sqrt(-cv_scores.mean())
        results['cv_std'] = np.sqrt(cv_scores.std())
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        X_prep = self.prepare_features(X, fit=False)
        return self.model.predict(X_prep)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

