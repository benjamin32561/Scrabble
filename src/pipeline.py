"""
End-to-end ML Pipeline for Scrabble Player Rating Prediction

This script orchestrates the complete workflow:
1. Load data
2. Preprocess and feature engineering
3. Train model
4. Evaluate performance
5. Generate predictions
"""

# Standard library imports
import sys
from pathlib import Path
from datetime import datetime

# Third-party imports
import pandas as pd
import numpy as np
import joblib

# Local imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelPipeline
from src.config import OUTPUT_DIR, RANDOM_STATE


class MLPipeline:
    """Complete machine learning pipeline"""
    
    def __init__(self, use_turns: bool = False, sample_turns: int = None):
        """
        Initialize the ML pipeline
        
        Args:
            use_turns: Whether to include turn-level features
            sample_turns: Number of turns to sample (None for all)
        """
        self.use_turns = use_turns
        self.sample_turns = sample_turns
        self.preprocessor = None
        self.feature_engineer = None
        self.model_pipeline = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.test_game_ids = None
        
    def load_and_preprocess(self):
        """Step 1: Load and preprocess data"""
        print("\n" + "="*70)
        print(" STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*70)
        
        self.preprocessor = DataPreprocessor(
            use_turns=self.use_turns,
            sample_turns=self.sample_turns
        )
        
        # Load data
        self.preprocessor.load_data()
        
        # Process training data
        self.X_train, self.y_train = self.preprocessor.process_train_data()
        
        # Process test data
        self.X_test = self.preprocessor.process_test_data()
        
        # Save game IDs for submission
        self.test_game_ids = self.X_test['game_id'].copy()
        
        print(f"\n✓ Training samples: {len(self.X_train)}")
        print(f"✓ Test samples: {len(self.X_test)}")
        print(f"✓ Number of features: {self.X_train.shape[1]}")
        
        return self
    
    def engineer_features(self):
        """Step 2: Apply feature engineering and transformations"""
        print("\n" + "="*70)
        print(" STEP 2: FEATURE ENGINEERING")
        print("="*70)
        
        self.feature_engineer = FeatureEngineer()
        
        # Create preprocessing pipeline
        print("Creating feature transformation pipeline...")
        self.feature_engineer.create_preprocessing_pipeline(self.X_train)
        
        # Fit and transform training data
        print("Transforming training data...")
        X_train_transformed = self.feature_engineer.fit_transform(self.X_train)
        
        # Transform test data
        print("Transforming test data...")
        X_test_transformed = self.feature_engineer.transform(self.X_test)
        
        # Convert to DataFrame for easier handling
        feature_names = self.feature_engineer.get_feature_names()
        self.X_train_transformed = pd.DataFrame(
            X_train_transformed,
            columns=feature_names,
            index=self.X_train.index
        )
        self.X_test_transformed = pd.DataFrame(
            X_test_transformed,
            columns=feature_names,
            index=self.X_test.index
        )
        
        print(f"\n✓ Features after transformation: {len(feature_names)}")
        print(f"✓ Training shape: {self.X_train_transformed.shape}")
        print(f"✓ Test shape: {self.X_test_transformed.shape}")
        
        return self
    
    def train_model(self, model_type: str = 'rf', **model_params):
        """Step 3: Train machine learning model"""
        print("\n" + "="*70)
        print(f" STEP 3: TRAINING MODEL ({model_type.upper()})")
        print("="*70)
        
        self.model_pipeline = ModelPipeline()
        
        # Train model (already preprocessed, so set fit=False for prepare_features)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train_transformed, self.y_train,
            test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Initialize model
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.model_pipeline.model = RandomForestRegressor(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 15),
                min_samples_split=model_params.get('min_samples_split', 5),
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == 'gbr':
            from sklearn.ensemble import GradientBoostingRegressor
            self.model_pipeline.model = GradientBoostingRegressor(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 5),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=RANDOM_STATE,
                verbose=1
            )
        elif model_type == 'lr':
            from sklearn.linear_model import LinearRegression
            self.model_pipeline.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        print("Training model...")
        self.model_pipeline.model.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred_train = self.model_pipeline.model.predict(X_train)
        y_pred_val = self.model_pipeline.model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        print("\n" + "-"*70)
        print(" TRAINING METRICS")
        print("-"*70)
        print(f"  RMSE: {train_rmse:.2f}")
        print(f"  MAE:  {train_mae:.2f}")
        print(f"  R²:   {train_r2:.4f}")
        
        print("\n" + "-"*70)
        print(" VALIDATION METRICS")
        print("-"*70)
        print(f"  RMSE: {val_rmse:.2f}")
        print(f"  MAE:  {val_mae:.2f}")
        print(f"  R²:   {val_r2:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.model_pipeline.model, 'feature_importances_'):
            print("\n" + "-"*70)
            print(" TOP 10 FEATURE IMPORTANCES")
            print("-"*70)
            importances = pd.DataFrame({
                'feature': self.X_train_transformed.columns,
                'importance': self.model_pipeline.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, row in importances.head(10).iterrows():
                print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        return self
    
    def generate_predictions(self):
        """Step 4: Generate predictions for test set"""
 
    
    def save_submission(self, predictions: np.ndarray, filename: str = None):
        """Step 5: Save predictions to submission file"""
 
    
    def save_model(self, filename: str = None):
        """Save trained model and preprocessing pipeline"""

    
    def run_full_pipeline(self, model_type: str = 'rf', save_model: bool = True, **model_params):
        """Run the complete pipeline from data loading to submission"""





