# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.config import OUTPUT_DIR

print("\n" + "="*70)
print(" QUICK PIPELINE TEST")
print("="*70)

# Step 1: Preprocess data
print("\nStep 1: Preprocessing data...")
preprocessor = DataPreprocessor(use_turns=False)
preprocessor.load_data()

X_train, y_train = preprocessor.process_train_data()
X_test = preprocessor.process_test_data()
test_game_ids = X_test['game_id'].copy()

print(f"✓ Training data: {X_train.shape}")
print(f"✓ Test data: {X_test.shape}")

# Step 2: Feature engineering
print("\nStep 2: Feature engineering...")
engineer = FeatureEngineer()
engineer.create_preprocessing_pipeline(X_train)

X_train_transformed = engineer.fit_transform(X_train)
X_test_transformed = engineer.transform(X_test)

print(f"✓ Transformed training data: {X_train_transformed.shape}")
print(f"✓ Transformed test data: {X_test_transformed.shape}")

# Step 3: Train baseline model (Linear Regression)
print("\nStep 3a: Training Linear Regression (baseline)...")
lr_model = LinearRegression()
lr_model.fit(X_train_transformed, y_train)

y_pred_lr = lr_model.predict(X_train_transformed)
lr_rmse = np.sqrt(mean_squared_error(y_train, y_pred_lr))
lr_mae = mean_absolute_error(y_train, y_pred_lr)
lr_r2 = r2_score(y_train, y_pred_lr)

print(f"  RMSE: {lr_rmse:.2f}")
print(f"  MAE:  {lr_mae:.2f}")
print(f"  R²:   {lr_r2:.4f}")

# Step 4: Train Random Forest
print("\nStep 3b: Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=50,  # Reduced for speed
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_transformed, y_train, test_size=0.2, random_state=42
)

rf_model.fit(X_tr, y_tr)

y_pred_train = rf_model.predict(X_tr)
y_pred_val = rf_model.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_tr, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"  Train RMSE: {train_rmse:.2f}")
print(f"  Val RMSE:   {val_rmse:.2f}")
print(f"  Train MAE:  {mean_absolute_error(y_tr, y_pred_train):.2f}")
print(f"  Val MAE:    {mean_absolute_error(y_val, y_pred_val):.2f}")
print(f"  Val R²:     {r2_score(y_val, y_pred_val):.4f}")

# Step 5: Feature importance
print("\nTop 10 Feature Importances:")
feature_names = engineer.get_feature_names()
importances = sorted(zip(feature_names, rf_model.feature_importances_), 
                     key=lambda x: x[1], reverse=True)
for i, (name, imp) in enumerate(importances[:10], 1):
    print(f"  {i:2d}. {name:30s}: {imp:.4f}")

# Step 6: Generate test predictions
print("\nStep 4: Generating test predictions...")
test_predictions = rf_model.predict(X_test_transformed)
print(f"  Mean prediction: {test_predictions.mean():.2f}")
print(f"  Std prediction:  {test_predictions.std():.2f}")

# Save submission
submission = pd.DataFrame({
    'game_id': test_game_ids,
    'rating': test_predictions
})

OUTPUT_DIR.mkdir(exist_ok=True)
submission_path = OUTPUT_DIR / 'test_submission.csv'
submission.to_csv(submission_path, index=False)

print(f"\n✓ Submission saved to: {submission_path}")

print("\n" + "="*70)
print(" TEST COMPLETE!")
print("="*70)
print(f"\n✓ Validation RMSE: {val_rmse:.2f}")
print(f"✓ Predictions generated: {len(test_predictions)}")
print(f"✓ Submission file created")

