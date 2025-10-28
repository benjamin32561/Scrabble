# Standard library imports
from typing import Dict, List, Tuple

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class FeatureEngineer:
    """Class for creating and engineering features"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
    
    def aggregate_game_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from games.csv"""
        df = games_df.copy()
        
        # Convert date to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Extract date features
        df['year'] = df['created_at'].dt.year
        df['month'] = df['created_at'].dt.month
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['hour'] = df['created_at'].dt.hour
        
        # Time control features
        df['has_overtime'] = (df['max_overtime_minutes'] > 0).astype(int)
        
        # Categorical encoding
        df['is_rated'] = (df['rating_mode'] == 'RATED').astype(int)
        
        return df
    
    def aggregate_turn_features(self, turns_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from turns.csv aggregated by game_id and nickname"""
        from src.config import TURNS_AGGREGATIONS
        
        df = turns_df.copy()
        
        # Basic aggregations
        turn_features = df.groupby(['game_id', 'nickname']).agg({
            'points': ['mean', 'max', 'min', 'std', 'sum'],
            'score': ['mean', 'max', 'std', 'last'],
            'turn_number': 'max'
        }).reset_index()
        
        # Flatten column names
        turn_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in turn_features.columns.values]
        
        return turn_features
    
    def create_game_player_features(self, train_df: pd.DataFrame, 
                                    games_df: pd.DataFrame) -> pd.DataFrame:
        """Merge training data with games data"""
        df = train_df.copy()
        
        # Merge with games data
        df = df.merge(games_df, on='game_id', how='left')
        
        return df
    
    def compute_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each game, compute features relative to opponent"""
        game_features = []
        
        for game_id in df['game_id'].unique():
            game_df = df[df['game_id'] == game_id].copy()
            
            if len(game_df) == 2:  # Should have 2 players
                for idx, row in game_df.iterrows():
                    opponent = game_df[game_df['nickname'] != row['nickname']].iloc[0]
                    
                    # Create relative features
                    features = {
                        'game_id': game_id,
                        'nickname': row['nickname'],
                        'score_diff': row['score'] - opponent['score'],
                        'score_ratio': row['score'] / (opponent['score'] + 1),  # +1 to avoid div by zero
                        'opponent_rating': opponent['rating'],
                        'opponent_score': opponent['score'],
                        'is_bot': self._is_bot(row['nickname'])
                    }
                    
                    game_features.append(features)
        
        return pd.DataFrame(game_features)
    
    @staticmethod
    def _is_bot(nickname: str) -> bool:
        """Check if nickname is a bot"""
        bots = ['BetterBot', 'STEEBot', 'HastyBot']
        return nickname in bots
    
    def select_features(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """Select specific features from dataframe"""
        available_features = [f for f in feature_list if f in df.columns]
        return df[available_features]
    
    def create_preprocessing_pipeline(self, X: pd.DataFrame, 
                                     numeric_features: List[str] = None,
                                     categorical_features: List[str] = None) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline
        
        Args:
            X: Feature dataframe
            numeric_features: List of numeric feature names (auto-detected if None)
            categorical_features: List of categorical feature names (auto-detected if None)
            
        Returns:
            ColumnTransformer pipeline
        """
        # Auto-detect if not provided
        if numeric_features is None:
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude game_id if present
            numeric_features = [f for f in numeric_features if f != 'game_id']
        
        if categorical_features is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop game_id and other unused columns
        )
        
        self.preprocessor = preprocessor
        
        return preprocessor
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessor and transform data"""
        if self.preprocessor is None:
            self.create_preprocessing_pipeline(X)
        
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        self._extract_feature_names()
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        return self.preprocessor.transform(X)
    
    def _extract_feature_names(self):
        """Extract feature names after transformation"""
        feature_names = []
        
        # Numeric features keep their names
        feature_names.extend(self.numeric_features)
        
        # Get categorical feature names from OneHotEncoder
        if self.categorical_features:
            cat_pipeline = self.preprocessor.named_transformers_['cat']
            onehot = cat_pipeline.named_steps['onehot']
            cat_names = onehot.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_names)
        
        self.feature_names = feature_names
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features after transformation"""
        return self.feature_names if self.feature_names else []
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values with different strategies per column
        
        Args:
            df: Input dataframe
            strategy: Dictionary mapping column names to strategies
                     ('mean', 'median', 'mode', 'drop', or a constant value)
        
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if strategy is None:
            # Default strategies
            strategy = {}
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    strategy[col] = 'median'
                else:
                    strategy[col] = 'mode'
        
        for col, strat in strategy.items():
            if col not in df.columns:
                continue
            
            if strat == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strat == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strat == 'mode':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
            elif strat == 'drop':
                df = df[df[col].notna()]
            else:
                # Assume it's a constant value
                df[col].fillna(strat, inplace=True)
        
        return df