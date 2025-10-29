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
        self.bot_names = ['BetterBot', 'STEEBot', 'HastyBot', 'Super']
        self.feature_stats = {}
    
    def aggregate_game_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from games.csv"""
        df = games_df.copy()
        
        # Time control features
        df['has_overtime'] = (df['max_overtime_minutes'] > 0).astype(int)
        
        # Categorical encoding
        df['is_rated'] = (df['rating_mode'] == 'RATED').astype(int)
        
        return df
    
    def aggregate_turn_features(self, turns_df: pd.DataFrame) -> pd.DataFrame:
        """Create streamlined features from turns.csv aggregated by game_id and nickname"""
        df = turns_df.copy()
        
        # Simplified aggregations
        turn_features = df.groupby(['game_id', 'nickname']).agg({
            'points': ['mean', 'sum', 'count'],
            'score': 'last'
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
                        'score': row['score'],
                        'rating': row['rating'],
                        'opponent_nickname': opponent['nickname'],
                        'opponent_score': opponent['score'],
                        'opponent_rating': opponent['rating'],
                        'score_diff': row['score'] - opponent['score'],
                        'score_ratio': row['score'] / (opponent['score'] + 1),  # +1 to avoid div by zero
                        'won_game': 1 if row['score'] > opponent['score'] else 0,
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
    
    def create_bot_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on bot interactions"""
        print("Creating bot interaction features...")
        
        df = df.copy()
        
        # Identify bot opponents
        df['opponent_is_bot'] = df['opponent_nickname'].isin(self.bot_names).astype(int)
        
        # Bot-specific opponent features
        for bot in self.bot_names:
            df[f'opponent_is_{bot}'] = (df['opponent_nickname'] == bot).astype(int)
        
        # Bot strength categories based on EDA findings
        def get_bot_strength(nickname):
            if nickname == 'STEEBot':
                return 3  # Strongest
            elif nickname == 'HastyBot':
                return 2  # Medium
            elif nickname == 'BetterBot':
                return 1  # Weakest
            elif nickname == 'Super':
                return 2  # Medium
            else:
                return 0  # Human
        
        df['opponent_bot_strength'] = df['opponent_nickname'].apply(get_bot_strength)
        
        print(f"✓ Added bot interaction features")
        return df
    
    def create_performance_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features measuring performance consistency"""
        print("Creating performance consistency features...")
        
        df = df.copy()
        
        # Score consistency relative to opponent
        df['score_consistency'] = df['score_diff'] / (df['opponent_score'] + 1)
        
        # Performance categories
        def categorize_performance(score_diff):
            if score_diff > 50:
                return 'dominant'
            elif score_diff > 10:
                return 'strong'
            elif score_diff > -10:
                return 'close'
            elif score_diff > -50:
                return 'weak'
            else:
                return 'poor'
        
        df['performance_category'] = df['score_diff'].apply(categorize_performance)
        
        # Binary performance indicators
        df['is_dominant_win'] = (df['score_diff'] > 50).astype(int)
        df['is_close_game'] = (abs(df['score_diff']) <= 10).astype(int)
        
        print(f"✓ Added performance consistency features")
        return df
    
    # Temporal features removed - showed little to no predictive value
    
    def create_game_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create game context features"""
        print("Creating game context features...")
        
        df = df.copy()
        
        # Time control categories based on EDA
        def categorize_time_control(time_name):
            if pd.isna(time_name):
                return 'unknown'
            time_name = str(time_name).lower()
            if 'ultra' in time_name:
                return 'ultra_blitz'
            elif 'blitz' in time_name:
                return 'blitz'
            elif 'rapid' in time_name:
                return 'rapid'
            else:
                return 'regular'
        
        df['time_category'] = df['time_control_name'].apply(categorize_time_control)
        
        # Lexicon categories
        def categorize_lexicon(lexicon):
            if pd.isna(lexicon):
                return 'unknown'
            lexicon = str(lexicon)
            if 'CSW' in lexicon:
                return 'csw'
            elif 'NWL' in lexicon:
                return 'nwl'
            elif 'ECWL' in lexicon:
                return 'ecwl'
            else:
                return 'other'
        
        df['lexicon_category'] = df['lexicon'].apply(categorize_lexicon)
        
        # Game mode features
        df['is_rated'] = (df['rating_mode'] == 'RATED').astype(int)
        df['has_overtime'] = (df['max_overtime_minutes'] > 1).astype(int)
        
        # Game duration features
        df['game_duration_minutes'] = df['game_duration_seconds'] / 60
        df['is_long_game'] = (df['game_duration_minutes'] > 15).astype(int)  # Based on EDA
        df['is_short_game'] = (df['game_duration_minutes'] < 5).astype(int)
        
        # Time pressure features
        df['time_pressure'] = df['initial_time_seconds'] / 60  # Convert to minutes
        df['high_time_pressure'] = (df['time_pressure'] < 10).astype(int)  # Less than 10 minutes
        
        print(f"✓ Added game context features")
        return df
    
    def create_turn_aggregation_features(self, df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
        """Create streamlined turn-level aggregation features (keeping only most predictive)"""
        print("Creating turn aggregation features...")
        
        # Get relevant games
        relevant_games = df['game_id'].unique()
        turns_filtered = turns_df[turns_df['game_id'].isin(relevant_games)].copy()
        
        # Handle missing values in turns data
        turns_filtered['points'] = turns_filtered['points'].fillna(0)
        turns_filtered['score'] = turns_filtered['score'].fillna(0)
        
        # Simplified aggregations - only keep most predictive features
        turn_agg = turns_filtered.groupby(['game_id', 'nickname']).agg({
            'points': ['mean', 'sum', 'count'],
            'score': 'last'
        }).reset_index()
        
        # Flatten column names
        turn_agg.columns = ['game_id', 'nickname', 'avg_points', 'total_points', 'turn_count', 'final_score']
        
        # Create key efficiency metrics (top correlated features)
        turn_agg['points_efficiency'] = turn_agg['total_points'] / (turn_agg['turn_count'] + 1)
        turn_agg['turn_efficiency'] = turn_agg['final_score'] / (turn_agg['turn_count'] + 1)
        
        # Merge with main dataframe
        df_merged = df.merge(turn_agg, on=['game_id', 'nickname'], how='left')
        
        print(f"✓ Added {len(turn_agg.columns)-2} turn aggregation features for {df_merged['avg_points'].notna().sum()} players")
        return df_merged
    
    def create_outlier_robust_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features robust to outliers"""
        print("Creating outlier-robust features...")
        
        df = df.copy()
        
        # Winsorized features (cap extreme values)
        def winsorize(series, limits=(0.01, 0.01)):
            return series.clip(
                lower=series.quantile(limits[0]),
                upper=series.quantile(1-limits[1])
            )
        
        # Apply winsorization to key features
        if 'score' in df.columns:
            df['score_winsorized'] = winsorize(df['score'])
        if 'game_duration_seconds' in df.columns:
            df['game_duration_winsorized'] = winsorize(df['game_duration_seconds'])
        
        # Log transformations for skewed features
        if 'game_duration_seconds' in df.columns:
            df['log_game_duration'] = np.log1p(df['game_duration_seconds'])
        
        # Robust scoring features
        if 'score' in df.columns:
            df['score_percentile'] = df['score'].rank(pct=True)
            df['score_zscore'] = (df['score'] - df['score'].mean()) / df['score'].std()
        
        print(f"✓ Added outlier-robust features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        print("Creating interaction features...")
        
        df = df.copy()
        
        # Score-time interactions
        if 'score' in df.columns and 'game_duration_minutes' in df.columns:
            df['score_per_minute'] = df['score'] / (df['game_duration_minutes'] + 1)
        
        # Rating-mode interactions
        if 'opponent_rating' in df.columns and 'is_rated' in df.columns:
            df['rated_opponent_rating'] = df['opponent_rating'] * df['is_rated']
        
        # Bot-strength interactions
        if 'opponent_bot_strength' in df.columns and 'score_diff' in df.columns:
            df['bot_strength_score_diff'] = df['opponent_bot_strength'] * df['score_diff']
        
        print(f"✓ Added interaction features")
        return df
    
    def engineer_all_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                             games_df: pd.DataFrame, turns_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete feature engineering pipeline"""
        print("Starting comprehensive feature engineering...")
        print("=" * 60)
        
        # Process training data
        print("\nProcessing training data...")
        train_processed = self._process_dataset(train_df, games_df, turns_df, is_test=False)
        
        # Process test data
        print("\nProcessing test data...")
        test_processed = self._process_dataset(test_df, games_df, turns_df, is_test=True)
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)
        print(f"Training features: {train_processed.shape[1]}")
        print(f"Test features: {test_processed.shape[1]}")
        
        return train_processed, test_processed
    
    def _process_dataset(self, df: pd.DataFrame, games_df: pd.DataFrame, 
                        turns_df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """Process a single dataset through the feature engineering pipeline"""
        
        # Start with basic opponent features
        df_processed = self.compute_opponent_features(df)
        
        # Merge with games data
        df_processed = df_processed.merge(games_df, on='game_id', how='left')
        
        # Apply all feature engineering steps
        df_processed = self.create_bot_interaction_features(df_processed)
        df_processed = self.create_performance_consistency_features(df_processed)
        df_processed = self.create_game_context_features(df_processed)
        df_processed = self.create_turn_aggregation_features(df_processed, turns_df)
        df_processed = self.create_outlier_robust_features(df_processed)
        df_processed = self.create_interaction_features(df_processed)
        
        return df_processed
    
    def prepare_features_for_modeling(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                     target_col: str = 'rating') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare features for machine learning models"""
        print("PREPARING FEATURES FOR MODELING")
        print("=" * 50)
        
        # Remove problematic features
        features_to_remove = [
            'game_id',  # ID column
            'nickname',  # High cardinality
            'opponent_nickname',  # High cardinality
            'created_at',  # Datetime (we have derived features)
            'first',  # High cardinality
            'time_control_name',  # We have time_category
            'lexicon',  # We have lexicon_category
            'game_end_reason',  # High cardinality
            'rating_mode'  # We have is_rated
        ]
        
        # Remove features that don't exist
        features_to_remove = [f for f in features_to_remove if f in train_df.columns]
        
        print(f"Removing {len(features_to_remove)} problematic features:")
        for feature in features_to_remove:
            print(f"  - {feature}")
        
        # Create clean datasets
        train_clean = train_df.drop(columns=features_to_remove)
        test_clean = test_df.drop(columns=[f for f in features_to_remove if f in test_df.columns])
        
        # Handle missing values
        print("\nHandling missing values...")
        
        # For numeric features, fill with median
        numeric_cols = train_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_col:
                median_val = train_clean[col].median()
                train_clean[col] = train_clean[col].fillna(median_val)
                if col in test_clean.columns:
                    test_clean[col] = test_clean[col].fillna(median_val)
        
        # For categorical features, fill with mode
        categorical_cols = train_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            mode_val = train_clean[col].mode()[0] if not train_clean[col].mode().empty else 'unknown'
            train_clean[col] = train_clean[col].fillna(mode_val)
            if col in test_clean.columns:
                test_clean[col] = test_clean[col].fillna(mode_val)
        
        # Separate features and target
        if target_col in train_clean.columns:
            X_train = train_clean.drop(columns=[target_col])
            y_train = train_clean[target_col]
        else:
            X_train = train_clean
            y_train = None
        
        X_test = test_clean
        
        print(f"\nFinal feature matrix shapes:")
        print(f"  Training: {X_train.shape}")
        print(f"  Test: {X_test.shape}")
        
        if y_train is not None:
            print(f"  Target: {y_train.shape}")
        
        return X_train, y_train, X_test