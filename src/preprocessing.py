import sys
from pathlib import Path

# Local imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, Optional, Dict

# Third-party imports
import pandas as pd
import numpy as np

from src.config import TRAIN_FILE, TEST_FILE, GAMES_FILE, TURNS_FILE


class DataPreprocessor:
    """Handles all data preprocessing and merging operations"""
    
    def __init__(self, use_turns: bool = True, sample_turns: Optional[int] = None):
        """
        Initialize the preprocessor
        
        Args:
            use_turns: Whether to include turn-level features (slower but more accurate)
            sample_turns: If provided, only use this many rows from turns.csv for faster processing
        """
        self.use_turns = use_turns
        self.sample_turns = sample_turns
        self.train_df = None
        self.test_df = None
        self.games_df = None
        self.turns_df = None
        self.bot_names = ['BetterBot', 'STEEBot', 'HastyBot', 'Super']
        
    def load_data(self):
        """Load all necessary data files"""
        print("Loading data files...")
        self.train_df = pd.read_csv(TRAIN_FILE)
        self.test_df = pd.read_csv(TEST_FILE)
        self.games_df = pd.read_csv(GAMES_FILE)
        
        if self.use_turns:
            if self.sample_turns:
                # Sample turns for faster processing during development
                print(f"Sampling {self.sample_turns} turns from turns.csv...")
                chunk_size = 100000
                chunks = []
                for chunk in pd.read_csv(TURNS_FILE, chunksize=chunk_size):
                    chunks.append(chunk)
                    if sum(len(c) for c in chunks) >= self.sample_turns:
                        break
                self.turns_df = pd.concat(chunks, ignore_index=True).iloc[:self.sample_turns]
            else:
                print("Loading full turns.csv (this may take a moment)...")
                self.turns_df = pd.read_csv(TURNS_FILE)
        
        print(f"✓ Loaded train: {len(self.train_df)} rows")
        print(f"✓ Loaded test: {len(self.test_df)} rows")
        print(f"✓ Loaded games: {len(self.games_df)} rows")
        if self.use_turns:
            print(f"✓ Loaded turns: {len(self.turns_df)} rows")
        
        return self
    
    def identify_target_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify which players in each game are the target (non-bot) players
        
        Args:
            df: DataFrame with game_id, nickname, score, rating columns
            
        Returns:
            DataFrame with additional 'is_bot' and 'is_target' columns
        """
        df = df.copy()
        df['is_bot'] = df['nickname'].isin(self.bot_names)
        
        # For test set, target players are those with NA ratings
        # For train set, all non-bot players are potential targets
        if 'rating' in df.columns:
            df['is_target'] = df['rating'].isna() | (~df['is_bot'])
        else:
            df['is_target'] = ~df['is_bot']
        
        return df
    
    def create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each player in a game, create features about their opponent
        
        Features created:
        - opponent_rating: Rating of the opponent
        - opponent_score: Score of the opponent
        - score_diff: player_score - opponent_score
        - score_ratio: player_score / opponent_score (with smoothing)
        - won_game: Whether the player won (1) or lost (0)
        """
        print("Creating opponent features...")
        
        result_rows = []
        
        for game_id in df['game_id'].unique():
            game_players = df[df['game_id'] == game_id].copy()
            
            if len(game_players) != 2:
                # Skip games without exactly 2 players
                continue
            
            player1 = game_players.iloc[0]
            player2 = game_players.iloc[1]
            
            # Create features for player 1
            p1_features = player1.to_dict()
            p1_features['opponent_rating'] = player2['rating']
            p1_features['opponent_score'] = player2['score']
            p1_features['opponent_nickname'] = player2['nickname']
            p1_features['score_diff'] = player1['score'] - player2['score']
            p1_features['score_ratio'] = player1['score'] / (player2['score'] + 1)  # +1 to avoid division by zero
            p1_features['won_game'] = 1 if player1['score'] > player2['score'] else 0
            
            # Create features for player 2
            p2_features = player2.to_dict()
            p2_features['opponent_rating'] = player1['rating']
            p2_features['opponent_score'] = player1['score']
            p2_features['opponent_nickname'] = player1['nickname']
            p2_features['score_diff'] = player2['score'] - player1['score']
            p2_features['score_ratio'] = player2['score'] / (player1['score'] + 1)
            p2_features['won_game'] = 1 if player2['score'] > player1['score'] else 0
            
            result_rows.append(p1_features)
            result_rows.append(p2_features)
        
        result_df = pd.DataFrame(result_rows)
        print(f"✓ Created opponent features for {len(result_df)} player-game pairs")
        
        return result_df
    
    def merge_with_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge player data with game metadata"""
        print("Merging with games data...")
        
        merged = df.merge(self.games_df, on='game_id', how='left')
        print(f"✓ Merged with games: {len(merged)} rows")
        
        return merged
    
    def aggregate_turn_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate turn-level statistics to game+player level
        
        Features created per player per game:
        - avg_points: Average points per turn
        - max_points: Maximum points in a single turn
        - std_points: Standard deviation of points (consistency)
        - total_turns: Number of turns played
        - total_points: Sum of all points
        """
        if not self.use_turns or self.turns_df is None:
            print("Skipping turn aggregation (turns data not loaded)")
            return df
        
        print("Aggregating turn-level features...")
        
        # Get unique game_ids from df to filter turns
        relevant_games = df['game_id'].unique()
        turns_filtered = self.turns_df[self.turns_df['game_id'].isin(relevant_games)]
        
        # Aggregate statistics
        turn_agg = turns_filtered.groupby(['game_id', 'nickname']).agg({
            'points': ['mean', 'max', 'std', 'sum'],
            'turn_number': 'max',
            'score': 'last'  # Final score from turns
        }).reset_index()
        
        # Flatten column names
        turn_agg.columns = ['game_id', 'nickname', 
                           'avg_points', 'max_points', 'std_points', 'total_points',
                           'total_turns', 'final_score']
        
        # Fill NaN std with 0 (happens when player has only 1 turn)
        turn_agg['std_points'] = turn_agg['std_points'].fillna(0)
        
        # Merge with main dataframe
        merged = df.merge(turn_agg, on=['game_id', 'nickname'], how='left')
        
        print(f"✓ Added turn features: {merged['avg_points'].notna().sum()} players with turn data")
        
        return merged
    
    def engineer_game_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from game metadata
        
        Features created:
        - is_rated: Binary flag for RATED vs CASUAL
        - has_overtime: Binary flag for overtime games
        - game_duration_minutes: Duration in minutes
        - time_per_turn: Average time per turn (if turn data available)
        - Date features: year, month, day_of_week, hour
        """
        print("Engineering game features...")
        
        df = df.copy()
        
        # Binary flags
        df['is_rated'] = (df['rating_mode'] == 'RATED').astype(int)
        df['has_overtime'] = (df['max_overtime_minutes'] > 1).astype(int)
        
        # Time features
        df['game_duration_minutes'] = df['game_duration_seconds'] / 60
        
        # Parse datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['year'] = df['created_at'].dt.year
        df['month'] = df['created_at'].dt.month
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['hour'] = df['created_at'].dt.hour
        
        # Time per turn (if turn data available)
        if 'total_turns' in df.columns:
            df['time_per_turn'] = df['game_duration_seconds'] / (df['total_turns'] + 1)
        
        # Categorize time controls
        def categorize_time_control(time_name):
            if pd.isna(time_name):
                return 'unknown'
            time_name = str(time_name).lower()
            if 'blitz' in time_name or 'ultra' in time_name:
                return 'blitz'
            elif 'rapid' in time_name:
                return 'rapid'
            else:
                return 'regular'
        
        df['time_category'] = df['time_control_name'].apply(categorize_time_control)
        
        print(f"✓ Engineered {df.shape[1]} total features")
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame, is_test: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final dataset for modeling
        
        Args:
            df: Processed dataframe
            is_test: Whether this is test data (determines which players to keep)
            
        Returns:
            X: Feature dataframe
            y: Target series (None for test set)
        """
        print(f"Preparing {'test' if is_test else 'train'} data for modeling...")
        
        df = df.copy()
        
        # For test set, keep only rows where we need to predict (rating is NA)
        # For train set, keep only non-bot players
        if is_test:
            df = df[df['rating'].isna()].copy()
        else:
            df = df[~df['is_bot']].copy()
        
        print(f"✓ Selected {len(df)} target players")
        
        # Define features to use
        numeric_features = [
            'score', 'opponent_rating', 'opponent_score', 
            'score_diff', 'score_ratio', 'won_game',
            'game_duration_minutes', 'initial_time_seconds',
            'is_rated', 'has_overtime', 'year', 'month', 'day_of_week', 'hour'
        ]
        
        # Add turn features if available
        if 'avg_points' in df.columns:
            numeric_features.extend([
                'avg_points', 'max_points', 'std_points', 
                'total_points', 'total_turns', 'time_per_turn'
            ])
        
        categorical_features = [
            'lexicon', 'time_category', 'first', 'game_end_reason'
        ]
        
        # Select features that exist in the dataframe
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        all_features = numeric_features + categorical_features + ['game_id']
        
        X = df[all_features].copy()
        
        # Get target if training data
        y = None if is_test else df['rating'].copy()
        
        print(f"✓ Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
        print(f"✓ Final shape: {X.shape}")
        
        return X, y
    
    def process_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete pipeline for training data"""
        print("\n" + "="*60)
        print("PROCESSING TRAINING DATA")
        print("="*60)
        
        # Identify players
        train_with_flags = self.identify_target_players(self.train_df)
        
        # Create opponent features
        train_with_opponent = self.create_opponent_features(train_with_flags)
        
        # Merge with games
        train_merged = self.merge_with_games(train_with_opponent)
        
        # Add turn features
        train_with_turns = self.aggregate_turn_features(train_merged)
        
        # Engineer additional features
        train_engineered = self.engineer_game_features(train_with_turns)
        
        # Prepare for modeling
        X_train, y_train = self.prepare_for_modeling(train_engineered, is_test=False)
        
        print("\n" + "="*60)
        print("TRAINING DATA READY")
        print("="*60)
        
        return X_train, y_train
    
    def process_test_data(self) -> pd.DataFrame:
        """Complete pipeline for test data"""
        print("\n" + "="*60)
        print("PROCESSING TEST DATA")
        print("="*60)
        
        # Identify players
        test_with_flags = self.identify_target_players(self.test_df)
        
        # Create opponent features
        test_with_opponent = self.create_opponent_features(test_with_flags)
        
        # Merge with games
        test_merged = self.merge_with_games(test_with_opponent)
        
        # Add turn features
        test_with_turns = self.aggregate_turn_features(test_merged)
        
        # Engineer additional features
        test_engineered = self.engineer_game_features(test_with_turns)
        
        # Prepare for modeling
        X_test, _ = self.prepare_for_modeling(test_engineered, is_test=True)
        
        print("\n" + "="*60)
        print("TEST DATA READY")
        print("="*60)
        
        return X_test


def main():
    """Example usage of the preprocessing pipeline"""
    # Initialize preprocessor (without turns for faster processing)
    preprocessor = DataPreprocessor(use_turns=False)
    
    # Load data
    preprocessor.load_data()
    
    # Process training data
    X_train, y_train = preprocessor.process_train_data()
    
    # Process test data
    X_test = preprocessor.process_test_data()
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"\nFeature list:")
    for i, col in enumerate(X_train.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return X_train, y_train, X_test


if __name__ == "__main__":
    X_train, y_train, X_test = main()

