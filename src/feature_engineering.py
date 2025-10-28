"""Feature engineering utilities"""

import pandas as pd
import numpy as np
from typing import Dict, List


class FeatureEngineer:
    """Class for creating and engineering features"""
    
    def __init__(self):
        pass
    
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