"""Data loading and preprocessing utilities"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from src.config import DATA_DIR, TRAIN_FILE, TEST_FILE, GAMES_FILE, TURNS_FILE


class DataLoader:
    """Class to load and manage dataset files"""
    
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.games_df = None
        self.turns_df = None
    
    def load_all_data(self):
        """Load all CSV files into memory"""
        print("Loading data files...")
        self.train_df = pd.read_csv(TRAIN_FILE)
        self.test_df = pd.read_csv(TEST_FILE)
        self.games_df = pd.read_csv(GAMES_FILE)
        self.turns_df = pd.read_csv(TURNS_FILE)
        
        print(f"Training data: {len(self.train_df)} rows")
        print(f"Test data: {len(self.test_df)} rows")
        print(f"Games data: {len(self.games_df)} rows")
        print(f"Turns data: {len(self.turns_df)} rows")
        
        return self
    
    def get_train_data(self) -> pd.DataFrame:
        """Get training data"""
        if self.train_df is None:
            self.train_df = pd.read_csv(TRAIN_FILE)
        return self.train_df
    
    def get_test_data(self) -> pd.DataFrame:
        """Get test data"""
        if self.test_df is None:
            self.test_df = pd.read_csv(TEST_FILE)
        return self.test_df
    
    def get_games_data(self) -> pd.DataFrame:
        """Get games metadata"""
        if self.games_df is None:
            self.games_df = pd.read_csv(GAMES_FILE)
        return self.games_df
    
    def get_turns_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Get turns data, optionally sampling"""
        if self.turns_df is None:
            if sample_size:
                # Read in chunks to sample
                chunk_list = []
                for chunk in pd.read_csv(TURNS_FILE, chunksize=10000):
                    chunk_list.append(chunk)
                    if len(chunk_list) >= 10:  # Sample first 100k rows
                        break
                self.turns_df = pd.concat(chunk_list, ignore_index=True)
                if sample_size:
                    self.turns_df = self.turns_df.sample(n=min(sample_size, len(self.turns_df)), 
                                                          random_state=42)
            else:
                self.turns_df = pd.read_csv(TURNS_FILE)
        return self.turns_df
    
    def get_basic_info(self):
        """Print basic information about all datasets"""
        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        
        for name, df in [("Train", self.train_df), ("Test", self.test_df), 
                         ("Games", self.games_df), ("Turns", self.turns_df)]:
            if df is not None:
                print(f"\n{name} DataFrame:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Missing values: {df.isnull().sum().sum()}")
                print(f"\n  First few rows:")
                print(df.head(3).to_string())

