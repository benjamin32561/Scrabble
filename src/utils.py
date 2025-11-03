"""Utility functions for model training and data processing"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


def delete_keys_from_dict(dict, keys):
    """Remove specified keys from a dictionary"""
    return {k: v for k, v in dict.items() if k not in keys}


def create_rating_bins(ratings, bins=None, labels=None):
    """
    Create rating bins for stratification and analysis
    
    Parameters:
    -----------
    ratings : array-like
        Rating values to bin
    bins : list, optional
        Bin edges. Default: [0, 1200, 1400, 1600, 1800, 2000, 2200, 3000]
    labels : list, optional
        Labels for bins. Default: ['<1200', '1200-1400', '1400-1600', 
                                   '1600-1800', '1800-2000', '2000-2200', '>2200']
    
    Returns:
    --------
    pd.Categorical : Binned ratings
    """
    if bins is None:
        bins = [0, 1200, 1400, 1600, 1800, 2000, 2200, 3000]
    if labels is None:
        labels = ['<1200', '1200-1400', '1400-1600', '1600-1800', 
                  '1800-2000', '2000-2200', '>2200']
    
    return pd.cut(ratings, bins=bins, labels=labels)


def compute_rating_sample_weights(ratings, bins=None):
    """
    Compute sample weights to balance rating ranges
    
    Gives higher weights to underrepresented rating ranges (e.g., <1200)
    to help the model learn better for rare cases.
    
    Parameters:
    -----------
    ratings : array-like
        Rating values
    bins : list, optional
        Bin edges for grouping ratings
    
    Returns:
    --------
    np.ndarray : Sample weights (higher for underrepresented ratings)
    """
    rating_bins = create_rating_bins(ratings, bins=bins)
    weights = compute_sample_weight('balanced', rating_bins)
    return weights


def stratified_train_val_split(X, y, test_size=0.2, random_state=42, bins=None):
    """
    Split data with stratification by rating bins
    
    Ensures similar rating distributions in train and validation sets.
    
    Parameters:
    -----------
    X : DataFrame or array-like
        Features
    y : Series or array-like
        Target ratings
    test_size : float
        Proportion for validation set
    random_state : int
        Random seed
    bins : list, optional
        Bin edges for stratification
    
    Returns:
    --------
    X_train, X_val, y_train, y_val : Split datasets
    """
    rating_strata = create_rating_bins(y, bins=bins)
    
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=rating_strata,
        random_state=random_state
    )


def evaluate_by_rating_range(y_true, y_pred, bins=None, labels=None):
    """
    Calculate MAPE and other metrics by rating range
    
    Helps identify which rating ranges the model struggles with.
    
    Parameters:
    -----------
    y_true : array-like
        True ratings
    y_pred : array-like
        Predicted ratings
    bins : list, optional
        Bin edges
    labels : list, optional
        Bin labels
    
    Returns:
    --------
    pd.DataFrame : MAPE, MAE, and sample counts by rating range
    """
    rating_bins = create_rating_bins(y_true, bins=bins, labels=labels)
    pct_errors = np.abs((y_true - y_pred) / y_true) * 100
    
    results = pd.DataFrame({
        'rating_bin': rating_bins,
        'pct_error': pct_errors,
        'abs_error': np.abs(y_true - y_pred)
    })
    
    summary = results.groupby('rating_bin', observed=True).agg({
        'pct_error': ['mean', 'std'],
        'abs_error': 'mean'
    }).round(2)
    
    summary.columns = ['MAPE (%)', 'Std (%)', 'MAE']
    summary['Count'] = results.groupby('rating_bin', observed=True).size()
    
    return summary