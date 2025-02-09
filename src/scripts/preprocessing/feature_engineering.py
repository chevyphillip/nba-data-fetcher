"""
Feature Engineering for NBA Player Statistics

This script focuses on creating meaningful features for NBA player prediction:
1. Rolling averages for recent performance
2. Season context features
3. Efficiency metrics
4. Position-based features
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent.parent
CLEANED_DIR = SCRIPT_DIR / "data" / "cleaned"
FEATURES_DIR = SCRIPT_DIR / "data" / "features"

# Configuration
ROLLING_WINDOWS = [5, 10]  # Last N games windows
TARGET_STATS = {
    'PTS': {'name': 'Points', 'col': 'PTS_per_game'},
    'TRB': {'name': 'Rebounds', 'col': 'TRB_per_game'},
    'AST': {'name': 'Assists', 'col': 'AST_per_game'},
    '3P': {'name': '3-Pointers Made', 'col': '3P_per_game'}
}

def load_cleaned_data() -> pd.DataFrame:
    """Load the most recent cleaned data file."""
    try:
        cleaned_files = list(CLEANED_DIR.glob("nba_player_stats_cleaned_*.csv"))
        if not cleaned_files:
            raise FileNotFoundError("No cleaned data files found")
        
        latest_file = max(cleaned_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading cleaned data from {latest_file}")
        return pd.read_csv(latest_file)
    except Exception as e:
        logger.error(f"Error loading cleaned data: {e}")
        raise

def add_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages for key statistics."""
    df_rolling = df.copy()
    
    # Sort by player and season for proper rolling calculations
    df_rolling = df_rolling.sort_values(['Player', 'Season'])
    
    # Calculate rolling averages for each target stat
    for stat in TARGET_STATS.keys():
        col = TARGET_STATS[stat]['col']
        if col in df_rolling.columns:
            for window in ROLLING_WINDOWS:
                # Calculate rolling average within each player group
                roll_col = f'{stat}_rolling_{window}g'
                df_rolling[roll_col] = df_rolling.groupby('Player')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
    
    return df_rolling

def add_season_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add season context features."""
    df_context = df.copy()
    
    # COVID-impacted seasons (convert boolean to int)
    df_context['is_covid_season'] = df_context['Season'].isin([2020, 2021]).astype(int)
    
    # Games played relative to typical season
    df_context['games_played_ratio'] = df_context['G'] / df_context.groupby('Season')['G'].transform('max')
    
    # Season progress (0-1 scale)
    df_context['season_progress'] = df_context.groupby(['Season', 'Player']).cumcount() / df_context.groupby('Season')['G'].transform('max')
    
    # Experience (years in league)
    df_context['experience'] = df_context.groupby('Player').cumcount()
    
    return df_context

def add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced efficiency metrics."""
    df_eff = df.copy()
    
    # Scoring efficiency
    if all(col in df_eff.columns for col in ['PTS', 'FGA', 'FTA']):
        df_eff['points_per_shot'] = df_eff['PTS'] / (df_eff['FGA'] + 0.44 * df_eff['FTA'])
    
    # Usage rate components
    if all(col in df_eff.columns for col in ['FGA', 'TOV', 'FTA', 'MP']):
        df_eff['usage_rate'] = (df_eff['FGA'] + 0.44 * df_eff['FTA'] + df_eff['TOV']) / df_eff['MP']
    
    # Assist ratio
    if all(col in df_eff.columns for col in ['AST', 'TOV']):
        df_eff['assist_ratio'] = df_eff['AST'] / (df_eff['AST'] + df_eff['TOV'])
    
    # Three-point reliance
    if all(col in df_eff.columns for col in ['3PA', 'FGA']):
        df_eff['three_point_rate'] = df_eff['3PA'] / df_eff['FGA']
    
    return df_eff

def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add position-based features."""
    df_pos = df.copy()
    
    # Calculate position-specific averages
    for stat in TARGET_STATS.keys():
        col = TARGET_STATS[stat]['col']
        if col in df_pos.columns:
            # Calculate league average for each position
            pos_avgs = df_pos.groupby('Pos')[col].transform('mean')
            # Calculate how player performs vs. position average
            df_pos[f'{stat}_vs_pos_avg'] = df_pos[col] - pos_avgs
    
    return df_pos

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features to appropriate scales."""
    df_norm = df.copy()
    
    # Normalize per-game stats by minutes played
    per_game_cols = [col for col in df_norm.columns if 'per_game' in col]
    for col in per_game_cols:
        if 'MPG' in df_norm.columns:
            df_norm[f'{col}_per_minute'] = df_norm[col] / df_norm['MPG']
    
    # Scale rolling averages relative to season averages
    rolling_cols = [col for col in df_norm.columns if 'rolling' in col]
    for col in rolling_cols:
        base_stat = col.split('_rolling_')[0]
        if f'{base_stat}_per_game' in df_norm.columns:
            season_avg = df_norm.groupby('Season')[f'{base_stat}_per_game'].transform('mean')
            df_norm[f'{col}_vs_avg'] = df_norm[col] - season_avg
    
    return df_norm

def main():
    """Main execution function."""
    try:
        # Create features directory if it doesn't exist
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load cleaned data
        df = load_cleaned_data()
        initial_shape = df.shape
        logger.info(f"Initial data shape: {initial_shape}")
        
        # Apply feature engineering steps
        df = add_rolling_averages(df)
        df = add_season_context(df)
        df = add_efficiency_metrics(df)
        df = add_position_features(df)
        df = normalize_features(df)
        
        # Save engineered features
        current_date = datetime.now().strftime("%Y%m%d")
        output_file = FEATURES_DIR / f"nba_player_stats_features_{current_date}.csv"
        df.to_csv(output_file, index=False)
        
        final_shape = df.shape
        logger.info(f"Final data shape: {final_shape}")
        logger.info(f"Engineered features saved to {output_file}")
        
        # Log feature summary
        numeric_features = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
        logger.info(f"\nFeature Summary:")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Numeric features: {len(numeric_features)}")
        logger.info(f"Categorical features: {len(df.columns) - len(numeric_features)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
