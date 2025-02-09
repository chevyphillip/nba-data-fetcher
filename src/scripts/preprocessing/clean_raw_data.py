"""
Clean and preprocess raw NBA statistics data.

This script performs initial cleaning and preprocessing of raw NBA statistics:
1. Removes duplicate entries
2. Handles missing values
3. Converts data types
4. Adds basic calculated fields
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = SCRIPT_DIR / "data" / "raw"
CLEANED_DIR = SCRIPT_DIR / "data" / "cleaned"

def load_raw_data() -> pd.DataFrame:
    """Load the most recent raw data file."""
    try:
        raw_files = list(RAW_DIR.glob("nba_player_stats_*.csv"))
        if not raw_files:
            raise FileNotFoundError("No raw data files found")
        
        latest_file = max(raw_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading raw data from {latest_file}")
        return pd.read_csv(latest_file)
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        raise

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries based on player, team, and season."""
    before_count = len(df)
    df = df.drop_duplicates(subset=['Player', 'Team', 'Season'], keep='last')
    after_count = len(df)
    
    if before_count > after_count:
        logger.info(f"Removed {before_count - after_count} duplicate entries")
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # Fill missing games/minutes with 0
    df['G'] = df['G'].fillna(0)
    df['MP'] = df['MP'].fillna(0)
    
    # Fill missing percentages with 0
    pct_columns = [col for col in df.columns if 'PCT' in col or 'Pct' in col]
    df[pct_columns] = df[pct_columns].fillna(0)
    
    # Fill missing counting stats with 0
    stat_columns = ['PTS', 'AST', 'TRB', 'STL', 'BLK', '3P', 'FG', 'FT']
    for col in stat_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Drop rows where player name or team is missing
    df = df.dropna(subset=['Player', 'Team'])
    
    return df

def convert_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types."""
    # Convert season to int
    df['Season'] = df['Season'].astype(int)
    
    # Convert numeric columns to float32 for efficiency
    numeric_columns = df.select_dtypes(include=['float64']).columns
    df[numeric_columns] = df[numeric_columns].astype('float32')
    
    # Convert categorical columns to category
    categorical_columns = ['Player', 'Team', 'Pos']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

def add_basic_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic calculated fields."""
    # Minutes per game
    df['MPG'] = df['MP'] / df['G']
    
    # Usage rate approximation
    df['Usage'] = df['FGA'] + 0.44 * df['FTA']
    
    # Per game statistics
    counting_stats = ['PTS', 'AST', 'TRB', 'STL', 'BLK', '3P']
    for stat in counting_stats:
        if stat in df.columns:
            df[f'{stat}_per_game'] = df[stat] / df['G']
    
    # Efficiency metrics
    df['AST_TO_ratio'] = df['AST'] / df['TOV'].replace(0, 1)  # Avoid division by zero
    
    return df

def clean_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize position data."""
    # Map positions to standard format
    pos_map = {
        'PG': 'PG', 'G': 'PG',
        'SG': 'SG', 'G-F': 'SG',
        'SF': 'SF', 'F': 'SF', 'F-G': 'SF',
        'PF': 'PF', 'F-C': 'PF',
        'C': 'C', 'C-F': 'C'
    }
    
    df['Pos'] = df['Pos'].map(pos_map).fillna('SF')  # Default to SF if unknown
    
    # Create position dummy variables
    pos_dummies = pd.get_dummies(df['Pos'], prefix='Pos')
    df = pd.concat([df, pos_dummies], axis=1)
    
    return df

def main():
    """Main execution function."""
    try:
        # Create cleaned directory if it doesn't exist
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        df = load_raw_data()
        initial_shape = df.shape
        logger.info(f"Initial data shape: {initial_shape}")
        
        # Apply cleaning steps
        df = remove_duplicates(df)
        df = handle_missing_values(df)
        df = convert_datatypes(df)
        df = add_basic_calculations(df)
        df = clean_positions(df)
        
        # Save cleaned data
        current_date = datetime.now().strftime("%Y%m%d")
        output_file = CLEANED_DIR / f"nba_player_stats_cleaned_{current_date}.csv"
        df.to_csv(output_file, index=False)
        
        final_shape = df.shape
        logger.info(f"Final data shape: {final_shape}")
        logger.info(f"Cleaned data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
