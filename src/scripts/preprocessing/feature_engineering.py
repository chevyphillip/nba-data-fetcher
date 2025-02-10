"""
Feature Engineering for NBA Player Statistics

This script handles both data cleaning and feature engineering:
1. Data validation and cleaning
2. Feature engineering
3. Position-based features
4. Rolling averages and statistics
5. Advanced metrics calculation
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = SCRIPT_DIR / "data" / "raw"
FEATURES_DIR = SCRIPT_DIR / "data" / "features"


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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the data."""
    logger.info("Cleaning and validating data...")

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=["Player", "Team", "Season"])

    # Convert datatypes
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    numeric_cols = ["PTS", "TRB", "AST", "3P", "MP", "G"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate ranges
    df = df[df["Season"] >= 2010]  # Only keep recent seasons
    df = df[df["G"] > 0]  # Must have played games
    df = df[df["MP"] > 0]  # Must have minutes played

    return df


def calculate_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic per-game and per-minute statistics."""
    logger.info("Calculating basic statistics...")

    df_stats = df.copy()

    # Calculate Minutes Per Game (MPG)
    if all(col in df_stats.columns for col in ["MP", "G"]):
        df_stats["MPG"] = df_stats["MP"] / df_stats["G"]

    # Calculate per-game statistics
    per_game_stats = ["PTS", "TRB", "AST", "3P", "FGA", "FTA", "TOV"]
    for stat in per_game_stats:
        if stat in df_stats.columns:
            df_stats[f"{stat}_per_game"] = df_stats[stat] / df_stats["G"]

    return df_stats


def add_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages for key statistics."""
    logger.info("Adding rolling averages...")

    df = df.sort_values(["Player", "Season"])

    # Calculate rolling averages for main stats
    stats = ["PTS", "TRB", "AST", "3P"]
    windows = [5, 10, 15]

    for stat in stats:
        if f"{stat}_per_game" in df.columns:
            for window in windows:
                df[f"{stat}_rolling_{window}g"] = df.groupby("Player")[
                    f"{stat}_per_game"
                ].transform(lambda x: x.rolling(window, min_periods=1).mean())

    return df


def add_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced basketball metrics."""
    logger.info("Adding advanced metrics...")

    # Usage Rate
    if all(col in df.columns for col in ["FGA", "TOV", "FTA", "MP"]):
        df["Usage"] = (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / df["MP"]

    # True Shooting Percentage
    if all(col in df.columns for col in ["PTS", "FGA", "FTA"]):
        df["TS%"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"]))

    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric features."""
    logger.info("Normalizing features...")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # Create scaler
    scaler = StandardScaler()

    # Scale numeric columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def main():
    """Main execution function."""
    try:
        # Create features directory
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        # Load and clean data
        df = load_raw_data()
        df = clean_data(df)

        # Feature engineering
        df = calculate_basic_stats(df)
        df = add_rolling_averages(df)
        df = add_advanced_metrics(df)
        df = normalize_features(df)

        # Save features
        output_file = (
            FEATURES_DIR
            / f"nba_player_stats_features_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        df.to_csv(output_file, index=False)
        logger.info(f"Saved features to {output_file}")

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


if __name__ == "__main__":
    main()
