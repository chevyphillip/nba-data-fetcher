"""
Feature Engineering for NBA Player Statistics

This script focuses on creating meaningful features for NBA player prediction:
1. Rolling averages for recent performance
2. Season context features
3. Efficiency metrics
4. Position-based features
5. Opponent adjustments
6. Home/Away splits
7. Rest days impact
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent.parent
CLEANED_DIR = SCRIPT_DIR / "data" / "cleaned"
FEATURES_DIR = SCRIPT_DIR / "data" / "features"

# Configuration
ROLLING_WINDOWS = [5, 10, 15]  # Last N games windows
TARGET_STATS = {
    "PTS": {"name": "Points", "col": "PTS_per_game"},
    "TRB": {"name": "Rebounds", "col": "TRB_per_game"},
    "AST": {"name": "Assists", "col": "AST_per_game"},
    "3P": {"name": "3-Pointers Made", "col": "3P_per_game"},
}


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

    # Calculate per-minute statistics
    per_minute_stats = ["PTS", "TRB", "AST", "3P"]
    for stat in per_minute_stats:
        if stat in df_stats.columns:
            df_stats[f"{stat}_per_minute"] = df_stats[stat] / df_stats["MP"]

    # Calculate Usage Rate
    if all(
        col in df_stats.columns for col in ["FGA", "TOV", "FTA", "MP", "Team", "Season"]
    ):
        team_poss = df_stats.groupby(["Team", "Season"])["MP"].transform("sum")
        df_stats["Usage"] = (
            100
            * (
                (df_stats["FGA"] + 0.44 * df_stats["FTA"] + df_stats["TOV"])
                * (team_poss / df_stats["MP"])
            )
            / team_poss
        )

    # Calculate assist-related metrics
    if all(col in df_stats.columns for col in ["AST", "MP", "TOV", "Usage"]):
        df_stats["assists_per_minute"] = df_stats["AST"] / df_stats["MP"]
        df_stats["assist_to_usage"] = np.where(
            df_stats["Usage"] > 0, df_stats["AST"] / df_stats["Usage"], 0
        )

    return df_stats


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
    """Add rolling averages for key statistics with exponential weighting."""
    df_rolling = df.copy()

    # Sort by player and season for proper rolling calculations
    df_rolling = df_rolling.sort_values(["Player", "Season"])

    # Calculate rolling averages for each target stat
    for stat in TARGET_STATS.keys():
        col = TARGET_STATS[stat]["col"]
        if col in df_rolling.columns:
            for window in ROLLING_WINDOWS:
                # Standard rolling average
                roll_col = f"{stat}_rolling_{window}g"
                df_rolling[roll_col] = df_rolling.groupby("Player")[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

                # Exponentially weighted moving average
                ewm_col = f"{stat}_ewm_{window}g"
                df_rolling[ewm_col] = df_rolling.groupby("Player")[col].transform(
                    lambda x: x.ewm(span=window, min_periods=1).mean()
                )

                # Rolling standard deviation (for consistency metrics)
                std_col = f"{stat}_rolling_{window}g_std"
                df_rolling[std_col] = df_rolling.groupby("Player")[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )

    return df_rolling


def add_season_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add season context features."""
    df_context = df.copy()

    # COVID-impacted seasons (convert boolean to int)
    df_context["is_covid_season"] = df_context["Season"].isin([2020, 2021]).astype(int)

    # Games played relative to typical season
    df_context["games_played_ratio"] = df_context["G"] / df_context.groupby("Season")[
        "G"
    ].transform("max")

    # Season progress (0-1 scale)
    df_context["season_progress"] = df_context.groupby(
        ["Season", "Player"]
    ).cumcount() / df_context.groupby("Season")["G"].transform("max")

    # Experience (years in league)
    df_context["experience"] = df_context.groupby("Player").cumcount()

    # Season performance trends
    for stat in TARGET_STATS.keys():
        col = TARGET_STATS[stat]["col"]
        if col in df_context.columns:
            # Calculate trend (slope) over last 10 games
            df_context[f"{stat}_trend"] = df_context.groupby("Player")[col].transform(
                lambda x: pd.Series(
                    np.polyfit(range(len(x[-10:])), x[-10:], 1)[0]
                    if len(x) >= 10
                    else 0
                )
            )

    return df_context


def add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced efficiency metrics."""
    df_eff = df.copy()

    try:
        # Scoring efficiency
        if all(col in df_eff.columns for col in ["PTS", "FGA", "FTA"]):
            # True Shooting Percentage (TS%)
            df_eff["true_shooting_pct"] = np.where(
                (df_eff["FGA"] + 0.44 * df_eff["FTA"]) > 0,
                df_eff["PTS"] / (2 * (df_eff["FGA"] + 0.44 * df_eff["FTA"])),
                0,
            )

            # Points Per Shot Attempt
            df_eff["points_per_shot"] = np.where(
                df_eff["FGA"] > 0, df_eff["PTS"] / df_eff["FGA"], 0
            )

            # Free Throw Rate
            df_eff["ft_rate"] = np.where(
                df_eff["FGA"] > 0, df_eff["FTA"] / df_eff["FGA"], 0
            )

        # Usage rate with team possession estimate
        if all(
            col in df_eff.columns
            for col in ["FGA", "TOV", "FTA", "MP", "Team", "Season"]
        ):
            # Estimate team possessions
            df_eff["team_poss"] = df_eff.groupby(["Team", "Season"])["MP"].transform(
                "sum"
            )

            # Advanced Usage Rate calculation
            df_eff["usage_rate"] = np.where(
                df_eff["MP"] > 0,
                100
                * (
                    (df_eff["FGA"] + 0.44 * df_eff["FTA"] + df_eff["TOV"])
                    * (df_eff["team_poss"] / df_eff["MP"])
                )
                / df_eff.groupby(["Team", "Season"])["MP"].transform("sum"),
                0,
            )
            df_eff.drop("team_poss", axis=1, inplace=True)

        # Assist metrics
        if all(col in df_eff.columns for col in ["AST", "TOV", "MP", "FGA"]):
            # Assist to Turnover Ratio
            df_eff["ast_to_tov"] = np.where(
                df_eff["TOV"] > 0,
                df_eff["AST"] / df_eff["TOV"],
                df_eff["AST"],  # If no turnovers, use raw assists
            )

            # Assist Ratio (percentage of possessions ending in assist)
            df_eff["assist_ratio"] = np.where(
                (df_eff["AST"] + df_eff["FGA"]) > 0,
                100 * df_eff["AST"] / (df_eff["AST"] + df_eff["FGA"]),
                0,
            )

        # Shooting efficiency
        if all(
            col in df_eff.columns for col in ["3PA", "FGA", "3P", "FG", "2P", "2PA"]
        ):
            # Three Point Rate
            df_eff["three_point_rate"] = np.where(
                df_eff["FGA"] > 0, df_eff["3PA"] / df_eff["FGA"], 0
            )

            # Effective Field Goal Percentage
            df_eff["effective_fg_pct"] = np.where(
                df_eff["FGA"] > 0,
                (df_eff["FG"] + 0.5 * df_eff["3P"]) / df_eff["FGA"],
                0,
            )

            # Two Point Percentage
            df_eff["two_point_pct"] = np.where(
                df_eff["2PA"] > 0, df_eff["2P"] / df_eff["2PA"], 0
            )

        # Rebounding rates
        if all(
            col in df_eff.columns
            for col in ["ORB", "DRB", "TRB", "MP", "Team", "Season"]
        ):
            # Team totals
            team_orb = df_eff.groupby(["Team", "Season"])["ORB"].transform("sum")
            team_drb = df_eff.groupby(["Team", "Season"])["DRB"].transform("sum")

            # Rebounding Percentages
            df_eff["orb_pct"] = np.where(
                team_orb > 0,
                100
                * (
                    df_eff["ORB"]
                    * (
                        df_eff.groupby(["Team", "Season"])["MP"].transform("sum")
                        / df_eff["MP"]
                    )
                )
                / team_orb,
                0,
            )
            df_eff["drb_pct"] = np.where(
                team_drb > 0,
                100
                * (
                    df_eff["DRB"]
                    * (
                        df_eff.groupby(["Team", "Season"])["MP"].transform("sum")
                        / df_eff["MP"]
                    )
                )
                / team_drb,
                0,
            )

        # Clean up any potential infinities or NaNs
        numeric_cols = df_eff.select_dtypes(include=[np.number]).columns
        df_eff[numeric_cols] = df_eff[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_eff[numeric_cols] = df_eff[numeric_cols].fillna(0)

        logger.info("Successfully added efficiency metrics")
        return df_eff

    except Exception as e:
        logger.error(f"Error in add_efficiency_metrics: {e}")
        raise


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add position-based features."""
    df_pos = df.copy()

    # Calculate position-specific averages and percentiles
    for stat in TARGET_STATS.keys():
        col = TARGET_STATS[stat]["col"]
        if col in df_pos.columns:
            # Calculate league average for each position
            pos_avgs = df_pos.groupby(["Season", "Pos"])[col].transform("mean")
            pos_stds = df_pos.groupby(["Season", "Pos"])[col].transform("std")

            # Calculate how player performs vs. position average (z-score)
            df_pos[f"{stat}_vs_pos_zscore"] = (df_pos[col] - pos_avgs) / pos_stds

            # Calculate percentile rank within position
            df_pos[f"{stat}_pos_percentile"] = (
                df_pos.groupby(["Season", "Pos"])[col].transform(
                    lambda x: pd.qcut(x, q=100, labels=False, duplicates="drop")
                )
                / 100
            )

    return df_pos


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features related to opponent and matchup context."""
    df_matchup = df.copy()

    # Calculate opponent defensive ratings
    for stat in TARGET_STATS.keys():
        col = TARGET_STATS[stat]["col"]
        if col in df_matchup.columns:
            # Calculate average stats allowed by each team
            team_defense = df_matchup.groupby(["Season", "Team"])[col].transform("mean")
            league_avg = df_matchup.groupby("Season")[col].transform("mean")

            # Calculate defensive rating (higher means team allows more points)
            df_matchup[f"opp_def_rating_{stat}"] = team_defense / league_avg

    return df_matchup


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features to appropriate scales."""
    df_norm = df.copy()

    # Normalize per-game stats by minutes played
    per_game_cols = [col for col in df_norm.columns if "per_game" in col]
    if "MPG" in df_norm.columns:
        per_minute_cols = {col: f"{col}_per_minute" for col in per_game_cols}
        df_norm = pd.concat(
            [
                df_norm,
                pd.DataFrame(
                    {
                        new_col: df_norm[col] / df_norm["MPG"]
                        for col, new_col in per_minute_cols.items()
                    },
                    index=df_norm.index,
                ),
            ],
            axis=1,
        )

    # Scale rolling averages relative to season averages
    rolling_cols = [col for col in df_norm.columns if "rolling" in col]
    vs_avg_cols = {}
    for col in rolling_cols:
        base_stat = col.split("_rolling_")[0]
        if f"{base_stat}_per_game" in df_norm.columns:
            season_avg = df_norm.groupby("Season")[f"{base_stat}_per_game"].transform(
                "mean"
            )
            vs_avg_cols[f"{col}_vs_avg"] = df_norm[col] - season_avg

    if vs_avg_cols:
        df_norm = pd.concat(
            [df_norm, pd.DataFrame(vs_avg_cols, index=df_norm.index)], axis=1
        )

    # Z-score normalization for all numeric features
    numeric_cols = df_norm.select_dtypes(include=["float64", "float32"]).columns
    zscore_cols = {}
    for col in numeric_cols:
        mean = df_norm[col].mean()
        std = df_norm[col].std()
        if std > 0:  # Avoid division by zero
            zscore_cols[f"{col}_zscore"] = (df_norm[col] - mean) / std

    if zscore_cols:
        df_norm = pd.concat(
            [df_norm, pd.DataFrame(zscore_cols, index=df_norm.index)], axis=1
        )

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
        df = calculate_basic_stats(df)
        df = add_rolling_averages(df)
        df = add_season_context(df)
        df = add_efficiency_metrics(df)
        df = add_position_features(df)
        df = add_matchup_features(df)
        df = normalize_features(df)

        # Save engineered features
        current_date = datetime.now().strftime("%Y%m%d")
        output_file = FEATURES_DIR / f"nba_player_stats_features_{current_date}.csv"
        df.to_csv(output_file, index=False)

        final_shape = df.shape
        logger.info(f"Final data shape: {final_shape}")
        logger.info(f"Engineered features saved to {output_file}")

        # Log feature summary
        numeric_features = df.select_dtypes(
            include=["float32", "float64", "int32", "int64"]
        ).columns
        logger.info(f"\nFeature Summary:")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Numeric features: {len(numeric_features)}")
        logger.info(f"Categorical features: {len(df.columns) - len(numeric_features)}")

        # Log most important features
        logger.info("\nKey Feature Groups:")
        logger.info("1. Rolling Averages and Trends")
        logger.info("2. Position-Based Performance")
        logger.info("3. Efficiency Metrics")
        logger.info("4. Matchup-Based Features")
        logger.info("5. Normalized Statistics")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
