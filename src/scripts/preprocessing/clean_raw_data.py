"""
Clean and preprocess raw NBA statistics data.

This script performs comprehensive cleaning and preprocessing of raw NBA statistics:
1. Data validation and quality checks
2. Outlier detection and handling
3. Missing value imputation
4. Data type conversion and standardization
5. Feature consistency checks
6. Position standardization and encoding
7. Feature normalization and scaling
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = SCRIPT_DIR / "data" / "raw"
CLEANED_DIR = SCRIPT_DIR / "data" / "cleaned"

# Configuration
VALID_POSITIONS = {"PG", "SG", "SF", "PF", "C", "G", "F", "G-F", "F-G", "F-C", "C-F"}
REQUIRED_COLUMNS = ["Player", "Team", "Season", "G", "MP"]

# Define target-specific feature groups
POINTS_FEATURES = [
    "PTS",
    "FGA",
    "FTA",
    "USG%",
    "TS%",
    "eFG%",  # Core scoring
    "FG%",
    "3P%",
    "FT%",  # Shooting splits
    "MP",
    "GS",  # Playing time
    "Team_pace",
    "Relative_team_pace",  # Team context
]

REBOUNDS_FEATURES = [
    "TRB",
    "ORB",
    "DRB",  # Core rebounding
    "ORB%",
    "DRB%",
    "TRB%",  # Rebounding percentages
    "MP",
    "GS",  # Playing time
    "Team_reb_rate",
    "Relative_team_reb_rate",  # Team context
]

ASSISTS_FEATURES = [
    "AST",
    "AST%",
    "TOV",
    "TOV%",  # Core playmaking
    "USG%",
    "MP",
    "GS",  # Usage and playing time
    "Team_ast_rate",
    "Relative_team_ast_rate",  # Team context
]

THREES_FEATURES = [
    "3P",
    "3PA",
    "3P%",  # Core 3-point shooting
    "eFG%",
    "MP",
    "GS",  # Efficiency and playing time
    "Team_pace",
    "Relative_team_pace",  # Team context
]

BLOCKS_FEATURES = [
    "BLK",
    "BLK%",  # Core blocking
    "MP",
    "GS",  # Playing time
    "Team_blk_rate",
    "Relative_team_blk_rate",  # Team context
]

STEALS_FEATURES = [
    "STL",
    "STL%",  # Core stealing
    "MP",
    "GS",  # Playing time
    "Team_stl_rate",
    "Relative_team_stl_rate",  # Team context
]

# Update NUMERIC_STATS to include new team context features
NUMERIC_STATS = [
    "PTS",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "FG",
    "FGA",
    "3P",
    "3PA",
    "2P",
    "2PA",
    "FT",
    "FTA",
    "ORB",
    "DRB",
    "MP",
    "GS",
    "Team_pace",
    "Team_reb_rate",
    "Team_ast_rate",
    "Team_blk_rate",
    "Team_stl_rate",
]

# Update PCT_COLUMNS to include relative team rates
PCT_COLUMNS = [
    "FG%",
    "3P%",
    "2P%",
    "FT%",
    "TS%",
    "eFG%",
    "ORB%",
    "DRB%",
    "TRB%",
    "AST%",
    "STL%",
    "BLK%",
    "TOV%",
    "USG%",
    "Relative_team_pace",
    "Relative_team_reb_rate",
    "Relative_team_ast_rate",
    "Relative_team_blk_rate",
    "Relative_team_stl_rate",
]

# Define column groups for better organization
PLAYER_INFO = ["Player", "Team", "Pos", "Age", "Season"]
PLAYING_TIME = ["G", "GS", "MP"]
CORE_STATS = ["PTS", "AST", "TRB", "STL", "BLK", "TOV", "PF"]
SHOOTING_STATS = [
    "FG",
    "FGA",
    "FG%",
    "3P",
    "3PA",
    "3P%",
    "2P",
    "2PA",
    "2P%",
    "eFG%",
    "FT",
    "FTA",
    "FT%",
]
ADVANCED_STATS = [
    "PER",
    "TS%",
    "3PAr",
    "FTr",
    "ORB%",
    "DRB%",
    "TRB%",
    "AST%",
    "STL%",
    "BLK%",
    "TOV%",
    "USG%",
    "OWS",
    "DWS",
    "WS",
    "WS/48",
    "OBPM",
    "DBPM",
    "BPM",
    "VORP",
    "ORtg",
    "DRtg",
]

# Columns to drop
COLUMNS_TO_DROP = [
    "Rk",
    "Awards",
    "Tab",
    "FetchDate",
    "Trp-Dbl",
    "Heaves_Att.",
    "Heaves_Md.",
]


def load_raw_data() -> pd.DataFrame:
    """Load the most recent raw data file."""
    try:
        raw_files = list(RAW_DIR.glob("nba_player_stats_*.csv"))
        if not raw_files:
            raise FileNotFoundError("No raw data files found")

        latest_file = max(raw_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading raw data from {latest_file}")
        df = pd.read_csv(
            latest_file, low_memory=False
        )  # Prevent mixed type inference warnings

        # Validate required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        raise


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform comprehensive data validation and quality checks."""
    logger.info("Performing data validation...")
    df_validated = df.copy()
    validation_warnings = []

    try:
        # 1. Basic Range Validations
        # Check for invalid seasons
        current_year = datetime.now().year
        invalid_seasons = df_validated[
            ~df_validated["Season"].between(1946, current_year)
        ]
        if not invalid_seasons.empty:
            msg = f"Found {len(invalid_seasons)} rows with invalid seasons"
            logger.warning(msg)
            validation_warnings.append(msg)
            df_validated = df_validated[
                df_validated["Season"].between(1946, current_year)
            ]

        # 2. Logical Validations
        # Games played validation
        if "G" in df_validated.columns:
            invalid_games = df_validated[
                (df_validated["G"] < 0) | (df_validated["G"] > 82)
            ]
            if not invalid_games.empty:
                msg = f"Found {len(invalid_games)} rows with invalid games played"
                logger.warning(msg)
                validation_warnings.append(msg)
                df_validated = df_validated[
                    (df_validated["G"] >= 0) & (df_validated["G"] <= 82)
                ]

        # Minutes played validation
        if "MP" in df_validated.columns:
            invalid_minutes = df_validated[
                (df_validated["MP"] < 0) | (df_validated["MP"] > 48 * df_validated["G"])
            ]
            if not invalid_minutes.empty:
                msg = f"Found {len(invalid_minutes)} rows with invalid minutes played"
                logger.warning(msg)
                validation_warnings.append(msg)
                df_validated = df_validated[
                    (df_validated["MP"] >= 0)
                    & (df_validated["MP"] <= 48 * df_validated["G"])
                ]

        # 3. Statistical Validations
        # Shooting percentages validation
        pct_columns = [col for col in df_validated.columns if col.endswith("%")]
        for col in pct_columns:
            invalid_pct = df_validated[
                (df_validated[col] < 0) | (df_validated[col] > 1)
            ]
            if not invalid_pct.empty:
                msg = f"Found {len(invalid_pct)} rows with invalid {col}"
                logger.warning(msg)
                validation_warnings.append(msg)
                df_validated.loc[df_validated[col] < 0, col] = 0
                df_validated.loc[df_validated[col] > 1, col] = 1

        # 4. Consistency Checks
        # Field goal consistency
        if all(col in df_validated.columns for col in ["FG", "2P", "3P"]):
            inconsistent_fg = df_validated[
                abs(df_validated["FG"] - (df_validated["2P"] + df_validated["3P"]))
                > 0.01
            ]
            if not inconsistent_fg.empty:
                msg = f"Found {len(inconsistent_fg)} rows with inconsistent field goal numbers"
                logger.warning(msg)
                validation_warnings.append(msg)
                # Fix inconsistency by using sum of 2P and 3P
                df_validated["FG"] = df_validated["2P"] + df_validated["3P"]

        # 5. Missing Value Analysis
        missing_data = df_validated[REQUIRED_COLUMNS].isnull().sum()
        if missing_data.any():
            for col, count in missing_data.items():
                if count > 0:
                    msg = f"Found {count} missing values in required column {col}"
                    logger.warning(msg)
                    validation_warnings.append(msg)

        # 6. Duplicate Check
        duplicates = df_validated.duplicated(
            subset=["Player", "Team", "Season"], keep=False
        )
        if duplicates.any():
            msg = f"Found {duplicates.sum()} duplicate player-team-season entries"
            logger.warning(msg)
            validation_warnings.append(msg)

        # 7. Data Type Validation
        expected_types = {
            "Player": np.dtype("O"),  # object type
            "Team": np.dtype("O"),  # object type
            "Season": np.dtype("int64"),
            "G": np.dtype("int64"),
            "MP": np.dtype("float64"),
        }
        for col, expected_type in expected_types.items():
            if col in df_validated.columns:
                current_type = df_validated[col].dtype
                if current_type != expected_type:
                    msg = f"Column {col} has type {current_type}, expected {expected_type}"
                    logger.warning(msg)
                    validation_warnings.append(msg)
                    try:
                        df_validated[col] = df_validated[col].astype(expected_type)
                    except Exception as e:
                        logger.error(f"Could not convert {col} to {expected_type}: {e}")

        # Log summary of validation results
        if validation_warnings:
            logger.warning("Validation completed with warnings:")
            for warning in validation_warnings:
                logger.warning(f"- {warning}")
        else:
            logger.info("Validation completed successfully with no warnings")

        return df_validated

    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        raise


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using appropriate scaling methods:
    - Standard scaling for raw counting stats
    - Min-max scaling for percentage stats
    - Robust scaling for advanced metrics
    """
    logging.info("Normalizing features...")

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Standard scaling for raw counting stats
    standard_scale_features = [
        "MP",
        "PTS",
        "TRB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "FG",
    ]

    # Min-max scaling for percentage stats
    minmax_scale_features = [
        "FG%",
        "3P%",
        "2P%",
        "FT%",
        "eFG%",
        "Team_pace",
        "Relative_team_pace",
        "Home_court_FG%",
        "Home_court_3P%",
        "Home_court_FT%",
        "Home_court_PTS",
        "Home_court_TRB",
        "Home_court_AST",
        "Home_court_STL",
        "Home_court_BLK",
    ]

    # Robust scaling for advanced metrics and rates
    robust_scale_features = [
        "PER",
        "TS%",
        "3PAr",
        "FTr",
        "ORB%",
        "DRB%",
        "TRB%",
        "AST%",
        "STL%",
        "BLK%",
        "TOV%",
        "USG%",
        "OWS",
        "DWS",
        "WS",
        "WS/48",
        "OBPM",
        "DBPM",
        "BPM",
        "VORP",
        "ORtg",
        "DRtg",
    ]

    # Filter features that exist in the DataFrame
    standard_scale_features = [f for f in standard_scale_features if f in df.columns]
    minmax_scale_features = [f for f in minmax_scale_features if f in df.columns]
    robust_scale_features = [f for f in robust_scale_features if f in df.columns]

    # Handle extreme values before scaling
    def clip_extreme_values(data):
        if data.dtype in [np.float64, np.float32]:
            q1 = data.quantile(0.01)
            q3 = data.quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            return data.clip(lower_bound, upper_bound)
        return data

    # Apply clipping to numeric features
    for features in [
        standard_scale_features,
        minmax_scale_features,
        robust_scale_features,
    ]:
        for col in features:
            if col in df.columns and df[col].dtype in [np.float64, np.float32]:
                df[col] = clip_extreme_values(df[col])

    if standard_scale_features:
        scaler = StandardScaler()
        df[standard_scale_features] = scaler.fit_transform(df[standard_scale_features])
        logging.info(
            f"Applied standard scaling to {len(standard_scale_features)} features"
        )

    if minmax_scale_features:
        scaler = MinMaxScaler()
        df[minmax_scale_features] = scaler.fit_transform(df[minmax_scale_features])
        logging.info(
            f"Applied min-max scaling to {len(minmax_scale_features)} features"
        )

    if robust_scale_features:
        scaler = RobustScaler()
        df[robust_scale_features] = scaler.fit_transform(df[robust_scale_features])
        logging.info(f"Applied robust scaling to {len(robust_scale_features)} features")

    return df


def detect_outliers(
    df: pd.DataFrame, columns: List[str], n_std: float = 3
) -> pd.DataFrame:
    """Detect and handle outliers using robust statistics."""
    df_clean = df.copy()

    for col in columns:
        if col in df.columns and df[col].dtype in ["int64", "float64"]:
            # Use RobustScaler for outlier detection
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(df[[col]])
            outliers = np.abs(scaled_values) > n_std

            if outliers.any():
                n_outliers = outliers.sum()
                logger.warning(f"Found {n_outliers} outliers in {col}")

                # Calculate robust statistics
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Cap outliers at bounds
                df_clean.loc[df[col] < lower_bound, col] = lower_bound
                df_clean.loc[df[col] > upper_bound, col] = upper_bound

                logger.info(
                    f"Capped {col} outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]"
                )

    return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using rolling averages and smart imputation."""
    logger.info("Handling missing values...")

    # Calculate missing value percentages
    missing_pct = df.isnull().sum() / len(df) * 100
    for col in df.columns:
        if missing_pct[col] > 0:
            logger.info(f"{col}: {missing_pct[col]:.2f}% missing")

    # Sort by Player and Season for proper rolling calculations
    df = df.sort_values(["Player", "Season"])

    # Define windows for rolling averages
    windows = [3, 5, 10]  # Last 3, 5, and 10 games/seasons

    # Stats to apply rolling averages
    rolling_stats = [
        "PTS",
        "TRB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "FG",
        "FGA",
        "FG%",
        "3P",
        "3PA",
        "3P%",
        "2P",
        "2PA",
        "2P%",
        "FT",
        "FTA",
        "FT%",
        "TS%",
        "eFG%",
        "ORB%",
        "DRB%",
        "TRB%",
        "AST%",
        "STL%",
        "BLK%",
        "TOV%",
        "USG%",
    ]

    # Calculate rolling averages for each window
    for stat in rolling_stats:
        if stat in df.columns:
            for window in windows:
                roll_col = f"{stat}_rolling_{window}"
                # Calculate rolling mean within each player group
                df[roll_col] = df.groupby("Player")[stat].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

    # Fill missing values with the most recent rolling average available
    for stat in rolling_stats:
        if stat in df.columns:
            mask = df[stat].isna()
            if mask.any():
                # Try different window sizes in order
                for window in windows:
                    roll_col = f"{stat}_rolling_{window}"
                    df.loc[mask, stat] = df.loc[mask, roll_col]
                    mask = df[stat].isna()  # Update mask

                # If still missing, use position-based median
                if mask.any():
                    df.loc[mask, stat] = (
                        df.loc[mask].groupby("Pos")[stat].transform("median")
                    )

                # If still missing (new position), use overall median
                df[stat] = df[stat].fillna(df[stat].median())

    # Clean up rolling columns we don't need anymore
    roll_cols = [col for col in df.columns if "_rolling_" in col]
    df = df.drop(columns=roll_cols)

    return df


def convert_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types with validation."""
    logger.info("Converting data types...")

    # Convert season to int with validation
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")

    # Convert numeric columns to appropriate types
    for col in df.select_dtypes(include=["float64"]).columns:
        if col in NUMERIC_STATS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        elif col in PCT_COLUMNS:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").clip(0, 1).astype("float32")
            )

    # Convert categorical columns
    categorical_columns = ["Player", "Team", "Pos"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def clean_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize position data with validation."""
    logger.info("Cleaning position data...")

    # Validate positions
    invalid_positions = set(df["Pos"].unique()) - VALID_POSITIONS
    if invalid_positions:
        logger.warning(f"Found invalid positions: {invalid_positions}")

    # Map positions to standard format
    pos_map = {
        "PG": "PG",
        "G": "PG",
        "SG": "SG",
        "G-F": "SG",
        "SF": "SF",
        "F": "SF",
        "F-G": "SF",
        "PF": "PF",
        "F-C": "PF",
        "C": "C",
        "C-F": "C",
    }

    df["Pos"] = df["Pos"].map(pos_map).fillna("SF")  # Default to SF if unknown

    # Create position dummy variables
    pos_dummies = pd.get_dummies(df["Pos"], prefix="Pos")
    df = pd.concat([df, pos_dummies], axis=1)

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataset."""
    logger.info("Adding derived features...")

    # Store categorical columns
    categorical_cols = df.select_dtypes(include=["category"]).columns

    # Convert categorical to string temporarily
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # Fill missing values
    df = df.fillna(0)

    # Convert back to category
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    # Calculate per-game statistics
    per_game_cols = [
        "PTS",
        "TRB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "FG",
        "FGA",
        "3P",
        "3PA",
        "2P",
        "2PA",
        "FT",
        "FTA",
        "ORB",
        "DRB",
    ]
    for col in per_game_cols:
        if col in df.columns:
            df[f"{col}_per_game"] = df[col] / df["G"]

    # Calculate advanced statistics
    if all(col in df.columns for col in ["FGA", "FTA", "PTS"]):
        # True Shooting Percentage (TS%)
        df["TS%"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"]))

    if all(col in df.columns for col in ["3PA", "FGA"]):
        # Three Point Attempt Rate (3PAr)
        df["3PAr"] = df["3PA"] / df["FGA"]

    if all(col in df.columns for col in ["FTA", "FGA"]):
        # Free Throw Attempt Rate (FTr)
        df["FTr"] = df["FTA"] / df["FGA"]

    # Usage Rate (USG%)
    if all(col in df.columns for col in ["FGA", "TOV", "FTA", "MP"]):
        df["USG%"] = (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / df["MP"]

    # Offensive and Defensive Rebound Rates
    if all(col in df.columns for col in ["ORB", "DRB", "MP"]):
        df["ORB%"] = df["ORB"] / df["MP"]
        df["DRB%"] = df["DRB"] / df["MP"]
        df["TRB%"] = (df["ORB"] + df["DRB"]) / df["MP"]

    # Assist Rate
    if all(col in df.columns for col in ["AST", "MP"]):
        df["AST%"] = df["AST"] / df["MP"]

    # Block and Steal Rates
    if "BLK" in df.columns and "MP" in df.columns:
        df["BLK%"] = df["BLK"] / df["MP"]
    if "STL" in df.columns and "MP" in df.columns:
        df["STL%"] = df["STL"] / df["MP"]

    # Turnover Rate
    if all(col in df.columns for col in ["TOV", "FGA", "FTA"]):
        df["TOV%"] = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries with logging."""
    logger.info("Removing duplicates...")

    before_count = len(df)
    df = df.drop_duplicates(subset=["Player", "Team", "Season"], keep="last")
    after_count = len(df)

    if before_count > after_count:
        logger.info(f"Removed {before_count - after_count} duplicate entries")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-essential columns and organize features."""
    logger.info("Cleaning and organizing features...")

    # Drop non-essential columns
    columns_to_drop = (
        COLUMNS_TO_DROP
        + [col for col in df.columns if col.startswith("Unnamed")]
        + [col for col in df.columns if "Heaves" in col]
    )
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Log retained columns by category
    logger.info("\nRetained columns by category:")
    logger.info(f"Player Info: {[col for col in PLAYER_INFO if col in df.columns]}")
    logger.info(f"Playing Time: {[col for col in PLAYING_TIME if col in df.columns]}")
    logger.info(f"Core Stats: {[col for col in CORE_STATS if col in df.columns]}")
    logger.info(
        f"Shooting Stats: {[col for col in SHOOTING_STATS if col in df.columns]}"
    )
    logger.info(
        f"Advanced Stats: {[col for col in ADVANCED_STATS if col in df.columns]}"
    )

    return df


def add_team_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add team context features with validation."""
    logger.info("Adding team context features...")

    # Calculate team-level statistics
    team_stats = (
        df.groupby(["Team", "Season"])
        .agg(
            {
                "PTS": "sum",
                "TRB": "sum",
                "AST": "sum",
                "BLK": "sum",
                "STL": "sum",
                "MP": "sum",
                "G": "max",  # Use max games for the team
            }
        )
        .reset_index()
    )

    # Calculate team rates per game
    team_stats["Team_pace"] = team_stats["PTS"] / team_stats["G"]
    team_stats["Team_reb_rate"] = team_stats["TRB"] / team_stats["G"]
    team_stats["Team_ast_rate"] = team_stats["AST"] / team_stats["G"]
    team_stats["Team_blk_rate"] = team_stats["BLK"] / team_stats["G"]
    team_stats["Team_stl_rate"] = team_stats["STL"] / team_stats["G"]

    # Calculate league averages per season
    league_avgs = (
        team_stats.groupby("Season")
        .agg(
            {
                "Team_pace": "mean",
                "Team_reb_rate": "mean",
                "Team_ast_rate": "mean",
                "Team_blk_rate": "mean",
                "Team_stl_rate": "mean",
            }
        )
        .add_prefix("League_avg_")
    )

    # Merge team stats back to player data
    df = df.merge(
        team_stats[
            [
                "Team",
                "Season",
                "Team_pace",
                "Team_reb_rate",
                "Team_ast_rate",
                "Team_blk_rate",
                "Team_stl_rate",
            ]
        ],
        on=["Team", "Season"],
        how="left",
    )

    # Merge league averages
    df = df.merge(league_avgs, on="Season", how="left")

    # Calculate relative team rates (compared to league average)
    for stat in ["pace", "reb_rate", "ast_rate", "blk_rate", "stl_rate"]:
        team_col = f"Team_{stat}"
        league_col = f"League_avg_Team_{stat}"
        relative_col = f"Relative_team_{stat}"

        # Calculate relative rate and clip to reasonable range
        df[relative_col] = (df[team_col] / df[league_col].clip(lower=0.1)).clip(
            0.5, 2.0
        )

    logger.info("Added team context features successfully")
    return df


def add_home_court_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add home court advantage features."""
    logger.info("Adding home court advantage features...")

    # Calculate team home court advantage
    team_stats = (
        df.groupby(["Team", "Season"])
        .agg(
            {
                "PTS": "mean",
                "FG%": "mean",
                "3P%": "mean",
                "FT%": "mean",
                "TRB": "mean",
                "AST": "mean",
                "STL": "mean",
                "BLK": "mean",
            }
        )
        .reset_index()
    )

    # Calculate league averages
    league_avgs = (
        team_stats.groupby("Season")
        .agg(
            {
                "PTS": "mean",
                "FG%": "mean",
                "3P%": "mean",
                "FT%": "mean",
                "TRB": "mean",
                "AST": "mean",
                "STL": "mean",
                "BLK": "mean",
            }
        )
        .add_prefix("league_avg_")
    )

    # Merge league averages back
    team_stats = team_stats.merge(league_avgs, on="Season")

    # Calculate home court factors
    for stat in ["PTS", "FG%", "3P%", "FT%", "TRB", "AST", "STL", "BLK"]:
        team_stats[f"home_advantage_{stat}"] = (
            team_stats[stat] / team_stats[f"league_avg_{stat}"]
        ).clip(
            0.8, 1.2
        )  # Clip to reasonable range

    # Keep only home advantage columns
    home_advantage_cols = [
        col for col in team_stats.columns if "home_advantage_" in col
    ]
    team_stats = team_stats[["Team", "Season"] + home_advantage_cols]

    # Merge back to player data
    df = df.merge(team_stats, on=["Team", "Season"], how="left")

    # Fill any missing values with 1.0 (neutral)
    for col in home_advantage_cols:
        df[col] = df[col].fillna(1.0)

    logger.info("Added home court advantage features successfully")
    return df


def main():
    """Main execution function with comprehensive error handling."""
    try:
        # Create cleaned directory if it doesn't exist
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)

        # Load and validate raw data
        df = load_raw_data()
        initial_shape = df.shape
        logger.info(f"Initial data shape: {initial_shape}")

        # Apply cleaning steps with validation
        df = validate_data(df)
        df = clean_data(df)
        df = detect_outliers(df, NUMERIC_STATS)
        df = handle_missing_values(df)
        df = convert_datatypes(df)
        df = clean_positions(df)
        df = add_derived_features(df)
        df = add_team_context(df)
        df = add_home_court_features(df)  # Add home court features
        df = normalize_features(df)
        df = remove_duplicates(df)

        # Final validation and save
        final_shape = df.shape
        logger.info(f"Final data shape: {final_shape}")

        # Log feature statistics
        numeric_cols = df.select_dtypes(include=["float32", "float64", "int64"]).columns
        logger.info("\nFeature Statistics:")
        for col in numeric_cols:
            stats = df[col].describe()
            logger.info(f"\n{col}:")
            logger.info(f"  Mean: {stats['mean']:.2f}")
            logger.info(f"  Std: {stats['std']:.2f}")
            logger.info(f"  Min: {stats['min']:.2f}")
            logger.info(f"  Max: {stats['max']:.2f}")

        # Save cleaned data
        current_date = datetime.now().strftime("%Y%m%d")
        output_file = CLEANED_DIR / f"nba_player_stats_cleaned_{current_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved to {output_file}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
