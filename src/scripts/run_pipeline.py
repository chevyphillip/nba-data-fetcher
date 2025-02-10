"""
NBA Stats Pipeline

This script runs the complete pipeline:
1. Fetch NBA data
2. Clean and preprocess data
3. Engineer features
4. Train models
5. Analyze props
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Local imports
from src.scripts.data_collection.nba_historical_stats_fetcher import fetch_nba_stats
from src.scripts.preprocessing.clean_raw_data import clean_data, main as clean_data_main
from src.scripts.preprocessing.feature_engineering import main as engineer_features
from src.scripts.modeling.train_and_save_models import train_and_save_models
from src.scripts.analysis.prop_analyzer import PropAnalyzer
from src.scripts.odds.odds_api import OddsAPI
from src.scripts.run_odds_analysis import PropsAnalysisRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "src/data/raw",
        "src/data/cleaned",
        "src/data/features",
        "src/models",
        "src/data/analysis",
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def run_data_collection():
    """Run the data collection step."""
    logger.info("Starting data collection...")
    try:
        fetch_nba_stats()  # Using the imported function directly
        logger.info("Data collection completed successfully")
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        raise


def run_data_cleaning():
    """Run the data cleaning step."""
    logger.info("Starting data cleaning...")
    try:
        clean_data_main()  # Using the imported main function
        logger.info("Data cleaning completed successfully")
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise


def run_feature_engineering():
    """Run the feature engineering step."""
    logger.info("Starting feature engineering...")
    try:
        engineer_features()
        logger.info("Feature engineering completed successfully")
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


def run_model_training():
    """Run the model training step."""
    logger.info("Starting model training...")
    try:
        train_and_save_models()  # Using the imported function
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


def run_prop_analysis():
    """Run the prop analysis step."""
    logger.info("Starting prop analysis...")
    try:
        # Use the PropsAnalysisRunner class
        runner = PropsAnalysisRunner()
        success = runner.run()

        if success:
            logger.info("Prop analysis completed successfully")
        else:
            logger.error("Prop analysis failed")

    except Exception as e:
        logger.error(f"Error in prop analysis: {e}")
        raise


def load_latest_features() -> pd.DataFrame:
    """Load the latest features file."""
    features_dir = project_root / "data" / "features"
    feature_files = list(features_dir.glob("nba_player_stats_features_*.csv"))
    if not feature_files:
        raise FileNotFoundError("No feature files found")

    latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading features from {latest_file}")
    return pd.read_csv(latest_file, index_col="Player")


def main():
    """Run the complete NBA stats pipeline."""
    try:
        logger.info("Starting NBA stats pipeline...")
        setup_directories()

        # Step 1: Fetch historical stats
        run_data_collection()

        # Step 2: Clean data
        run_data_cleaning()

        # Step 3: Engineer features
        run_feature_engineering()

        # Step 4: Train models
        run_model_training()

        # Step 5: Analyze props
        run_prop_analysis()

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
