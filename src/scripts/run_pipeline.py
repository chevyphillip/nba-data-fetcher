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
        stats_fetcher = NBAStatsFetcher()
        stats_fetcher.fetch_stats()
        logger.info("Data collection completed successfully")
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        raise


def run_data_cleaning():
    """Run the data cleaning step."""
    logger.info("Starting data cleaning...")
    try:
        clean_data()
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
        train_models()
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


def run_prop_analysis():
    """Run the prop analysis step."""
    logger.info("Starting prop analysis...")
    try:
        # Load player features
        features_dir = Path("src/data/features")
        latest_features = max(
            features_dir.glob("nba_player_stats_features_*.csv"),
            key=lambda x: x.stat().st_mtime,
        )
        features_df = pd.read_csv(latest_features, index_col="Player")

        # Initialize APIs and analyzer
        odds_api = OddsAPI()
        prop_analyzer = PropAnalyzer()

        # Fetch props
        logger.info("Fetching props from OddsAPI...")
        props = odds_api.fetch_props()
        odds_api.save_props(props)

        # Analyze props
        logger.info("Analyzing props...")
        analyzed_props = prop_analyzer.analyze_props(props, features_df)

        # Find best edges
        logger.info("Finding best edges...")
        best_props = prop_analyzer.find_best_edges(analyzed_props, min_edge=5.0)

        # Save analysis
        logger.info("Saving analysis...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = project_root / "data" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_file = analysis_dir / f"prop_analysis_{timestamp}.json"

        with open(analysis_file, "w") as f:
            json.dump(best_props, f, indent=2)
        logger.info(f"Analysis saved to {analysis_file}")

        # Print results
        logger.info("\nTop 10 Props with Best Edges:")
        logger.info("-" * 100)
        for prop in best_props[:10]:
            logger.info(
                f"{prop['player_name']} - {prop['market']} - "
                f"Line: {prop['line']}, Prediction: {prop['prediction']:.1f}, "
                f"Price: {prop['price']:+d}, Edge: {prop['edge']:.1f}%, "
                f"Confidence: {prop['confidence']:.2f}, Score: {prop['score']:.1f}"
            )

        # Print summary statistics
        total_props = len(analyzed_props)
        props_with_edge = len([p for p in analyzed_props if abs(p["edge"]) >= 5.0])
        avg_edge = sum(abs(p["edge"]) for p in analyzed_props) / total_props

        logger.info(f"\nSummary:")
        logger.info(f"Total props analyzed: {total_props}")
        logger.info(
            f"Props with edge >= 5%: {props_with_edge} ({props_with_edge/total_props*100:.1f}%)"
        )
        logger.info(f"Average absolute edge: {avg_edge:.1f}%")

        logger.info("Prop analysis completed successfully")
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

        # Step 1: Fetch historical stats
        logger.info("Fetching historical stats...")
        master_df = fetch_nba_stats()
        if master_df is not None:
            filename = f"nba_player_stats_{datetime.now().strftime('%Y%m%d')}.csv"
            raw_dir = project_root / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            master_df.to_csv(raw_dir / filename, index=False)

        # Step 2: Clean data
        logger.info("Cleaning data...")
        clean_data_main()  # This will load the raw data, clean it, and save it

        # Step 3: Engineer features
        logger.info("Engineering features...")
        engineer_features()

        # Step 4: Train models
        logger.info("Training models...")
        train_and_save_models()

        # Step 5: Analyze props
        logger.info("Analyzing props...")

        # Load latest features
        player_features = load_latest_features()

        # Initialize components
        odds_api = OddsAPI()
        prop_analyzer = PropAnalyzer()

        # Fetch and analyze props
        props = odds_api.get_all_props()
        if props:
            analyzed_props = prop_analyzer.analyze_props(props, player_features)
            logger.info(f"Found {len(analyzed_props)} props with potential edges")

            # Save analysis results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_dir = project_root / "data" / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            analysis_file = analysis_dir / f"prop_analysis_{timestamp}.json"

            with open(analysis_file, "w") as f:
                json.dump(analyzed_props, f, indent=2)
            logger.info(f"Analysis saved to {analysis_file}")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
