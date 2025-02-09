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
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from data_collection.nba_historical_stats_fetcher import fetch_nba_stats, save_data
from preprocessing.clean_raw_data import main as clean_data
from preprocessing.feature_engineering import main as engineer_features
from modeling.train_and_save_models import train_and_save_models
from analysis.prop_analyzer import PropAnalyzer
from odds.odds_api import OddsAPI

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
        master_df = fetch_nba_stats()
        if master_df is not None:
            filename = f"nba_player_stats_{datetime.now().strftime('%Y%m%d')}.csv"
            save_data(master_df, filename)
            logger.info("Data collection completed successfully")
        else:
            raise Exception("No data collected")
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
        train_and_save_models()
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
        analyzer = PropAnalyzer()

        # Fetch props
        logger.info("Fetching props from OddsAPI...")
        props = odds_api.get_all_props()
        odds_api.save_props(props)

        # Analyze props
        logger.info("Analyzing props...")
        analyzed_props = analyzer.analyze_props(props, features_df)

        # Find best edges
        logger.info("Finding best edges...")
        best_props = analyzer.find_best_edges(analyzed_props, min_edge=5.0)

        # Save analysis
        analyzer.save_analysis(best_props)

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


def main():
    """Run the complete pipeline."""
    try:
        # Create necessary directories
        setup_directories()

        # Run pipeline steps
        run_data_collection()
        run_data_cleaning()
        run_feature_engineering()
        run_model_training()
        run_prop_analysis()

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
