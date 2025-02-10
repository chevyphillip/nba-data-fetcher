"""
NBA Props Analysis Runner

This script orchestrates the prop analysis workflow:
1. Fetches odds data
2. Loads player features
3. Analyzes props
4. Reports and saves results
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scripts.odds.odds_api import OddsAPI
from src.scripts.analysis.prop_analyzer import PropAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PropsAnalysisRunner:
    def __init__(self):
        """Initialize the analysis runner."""
        self.odds_api = OddsAPI()
        self.prop_analyzer = PropAnalyzer()
        self.features_df = None

    def load_features(self) -> bool:
        """Load the latest player features."""
        try:
            features_dir = Path("src/data/features")
            latest_features = max(
                features_dir.glob("nba_player_stats_features_*.csv"),
                key=lambda x: x.stat().st_mtime,
            )
            logger.info(f"Loading features from {latest_features}")
            self.features_df = pd.read_csv(latest_features, index_col="Player")
            return True
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return False

    def fetch_and_save_props(self) -> Optional[List[Dict]]:
        """Fetch props from odds API and save raw data."""
        try:
            logger.info("Fetching props from odds API...")
            props = self.odds_api.get_all_props()

            if not props:
                logger.warning("No props found to analyze")
                return None

            # Save raw props
            self.odds_api.save_props(props)
            logger.info(f"Fetched {len(props)} props")
            return props
        except Exception as e:
            logger.error(f"Failed to fetch props: {e}")
            return None

    def analyze_props(self, props: List[Dict]) -> Optional[List[Dict]]:
        """Analyze props using PropAnalyzer."""
        try:
            if not self.features_df is not None:
                logger.error("Features not loaded")
                return None

            logger.info(f"Analyzing {len(props)} props...")
            analyzed_props = self.prop_analyzer.analyze_props(props, self.features_df)
            best_props = self.prop_analyzer.find_best_edges(
                analyzed_props, min_edge=5.0
            )
            return best_props
        except Exception as e:
            logger.error(f"Failed to analyze props: {e}")
            return None

    def save_results(
        self, props: List[Dict], directory: str = "src/data/analysis"
    ) -> Optional[str]:
        """Save analysis results."""
        try:
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prop_analysis_{timestamp}.json"
            filepath = os.path.join(directory, filename)

            with open(filepath, "w") as f:
                json.dump(props, f, indent=2)

            logger.info(f"Saved analysis to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None

    def print_results(self, props: List[Dict], top_n: int = 10) -> None:
        """Print analysis results."""
        if not props:
            logger.info("No props to display")
            return

        logger.info(f"\nTop {top_n} Props with Best Edges:")
        logger.info("-" * 100)

        for prop in props[:top_n]:
            logger.info(
                f"{prop['player_name']} - {prop['market']} - "
                f"Line: {prop['line']}, Prediction: {prop['prediction']:.1f}, "
                f"Price: {prop['price']:+d}, Edge: {prop['edge']:.1f}%, "
                f"Confidence: {prop['confidence']:.2f}, Score: {prop['score']:.1f}"
            )

        # Print summary statistics
        total_props = len(props)
        props_with_edge = len([p for p in props if abs(p["edge"]) >= 5.0])
        avg_edge = sum(abs(p["edge"]) for p in props) / total_props if props else 0

        logger.info(f"\nSummary:")
        logger.info(f"Total props analyzed: {total_props}")
        logger.info(
            f"Props with edge >= 5%: {props_with_edge} ({props_with_edge/total_props*100:.1f}%)"
        )
        logger.info(f"Average edge: {avg_edge:.1f}%")

    def run(self) -> bool:
        """Run the complete analysis pipeline."""
        try:
            # Step 1: Load features
            if not self.load_features():
                return False

            # Step 2: Fetch props
            props = self.fetch_and_save_props()
            if not props:
                return False

            # Step 3: Analyze props
            analyzed_props = self.analyze_props(props)
            if not analyzed_props:
                return False

            # Step 4: Save and display results
            self.save_results(analyzed_props)
            self.print_results(analyzed_props)

            return True

        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            return False


def main():
    """Main entry point."""
    runner = PropsAnalysisRunner()
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
