import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scripts.odds.odds_api import OddsAPI


def main():
    # Initialize the API
    odds_api = OddsAPI()

    # Get and analyze props
    best_props = odds_api.analyze_props(min_edge=5.0, min_confidence=0.7)

    # Print results
    odds_api.print_best_edges(best_props, top_n=10)

    # Save results
    if best_props:
        odds_api.save_props(best_props)


if __name__ == "__main__":
    main()
