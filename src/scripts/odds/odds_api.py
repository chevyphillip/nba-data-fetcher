import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
from src.scripts.analysis.prop_analyzer import PropAnalyzer


class OddsAPI:
    BASE_URL = "https://api.the-odds-api.com/v4"
    API_KEY = os.getenv("ODDS_API_KEY", "bcab2a03da8de48a2a68698a40b78b4c")

    PROP_MARKETS = [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
    ]

    BOOKMAKERS = [
        "pinnacle",
        "betonlineag",
        "fanduel",
        "draftkings",
        "fliff",
        "underdog",
        "betmgm",
        "betrivers",
        "ballybet",
        "espnbet",
        "novig",
        "prophetx",
    ]

    def __init__(self, model_dir: str = "src/models"):
        """Initialize OddsAPI with PropAnalyzer."""
        self.session = requests.Session()
        self.analyzer = PropAnalyzer(model_dir)

        # Load latest player features
        features_dir = Path("src/data/features")
        try:
            latest_features = max(
                features_dir.glob("nba_player_stats_features_*.csv"),
                key=lambda x: x.stat().st_mtime,
            )
            self.features_df = pd.read_csv(latest_features, index_col="Player")
            print(f"Loaded player features from {latest_features}")
        except Exception as e:
            print(f"Error loading player features: {str(e)}")
            self.features_df = None

    def get_nba_events(self) -> List[Dict]:
        """Fetch all NBA events."""
        url = f"{self.BASE_URL}/sports/basketball_nba/events"
        params = {"apiKey": self.API_KEY}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def get_event_odds(self, event_id: str) -> Dict:
        """Fetch player props for a specific event."""
        url = f"{self.BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
        params = {
            "apiKey": self.API_KEY,
            "regions": "us,us2,us_dfs,eu,us_ex",
            "markets": ",".join(self.PROP_MARKETS),
            "oddsFormat": "american",
            "bookmakers": ",".join(self.BOOKMAKERS),
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def parse_player_props(self, odds_data: Dict) -> List[Dict]:
        """Parse odds data into a list of player props."""
        props = []

        for bookmaker in odds_data.get("bookmakers", []):
            book_name = bookmaker["title"]

            for market in bookmaker.get("markets", []):
                market_key = market["key"]

                for outcome in market.get("outcomes", []):
                    player_name = outcome["description"]
                    line = float(outcome.get("point", 0))
                    price = int(outcome.get("price", 0))

                    props.append(
                        {
                            "event_id": odds_data["id"],
                            "event_time": odds_data["commence_time"],
                            "bookmaker": book_name,
                            "player_name": player_name,
                            "market": market_key,
                            "line": line,
                            "price": price,
                            "home_team": odds_data["home_team"],
                            "away_team": odds_data["away_team"],
                        }
                    )

        return props

    def get_all_props(self) -> List[Dict]:
        """Get all available player props for NBA games."""
        all_props = []

        try:
            # Get all events
            events = self.get_nba_events()
            print(f"Found {len(events)} NBA events")

            # Get odds for each event
            for event in events:
                try:
                    odds_data = self.get_event_odds(event["id"])
                    props = self.parse_player_props(odds_data)
                    all_props.extend(props)
                    print(
                        f"Fetched {len(props)} props for {event['home_team']} vs {event['away_team']}"
                    )
                except Exception as e:
                    print(f"Error fetching odds for event {event['id']}: {str(e)}")
                    continue

            return all_props

        except Exception as e:
            print(f"Error fetching events: {str(e)}")
            return []

    def analyze_props(
        self, min_edge: float = 5.0, min_confidence: float = 0.7
    ) -> List[Dict]:
        """
        Fetch props, analyze them using the PropAnalyzer, and return the best edges.

        Args:
            min_edge: Minimum edge percentage to consider (default: 5.0)
            min_confidence: Minimum confidence score to consider (default: 0.7)

        Returns:
            List of analyzed props with edges above the threshold, sorted by edge
        """
        if self.features_df is None:
            print("Error: Player features not loaded. Cannot analyze props.")
            return []

        try:
            # Fetch latest props
            props = self.get_all_props()
            if not props:
                print("No props found to analyze")
                return []

            print(f"\nAnalyzing {len(props)} props...")

            # Analyze props using PropAnalyzer
            analyzed_props = self.analyzer.analyze_props(props, self.features_df)

            # Filter and sort props
            best_props = []
            for prop in analyzed_props:
                if (
                    abs(prop.get("edge", 0)) >= min_edge
                    and prop.get("confidence", 0) >= min_confidence
                ):
                    best_props.append(prop)

            # Sort by edge * confidence score
            best_props.sort(
                key=lambda x: abs(x.get("edge", 0)) * x.get("confidence", 0),
                reverse=True,
            )

            return best_props

        except Exception as e:
            print(f"Error analyzing props: {str(e)}")
            return []

    def save_props(self, props: List[Dict], directory: str = "src/data/odds") -> str:
        """Save props to a JSON file."""
        os.makedirs(directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nba_props_{timestamp}.json"
        filepath = os.path.join(directory, filename)

        with open(filepath, "w") as f:
            json.dump(props, f, indent=2)

        print(f"Saved {len(props)} props to {filepath}")
        return filepath

    def print_best_edges(self, analyzed_props: List[Dict], top_n: int = 10) -> None:
        """Print the top N props with the best edges."""
        if not analyzed_props:
            print("No props to display")
            return

        print(f"\nTop {top_n} Props with Best Edges:")
        print("-" * 100)

        for prop in analyzed_props[:top_n]:
            print(
                f"{prop['player_name']} - {prop['market']} - "
                f"Line: {prop['line']}, Prediction: {prop['prediction']:.1f}, "
                f"Price: {prop['price']:+d}, Edge: {prop['edge']:.1f}%, "
                f"Confidence: {prop['confidence']:.2f}, Score: {prop['score']:.1f}"
            )

        # Print summary statistics
        total_props = len(analyzed_props)
        props_with_edge = len([p for p in analyzed_props if abs(p["edge"]) >= 5.0])
        avg_edge = (
            sum(abs(p["edge"]) for p in analyzed_props) / total_props
            if analyzed_props
            else 0
        )

        print(f"\nSummary:")
        print(f"Total props analyzed: {total_props}")
        print(
            f"Props with edge >= 5%: {props_with_edge} ({props_with_edge/total_props*100:.1f}%)"
        )
        print(f"Average edge: {avg_edge:.1f}%")
