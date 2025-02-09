import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional

class OddsAPI:
    BASE_URL = "https://api.the-odds-api.com/v4"
    API_KEY = "bcab2a03da8de48a2a68698a40b78b4c"

    PROP_MARKETS = [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_blocks",
        "player_steals"
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
        "prophetx"
    ]

    def __init__(self):
        self.session = requests.Session()

    def get_nba_events(self) -> List[Dict]:
        """Fetch all NBA events."""
        url = f"{self.BASE_URL}/sports/basketball_nba/events"
        params = {
            "apiKey": self.API_KEY
        }
        
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
            "bookmakers": ",".join(self.BOOKMAKERS)
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
                    
                    props.append({
                        "event_id": odds_data["id"],
                        "event_time": odds_data["commence_time"],
                        "bookmaker": book_name,
                        "player_name": player_name,
                        "market": market_key,
                        "line": line,
                        "price": price,
                        "home_team": odds_data["home_team"],
                        "away_team": odds_data["away_team"]
                    })
        
        return props

    def get_all_props(self) -> List[Dict]:
        """Get all available player props for NBA games."""
        all_props = []
        
        try:
            # Get all events
            events = self.get_nba_events()
            
            # Get odds for each event
            for event in events:
                try:
                    odds_data = self.get_event_odds(event["id"])
                    props = self.parse_player_props(odds_data)
                    all_props.extend(props)
                except Exception as e:
                    print(f"Error fetching odds for event {event['id']}: {str(e)}")
                    continue
            
            return all_props
        
        except Exception as e:
            print(f"Error fetching events: {str(e)}")
            return []

    def save_props(self, props: List[Dict], directory: str = "src/data/odds"):
        """Save props to a JSON file."""
        os.makedirs(directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nba_props_{timestamp}.json"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, "w") as f:
            json.dump(props, f, indent=2)
        
        print(f"Saved {len(props)} props to {filepath}")
        return filepath
