import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.stats import norm
from pathlib import Path

from src.scripts.modeling.custom_models import ScaledGradientBoostingRegressor


class PropAnalyzer:
    def __init__(self, model_dir: str = "src/models"):
        """Initialize PropAnalyzer with trained models."""
        try:
            self.models = self._load_models(model_dir)
            feature_names_path = os.path.join(model_dir, "feature_names.joblib")
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
            else:
                print(f"Warning: Feature names not found at {feature_names_path}")
                self.feature_names = None

            # Load historical performance data if available
            self.historical_performance = self._load_historical_performance()
        except Exception as e:
            print(f"Warning: Error loading models: {str(e)}")
            self.models = {}
            self.feature_names = None
            self.historical_performance = None

    def _load_historical_performance(self) -> Optional[pd.DataFrame]:
        """Load historical performance data for calibration."""
        try:
            history_file = "src/data/analysis/historical_performance.csv"
            if os.path.exists(history_file):
                return pd.read_csv(history_file)
        except Exception as e:
            print(f"Warning: Could not load historical performance: {e}")
        return None

    def _load_models(self, model_dir: str) -> Dict:
        """Load all trained models and their feature groups."""
        models = {}

        # Get latest date from model files
        model_files = list(Path(model_dir).glob("*_model_*.joblib"))
        if not model_files:
            return {}

        latest_date = max(f.stem.split("_")[-1] for f in model_files)

        # Load feature groups
        feature_groups_path = os.path.join(model_dir, "feature_groups.joblib")
        if not os.path.exists(feature_groups_path):
            print(f"Error: Feature groups not found at {feature_groups_path}")
            return {}

        feature_groups = joblib.load(feature_groups_path)

        stat_mapping = {
            "PTS": "points",
            "TRB": "rebounds",
            "AST": "assists",
            "3P": "threes",
        }

        for stat_key, stat_name in stat_mapping.items():
            model_path = os.path.join(
                model_dir, f"{stat_key}_model_{latest_date}.joblib"
            )
            feature_names_path = os.path.join(
                model_dir, f"{stat_key}_feature_names_{latest_date}.joblib"
            )

            if os.path.exists(model_path) and os.path.exists(feature_names_path):
                try:
                    # Load model and feature names
                    model = joblib.load(model_path)
                    feature_names = joblib.load(feature_names_path)

                    # Verify the model pipeline is complete
                    if (
                        not hasattr(model, "named_steps")
                        or "regressor" not in model.named_steps
                    ):
                        print(f"Error: Invalid model pipeline for {stat_key}")
                        continue

                    # Store model components
                    models[stat_name] = {
                        "model": model,
                        "feature_names": feature_names,
                        "feature_group": feature_groups[stat_key],
                    }
                    print(f"Loaded model and feature group for {stat_key}")
                except Exception as e:
                    print(f"Error loading model for {stat_key}: {str(e)}")
                    continue

        return models

    def _convert_market_to_stat(self, market: str) -> Optional[str]:
        """Convert market name to stat name."""
        market_mapping = {
            "player_points": "points",
            "player_rebounds": "rebounds",
            "player_assists": "assists",
            "player_threes": "threes",
            "player_blocks": "blocks",
            "player_steals": "steals",
        }
        return market_mapping.get(market)

    def _american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))

    def _american_to_implied_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def _remove_vig(self, probabilities: List[float]) -> List[float]:
        """Remove the vig from probabilities."""
        total_probability = sum(probabilities)
        return [p / total_probability for p in probabilities]

    def _calculate_prediction_interval(
        self, prediction: float, model_dict: Dict
    ) -> Tuple[float, float]:
        """Calculate prediction interval using model uncertainty."""
        model = model_dict["model"]

        # Get the residuals from the training data if available
        if hasattr(model, "named_steps") and "regressor" in model.named_steps:
            regressor = model.named_steps["regressor"]
            if hasattr(regressor, "train_score_"):
                # Use the training score to estimate prediction uncertainty
                mse = 1 - regressor.train_score_
                std_dev = np.sqrt(mse) * prediction
                return std_dev, mse

        # Fallback to default uncertainty estimation
        return prediction * 0.15, 0.15  # 15% uncertainty as default

    def _calculate_edge(
        self, prediction: float, line: float, price: int, uncertainty: float
    ) -> Tuple[float, float]:
        """Calculate edge and probability for over/under with uncertainty consideration."""
        # Use prediction uncertainty for the standard deviation
        std_dev = max(0.5, uncertainty * prediction)  # Minimum std_dev of 0.5

        # Calculate probability of going over
        prob_over = 1 - norm.cdf(line, prediction, std_dev)
        prob_under = 1 - prob_over

        # Get implied probabilities from odds
        implied_prob = self._american_to_implied_prob(price)

        # For over bets, use the original implied prob
        # For under bets, calculate the other side's implied prob
        if price < 0:  # If odds are negative, they represent the favorite
            implied_prob_over = implied_prob
            implied_prob_under = 1 - implied_prob_over
        else:  # If odds are positive, they represent the underdog
            implied_prob_under = implied_prob
            implied_prob_over = 1 - implied_prob_under

        # Remove vig from implied probabilities
        no_vig_probs = self._remove_vig([implied_prob_over, implied_prob_under])
        implied_prob_over_no_vig, implied_prob_under_no_vig = no_vig_probs

        # Calculate edge based on whether we're betting over or under
        if price > 0:  # If betting over
            edge = (prob_over - implied_prob_over_no_vig) * 100
            prob = prob_over
        else:  # If betting under
            edge = (prob_under - implied_prob_under_no_vig) * 100
            prob = prob_under

        return float(edge), float(prob)

    def _calculate_confidence_score(
        self, prop: Dict, prediction: float, uncertainty: float
    ) -> float:
        """Calculate a confidence score for the prediction."""
        confidence = 1.0

        # Factor 1: Model uncertainty
        confidence *= max(0, 1 - uncertainty)

        # Factor 2: Historical accuracy for similar predictions
        if self.historical_performance is not None:
            similar_props = self.historical_performance[
                (self.historical_performance["market"] == prop["market"])
                & (
                    abs(self.historical_performance["prediction"] - prediction)
                    < prediction * 0.1
                )
            ]
            if len(similar_props) > 0:
                accuracy = len(similar_props[similar_props["result"] == "win"]) / len(
                    similar_props
                )
                confidence *= accuracy

        # Factor 3: Line movement (if available)
        if "opening_line" in prop and "opening_price" in prop:
            line_movement = abs(prop["line"] - prop["opening_line"])
            price_movement = abs(prop["price"] - prop["opening_price"])
            if line_movement > 0 or price_movement > 0:
                confidence *= 0.9  # Reduce confidence if there's significant movement

        # Factor 4: Sample size
        if "games_played" in prop and prop["games_played"] < 10:
            confidence *= 0.8  # Reduce confidence for small sample sizes

        return confidence

    def analyze_props(
        self, props: List[Dict], player_features: pd.DataFrame
    ) -> List[Dict]:
        """Analyze props and find edges."""
        if not self.models:
            print("Error: No models loaded. Cannot analyze props.")
            return []

        analyzed_props = []
        skipped_players = set()
        skipped_markets = set()

        # Add position columns if they don't exist
        position_cols = ["Pos_C", "Pos_PF", "Pos_PG", "Pos_SF", "Pos_SG"]
        for col in position_cols:
            if col not in player_features.columns:
                player_features[col] = 0

        for prop in props:
            try:
                stat = self._convert_market_to_stat(prop["market"])
                if stat not in self.models:
                    skipped_markets.add(prop["market"])
                    continue

                player_name = prop["player_name"]
                if player_name not in player_features.index:
                    skipped_players.add(player_name)
                    continue

                # Get model components and feature groups
                model_dict = self.models[stat]
                model = model_dict["model"]
                feature_group = model_dict["feature_group"]

                # Get all required features
                all_features = (
                    feature_group["base"]
                    + feature_group["rolling"]
                    + feature_group["other"]
                    + [col for col in position_cols if col in player_features.columns]
                )

                # Check if we have all required features
                missing_cols = [
                    col for col in all_features if col not in player_features.columns
                ]
                if missing_cols:
                    print(f"Error: Missing required columns for {stat}: {missing_cols}")
                    continue

                # Select features for this player
                features = player_features.loc[player_name][all_features]

                # Make prediction
                X = features.values.reshape(1, -1)
                raw_prediction = model.predict(X)[0]  # Model handles scaling internally
                prediction = max(0, raw_prediction)  # Ensure non-negative

                # Calculate prediction uncertainty
                uncertainty, mse = self._calculate_prediction_interval(
                    prediction, model_dict
                )

                # Calculate edge and probability
                edge, prob = self._calculate_edge(
                    prediction, prop["line"], prop["price"], uncertainty
                )

                # Calculate confidence score
                confidence = self._calculate_confidence_score(
                    prop, prediction, uncertainty
                )

                analyzed_prop = {
                    **prop,
                    "prediction": prediction,
                    "uncertainty": uncertainty,
                    "edge": edge,
                    "probability": prob,
                    "confidence": confidence,
                    "mse": mse,
                }
                analyzed_props.append(analyzed_prop)

                print(f"Analysis for {player_name} {stat}:")
                print(f"Prediction: {prediction:.2f} ± {uncertainty:.2f}")
                print(f"Edge: {edge:.2f}%")
                print(f"Confidence: {confidence:.2f}")

            except Exception as e:
                print(
                    f"Error analyzing prop for {prop.get('player_name', 'Unknown')} - {prop.get('market', 'Unknown')}: {str(e)}"
                )
                continue

        if skipped_markets:
            print(
                f"\nSkipped markets (no model available): {', '.join(skipped_markets)}"
            )
        if skipped_players:
            print(
                f"\nSkipped players (no features available): {', '.join(list(skipped_players)[:5])}"
            )
            if len(skipped_players) > 5:
                print(f"...and {len(skipped_players) - 5} more players")

        return analyzed_props

    def find_best_edges(
        self, analyzed_props: List[Dict], min_edge: float = 5.0
    ) -> List[Dict]:
        """Find props with the best edges."""
        # Calculate Kelly stake and combined score for each prop
        for prop in analyzed_props:
            # Calculate optimal Kelly stake
            if prop["price"] > 0:
                decimal_odds = self._american_to_decimal(prop["price"])
                prob = prop["probability"]
                kelly = (decimal_odds * prob - 1) / (decimal_odds - 1)
            else:
                decimal_odds = self._american_to_decimal(abs(prop["price"]))
                prob = 1 - prop["probability"]
                kelly = (decimal_odds * prob - 1) / (decimal_odds - 1)

            kelly = max(0, kelly)  # Ensure non-negative Kelly

            # Calculate combined score using edge, confidence, and Kelly criterion
            prop["kelly"] = kelly
            prop["score"] = abs(prop["edge"]) * prop["confidence"] * kelly

        # Sort props by score
        sorted_props = sorted(analyzed_props, key=lambda x: x["score"], reverse=True)

        # Filter props with minimum edge and positive Kelly
        best_props = [
            prop
            for prop in sorted_props
            if abs(prop["edge"]) >= min_edge and prop["kelly"] > 0
        ]

        return best_props

    def save_analysis(
        self, analyzed_props: List[Dict], directory: str = "src/data/analysis"
    ) -> str:
        """Save analyzed props to a JSON file."""
        os.makedirs(directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prop_analysis_{timestamp}.json"
        filepath = os.path.join(directory, filename)

        # Convert numpy types to native Python types
        serializable_props = self._make_serializable(analyzed_props)

        with open(filepath, "w") as f:
            json.dump(serializable_props, f, indent=2)

        print(f"Saved analysis of {len(analyzed_props)} props to {filepath}")
        return filepath

    def _make_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
