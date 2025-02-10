"""
PropAnalyzer: Core class for analyzing NBA player props using trained models.

This class handles:
1. Loading and managing trained models
2. Analyzing props using player features
3. Calculating edges and confidence scores
4. Finding best betting opportunities
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.stats import norm
from pathlib import Path
import logging

from ..modeling.custom_models import ScaledGradientBoostingRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PropAnalyzer:
    def __init__(self, model_dir: str = "src/models"):
        """Initialize PropAnalyzer with trained models."""
        try:
            self.models = self._load_models(model_dir)
            self.feature_names = self._load_feature_names(model_dir)
            logger.info("Successfully initialized PropAnalyzer")
        except Exception as e:
            logger.error(f"Error initializing PropAnalyzer: {e}")
            raise

    def _load_models(self, model_dir: str) -> Dict:
        """Load all trained models."""
        models = {}
        try:
            # Get latest date from model files
            model_files = list(Path(model_dir).glob("*_model_*.joblib"))
            if not model_files:
                raise FileNotFoundError("No model files found")

            latest_date = max(f.stem.split("_")[-1] for f in model_files)
            logger.info(f"Loading models from date: {latest_date}")

            # Load models for each statistic
            for stat in ["PTS", "TRB", "AST", "3P"]:
                model_path = os.path.join(
                    model_dir, f"{stat}_model_{latest_date}.joblib"
                )
                if os.path.exists(model_path):
                    models[stat.lower()] = joblib.load(model_path)
                    logger.info(f"Loaded model for {stat}")

            return models
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _load_feature_names(self, model_dir: str) -> Dict[str, List[str]]:
        """Load feature names for each model."""
        feature_names = {}
        try:
            latest_date = max(
                f.stem.split("_")[-1]
                for f in Path(model_dir).glob("*_feature_names_*.joblib")
            )

            for stat in ["PTS", "TRB", "AST", "3P"]:
                feature_path = os.path.join(
                    model_dir, f"{stat}_feature_names_{latest_date}.joblib"
                )
                if os.path.exists(feature_path):
                    feature_names[stat.lower()] = joblib.load(feature_path)

            return feature_names
        except Exception as e:
            logger.error(f"Error loading feature names: {e}")
            raise

    def _convert_market_to_stat(self, market: str) -> Optional[str]:
        """Convert market name to stat name."""
        market_mapping = {
            "player_points": "points",
            "player_rebounds": "rebounds",
            "player_assists": "assists",
            "player_threes": "threes",
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
        """Calculate edge and probability for over/under."""
        # Use prediction uncertainty for standard deviation
        std_dev = max(0.5, uncertainty * prediction)

        # Calculate probability of going over
        prob_over = 1 - norm.cdf(line, prediction, std_dev)
        prob_under = 1 - prob_over

        # Convert odds to probability
        if price > 0:
            implied_prob = 100 / (price + 100)
        else:
            implied_prob = abs(price) / (abs(price) + 100)

        # Calculate edge
        if price > 0:  # If betting over
            edge = (prob_over - implied_prob) * 100
            prob = prob_over
        else:  # If betting under
            edge = (prob_under - implied_prob) * 100
            prob = prob_under

        return float(edge), float(prob)

    def _calculate_confidence_score(
        self, prediction: float, uncertainty: float, games_played: int
    ) -> float:
        """Calculate confidence score based on multiple factors."""
        confidence = 1.0

        # Factor 1: Model uncertainty
        confidence *= max(0, 1 - uncertainty)

        # Factor 2: Games played
        if games_played < 10:
            confidence *= 0.8  # Reduce confidence for small sample sizes

        # Factor 3: Prediction magnitude
        if prediction > 0:
            confidence *= min(1.0, prediction / 50)  # Scale based on prediction size

        return confidence

    def analyze_props(
        self, props: List[Dict], player_features: pd.DataFrame
    ) -> List[Dict]:
        """Analyze props and find edges."""
        if not self.models:
            logger.error("No models loaded. Cannot analyze props.")
            return []

        analyzed_props = []
        skipped_players = set()
        skipped_markets = set()

        for prop in props:
            try:
                player_name = prop["player_name"]
                market = prop["market"]
                stat = self._convert_market_to_stat(market)

                if stat not in self.models:
                    skipped_markets.add(market)
                    continue

                if player_name not in player_features.index:
                    skipped_players.add(player_name)
                    continue

                # Get features for prediction
                features = player_features.loc[[player_name]]
                model = self.models[stat]

                # Make prediction
                prediction = model.predict(features)[0]
                uncertainty = 0.15  # Base uncertainty

                # Calculate edge and probability
                edge, probability = self._calculate_edge(
                    prediction, prop["line"], prop["price"], uncertainty
                )

                # Calculate confidence
                confidence = self._calculate_confidence_score(
                    prediction, uncertainty, features.get("G", [10])[0]
                )

                analyzed_prop = {
                    **prop,
                    "prediction": prediction,
                    "edge": edge,
                    "probability": probability,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                }
                analyzed_props.append(analyzed_prop)

            except Exception as e:
                logger.error(
                    f"Error analyzing prop for {prop.get('player_name', 'Unknown')}: {e}"
                )
                continue

        if skipped_players:
            logger.warning(
                f"Skipped players (no features): {', '.join(sorted(skipped_players)[:5])}"
            )
        if skipped_markets:
            logger.warning(
                f"Skipped markets (no model): {', '.join(sorted(skipped_markets))}"
            )

        return analyzed_props

    def find_best_edges(
        self, analyzed_props: List[Dict], min_edge: float = 5.0
    ) -> List[Dict]:
        """Find props with the best edges."""
        try:
            # Calculate combined score for each prop
            scored_props = []
            for prop in analyzed_props:
                if abs(prop["edge"]) >= min_edge:
                    # Calculate Kelly stake
                    if prop["price"] > 0:
                        decimal_odds = 1 + (prop["price"] / 100)
                    else:
                        decimal_odds = 1 + (100 / abs(prop["price"]))

                    kelly = (decimal_odds * prop["probability"] - 1) / (
                        decimal_odds - 1
                    )
                    kelly = max(0, kelly)  # Ensure non-negative Kelly

                    # Calculate combined score
                    score = abs(prop["edge"]) * prop["confidence"] * kelly
                    scored_props.append({**prop, "kelly": kelly, "score": score})

            # Sort by score
            return sorted(scored_props, key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Error finding best edges: {e}")
            return []

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
