"""
NBA Player Statistics Prediction Models

This script trains separate models for predicting different player statistics using:
1. Proper train/validation/test splits by season
2. Scikit-learn pipelines for preprocessing and modeling
3. Feature selection and importance analysis
4. Clear evaluation metrics and model persistence
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent.parent
FEATURES_DIR = SCRIPT_DIR / "data" / "features"
MODELS_DIR = SCRIPT_DIR / "models"

# Configuration
TARGET_STATS = {
    'PTS': {'name': 'Points', 'col': 'PTS_per_game'},
    'TRB': {'name': 'Rebounds', 'col': 'TRB_per_game'},
    'AST': {'name': 'Assists', 'col': 'AST_per_game'},
    '3P': {'name': '3-Pointers Made', 'col': '3P_per_game'}
}

# Model parameters
MODEL_PARAMS = {
    'PTS': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8
    },
    'TRB': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8
    },
    'AST': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8
    },
    '3P': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8
    }
}

def load_feature_data() -> pd.DataFrame:
    """Load the most recent feature-engineered data."""
    try:
        feature_files = list(FEATURES_DIR.glob("nba_player_stats_features_*.csv"))
        if not feature_files:
            raise FileNotFoundError("No feature-engineered data files found")
        
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading feature data from {latest_file}")
        return pd.read_csv(latest_file)
    except Exception as e:
        logger.error(f"Error loading feature data: {e}")
        raise

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for modeling."""
    # Drop non-feature columns
    drop_cols = ['Player', 'Team', 'Pos']  # Keep position dummy variables
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Convert all columns to numeric, dropping those that can't be converted
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    dropped_cols = set(df.columns) - set(numeric_df.columns)
    if dropped_cols:
        logger.warning(f"Dropped non-numeric columns: {dropped_cols}")
    
    return numeric_df

def split_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and validation sets by season."""
    # Use 2024 season as validation
    train_df = df[df['Season'] < 2024].copy()
    valid_df = df[df['Season'] == 2024].copy()
    
    # Remove target column and related columns from features
    target_related = [col for col in df.columns if target_col.split('_')[0] in col]
    feature_cols = [col for col in df.columns if col not in target_related + ['Season']]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]
    
    return X_train, X_valid, y_train, y_valid

def create_pipeline(params: Dict) -> Pipeline:
    """Create a scikit-learn pipeline for preprocessing and modeling."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Remove add_indicator
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(
            random_state=42,
            **params
        ))
    ])

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    target: str
) -> Tuple[Pipeline, Dict]:
    """Train a model for a specific target statistic."""
    logger.info(f"\nTraining model for {TARGET_STATS[target]['name']}...")
    
    # Create and train pipeline
    pipeline = create_pipeline(MODEL_PARAMS[target])
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_valid)
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_valid, y_pred)),
        'mae': mean_absolute_error(y_valid, y_pred),
        'r2': r2_score(y_valid, y_pred)
    }
    
    # Log metrics
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"R2: {metrics['r2']:.4f}")
    
    # Get feature importance for available features
    model = pipeline.named_steps['regressor']
    
    # Get features that have data
    features = [col for col in X_train.columns if X_train[col].notna().any()]
    
    # Log feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 10 Important Features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.2%}")
    
    return pipeline, metrics

def save_model(pipeline: Pipeline, target: str, metrics: Dict):
    """Save trained model and its metrics."""
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    current_date = datetime.now().strftime("%Y%m%d")
    model_file = MODELS_DIR / f"{target}_model_{current_date}.joblib"
    joblib.dump(pipeline, model_file)
    
    # Save metrics
    metrics_file = MODELS_DIR / f"{target}_metrics_{current_date}.json"
    pd.Series(metrics).to_json(metrics_file)
    
    logger.info(f"Model and metrics saved to {MODELS_DIR}")

def main():
    """Main execution function."""
    try:
        # Load and prepare data
        df = load_feature_data()
        df = prepare_features(df)
        
        # Train models for each target
        for target, info in TARGET_STATS.items():
            try:
                # Split data
                X_train, X_valid, y_train, y_valid = split_data(df, info['col'])
                
                # Train and evaluate model
                pipeline, metrics = train_model(X_train, y_train, X_valid, y_valid, target)
                
                # Save model and metrics
                save_model(pipeline, target, metrics)
                
            except Exception as e:
                logger.error(f"Error training model for {target}: {e}")
                continue
        
        logger.info("\nModel training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
