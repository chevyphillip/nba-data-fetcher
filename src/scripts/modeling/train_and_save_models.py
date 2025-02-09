import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def train_and_save_models(features_file: str = "src/data/features/nba_player_stats_features_20250208.csv",
                         model_dir: str = "src/models"):
    """Train models for each stat and save them."""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load features
    df = pd.read_csv(features_file)
    
    # Define feature groups
    FEATURE_GROUPS = {
        'PTS': {
            'base': ['PTS_per_game', 'MPG', 'Usage', 'points_per_shot', 'usage_rate'],
            'rolling': ['PTS_rolling_5g', 'PTS_rolling_10g'],
            'other': ['games_played_ratio', 'season_progress', 'experience']
        },
        'TRB': {
            'base': ['TRB_per_game', 'MPG', 'Usage', 'usage_rate'],
            'rolling': ['TRB_rolling_5g', 'TRB_rolling_10g'],
            'other': ['games_played_ratio', 'season_progress', 'experience']
        },
        'AST': {
            'base': ['AST_per_game', 'MPG', 'Usage', 'AST_TO_ratio', 'assist_ratio'],
            'rolling': ['AST_rolling_5g', 'AST_rolling_10g'],
            'other': ['games_played_ratio', 'season_progress', 'experience']
        },
        '3P': {
            'base': ['3P_per_game', 'MPG', 'Usage', 'three_point_rate'],
            'rolling': ['3P_rolling_5g', '3P_rolling_10g'],
            'other': ['games_played_ratio', 'season_progress', 'experience']
        }
    }
    
    # Save feature groups for use during prediction
    feature_groups_file = os.path.join(model_dir, "feature_groups.joblib")
    joblib.dump(FEATURE_GROUPS, feature_groups_file)
    
    # Train and save models for each stat
    for stat_key, feature_group in FEATURE_GROUPS.items():
        print(f"\nTraining model for {stat_key}...")
        
        # Combine all features for this stat
        feature_cols = (
            feature_group['base'] +
            feature_group['rolling'] +
            feature_group['other']
        )
        
        # Add position columns
        position_cols = ['Pos_C', 'Pos_PF', 'Pos_PG', 'Pos_SF', 'Pos_SG']
        categorical_cols = [col for col in position_cols if col in df.columns]
        
        # Convert boolean columns to float
        bool_cols = [col for col in categorical_cols if df[col].dtype == bool]
        for col in bool_cols:
            df[col] = df[col].astype(float)
        
        # Get target column
        target_col = f"{stat_key}_per_game"
        
        # Remove target from features if present
        feature_cols = [col for col in feature_cols if col != target_col]
        
        # Split features into numeric and categorical
        numeric_features = [col for col in feature_cols if col in df.columns]
        
        print(f"\nTraining model for {stat_key} ({target_col})")
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_cols}")
        
        # Create preprocessor for this specific feature set
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Create preprocessor
        feature_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Create pipeline with optimized hyperparameters
        model = Pipeline([
            ('preprocessor', feature_preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_split=5,
                random_state=42
            ))
        ])
        # Prepare data
        X = df[numeric_features + categorical_cols]
        y = df[target_col]
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print(f"No valid data for {stat_key} model after removing NaN values")
            continue
        
        # Train test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Save model and feature names
        model_path = os.path.join(model_dir, f"{stat_key}_model_{datetime.now().strftime('%Y%m%d')}.joblib")
        feature_names_path = os.path.join(model_dir, f"{stat_key}_feature_names_{datetime.now().strftime('%Y%m%d')}.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(numeric_features + categorical_cols, feature_names_path)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"\nModel performance for {stat_key}:")
        print(f"Training R² score: {train_score:.3f}")
        print(f"Testing R² score: {test_score:.3f}")
        print(f"Saved model to {model_path}")
        print(f"Saved feature names to {feature_names_path}")

if __name__ == "__main__":
    train_and_save_models()
