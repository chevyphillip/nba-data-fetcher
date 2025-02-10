import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import optuna


def create_feature_groups():
    """Define feature groups for each target statistic."""
    feature_groups = {
        "PTS": {
            "base": [
                "PTS_per_game",
                "MPG",
                "Usage",
                "points_per_shot",
                "usage_rate",
                "true_shooting_pct",
            ],
            "rolling": [
                "PTS_rolling_5g",
                "PTS_rolling_10g",
                "PTS_rolling_15g",
                "PTS_ewm_5g",
                "PTS_trend",
            ],
            "other": [
                "games_played_ratio",
                "season_progress",
                "experience",
                "PTS_vs_pos_zscore",
                "opp_def_rating_PTS",
            ],
        },
        "TRB": {
            "base": ["TRB_per_game", "MPG", "Usage", "usage_rate"],
            "rolling": [
                "TRB_rolling_5g",
                "TRB_rolling_10g",
                "TRB_rolling_15g",
                "TRB_ewm_5g",
                "TRB_trend",
            ],
            "other": [
                "games_played_ratio",
                "season_progress",
                "experience",
                "TRB_vs_pos_zscore",
                "opp_def_rating_TRB",
            ],
        },
        "AST": {
            "base": [
                "AST_per_game",
                "MPG",
                "Usage",
                "assist_ratio",
                "assists_per_minute",
                "assist_to_usage",
            ],
            "rolling": [
                "AST_rolling_5g",
                "AST_rolling_10g",
                "AST_rolling_15g",
                "AST_ewm_5g",
                "AST_trend",
            ],
            "other": [
                "games_played_ratio",
                "season_progress",
                "experience",
                "AST_vs_pos_zscore",
                "opp_def_rating_AST",
            ],
        },
        "3P": {
            "base": [
                "3P_per_game",
                "MPG",
                "Usage",
                "three_point_rate",
                "effective_fg_pct",
            ],
            "rolling": [
                "3P_rolling_5g",
                "3P_rolling_10g",
                "3P_rolling_15g",
                "3P_ewm_5g",
                "3P_trend",
            ],
            "other": [
                "games_played_ratio",
                "season_progress",
                "experience",
                "3P_vs_pos_zscore",
                "opp_def_rating_3P",
            ],
        },
    }

    # Add position features as a separate group for each stat
    position_features = [f'Pos_{pos}' for pos in ['C', 'PF', 'PG', 'SF', 'SG']]
    for stat_group in feature_groups.values():
        stat_group['position'] = position_features.copy()

    return feature_groups


def objective(trial, X, y, cv_splits):
    """Optuna objective function for hyperparameter optimization."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }

    model = GradientBoostingRegressor(**params, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv_splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        cv_scores.append(r2_score(y_val, val_pred))

    return float(np.mean(cv_scores))


def train_and_save_models(
    features_file: str = "src/data/features/nba_player_stats_features_20250209.csv",
    model_dir: str = "src/models",
):
    """Train models for each stat and save them."""
    os.makedirs(model_dir, exist_ok=True)

    # Load features
    df = pd.read_csv(features_file)
    print(f"Loaded features with shape: {df.shape}")

    # Get feature groups
    FEATURE_GROUPS = create_feature_groups()

    # Save feature groups for use during prediction
    feature_groups_file = os.path.join(model_dir, "feature_groups.joblib")
    joblib.dump(FEATURE_GROUPS, feature_groups_file)
    print(f"Saved feature groups to {feature_groups_file}")

    # Train and save models for each stat
    for stat_key, feature_group in FEATURE_GROUPS.items():
        print(f"\nTraining model for {stat_key}...")

        # Combine all features for this stat
        feature_cols = (
            feature_group["base"] 
            + feature_group["rolling"] 
            + feature_group["other"]
        )
        
        # Get position features
        categorical_cols = feature_group["position"]

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

        print(f"Features for {stat_key}:")
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_cols)}")

        # Create preprocessor
        numeric_transformer = Pipeline(
            [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

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

        # Create time series cross-validation splits
        tscv = TimeSeriesSplit(n_splits=5)
        cv_splits = list(tscv.split(X))

        # Optimize hyperparameters
        print("Optimizing hyperparameters...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X, y, cv_splits), n_trials=20)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

        # Create final pipeline with best parameters
        final_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "feature_selector",
                    SelectFromModel(
                        GradientBoostingRegressor(random_state=42), threshold="mean"
                    ),
                ),
                (
                    "regressor",
                    GradientBoostingRegressor(**best_params, random_state=42),
                ),
            ]
        )

        # Train test split (using the most recent data for testing)
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Fit final model
        final_pipeline.fit(X_train, y_train)

        # Save model and feature names
        model_path = os.path.join(
            model_dir, f"{stat_key}_model_{datetime.now().strftime('%Y%m%d')}.joblib"
        )
        feature_names_path = os.path.join(
            model_dir,
            f"{stat_key}_feature_names_{datetime.now().strftime('%Y%m%d')}.joblib",
        )

        joblib.dump(final_pipeline, model_path)
        joblib.dump(numeric_features + categorical_cols, feature_names_path)

        # Evaluate model
        train_pred = final_pipeline.predict(X_train)
        test_pred = final_pipeline.predict(X_test)

        metrics = {
            "r2": r2_score(y_test, test_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "mae": mean_absolute_error(y_test, test_pred),
        }

        # Save metrics
        metrics_path = os.path.join(
            model_dir, f"{stat_key}_metrics_{datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"\nModel performance for {stat_key}:")
        print(f"Training R² score: {r2_score(y_train, train_pred):.3f}")
        print(f"Testing R² score: {metrics['r2']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"MAE: {metrics['mae']:.3f}")
        print(f"Saved model to {model_path}")
        print(f"Saved feature names to {feature_names_path}")
        print(f"Saved metrics to {metrics_path}")

        # Feature importance analysis
        feature_selector = final_pipeline.named_steps["feature_selector"]
        selected_features = np.array(numeric_features + categorical_cols)[
            feature_selector.get_support()
        ]
        print("\nTop selected features:")
        for feature in selected_features:
            print(f"- {feature}")


if __name__ == "__main__":
    train_and_save_models()
