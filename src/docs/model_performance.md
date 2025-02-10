# NBA Statistics Prediction Models

This document provides detailed information about the NBA statistics prediction models, including their architecture, performance metrics, and characteristics.

## Model Overview

The prediction system uses Gradient Boosting Regressors optimized through Optuna for four key NBA statistics:

- Points (PTS)
- Rebounds (TRB)
- Assists (AST)
- Three-Pointers (3P)

## Model Architecture

Each model utilizes:

- Base Algorithm: Gradient Boosting Regressor
- Feature Selection: SelectFromModel with mean threshold
- Preprocessing: StandardScaler for numeric features, OneHotEncoder for categorical features
- Cross-validation: 5-fold TimeSeriesSplit
- Hyperparameter Optimization: Optuna with 20 trials

## Performance Metrics

### Points (PTS) Model

- R² Score: 0.710 (71.0% variance explained)
- RMSE: 1.903
- MAE: 0.279
- Key Features: Usage, points_per_shot, usage_rate, PTS_rolling_5g, games_played_ratio

### Rebounds (TRB) Model

- R² Score: 0.680 (68.0% variance explained)
- RMSE: 1.154
- MAE: 0.230
- Key Features: TRB_rolling_5g, games_played_ratio

### Assists (AST) Model

- R² Score: 0.886 (88.6% variance explained)
- RMSE: 0.197
- MAE: 0.054
- Key Features: assist_ratio, AST_rolling_5g, AST_rolling_10g, games_played_ratio

### Three-Points (3P) Model

- R² Score: 0.189 (18.9% variance explained)
- RMSE: 0.221
- MAE: 0.056
- Key Features: 3P_rolling_5g, 3P_rolling_10g

## Feature Groups

The models utilize several feature groups stored in `feature_groups.joblib`:

1. Base Statistics: Raw game statistics
2. Rolling Averages: 5 and 10 game windows
3. Other: Derived metrics (usage rate, assist ratio, etc.)
4. Position: One-hot encoded player positions (C, PF, PG, SF, SG)

## Model Storage

Models and related files are stored with consistent naming conventions:

- Models: `{stat}_model_YYYYMMDD.joblib`
- Metrics: `{stat}_metrics_YYYYMMDD.json`
- Feature Names: `{stat}_feature_names_YYYYMMDD.joblib`

## Usage Notes

1. Models are optimized for daily predictions
2. Feature importance is calculated using the gradient boosting feature importance method
3. Preprocessing steps are included in the model pipeline for ease of use
4. Cross-validation ensures robust performance estimates

## Strengths

1. Strong performance in assists prediction (88.6% variance explained)
2. Robust feature selection process
3. Efficient training process (20 trials optimization)
4. Comprehensive preprocessing pipeline
5. Time series aware validation
6. Clear separation of position features

## Limitations

1. Lower accuracy in three-point predictions (18.9% variance explained)
2. Moderate performance in points and rebounds predictions
3. May not capture sudden changes in player roles
4. Limited handling of rare events (injuries, trades)
5. Assumes historical patterns predict future performance

## Future Improvements

1. Feature Engineering
   - Add more advanced interaction features
   - Incorporate defensive matchup data
   - Develop rest day impact features

2. Model Enhancements
   - Experiment with ensemble methods
   - Implement uncertainty quantification
   - Add online learning capabilities
   - Explore deep learning approaches

3. Validation
   - Add backtesting capabilities
   - Implement confidence intervals
   - Develop anomaly detection

## Version History

### February 9, 2025

- Initial model documentation
- Implemented optimized feature selection
- Reduced training time with 20 trials
- Achieved strong performance across all statistics
