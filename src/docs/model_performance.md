# NBA Statistics Prediction Models

This document provides detailed information about the NBA statistics prediction models, including their architecture, performance metrics, and characteristics.

## Model Overview

The prediction system uses Gradient Boosting Regressors optimized through Optuna for four key NBA statistics:

- Points (PTS)
- Rebounds (TRB)
- Assists (AST)
- Three-Pointers (3P)

## Model Architecture

### Core Components

- **Base Algorithm**: Gradient Boosting Regressor
- **Feature Selection**: SelectFromModel with mean threshold
- **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categorical features
- **Cross-validation**: 5-fold TimeSeriesSplit
- **Hyperparameter Optimization**: Optuna with 20 trials

### Feature Groups

Each model utilizes specialized feature groups:

1. **Base Statistics**
   - Per-game averages
   - Usage rates
   - Efficiency metrics

2. **Rolling Statistics**
   - 5-game window
   - 10-game window
   - 15-game window

3. **Position Features**
   - One-hot encoded positions
   - Position-specific metrics

4. **Team Context**
   - Team pace
   - Team efficiency metrics
   - Home/Away splits

## Performance Metrics

### Points (PTS) Model

- R² Score: 0.710 (71.0% variance explained)
- RMSE: 1.903
- MAE: 0.279
- Key Features:
  - Points per game
  - Usage rate
  - True shooting percentage
  - Rolling averages
  - Team pace

### Rebounds (TRB) Model

- R² Score: 0.680 (68.0% variance explained)
- RMSE: 1.154
- MAE: 0.230
- Key Features:
  - Rebounds per game
  - Minutes played
  - Position indicators
  - Team rebounding rate
  - Rolling averages

### Assists (AST) Model

- R² Score: 0.886 (88.6% variance explained)
- RMSE: 0.197
- MAE: 0.054
- Key Features:
  - Assists per game
  - Usage rate
  - Team assist rate
  - Rolling averages
  - Position indicators

### Three-Points (3P) Model

- R² Score: 0.189 (18.9% variance explained)
- RMSE: 0.221
- MAE: 0.056
- Key Features:
  - Three-point attempts
  - Three-point percentage
  - Team pace
  - Rolling averages

## Model Storage

Models and related files are stored with consistent naming conventions:

```
src/models/
├── {stat}_model_YYYYMMDD.joblib       # Trained model pipeline
├── {stat}_metrics_YYYYMMDD.json       # Performance metrics
├── {stat}_feature_names_YYYYMMDD.joblib  # Selected features
└── feature_groups.joblib              # Feature group definitions
```

## Prediction Pipeline

1. **Feature Preparation**
   - Load latest player features
   - Apply preprocessing transformations
   - Generate rolling statistics

2. **Model Application**
   - Load appropriate model for statistic
   - Generate predictions
   - Calculate uncertainty estimates

3. **Edge Calculation**
   - Compare predictions to lines
   - Calculate edge percentages
   - Apply Kelly criterion
   - Generate confidence scores

## Strengths

1. **Assists Model**
   - Highest accuracy (88.6% variance explained)
   - Consistent performance across different players
   - Strong predictive power for primary ball handlers

2. **Points Model**
   - Good balance of accuracy and coverage
   - Effective usage of team context
   - Reliable for high-usage players

3. **Rebounds Model**
   - Strong position-based predictions
   - Effective use of team rebounding rates
   - Good performance for centers and power forwards

## Limitations

1. **Three-Point Model**
   - Lower accuracy (18.9% variance explained)
   - High variance in predictions
   - Sensitive to game script and matchups

2. **General Limitations**
   - Limited handling of injuries
   - No direct opponent adjustment
   - Sensitive to role changes
   - Limited historical data for new players

## Recent Improvements

1. **Feature Engineering**
   - Added team context features
   - Improved rolling average calculations
   - Enhanced position-based features
   - Added home/away splits

2. **Model Training**
   - Optimized feature selection
   - Improved hyperparameter tuning
   - Enhanced cross-validation strategy
   - Better handling of categorical variables

3. **Prediction Pipeline**
   - Added uncertainty estimation
   - Improved confidence scoring
   - Enhanced edge calculation
   - Better handling of missing data

## Future Enhancements

1. **Model Architecture**
   - Experiment with ensemble methods
   - Implement neural networks for specific cases
   - Add recurrent layers for time series
   - Explore transfer learning

2. **Feature Engineering**
   - Add defensive matchup features
   - Implement rest day impact
   - Add player similarity scores
   - Include injury history

3. **Prediction Pipeline**
   - Add backtesting capabilities
   - Implement confidence intervals
   - Add anomaly detection
   - Real-time performance monitoring

## Validation Strategy

1. **Time Series Validation**
   - 5-fold time series cross-validation
   - Forward-chaining prediction
   - Out-of-time testing

2. **Performance Monitoring**
   - Daily prediction tracking
   - Edge realization analysis
   - Model drift detection
   - Feature importance tracking

3. **Quality Checks**
   - Input data validation
   - Prediction range checks
   - Uncertainty estimation
   - Confidence thresholds

## Usage Guidelines

1. **Best Practices**
   - Use rolling window predictions
   - Consider confidence scores
   - Monitor edge realization
   - Track model performance

2. **Risk Management**
   - Apply Kelly criterion
   - Consider prediction uncertainty
   - Use confidence thresholds
   - Monitor bankroll exposure

3. **Monitoring**
   - Track daily performance
   - Monitor feature drift
   - Check prediction distributions
   - Validate edge realization

## Version History

### February 9, 2025

- Initial model documentation
- Implemented optimized feature selection
- Reduced training time with 20 trials
- Achieved strong performance across all statistics
