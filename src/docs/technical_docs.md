# NBA Stats Predictor - Technical Documentation

## Architecture Overview

The NBA Stats Predictor is a machine learning system designed to predict various basketball statistics for NBA players. The system follows a modular pipeline architecture:

1. **Data Collection**: Web scraping with BeautifulSoup4 and robust retry mechanisms
2. **Data Cleaning**: Preprocessing and standardization of raw data
3. **Feature Engineering**: Advanced metrics and rolling statistics
4. **Model Training**: Gradient Boosting with hyperparameter optimization
5. **Prop Analysis**: Edge calculation with uncertainty quantification
6. **Visualization**: Performance metrics and model insights

## Data Pipeline

### Data Collection (`nba_historical_stats_fetcher.py`)

- Uses BeautifulSoup4 with retry mechanism and exponential backoff
- Implements rate limiting detection and handling
- Rotates user agents automatically
- Stores raw data in CSV format with metadata
- Comprehensive error handling and logging

### Data Cleaning

- Removes duplicate entries
- Handles missing values through imputation
- Validates data types and ranges
- Standardizes column names
- Output: Clean dataset ready for feature engineering

### Feature Engineering (`feature_engineering.py`)

- Creates rolling averages with multiple windows (5, 10, 15 games)
- Generates exponentially weighted moving averages
- Implements position-based features and percentiles
- Calculates advanced efficiency metrics
- Adds season context and matchup features
- Output: Feature-rich dataset for model training

## Model Architecture

### Training Pipeline (`train_and_save_models.py`)

- **Algorithm**: Gradient Boosting Regressor with optimized hyperparameters
- **Feature Selection**: SelectFromModel with importance thresholds
- **Preprocessing**: StandardScaler and categorical encoding
- **Cross-validation**: Time series split for temporal validation
- **Target Variables**: Points (PTS), Rebounds (TRB), Assists (AST), Three-Pointers (3P)

### Model Performance

Current model performance metrics (as of February 9, 2025):

| Statistic     | R² Score | RMSE  | MAE    |
|--------------|----------|--------|--------|
| Points       | 0.907    | 0.875  | 0.266  |
| Rebounds     | 0.815    | 0.670  | 0.169  |
| Assists      | 0.888    | 0.196  | 0.057  |
| Three-Points | 0.913    | 0.071  | 0.022  |

For detailed model analysis, feature importance, and performance characteristics, refer to [Model Performance Documentation](model_performance.md).

The models demonstrate strong predictive power across all statistics, with the Three-Pointers model showing the highest accuracy (91.3% variance explained). Each model utilizes specialized feature groups including base statistics, rolling averages, and contextual features.

Key improvements in this version:

- Optimized feature selection using gradient boosting importance
- Enhanced preprocessing pipeline with robust scaling
- Improved handling of categorical variables
- Reduced training time with optimized number of trials (20)
- Implementation of mean threshold for feature selection

## Prop Analysis System

### PropAnalyzer Class

- Loads and manages trained models
- Calculates prediction intervals and uncertainty
- Implements Kelly criterion for optimal bet sizing
- Considers historical performance for calibration
- Generates confidence scores based on multiple factors

### Edge Calculation

- Removes vigorish from odds
- Accounts for model uncertainty
- Considers line movements
- Incorporates historical accuracy
- Provides confidence scoring

## File Structure

```
src/
├── data/
│   ├── raw/          # Raw scraped data
│   ├── cleaned/      # Cleaned dataset
│   ├── features/     # Feature-engineered data
│   └── analysis/     # Prop analysis results
├── models/           # Trained models and metrics
├── scripts/
│   ├── data_collection/  # Data collection scripts
│   ├── preprocessing/    # Data processing scripts
│   ├── modeling/        # Model training scripts
│   └── analysis/       # Prop analysis scripts
└── docs/            # Documentation
```

## Dependencies

Core dependencies:

- pandas >= 2.2.3: Data manipulation
- scikit-learn: Machine learning and model training
- beautifulsoup4 >= 4.13.3: Web scraping
- requests >= 2.32.3: HTTP requests
- fake-useragent >= 1.4.0: User agent rotation
- optuna: Hyperparameter optimization
- numpy: Numerical computations

## Performance Optimization

- Efficient data processing through vectorized operations
- Optimized feature engineering pipeline
- Automated hyperparameter tuning with Optuna
- Time series cross-validation for robust evaluation
- Caching of intermediate results
- Retry mechanisms for robust data collection

## Output Files

### Models Directory (`/models/`)

- `{stat}_model_YYYYMMDD.joblib`: Trained models
- `{stat}_metrics_YYYYMMDD.json`: Performance metrics
- `{stat}_feature_names_YYYYMMDD.joblib`: Selected features
- `feature_groups.joblib`: Feature group definitions

### Data Directory (`/data/`)

- `/raw/`: Original scraped data
- `/cleaned/`: Preprocessed data
- `/features/`: Engineered features
- `/analysis/`: Prop analysis results and historical performance

## Error Handling

- Comprehensive logging throughout the pipeline
- Retry mechanism for failed HTTP requests
- Validation of data at each processing step
- Graceful handling of missing data
- Rate limiting detection and backoff
