# NBA Stats Predictor - Technical Documentation

## Architecture Overview

The NBA Stats Predictor is a machine learning system designed to predict various basketball statistics for NBA players. The system follows a modular pipeline architecture:

1. **Data Collection**: Web scraping of NBA statistics using BeautifulSoup4
2. **Data Cleaning**: Removal of duplicates and handling of missing values
3. **Feature Engineering**: Creation of advanced metrics and rolling averages
4. **Model Training**: Gradient Boosting Regression for multiple statistics
5. **Visualization**: Performance metrics and model insights
6. **Prediction**: Real-time predictions using trained models

## Data Pipeline

### Data Collection
- Uses BeautifulSoup4 for web scraping
- Implements rate limiting and user-agent rotation
- Stores raw data in CSV format

### Data Cleaning
- Removes duplicate entries
- Handles missing values through imputation
- Validates data types and ranges
- Output: Clean dataset ready for feature engineering

### Feature Engineering
- Creates rolling averages for key statistics
- Generates advanced metrics (efficiency, per-minute stats)
- Implements position-based features
- Output: Feature-rich dataset for model training

## Model Architecture

### Training Pipeline
- **Algorithm**: Gradient Boosting Regressor
- **Features**: Combination of raw stats and engineered features
- **Target Variables**: Points (PTS), Rebounds (TRB), Assists (AST), Three-Pointers (3P)
- **Preprocessing**: StandardScaler for feature normalization

### Model Performance

Current model performance metrics:

| Statistic | R² Score | RMSE | MAE |
|-----------|----------|------|-----|
| Points    | 0.96     | 2.31 | 1.75|
| Rebounds  | 0.89     | 1.89 | 1.42|
| Assists   | 0.82     | 1.65 | 1.23|
| 3-Pointers| 0.88     | 0.85 | 0.64|

## Visualization Components

### Performance Metrics
- R² Score comparison across models
- Error metrics (RMSE and MAE) visualization
- Located in `src/scripts/visualization/model_metrics.py`

### Generated Plots
1. `model_r2_comparison.png`: Bar plot of R² scores
2. `model_error_comparison.png`: Comparison of RMSE and MAE

## File Structure

```
src/
├── data/
│   ├── raw/          # Raw scraped data
│   ├── cleaned/      # Cleaned dataset
│   └── features/     # Feature-engineered data
├── models/           # Trained models and metrics
├── scripts/
│   ├── data/         # Data processing scripts
│   ├── modeling/     # Model training scripts
│   └── visualization/# Visualization scripts
└── docs/            # Documentation
```

## Dependencies

- pandas: Data manipulation
- scikit-learn: Machine learning
- matplotlib/seaborn: Visualization
- beautifulsoup4: Web scraping
- joblib: Model persistence

## Performance Optimization

- Efficient data processing through vectorized operations
- Optimized feature engineering pipeline
- Model hyperparameter tuning
- Caching of intermediate results

## Architecture Overview

The NBA Stats Predictor is organized into three main components:

### 1. Data Collection (`nba_historical_stats_fetcher.py`)
- Fetches historical NBA player statistics from basketball-reference.com
- Uses rotating user agents and request delays to respect the website's rate limits
- Saves raw data in CSV format to `src/data/raw/`

### 2. Data Processing Pipeline
#### 2.1 Data Cleaning (`clean_raw_data.py`)
- Removes duplicates and handles missing values
- Converts data types and standardizes formats
- Saves cleaned data to `src/data/cleaned/`

#### 2.2 Feature Engineering (`feature_engineering.py`)
- Creates advanced features for model training:
  - **Rolling Averages**: 5-game and 10-game rolling stats
  - **Position-Based**: Stats relative to position averages
  - **Efficiency Metrics**: Points per shot, assist ratio, etc.
  - **Season Context**: Games played ratio, minutes per game
- Saves engineered features to `src/data/features/`

### 3. Model Training (`train_models.py`)
- Uses scikit-learn's GradientBoostingRegressor
- Implements a robust pipeline with:
  - Missing value imputation
  - Feature scaling
  - Model training and evaluation
- Trains separate models for:
  - Points (PTS): R² = 0.96
  - Rebounds (TRB): R² = 0.89
  - Assists (AST): R² = 0.82
  - Three Pointers Made (3P): R² = 0.88

## Data Flow

1. **Raw Data Collection**
   ```
   basketball-reference.com → nba_historical_stats_fetcher.py → /data/raw/
   ```

2. **Data Processing**
   ```
   /data/raw/ → clean_raw_data.py → /data/cleaned/ → feature_engineering.py → /data/features/
   ```

3. **Model Training**
   ```
   /data/features/ → train_models.py → /models/
   ```

## Model Architecture

Each prediction model uses a GradientBoostingRegressor with:
- Mean imputation for missing values
- StandardScaler for feature normalization
- Hyperparameters optimized for each statistic

### Key Features by Importance
- **Points**: FG attempts, games played, usage rate
- **Rebounds**: Defensive rebounds, offensive rebounds, games played ratio
- **Assists**: Assist ratio, minutes played, games played ratio
- **3-Pointers**: Points scored, points per shot, position-relative stats

## Dependencies

Core dependencies:
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Model training and evaluation
- `numpy`: Numerical computations
- `beautifulsoup4`: Web scraping
- `fake-useragent`: Web scraping user agent rotation
- `joblib`: Model persistence

## Output Files

### Models Directory (`/models/`)
- `{stat}_model_YYYYMMDD.joblib`: Trained models
- `{stat}_metrics_YYYYMMDD.json`: Performance metrics

### Data Directory (`/data/`)
- `/raw/`: Original scraped data
- `/cleaned/`: Cleaned and preprocessed data
- `/features/`: Engineered features for model training
