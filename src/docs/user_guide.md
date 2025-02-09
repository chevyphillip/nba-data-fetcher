# NBA Stats Predictor - User Guide

## Overview

The NBA Stats Predictor is a machine learning tool that predicts various basketball statistics for NBA players using Gradient Boosting Regression. This guide will help you understand how to use the system effectively.

## Setup

1. **Environment Setup**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Directory Structure**

   ```bash
   mkdir -p src/data/{raw,cleaned,features,analysis}
   mkdir -p src/models
   ```

## Running the Pipeline

### 1. Data Collection

```bash
python src/scripts/data_collection/nba_historical_stats_fetcher.py
```

This script:

- Fetches NBA player statistics with retry mechanism
- Implements rate limiting and user agent rotation
- Saves raw data to `src/data/raw/`
- Logs collection progress and errors

### 2. Data Cleaning

```bash
python src/scripts/preprocessing/clean_raw_data.py
```

This script:

- Removes duplicates and invalid entries
- Handles missing values through imputation
- Standardizes column names and formats
- Validates data types and ranges
- Saves cleaned data to `src/data/cleaned/`

### 3. Feature Engineering

```bash
python src/scripts/preprocessing/feature_engineering.py
```

This script:

- Creates rolling averages (5, 10, 15-game windows)
- Calculates advanced efficiency metrics
- Generates position-specific features
- Adds matchup-based adjustments
- Saves engineered features to `src/data/features/`

### 4. Model Training

```bash
python src/scripts/modeling/train_and_save_models.py
```

This script:

- Trains Gradient Boosting models for each statistic
- Optimizes hyperparameters using Optuna
- Implements time series cross-validation
- Generates performance metrics and visualizations
- Saves models and metrics to `src/models/`

### 5. Prop Analysis

```bash
python src/scripts/analysis/analyze_props.py
```

This script:

- Analyzes betting propositions
- Calculates prediction intervals
- Determines optimal bet sizing
- Generates confidence scores
- Saves analysis to `src/data/analysis/`

## Understanding Results

### Model Performance

Current model performance metrics:

1. **Points (PTS)**
   - R² Score: 0.96 (96% accuracy)
   - RMSE: ±2.31 points
   - MAE: ±1.75 points

2. **Rebounds (TRB)**
   - R² Score: 0.89 (89% accuracy)
   - RMSE: ±1.89 rebounds
   - MAE: ±1.42 rebounds

3. **Assists (AST)**
   - R² Score: 0.82 (82% accuracy)
   - RMSE: ±1.65 assists
   - MAE: ±1.23 assists

4. **Three-Pointers (3P)**
   - R² Score: 0.88 (88% accuracy)
   - RMSE: ±0.85 threes
   - MAE: ±0.64 threes

### Output Files

1. **Models Directory** (`/models/`)
   - `{stat}_model_YYYYMMDD.joblib`: Trained models
   - `{stat}_metrics_YYYYMMDD.json`: Performance metrics
   - `{stat}_feature_names_YYYYMMDD.joblib`: Selected features
   - `feature_groups.joblib`: Feature group definitions

2. **Analysis Directory** (`/data/analysis/`)
   - Historical performance data
   - Prediction intervals
   - Confidence scores
   - Edge calculations

## Troubleshooting

### Common Issues

1. **Data Collection Failures**
   - Check internet connection
   - Verify website accessibility
   - Review rate limiting logs
   - Check retry mechanism logs

2. **Processing Errors**
   - Verify data file existence
   - Check file permissions
   - Monitor memory usage
   - Review error logs

3. **Model Training Issues**
   - Check feature completeness
   - Monitor memory consumption
   - Review training logs
   - Verify data quality

### Solutions

1. **Data Collection**

   ```bash
   # Check logs
   tail -f logs/data_collection.log
   
   # Retry with increased delay
   python src/scripts/data_collection/nba_historical_stats_fetcher.py --delay 5
   ```

2. **Processing**

   ```bash
   # Validate data
   python src/scripts/preprocessing/validate_data.py
   
   # Clean specific files
   python src/scripts/preprocessing/clean_raw_data.py --file specific_date.csv
   ```

3. **Model Training**

   ```bash
   # Train specific model
   python src/scripts/modeling/train_and_save_models.py --stat PTS
   
   # Review metrics
   python src/scripts/analysis/review_metrics.py
   ```

## Best Practices

1. **Data Collection**
   - Run during off-peak hours
   - Monitor rate limiting
   - Check data completeness
   - Review error logs

2. **Data Processing**
   - Validate input data
   - Check feature distributions
   - Monitor memory usage
   - Review output quality

3. **Model Training**
   - Review feature importance
   - Monitor convergence
   - Check performance metrics
   - Validate predictions

4. **Prop Analysis**
   - Consider uncertainty
   - Review confidence scores
   - Check historical accuracy
   - Monitor edge calculations

## Getting Help

1. **Documentation**
   - Review technical docs in `docs/technical_docs.md`
   - Check maintenance guide in `docs/maintenance_guide.md`
   - Read API documentation in `docs/api_docs.md`

2. **Support**
   - File issues on GitHub
   - Check existing issues
   - Review pull requests
   - Contact maintainers

3. **Updates**
   - Check for new releases
   - Review changelog
   - Update dependencies
   - Test new features
