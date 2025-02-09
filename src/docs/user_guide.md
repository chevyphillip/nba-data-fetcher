# NBA Stats Predictor - User Guide

## Overview

The NBA Stats Predictor is a machine learning tool that predicts various basketball statistics for NBA players. This guide will help you understand how to use the system effectively.

## Setup

1. **Environment Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Directory Structure**
   - Ensure all required directories exist:
     ```bash
     mkdir -p src/data/{raw,cleaned,features}
     mkdir -p src/models
     ```

## Using the Predictor

### 1. Data Collection
```bash
cd src/scripts/data
python nba_historical_stats_fetcher.py
```

### 2. Data Processing
```bash
python clean_raw_data.py
python feature_engineering.py
```

### 3. Model Training
```bash
cd ../modeling
python train_models.py
```

### 4. Viewing Model Performance
```bash
cd ../visualization
python model_metrics.py
```

## Understanding the Results

### Model Performance Metrics

The system generates two main visualization plots:

1. **R² Score Comparison** (`model_r2_comparison.png`)
   - Shows how well each model predicts its target statistic
   - Scale: 0 to 1 (higher is better)
   - Current performance:
     - Points: 0.96 (96% accuracy)
     - Rebounds: 0.89 (89% accuracy)
     - Assists: 0.82 (82% accuracy)
     - 3-Pointers: 0.88 (88% accuracy)

2. **Error Metrics** (`model_error_comparison.png`)
   - Shows RMSE and MAE for each model
   - Lower values indicate better performance
   - Helps understand the typical prediction error range

### Interpreting Predictions

- Predictions are most accurate for points and rebounds
- Typical error ranges:
  - Points: ±2.31 points
  - Rebounds: ±1.89 rebounds
  - Assists: ±1.65 assists
  - 3-Pointers: ±0.85 threes

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Not Found**
   - Ensure all directories exist
   - Run data collection script first

3. **Visualization Errors**
   - Check if models are trained
   - Verify matplotlib/seaborn installation

## Best Practices

1. **Regular Updates**
   - Collect new data weekly
   - Retrain models monthly

2. **Data Validation**
   - Check raw data quality
   - Verify feature engineering output

3. **Performance Monitoring**
   - Review visualization plots regularly
   - Track model metrics over time

## Getting Help

- Check the technical documentation for detailed information
- Review the maintenance guide for system upkeep
- File issues on the repository for bugs or feature requests

## Quick Start

1. **Setup Environment**
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/nba-data-fetcher.git
   cd nba-data-fetcher

   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install -e .
   ```

2. **Run the Pipeline**

   a. **Collect Fresh Data**
   ```bash
   python src/scripts/data_collection/nba_historical_stats_fetcher.py
   ```
   This fetches the latest NBA player statistics.

   b. **Clean the Data**
   ```bash
   python src/scripts/preprocessing/clean_raw_data.py
   ```
   This removes duplicates and handles missing values.

   c. **Generate Features**
   ```bash
   python src/scripts/preprocessing/feature_engineering.py
   ```
   This creates advanced features for model training.

   d. **Train Models**
   ```bash
   python src/scripts/modeling/train_models.py
   ```
   This trains prediction models for all statistics.

## Understanding the Output

### Model Performance
After training, you'll find these files in the `models` directory:

1. **Model Files** (`{stat}_model_YYYYMMDD.joblib`)
   - Trained models that can be loaded for predictions
   - One file per statistic (PTS, TRB, AST, 3P)

2. **Metrics Files** (`{stat}_metrics_YYYYMMDD.json`)
   - Contains model performance metrics
   - Key metrics: RMSE, MAE, R²

### Current Model Performance

- **Points Model**: R² = 0.96
  - Most important features: FG attempts, games played, usage rate

- **Rebounds Model**: R² = 0.89
  - Most important features: Defensive rebounds, offensive rebounds, games played ratio

- **Assists Model**: R² = 0.82
  - Most important features: Assist ratio, minutes played, games played ratio

- **3-Pointers Model**: R² = 0.88
  - Most important features: Points scored, points per shot, position-relative stats

## Best Practices

1. **Data Collection**
   - Run collection during off-peak hours
   - Respect basketball-reference.com's rate limits
   - Verify raw data files after collection

2. **Data Processing**
   - Check cleaned data for missing values
   - Verify feature engineering outputs
   - Monitor data types and formats

3. **Model Training**
   - Review log output for warnings
   - Check feature importance rankings
   - Compare metrics across models

## Troubleshooting

### Common Issues

1. **Data Collection Fails**
   - Check internet connection
   - Verify basketball-reference.com is accessible
   - Try increasing delay between requests

2. **Data Processing Errors**
   - Ensure raw data files exist and are readable
   - Check for correct column names
   - Verify data types match expectations

3. **Model Training Issues**
   - Check for missing or invalid feature values
   - Verify sufficient memory is available
   - Review log files for specific error messages

### Getting Help

- Review the technical documentation in `docs/technical_docs.md`
- Check the codebase on GitHub
- Submit issues for bugs or feature requests
