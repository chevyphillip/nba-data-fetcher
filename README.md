# NBA Stats Predictor

A machine learning pipeline for predicting NBA player statistics using H2O AutoML. This system processes historical NBA data and generates predictions for points, rebounds, assists, and three-pointers using advanced feature engineering and time series cross-validation.

## Overview

The NBA Stats Predictor uses a sophisticated pipeline to:
1. Collect historical NBA player statistics from basketball-reference.com
2. Engineer advanced features including rolling averages and position-specific indicators
3. Train and evaluate models using H2O AutoML with time series cross-validation
4. Generate comprehensive performance metrics and visualizations

## Project Structure

```
nba-data-fetcher/
├── src/
│   ├── data/
│   │   ├── raw/          # Raw NBA statistics
│   │   └── processed/    # Enhanced feature sets
│   ├── docs/
│   │   ├── technical_docs.md    # Technical documentation
│   │   ├── user_guide.md        # User guide
│   │   └── maintenance_guide.md # Maintenance guide
│   ├── models/                  # Trained models and metrics
│   │   ├── {stat}_all_metrics.csv   # Performance metrics
│   │   ├── {stat}_cv_results.png    # CV visualizations
│   │   └── best_{stat}_model/       # Saved models
│   └── scripts/
│       ├── data_collection/
│       │   └── nba_historical_stats_fetcher.py  # Data collection
│       ├── preprocessing/
│       │   └── enhance_features.py              # Feature engineering
│       └── modeling/
│           └── train_multistat_models.py        # Model training
├── pyproject.toml    # Project dependencies
└── README.md        # Project documentation
```

## Features

### Data Collection
- Automated scraping from basketball-reference.com
- Rate-limited requests with rotating user agents
- Comprehensive player statistics from 2010-2024

### Feature Engineering
- 26 carefully selected features including:
  - Core player attributes (age, position, minutes played)
  - Rolling averages (5-game window)
  - Advanced metrics (TS%, eFG%, AST/TO ratio)
  - Position-specific indicators

### Model Training
- H2O AutoML for automated model selection
- Time series cross-validation
- Feature importance analysis
- Performance visualization

### Predictions
- Points per Game (PTS): RMSE ≈ 0.052
- Rebounds per Game (TRB)
- Assists per Game (AST)
- Three Pointers Made (3P): RMSE ≈ 0.30

## Requirements

- Python >= 3.11.11
- Dependencies:
  - beautifulsoup4 >= 4.13.3
  - pandas >= 2.2.3
  - requests >= 2.32.3
  - fake-useragent >= 1.4.0
  - lxml >= 5.1.0
  - h2o >= 3.46.0.6
  - scikit-learn
  - matplotlib
  - seaborn
  - numpy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nba-data-fetcher.git
cd nba-data-fetcher
```

2. Install dependencies:

```bash
pip install -e .
```

## Pipeline Execution Order

Follow these steps to run the complete pipeline:

### 1. Data Collection

```bash
python src/scripts/data_collection/nba_historical_stats_fetcher.py
```

This script:

- Fetches NBA player statistics from 2010 to 2024
- Collects multiple statistical categories (totals, per game, advanced, etc.)
- Saves raw data to `src/data/raw/`

### 2. Data Cleaning

```bash
python src/scripts/preprocessing/clean_nba_stats.py
```

This script:

- Selects relevant columns
- Handles missing values
- Standardizes column names
- Adds derived statistics
- Saves cleaned data to `src/data/processed/`

### 3. Feature Engineering

```bash
python src/scripts/preprocessing/enhance_features.py
```

This script:

- Creates rolling averages for key statistics (5-game window)
- Generates position-specific indicators
- Calculates advanced metrics (TS%, eFG%, AST/TO ratio)
- Produces a simplified, high-impact feature set (26 features)

### 4. Model Training

```bash
python src/scripts/modeling/train_multistat_models.py
```

This script trains models to predict multiple player statistics:

- Points per Game (PTS)
- Rebounds per Game (TRB)
- Assists per Game (AST)
- Three Pointers Made (3P)

Key features:
- Uses H2O AutoML for automated model selection and tuning
- Implements time series cross-validation for robust evaluation
- Generates feature importance analysis for each statistic
- Produces performance metrics and visualization plots
- Encodes categorical variables
- Saves preprocessed data to `src/data/processed/`

### 4. Feature Engineering

```bash
python src/scripts/preprocessing/feature_engineering.py
```

This script:

- Calculates rolling averages
- Adds opponent-specific adjustments
- Creates performance trends
- Generates season-level aggregations
- Saves engineered features to `src/data/processed/`

### 5. ML Data Preparation

```bash
python src/scripts/preprocessing/prepare_ml_data.py
```

This script:

- Performs feature selection
- Analyzes feature importance
- Splits data into train/validation/test sets
- Saves ML-ready datasets to `src/data/processed/`

### 6. Model Training

You can choose between two training approaches:

#### Traditional ML Models

```bash
python src/scripts/modeling/train_models.py
```

This trains:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- XGBoost
- SVR

#### H2O AutoML

```bash
python src/scripts/modeling/train_automl.py
```

This automatically:

- Tries multiple algorithms
- Performs hyperparameter tuning
- Selects the best model
- Saves model artifacts to `src/models/`

### 7. Visualization

```bash
python src/scripts/visualization/visualize_results.py
```

This creates:

- Model performance comparisons
- Feature importance plots
- Prediction error analysis
- Player performance trends

## Data Description

The collected and processed data includes:

### Raw Data Features

- Basic statistics (points, rebounds, assists, etc.)
- Advanced metrics (true shooting %, efficiency, etc.)
- Per game and per minute statistics
- Shooting percentages and distributions

### Engineered Features

- Rolling averages over different game windows
- Performance trends and momentum indicators
- Opponent-adjusted statistics
- Season-level aggregations

### Target Variable

- Points (PTS) - The primary prediction target

## Model Outputs

The models generate several outputs in `src/models/`:

- Trained model files
- Performance metrics (RMSE, R², MAE)
- Feature importance rankings
- Predictions on test set
- Model comparison results

## Visualization Outputs

The visualization script generates plots in `src/plots/`:

- Model performance comparisons
- Feature importance charts
- Prediction accuracy plots
- Error distribution analysis
- Player trend visualizations

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Your chosen license]

## Acknowledgments

- Data sourced from [Basketball Reference](https://www.basketball-reference.com/)
- Built with Python, pandas, scikit-learn, and H2O AutoML
