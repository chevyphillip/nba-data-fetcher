# NBA Data Fetcher and Predictor

A machine learning system for NBA player statistics prediction and prop bet analysis.

## Project Overview

This project implements a complete pipeline for:

1. Collecting NBA player statistics
2. Processing and engineering features
3. Training prediction models
4. Analyzing betting propositions
5. Visualizing results and insights

## Project Structure

```
nba-data-fetcher/
├── src/
│   ├── data/
│   │   ├── raw/              # Raw NBA statistics data
│   │   ├── features/         # Feature-engineered datasets
│   │   └── analysis/         # Prop analysis results
│   │
│   ├── models/              # Trained models and metrics
│   │   ├── {stat}_model_YYYYMMDD.joblib
│   │   ├── {stat}_metrics_YYYYMMDD.json
│   │   └── feature_groups.joblib
│   │
│   ├── docs/               # Documentation
│   │   ├── technical_docs.md
│   │   └── model_performance.md
│   │
│   └── scripts/
│       ├── data_collection/
│       │   └── nba_historical_stats_fetcher.py
│       │
│       ├── preprocessing/
│       │   └── feature_engineering.py
│       │
│       ├── modeling/
│       │   └── train_and_save_models.py
│       │
│       ├── analysis/
│       │   └── prop_analyzer.py
│       │
│       ├── odds/
│       │   └── odds_api.py
│       │
│       └── run_pipeline.py
│
├── requirements.txt        # Python dependencies
└── .env                   # Environment variables (not in repo)
```

## Model Performance

Current model performance metrics (as of February 9, 2025):

| Statistic     | R² Score | RMSE  | MAE    |
|--------------|----------|--------|--------|
| Points       | 0.710    | 1.903  | 0.279  |
| Rebounds     | 0.680    | 1.154  | 0.230  |
| Assists      | 0.886    | 0.197  | 0.054  |
| Three-Points | 0.189    | 0.221  | 0.056  |

For detailed model analysis, see [Model Performance Documentation](src/docs/model_performance.md).

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nba-data-fetcher.git
cd nba-data-fetcher
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a .env file with required API keys:

```
ODDS_API_KEY=your_key_here
```

## Usage

1. Run the complete pipeline:

```bash
python src/scripts/run_pipeline.py
```

2. Run individual components:

```bash
# Data collection
python src/scripts/data_collection/nba_historical_stats_fetcher.py

# Data cleaning
python src/scripts/preprocessing/clean_raw_data.py

# Feature engineering
python src/scripts/preprocessing/feature_engineering.py

# Model training
python src/scripts/modeling/train_and_save_models.py

# Prop analysis
python src/scripts/analysis/prop_analyzer.py
```

## Key Features

- Comprehensive NBA statistics collection
- Advanced feature engineering
- Gradient Boosting models with optimized feature selection
- Time series cross-validation
- Prop bet edge analysis
- Performance visualization tools

## Recent Improvements

- Optimized feature selection using gradient boosting importance
- Enhanced preprocessing pipeline with robust scaling
- Improved handling of categorical variables
- Reduced training time with optimized number of trials (20)
- Implementation of mean threshold for feature selection

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
