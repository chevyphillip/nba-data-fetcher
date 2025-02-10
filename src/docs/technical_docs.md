# NBA Stats Predictor - Technical Documentation

## Architecture Overview

The NBA Stats Predictor is a machine learning system designed to predict various basketball statistics for NBA players. The system follows a modular pipeline architecture with clear separation of concerns:

1. **Data Collection**: Automated NBA statistics fetching with error handling
2. **Data Processing**: Comprehensive cleaning and feature engineering pipeline
3. **Model Training**: Gradient Boosting models with optimized feature selection
4. **Props Analysis**: Edge calculation and confidence scoring system
5. **Pipeline Orchestration**: Centralized pipeline management

## Pipeline Components

### Data Collection (`nba_historical_stats_fetcher.py`)

- Fetches historical NBA player statistics
- Implements robust error handling and logging
- Saves raw data with timestamps for versioning

### Data Processing

#### Cleaning (`clean_raw_data.py`)

- Validates data types and ranges
- Handles missing values through smart imputation
- Removes duplicates and inconsistencies
- Standardizes position encoding
- Implements comprehensive error logging

#### Feature Engineering (`feature_engineering.py`)

- Calculates rolling averages (5, 10, 15 game windows)
- Generates advanced basketball metrics
- Implements position-based features
- Adds team context features
- Normalizes features using appropriate scaling methods

### Model Training (`train_and_save_models.py`)

- Implements Gradient Boosting with hyperparameter optimization
- Uses Optuna for automated parameter tuning
- Performs feature selection with importance thresholds
- Implements time-series cross-validation
- Saves models with versioning

### Props Analysis

#### PropAnalyzer (`prop_analyzer.py`)

- Loads and manages trained models
- Calculates prediction intervals
- Implements Kelly criterion for bet sizing
- Provides confidence scoring
- Handles edge calculation with uncertainty

#### PropsAnalysisRunner (`run_odds_analysis.py`)

- Orchestrates the props analysis workflow
- Manages feature loading and prop fetching
- Coordinates analysis and result saving
- Implements comprehensive error handling
- Provides detailed logging

### Pipeline Orchestration (`run_pipeline.py`)

- Manages the complete analysis pipeline
- Creates necessary directory structure
- Coordinates component execution
- Implements error handling and logging
- Provides pipeline status updates

## File Structure

```
src/
├── data/
│   ├── raw/          # Raw NBA statistics
│   ├── features/     # Engineered features
│   └── analysis/     # Props analysis results
├── models/           # Trained models and metrics
├── scripts/
│   ├── data_collection/
│   │   └── nba_historical_stats_fetcher.py
│   ├── preprocessing/
│   │   ├── clean_raw_data.py
│   │   └── feature_engineering.py
│   ├── modeling/
│   │   └── train_and_save_models.py
│   ├── analysis/
│   │   └── prop_analyzer.py
│   ├── odds/
│   │   └── odds_api.py
│   └── run_pipeline.py
└── docs/
    ├── technical_docs.md
    └── model_performance.md
```

## Key Features

### Error Handling and Logging

- Comprehensive logging throughout the pipeline
- Detailed error messages and stack traces
- Status updates for long-running processes
- Pipeline execution validation

### Data Validation

- Input data validation at each step
- Feature consistency checks
- Model input validation
- Props data validation

### Performance Optimization

- Efficient data processing with pandas
- Optimized feature engineering pipeline
- Smart caching of intermediate results
- Parallel processing where applicable

### Model Management

- Version control for models
- Performance metrics tracking
- Feature importance analysis
- Model validation and testing

## Dependencies

Core dependencies and their purposes:

```
pandas>=2.2.3        # Data manipulation and analysis
scikit-learn>=1.4.0  # Machine learning algorithms
numpy>=1.26.4        # Numerical computations
joblib>=1.3.2        # Model serialization
optuna>=3.5.0        # Hyperparameter optimization
requests>=2.31.0     # HTTP requests for data fetching
python-dotenv>=1.0.1 # Environment variable management
```

## Configuration

The system uses environment variables for configuration:

```
ODDS_API_KEY=your_key_here  # API key for odds data
LOG_LEVEL=INFO              # Logging level
MODEL_DIR=src/models        # Model storage directory
```

## Error Handling Strategy

1. **Data Collection**
   - Retries for failed requests
   - Validation of received data
   - Logging of failed requests

2. **Data Processing**
   - Validation of input data
   - Handling of missing values
   - Logging of data quality issues

3. **Model Training**
   - Validation of training data
   - Monitoring of training metrics
   - Model performance validation

4. **Props Analysis**
   - Validation of input props
   - Handling of missing features
   - Logging of analysis issues

## Future Improvements

1. **Data Collection**
   - Add support for real-time updates
   - Implement more data sources
   - Add data quality metrics

2. **Feature Engineering**
   - Add more advanced metrics
   - Implement feature selection optimization
   - Add automated feature discovery

3. **Model Training**
   - Implement ensemble methods
   - Add online learning capabilities
   - Improve uncertainty estimation

4. **Props Analysis**
   - Add more sophisticated edge calculation
   - Implement portfolio optimization
   - Add real-time monitoring
