import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_model_metrics():
    """Plot model performance metrics comparison."""
    today = datetime.now().strftime("%Y%m%d")
    stats = ['PTS', 'TRB', 'AST', '3P']
    metrics_data = []
    
    for stat in stats:
        with open(f"src/models/{stat}_metrics_{today}.json", 'r') as f:
            metrics = json.load(f)
            metrics_data.append({
                'Statistic': stat,
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot R² scores
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Statistic', y='R²')
    plt.title('Model Performance Comparison (R² Score)')
    plt.ylim(0, 1)  # R² is between 0 and 1
    for i, v in enumerate(metrics_df['R²']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig('src/models/model_r2_comparison.png')
    plt.close()
    
    # Plot Error Metrics
    plt.figure(figsize=(12, 6))
    metrics_long = pd.melt(metrics_df, 
                          id_vars=['Statistic'], 
                          value_vars=['RMSE', 'MAE'],
                          var_name='Metric',
                          value_name='Value')
    
    sns.barplot(data=metrics_long, x='Statistic', y='Value', hue='Metric')
    plt.title('Model Error Metrics Comparison')
    for i, v in enumerate(metrics_long['Value']):
        plt.text(i // 2, v, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('src/models/model_error_comparison.png')
    plt.close()

def plot_predictions_vs_actual():
    """Plot predicted vs actual values for each model."""
    today = datetime.now().strftime("%Y%m%d")
    stats = ['PTS', 'TRB', 'AST', '3P']
    
    # Load test data
    test_data = pd.read_csv('src/data/test/test_data_20250208.csv')
    
    for stat in stats:
        # Load model
        pipeline = joblib.load(f"src/models/{stat}_model_{today}.joblib")
        
        # Get predictions
        y_pred = pipeline.predict(test_data)
        y_true = test_data[stat]
        
        # Create scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(max(y_true), max(y_pred))
        plt.plot([0, max_val], [0, max_val], 'r--')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual {stat}')
        plt.tight_layout()
        plt.savefig(f'src/models/{stat}_predictions.png')
        plt.close()

def main():
    """Generate all visualizations."""
    os.makedirs('src/models', exist_ok=True)
    
    # Plot model metrics
    plot_model_metrics()
    
    # Plot predictions vs actual
    plot_predictions_vs_actual()
    
    print("Visualizations have been generated in the models directory!")

if __name__ == "__main__":
    main()
