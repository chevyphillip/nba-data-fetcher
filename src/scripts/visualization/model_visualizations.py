import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model_and_metrics(stat_name):
    """Load model and metrics for a given statistic."""
    today = datetime.now().strftime("%Y%m%d")
    model_path = f"src/models/{stat_name}_model_{today}.joblib"
    metrics_path = f"src/models/{stat_name}_metrics_{today}.json"
    
    model = joblib.load(model_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return model, metrics

def plot_model_comparison():
    """Plot performance comparison of all models."""
    stats = ['PTS', 'TRB', 'AST', '3P']
    metrics_data = []
    
    for stat in stats:
        _, metrics = load_model_and_metrics(stat)
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
    
    # Plot RMSE and MAE
    metrics_long = pd.melt(metrics_df, 
                          id_vars=['Statistic'], 
                          value_vars=['RMSE', 'MAE'],
                          var_name='Metric',
                          value_name='Value')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_long, x='Statistic', y='Value', hue='Metric')
    plt.title('Model Error Metrics Comparison')
    for i, v in enumerate(metrics_long['Value']):
        plt.text(i // 2, v, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig('src/models/model_error_comparison.png')
    plt.close()

def main():
    """Generate all visualizations."""
    # Ensure the models directory exists
    os.makedirs('src/models', exist_ok=True)
    
    # Create model comparison plots
    plot_model_comparison()
    
    print("Visualizations have been generated in the models directory!")

if __name__ == "__main__":
    main()
